"""
train.py — JAX / Equinox / Optax training for SlotODEModel on CLEVR-with-masks.

================================================================================
JAX TRAINING LOOP PRIMER (read if you're coming from PyTorch)
================================================================================

In PyTorch, training mutates state in-place:
    loss.backward()        # accumulates gradients on .grad attributes
    optimizer.step()       # mutates parameters in-place
    model.train()          # sets mutable training flag

In JAX, everything is functional — no mutation:

1. COMPUTING GRADIENTS:
   `eqx.filter_grad(loss_fn)(model, batch)` returns a pytree of gradients
   with the same structure as `model`. Only array leaves get gradients;
   non-array fields (ints, bools, strings) are ignored.

2. OPTIMIZER STATE:
   optax optimizers are stateless transforms. You create one:
       optimizer = optax.adam(lr)
       opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
   Then each step returns NEW state (no mutation):
       updates, opt_state = optimizer.update(grads, opt_state, params)
       model = eqx.apply_updates(model, updates)

3. JIT COMPILATION:
   `eqx.filter_jit` compiles a function, tracing arrays and treating
   non-arrays as static. First call is slow (compilation), subsequent
   calls are fast. The compiled function must be pure (no side effects).

4. PRNG KEYS:
   Every stochastic operation needs an explicit key. We split keys:
       key, subkey = jax.random.split(key)
       slots = initialize_slots(subkey)
   This ensures reproducibility even under JIT.

5. DATA LOADING:
   We use tensorflow.data to read directly from CLEVR-with-masks TFRecords.
   tf.data handles decompression, parsing, resizing, shuffling, and batching
   in a pipelined fashion. Tensors are converted to JAX arrays per batch.

================================================================================
OPTAX SCHEDULE NOTES
================================================================================

optax.warmup_cosine_decay_schedule combines warmup + cosine decay:
  - Linearly ramps lr from 0 to peak over `warmup_steps`
  - Then cosine-decays to 0 over remaining steps
This replaces PyTorch's LambdaLR + manual cosine function.

================================================================================

Usage:
    python train.py                       # full run with defaults
    python train.py --total_steps 200     # quick smoke-test
    python train.py --model baseline      # train baseline SA instead
    python train.py --resume checkpoints/best.eqx
"""

import argparse
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from model import SlotODEModel
from model_baseline import SlotAttentionModel


# ---------------------------------------------------------------------------
# Dataset — TFRecords via tf.data
# ---------------------------------------------------------------------------

def build_tfrecord_dataset(tfrecords_path, resolution, batch_size, shuffle=True,
                           shuffle_buffer=10000, skip=0, take=-1):
    """Build a tf.data pipeline reading from CLEVR-with-masks TFRecords.

    Returns batches of (images, masks, visibility):
      images: [B, 3, H, W] float32 in [0, 1]
      masks:  [B, 11, H, W] uint8 binary {0, 255}
      visibility: [B, 11] float32
    """
    import tensorflow as tf

    def parse_fn(raw):
        features = tf.io.parse_single_example(raw, {
            'image': tf.io.VarLenFeature(tf.string),
            'mask': tf.io.VarLenFeature(tf.string),
            'visibility': tf.io.FixedLenFeature([11], tf.float32),
        })

        # image: join single-byte entries -> [240, 320, 3] -> resize -> CHW
        img_str = tf.strings.reduce_join(
            tf.sparse.to_dense(features['image'], default_value=b'\x00'))
        image = tf.io.decode_raw(img_str, tf.uint8)
        image = tf.cast(tf.reshape(image, [240, 320, 3]), tf.float32) / 127.5 - 1.0
        image = tf.image.resize(image, [resolution, resolution])
        image = tf.transpose(image, [2, 0, 1])  # [3, H, W]

        # mask: join -> [11, 240, 320] -> resize nearest -> [11, H, W]
        mask_str = tf.strings.reduce_join(
            tf.sparse.to_dense(features['mask'], default_value=b'\x00'))
        mask = tf.io.decode_raw(mask_str, tf.uint8)
        mask = tf.reshape(mask, [11, 240, 320])
        mask_t = tf.transpose(mask, [1, 2, 0])  # [H, W, 11]
        mask_t = tf.image.resize(mask_t, [resolution, resolution], method='nearest')
        mask_t = tf.transpose(mask_t, [2, 0, 1])  # [11, H, W]
        mask_t = tf.cast(mask_t, tf.uint8)

        return image, mask_t, features['visibility']

    ds = tf.data.TFRecordDataset(tfrecords_path, compression_type='GZIP')
    if skip > 0:
        ds = ds.skip(skip)
    if take > 0:
        ds = ds.take(take)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def count_tfrecords(tfrecords_path):
    """Count total records in a TFRecords file."""
    import tensorflow as tf
    ds = tf.data.TFRecordDataset(tfrecords_path, compression_type='GZIP')
    count = 0
    for _ in ds:
        count += 1
    return count


def tf_to_jax(tensor):
    """Convert TF tensor to JAX array (via numpy, zero-copy for CPU)."""
    return jnp.array(tensor.numpy())


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def mse_loss(recon: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error, averaged over all elements."""
    return jnp.mean((recon - target) ** 2)


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step(model, opt_state, optimizer, images, key):
    """Single training step. Returns (model, opt_state, loss).

    eqx.filter_jit automatically traces array leaves and treats
    non-array fields (like num_slots, resolution) as static constants.

    eqx.filter_grad differentiates only w.r.t. array leaves of `model`.
    """

    def loss_fn(model):
        recon, masks, slots = model(images, key=key)
        loss = mse_loss(recon, images)
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@eqx.filter_jit
def eval_step(model, images, key):
    """Compute MSE loss for a single batch (no gradients)."""
    recon, masks, slots = model(images, key=key)
    return mse_loss(recon, images)


def eval_loss(model, val_ds, key, max_batches=50):
    """Average validation loss over up to max_batches."""
    total, count = 0.0, 0
    for batch in val_ds:
        if count >= max_batches:
            break
        imgs, _, _ = batch
        imgs_jax = tf_to_jax(imgs)
        key, subkey = jax.random.split(key)
        loss = eval_step(model, imgs_jax, subkey)
        total += float(loss)
        count += 1
    return total / max(1, count)


# ---------------------------------------------------------------------------
# Image grid logging
# ---------------------------------------------------------------------------

def log_image_grid(model, imgs_jax, key, step):
    """Log input/reconstruction/mask grid to MLflow."""
    recon, masks, slots = model(imgs_jax, key=key)

    imgs_np = jax.device_get(imgs_jax)
    recon_np = jax.device_get(recon)
    masks_np = jax.device_get(masks)

    B = imgs_np.shape[0]
    N_slots = masks_np.shape[1]
    n_rows = 2 + N_slots

    fig, axes = plt.subplots(n_rows, B, figsize=(3 * B, 3 * n_rows))
    if B == 1:
        axes = axes[:, None]

    def to_img(t):
        return ((t.transpose(1, 2, 0) + 1) / 2).clip(0, 1)

    for j in range(B):
        axes[0, j].imshow(to_img(imgs_np[j]))
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("Input", fontsize=8)

        axes[1, j].imshow(to_img(recon_np[j]))
        axes[1, j].axis("off")
        if j == 0:
            axes[1, j].set_ylabel("Recon", fontsize=8)

        for s in range(N_slots):
            axes[2 + s, j].imshow(masks_np[j, s].clip(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[2 + s, j].axis("off")
            if j == 0:
                axes[2 + s, j].set_ylabel(f"Slot {s}", fontsize=8)

    plt.suptitle(f"Step {step}", fontsize=10)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        fig.savefig(tmp_path, dpi=80, bbox_inches="tight")
        mlflow.log_artifact(tmp_path, artifact_path="images")
    finally:
        plt.close(fig)
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model, opt_state, global_step: int,
                    best_val_loss: float, args):
    """Save model + optimizer state using equinox serialization."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    eqx.tree_serialise_leaves(path, model)

    import pickle
    meta_path = path.replace('.eqx', '_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'opt_state': jax.device_get(opt_state),
            'step': global_step,
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }, f)


def load_checkpoint(path: str, model, optimizer):
    """Load model + optimizer state. Returns (model, opt_state, step, best_val_loss)."""
    import pickle

    model = eqx.tree_deserialise_leaves(path, model)

    meta_path = path.replace('.eqx', '_meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    opt_state = meta['opt_state']
    opt_state = jax.tree.map(
        lambda x: jnp.array(x) if hasattr(x, 'shape') else x,
        opt_state
    )

    return model, opt_state, meta['step'], meta['best_val_loss']


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Slot Attention models on CLEVR-with-masks (JAX)")
    p.add_argument("--model",          default="slot_ode", choices=["slot_ode", "baseline"])
    p.add_argument("--solver",         default="tsit5", choices=["euler", "tsit5", "dopri5"],
                   help="ODE solver for slot_ode model (ignored for baseline)")
    p.add_argument("--tfrecords",      default="clevr_with_masks_clevr_with_masks_train.tfrecords",
                   help="Path to CLEVR-with-masks TFRecords file")
    p.add_argument("--val_size",       type=int,   default=5000,
                   help="Number of images held out for validation (last N records)")
    p.add_argument("--resolution",     type=int,   default=64,
                   help="Image resolution (both H and W)")
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=4e-4)
    p.add_argument("--warmup_steps",   type=int,   default=10_000)
    p.add_argument("--decay_steps",    type=int,   default=100_000)
    p.add_argument("--decay_rate",     type=float, default=0.5)
    p.add_argument("--total_steps",    type=int,   default=500_000)
    p.add_argument("--num_slots",      type=int,   default=7)
    p.add_argument("--slot_dim",       type=int,   default=64)
    p.add_argument("--enc_hidden_dim", type=int,   default=64)
    p.add_argument("--num_iter",       type=int,   default=3)
    p.add_argument("--log_every",      type=int,   default=100)
    p.add_argument("--val_every",      type=int,   default=5_000)
    p.add_argument("--img_every",      type=int,   default=5_000)
    p.add_argument("--ckpt_every",     type=int,   default=10_000)
    p.add_argument("--ckpt_dir",       default="checkpoints")
    p.add_argument("--experiment",     default="slot_ode_jax")
    p.add_argument("--run_name",       default=None)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--shuffle_buffer", type=int,   default=10000)
    p.add_argument("--resume",         default=None,
                   help="Path to a .eqx checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    import tensorflow as tf
    # suppress TF warnings, we only use it for data loading
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print(f"JAX devices: {jax.devices()}")

    # ---- PRNG setup -------------------------------------------------------
    key = jax.random.key(args.seed)
    key, model_key = jax.random.split(key)

    # ---- count dataset & split --------------------------------------------
    print(f"Counting records in {args.tfrecords}...")
    total = count_tfrecords(args.tfrecords)
    train_size = total - args.val_size
    print(f"Total: {total}  Train: {train_size}  Val: {args.val_size}")

    # ---- datasets ---------------------------------------------------------
    res = args.resolution
    train_ds = build_tfrecord_dataset(
        args.tfrecords, res, args.batch_size,
        shuffle=True, shuffle_buffer=args.shuffle_buffer,
        take=train_size,
    )
    val_ds = build_tfrecord_dataset(
        args.tfrecords, res, args.batch_size,
        shuffle=False,
        skip=train_size, take=args.val_size,
    )

    # ---- model ------------------------------------------------------------
    if args.model == "slot_ode":
        model = SlotODEModel(
            resolution=(res, res), num_slots=args.num_slots,
            slot_dim=args.slot_dim, enc_hidden_dim=args.enc_hidden_dim,
            num_iter=args.num_iter, solver=args.solver, key=model_key
        )
    else:
        model = SlotAttentionModel(
            resolution=(res, res), num_slots=args.num_slots,
            slot_dim=args.slot_dim, enc_hidden_dim=args.enc_hidden_dim,
            num_iter=args.num_iter, key=model_key
        )

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {args.model}  |  Parameters: {n_params:,}")

    # ---- optimizer --------------------------------------------------------
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, args.lr, args.warmup_steps),
         optax.exponential_decay(
             init_value=args.lr,
             transition_steps=args.decay_steps,
             decay_rate=args.decay_rate,
         )],
        boundaries=[args.warmup_steps],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(schedule),
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ---- state ------------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")

    # ---- resume -----------------------------------------------------------
    if args.resume is not None:
        model, opt_state, global_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer
        )
        print(f"Resumed from {args.resume} (step {global_step})")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- fixed batch for visualization ------------------------------------
    img_batch = None
    for batch in train_ds.take(1):
        imgs, _, _ = batch
        img_batch = tf_to_jax(imgs)[:4]
        break

    # ---- MLflow -----------------------------------------------------------
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(vars(args))

        done = False
        while not done:
            for batch in train_ds:
                imgs, _, _ = batch
                imgs_jax = tf_to_jax(imgs)

                key, subkey = jax.random.split(key)

                model, opt_state, loss = train_step(
                    model, opt_state, optimizer, imgs_jax, subkey
                )
                global_step += 1
                loss_val = float(loss)

                # ---- logging ----------------------------------------------
                if global_step % args.log_every == 0:
                    current_lr = float(schedule(global_step))
                    print(f"[step {global_step:>7d}]  train_loss={loss_val:.5f}  lr={current_lr:.2e}")
                    mlflow.log_metrics(
                        {"train/loss": loss_val, "train/lr": current_lr},
                        step=global_step,
                    )

                # ---- validation -------------------------------------------
                if global_step % args.val_every == 0:
                    key, val_key = jax.random.split(key)
                    val_loss_val = eval_loss(model, val_ds, val_key, max_batches=50)
                    mlflow.log_metric("val/loss", val_loss_val, step=global_step)
                    print(f"[step {global_step:>7d}]  val_loss={val_loss_val:.5f}"
                          f"  best={best_val_loss:.5f}")
                    if val_loss_val < best_val_loss:
                        best_val_loss = val_loss_val
                        save_checkpoint(
                            str(ckpt_dir / "best.eqx"),
                            model, opt_state,
                            global_step, best_val_loss, args,
                        )
                        print(f"  -> new best saved")

                # ---- image grid -------------------------------------------
                if global_step % args.img_every == 0:
                    key, img_key = jax.random.split(key)
                    log_image_grid(model, img_batch, img_key, global_step)

                # ---- periodic checkpoint ----------------------------------
                if global_step % args.ckpt_every == 0:
                    save_checkpoint(
                        str(ckpt_dir / f"step_{global_step:07d}.eqx"),
                        model, opt_state,
                        global_step, best_val_loss, args,
                    )
                    print(f"[step {global_step:>7d}]  checkpoint saved")

                # ---- termination ------------------------------------------
                if global_step >= args.total_steps:
                    done = True
                    break

        print(f"Training complete. Total steps: {global_step}  "
              f"Best val loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
