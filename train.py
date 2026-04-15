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
   Pre-converted PNG images and numpy masks are loaded into RAM at startup.
   Convert from TFRecords first: python convert_tfrecords.py

================================================================================

Usage:
    python train.py --data_dir CLEVR_64       # full run with defaults
    python train.py --total_steps 200         # quick smoke-test
    python train.py --model baseline          # train baseline SA instead
    python train.py --resume checkpoints/best.eqx
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from model import SlotODEModel
from model_baseline import SlotAttentionModel
from evaluate import compute_ari_fg


# ---------------------------------------------------------------------------
# Dataset — preloaded PNGs + numpy masks
# ---------------------------------------------------------------------------

class Dataset:
    """Loads pre-converted PNGs + npy masks into RAM for fast training.

    Expects output from convert_tfrecords.py:
        data_dir/images/{split}/*.png     [H, W, 3] uint8
        data_dir/masks/{split}/*.npy      [11, H, W] uint8
        data_dir/visibility/{split}/*.npy [11] float32

    Images are normalized to [-1, 1] NCHW at load time.
    """
    def __init__(self, data_dir, split, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        img_dir = Path(data_dir) / 'images' / split
        mask_dir = Path(data_dir) / 'masks' / split
        vis_dir = Path(data_dir) / 'visibility' / split

        self.img_files = sorted(img_dir.glob('*.png'))
        self.mask_files = sorted(mask_dir.glob('*.npy'))
        self.vis_files = sorted(vis_dir.glob('*.npy'))
        self.n = len(self.img_files)
        assert self.n > 0, f"No images found in {img_dir}"
        assert len(self.mask_files) == self.n, "Image/mask count mismatch"

        # determine resolution from first image
        sample = np.array(Image.open(self.img_files[0]))
        H, W = sample.shape[:2]

        # preload everything into RAM
        print(f"  Loading {split}: {self.n} images ({H}x{W})...")
        self.images = np.zeros((self.n, 3, H, W), dtype=np.float32)
        self.masks = np.zeros((self.n, 11, H, W), dtype=np.uint8)
        self.visibility = np.zeros((self.n, 11), dtype=np.float32)
        for i in range(self.n):
            img = np.array(Image.open(self.img_files[i]))  # [H, W, 3] uint8
            self.images[i] = img.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
            self.masks[i] = np.load(self.mask_files[i])
            self.visibility[i] = np.load(self.vis_files[i])
            if (i + 1) % 10000 == 0:
                print(f"    {i + 1}/{self.n}")
        print(f"  Done. ~{self.images.nbytes / 1e9:.1f} GB in RAM")

    def __iter__(self):
        idx = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, self.n - self.batch_size + 1, self.batch_size):
            sl = idx[start:start + self.batch_size]
            yield self.images[sl], self.masks[sl], self.visibility[sl]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def mse_loss(recon: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error, averaged over all elements."""
    return jnp.mean((recon - target) ** 2)


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

def _cast_tree(tree, dtype):
    """Cast all float arrays in a pytree to dtype."""
    def _cast(x):
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x
    return jax.tree.map(_cast, tree)

_USE_FP16 = jax.devices()[0].platform == "gpu"

@eqx.filter_jit
def train_step(model, opt_state, optimizer, images, key):
    """Single training step. Returns (model, opt_state, loss)."""

    def loss_fn(model):
        # mixed precision: cast model + images to fp16 for forward pass
        if _USE_FP16:
            params, static = eqx.partition(model, eqx.is_array)
            params = jax.tree.map(
                lambda x: x.astype(jnp.float16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                params
            )
            model = eqx.combine(params, static)
            images_fp = images.astype(jnp.float16)
        else:
            images_fp = images
        recon, masks, slots = model(images_fp, key=key)
        loss = mse_loss(recon, images_fp)
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    # cast grads back to float32 for stable optimizer update
    if _USE_FP16:
        grads = _cast_tree(grads, jnp.float32)

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
    """Compute MSE loss + predicted masks for a single batch."""
    recon, masks, slots = model(images, key=key)
    return mse_loss(recon, images), masks


def eval_metrics(model, val_ds, key, max_batches=50):
    """Average validation loss and ARI-FG over up to max_batches."""
    total_loss, total_ari, n_imgs = 0.0, 0.0, 0
    count = 0
    for imgs, gt_masks, _ in val_ds:
        if count >= max_batches:
            break
        key, subkey = jax.random.split(key)
        loss, pred_masks = eval_step(model, jnp.array(imgs), subkey)
        total_loss += float(loss)
        count += 1

        pred_masks_np = np.array(pred_masks)
        for i in range(pred_masks_np.shape[0]):
            pred_seg = pred_masks_np[i].argmax(axis=0)
            gt_seg = gt_masks[i].argmax(axis=0) if gt_masks[i].ndim == 3 else gt_masks[i]
            total_ari += compute_ari_fg(pred_seg, gt_seg)
            n_imgs += 1

    avg_loss = total_loss / max(1, count)
    avg_ari = total_ari / max(1, n_imgs)
    return avg_loss, avg_ari


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

def _gcs_sync(local_path: str, gcs_dir: str):
    """Upload a file to GCS. Logs warning on failure."""
    import subprocess
    dst = f"{gcs_dir.rstrip('/')}/{os.path.basename(local_path)}"
    try:
        subprocess.run(["gsutil", "-q", "cp", local_path, dst],
                       check=True, timeout=120)
    except Exception as e:
        print(f"  WARNING: GCS sync failed for {local_path}: {e}")


def save_checkpoint(path: str, model, opt_state, global_step: int,
                    best_val_loss: float, args, gcs_ckpt: str = None):
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

    if gcs_ckpt:
        _gcs_sync(path, gcs_ckpt)
        _gcs_sync(meta_path, gcs_ckpt)


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
    p.add_argument("--data_dir",       default="CLEVR_64",
                   help="Path to pre-converted dataset (from convert_tfrecords.py)")
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
    p.add_argument("--dt",             type=float, default=None,
                   help="ODE integration step size (default: 1.0)")
    p.add_argument("--log_every",      type=int,   default=100)
    p.add_argument("--val_every",      type=int,   default=5_000)
    p.add_argument("--img_every",      type=int,   default=5_000)
    p.add_argument("--ckpt_every",     type=int,   default=10_000)
    p.add_argument("--ckpt_dir",       default="checkpoints")
    p.add_argument("--experiment",     default="slot_ode_jax")
    p.add_argument("--run_name",       default=None)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--resume",         default=None,
                   help="Path to a .eqx checkpoint to resume from")
    p.add_argument("--gcs_ckpt",       default=None,
                   help="GCS path to sync checkpoints (e.g. gs://omkos-slotode/checkpoints)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    print(f"JAX devices: {jax.devices()}")

    # ---- PRNG setup -------------------------------------------------------
    key = jax.random.key(args.seed)
    key, model_key = jax.random.split(key)

    # ---- datasets ---------------------------------------------------------
    res = args.resolution
    train_ds = Dataset(args.data_dir, 'train', args.batch_size, shuffle=True)
    val_ds = Dataset(args.data_dir, 'val', args.batch_size, shuffle=False)

    # ---- model ------------------------------------------------------------
    if args.model == "slot_ode":
        dt0 = args.dt if args.dt is not None else 1.0
        model = SlotODEModel(
            resolution=(res, res), num_slots=args.num_slots,
            slot_dim=args.slot_dim, enc_hidden_dim=args.enc_hidden_dim,
            num_iter=args.num_iter, dt0=dt0,
            key=model_key
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
        optax.adam(schedule, b2=0.95),
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
    for imgs, _, _ in train_ds:
        img_batch = jnp.array(imgs[:4])
        break

    # ---- MLflow -----------------------------------------------------------
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(vars(args))

        done = False
        _t0 = time.time()
        while not done:
            for imgs, _, _ in train_ds:
                imgs_jax = jnp.array(imgs)

                key, subkey = jax.random.split(key)

                model, opt_state, loss = train_step(
                    model, opt_state, optimizer, imgs_jax, subkey
                )
                global_step += 1
                loss_val = float(loss)

                # ---- logging ----------------------------------------------
                if global_step % args.log_every == 0:
                    _elapsed = time.time() - _t0
                    _sps = _elapsed / args.log_every
                    _t0 = time.time()
                    current_lr = float(schedule(global_step))
                    print(f"[step {global_step:>7d}]  loss={loss_val:.5f}  lr={current_lr:.2e}  {_sps:.3f}s/step")
                    mlflow.log_metrics(
                        {"train/loss": loss_val, "train/lr": current_lr},
                        step=global_step,
                    )

                # ---- validation -------------------------------------------
                if global_step % args.val_every == 0:
                    key, val_key = jax.random.split(key)
                    val_loss_val, val_ari = eval_metrics(model, val_ds, val_key, max_batches=50)
                    mlflow.log_metrics(
                        {"val/loss": val_loss_val, "val/ari_fg": val_ari},
                        step=global_step,
                    )
                    print(f"[step {global_step:>7d}]  val_loss={val_loss_val:.5f}  ARI-FG={val_ari:.4f}"
                          f"  best={best_val_loss:.5f}")
                    if val_loss_val < best_val_loss:
                        best_val_loss = val_loss_val
                        save_checkpoint(
                            str(ckpt_dir / "best.eqx"),
                            model, opt_state,
                            global_step, best_val_loss, args,
                            gcs_ckpt=args.gcs_ckpt,
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
                        gcs_ckpt=args.gcs_ckpt,
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