"""
Evaluate a single trained SlotODE checkpoint at multiple test-time dt values.

Tests whether the learned update is a genuine vector field (curves collapse
across dt at matched T) or a step-count-dependent recurrence.

Usage:
    python dt_sweep.py --ckpt runs_data/omkos-slotode/slot_ode_11_slots_euler_T3_dt1/checkpoints/best.eqx \
                       --tfrecords clevr_with_masks_clevr_with_masks_train.tfrecords \
                       --num_samples 200
"""
import argparse
import pickle

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from model import SlotODEModel
from evaluate import (
    iter_tfrecords, preprocess_image, preprocess_mask,
    compute_ari_fg, compute_miou,
)


def build_model(ckpt_path, dt0_override, resolution):
    meta_path = ckpt_path.replace(".eqx", "_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    a = meta["args"]
    key = jax.random.key(0)
    model = SlotODEModel(
        resolution=resolution,
        num_slots=a["num_slots"],
        slot_dim=a["slot_dim"],
        enc_hidden_dim=a.get("enc_hidden_dim", 64),
        num_iter=a["num_iter"],
        dt0=dt0_override,
        key=key,
    )
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    return model, a, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tfrecords", required=True)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--dts", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.1])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    resolution = (args.resolution, args.resolution)

    # Load images/gt once, reuse across dt sweep
    import tensorflow as tf
    ds = tf.data.TFRecordDataset(args.tfrecords, compression_type="GZIP")
    total = sum(1 for _ in ds)
    val_offset = total - args.val_size
    print(f"Total records: {total}, val offset: {val_offset}")

    print(f"Loading {args.num_samples} val samples...")
    imgs, segs = [], []
    for img_np, seg_np in iter_tfrecords(args.tfrecords, args.num_samples, val_offset=val_offset):
        imgs.append(preprocess_image(img_np, resolution))
        segs.append(preprocess_mask(seg_np, resolution))
    imgs_arr = np.stack(imgs)  # [N, 3, H, W]
    print(f"Loaded {imgs_arr.shape}")

    # Peek meta for training dt
    _, ckpt_args, meta = build_model(args.ckpt, dt0_override=1.0, resolution=resolution)
    train_dt = ckpt_args.get("dt", 1.0)
    T = float(ckpt_args["num_iter"])
    num_slots = ckpt_args["num_slots"]
    print(f"Checkpoint: T={T}, train dt={train_dt}, step {meta['step']}")

    @eqx.filter_jit
    def predict(model, batch, key):
        recon, masks, _ = model(batch, key=key)
        return recon, masks

    print(f"\n{'dt':>6} {'n_steps':>8} {'ARI-FG':>10} {'mIoU':>10} {'MSE':>10}")
    print("-" * 50)

    results = {}
    for dt in args.dts:
        model, _, _ = build_model(args.ckpt, dt0_override=dt, resolution=resolution)

        ari_scores, miou_scores, mse_scores = [], [], []
        key = jax.random.key(args.seed)
        B = args.batch_size
        for start in range(0, len(imgs_arr), B):
            batch = jnp.array(imgs_arr[start:start + B])
            key, sk = jax.random.split(key)
            recon, masks = predict(model, batch, sk)
            recon = np.array(recon); masks = np.array(masks)
            for i in range(batch.shape[0]):
                pred_mask = masks[i].argmax(axis=0)
                gt = segs[start + i]
                ari_scores.append(compute_ari_fg(pred_mask, gt))
                miou_scores.append(compute_miou(pred_mask, gt, num_slots))
                mse_scores.append(float(((recon[i] - np.array(batch[i])) ** 2).mean()))

        ari = float(np.mean(ari_scores))
        miou = float(np.mean(miou_scores))
        mse = float(np.mean(mse_scores))
        n_steps = int(T / dt)
        marker = "  <- train" if abs(dt - train_dt) < 1e-6 else ""
        print(f"{dt:>6.3f} {n_steps:>8d} {ari:>10.4f} {miou:>10.4f} {mse:>10.5f}{marker}")
        results[dt] = {"ari_fg": ari, "miou": miou, "mse": mse, "n_steps": n_steps}

    print("\nResult dict:")
    print(results)


if __name__ == "__main__":
    main()
