"""
Reconstruct training-time convergence curves post-hoc by evaluating each
saved checkpoint on a fixed val subset.

Usage:
    # Slot ODE, every 10k-step checkpoint:
    python convergence_curve.py \
        --run_dir runs_data/omkos-slotode/slot_ode_11_slots_euler_T3_dt1/checkpoints \
        --model slot_ode \
        --step_multiple 10000 \
        --num_samples 200 \
        --out slotode_T3_dt1_curve.json

    # Baseline:
    python convergence_curve.py \
        --run_dir runs_data/omkos-baseline/baseline_sa_s11_T3 \
        --model baseline \
        --step_multiple 10000 \
        --num_samples 200 \
        --out baseline_T3_curve.json
"""
import argparse
import glob
import json
import pickle
import re
import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from model import SlotODEModel
from model_baseline import SlotAttentionModel
from evaluate import iter_pngs, compute_ari_fg, compute_miou


def build_model(ckpt_path, model_type, resolution):
    meta_path = ckpt_path.replace(".eqx", "_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    a = meta["args"]
    key = jax.random.key(0)
    if model_type == "slot_ode":
        model = SlotODEModel(
            resolution=resolution,
            num_slots=a["num_slots"],
            slot_dim=a["slot_dim"],
            enc_hidden_dim=a.get("enc_hidden_dim", 64),
            num_iter=a["num_iter"],
            dt0=a.get("dt", 1.0),
            key=key,
        )
    else:
        model = SlotAttentionModel(
            resolution=resolution,
            num_slots=a["num_slots"],
            slot_dim=a["slot_dim"],
            enc_hidden_dim=a.get("enc_hidden_dim", 64),
            num_iter=a["num_iter"],
            key=key,
        )
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    return model, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model", choices=["slot_ode", "baseline"], required=True)
    p.add_argument("--data_root", default="CLEVR_64")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--step_multiple", type=int, default=10000,
                   help="Only eval checkpoints whose step is a multiple of this.")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    resolution = (args.resolution, args.resolution)

    # discover checkpoints
    paths = sorted(glob.glob(f"{args.run_dir}/step_*.eqx"))
    def step_of(path):
        m = re.search(r"step_(\d+)\.eqx", path)
        return int(m.group(1)) if m else -1
    paths = [pp for pp in paths if step_of(pp) % args.step_multiple == 0 and step_of(pp) > 0]
    print(f"Evaluating {len(paths)} checkpoints from {args.run_dir}")
    if not paths:
        raise SystemExit("No matching checkpoints found.")

    # load train + val subsets once
    def load_split(split):
        imgs, segs = [], []
        for img_np, seg_np in iter_pngs(args.data_root, split, args.num_samples, resolution):
            arr = img_np.astype(np.float32) / 127.5 - 1.0
            imgs.append(np.transpose(arr, (2, 0, 1)))
            segs.append(seg_np.astype(np.int32))
        return np.stack(imgs), segs

    print(f"Loading {args.num_samples} samples from train and val ...")
    train_imgs, train_segs = load_split("train")
    val_imgs, val_segs = load_split("val")
    print(f"Train {train_imgs.shape}, Val {val_imgs.shape}")

    # model meta for num_slots etc
    _, first_args = build_model(paths[0], args.model, resolution)
    num_slots = first_args["num_slots"]

    @eqx.filter_jit
    def predict(model, batch, key):
        recon, masks, _ = model(batch, key=key)
        return recon, masks

    def eval_split(model, imgs_arr, segs):
        ari, miou, mse = [], [], []
        key = jax.random.key(args.seed)
        B = args.batch_size
        for start in range(0, len(imgs_arr), B):
            batch = jnp.array(imgs_arr[start:start + B])
            key, sk = jax.random.split(key)
            recon, masks = predict(model, batch, sk)
            recon = np.array(recon); masks = np.array(masks); batch_np = np.array(batch)
            for j in range(batch.shape[0]):
                pred_mask = masks[j].argmax(axis=0)
                gt = segs[start + j]
                ari.append(compute_ari_fg(pred_mask, gt))
                miou.append(compute_miou(pred_mask, gt, num_slots))
                mse.append(float(((recon[j] - batch_np[j]) ** 2).mean()))
        return {"ari_fg": float(np.mean(ari)), "miou": float(np.mean(miou)), "mse": float(np.mean(mse))}

    results = {}
    t0 = time.time()
    for i, ckpt_path in enumerate(paths):
        step = step_of(ckpt_path)
        model, _ = build_model(ckpt_path, args.model, resolution)
        train_metrics = eval_split(model, train_imgs, train_segs)
        val_metrics = eval_split(model, val_imgs, val_segs)
        results[step] = {"train": train_metrics, "val": val_metrics}
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(paths) - i - 1)
        print(f"[{i+1}/{len(paths)}] step={step:>7d}  "
              f"train ARI={train_metrics['ari_fg']:.4f} MSE={train_metrics['mse']:.5f}  |  "
              f"val ARI={val_metrics['ari_fg']:.4f} MSE={val_metrics['mse']:.5f}  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
