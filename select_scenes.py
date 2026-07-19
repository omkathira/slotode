"""
Select high-quality val scenes for the qualitative figure.

Picks candidates by: low MSE, high ARI-FG, and target object counts (3 and 7-9).
Prints top-5 indices per bucket so you can hand-pick in the notebook.

Usage:
    python select_scenes.py --ckpt runs_data/omkos-slotode/slot_ode_11_slots_euler_T3_dt1/checkpoints/best.eqx
"""

import argparse
import pickle

import jax

jax.config.update("jax_default_matmul_precision", "highest")
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from evaluate import compute_ari_fg, iter_pngs
from model import SlotODEModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", default="CLEVR_64")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--resolution", type=int, default=64)
    args = p.parse_args()

    resolution = (args.resolution, args.resolution)

    meta = pickle.load(open(args.ckpt.replace(".eqx", "_meta.pkl"), "rb"))
    a = meta["args"]
    key = jax.random.key(0)
    model = SlotODEModel(
        resolution=resolution,
        num_slots=a["num_slots"],
        slot_dim=a["slot_dim"],
        enc_hidden_dim=a.get("enc_hidden_dim", 64),
        num_iter=a["num_iter"],
        dt0=a.get("dt", 1.0),
        key=key,
    )
    model = eqx.tree_deserialise_leaves(args.ckpt, model)

    print(f"Loading {args.num_samples} val samples...")
    imgs, segs, obj_counts = [], [], []
    for idx, (img_np, seg_np) in enumerate(
        iter_pngs(args.data_root, "val", args.num_samples, resolution)
    ):
        arr = img_np.astype(np.float32) / 127.5 - 1.0
        imgs.append(np.transpose(arr, (2, 0, 1)))
        segs.append(seg_np.astype(np.int32))
        obj_counts.append(len(np.unique(seg_np)) - 1)  # subtract background
    imgs_arr = np.stack(imgs)

    @eqx.filter_jit
    def predict(model, batch, key):
        recon, masks, _ = model(batch, key=key)
        return recon, masks

    ari_all, mse_all = [], []
    kk = jax.random.key(0)
    B = args.batch_size
    for start in range(0, len(imgs_arr), B):
        batch = jnp.array(imgs_arr[start : start + B])
        kk, sk = jax.random.split(kk)
        recon, masks = predict(model, batch, sk)
        recon = np.array(recon)
        masks = np.array(masks)
        bn = np.array(batch)
        for j in range(batch.shape[0]):
            i = start + j
            pred = masks[j].argmax(axis=0)
            ari_all.append(compute_ari_fg(pred, segs[i]))
            mse_all.append(float(((recon[j] - bn[j]) ** 2).mean()))

    ari = np.array(ari_all)
    mse = np.array(mse_all)
    oc = np.array(obj_counts)

    # composite score: high ARI, low MSE
    score = (
        ari - 30 * mse
    )  # weights tuned roughly; MSE usually ~1e-3 so *30 puts it on same scale

    for label, mask in [("3 objects", oc == 3), ("7-9 objects", (oc >= 7) & (oc <= 9))]:
        print(f"\n=== {label}: {mask.sum()} candidates ===")
        if not mask.any():
            print("  (none found)")
            continue
        idxs = np.where(mask)[0]
        ranked = idxs[np.argsort(-score[idxs])]
        print(f"  top 8 (idx | ARI | MSE | obj_count):")
        for i in ranked[:8]:
            print(f"    {i:4d}  ARI={ari[i]:.4f}  MSE={mse[i]:.5f}  n_obj={oc[i]}")


if __name__ == "__main__":
    main()
