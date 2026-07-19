"""Modal: run threshold sweep on a checkpoint, dump masks + sweep curve to volume."""

import modal

app = modal.App("slotode-threshold-sweep")

data_vol = modal.Volume.from_name("slotode-data")
ckpt_vol = modal.Volume.from_name("slotode-ckpts")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "jax[cuda12]",
        "equinox",
        "optax",
        "diffrax",
        "numpy",
        "Pillow",
        "matplotlib",
        "mlflow",
        "scikit-learn",
        "scipy",
    )
    .add_local_dir(".", remote_path="/root/slotode", ignore=lambda p: not p.name.endswith(".py"))
)


def _ensure_data():
    import os
    import subprocess

    if not os.path.exists("/data/CLEVR_64/images/val"):
        os.makedirs("/data/CLEVR_64", exist_ok=True)
        print("Extracting clevr_npz.tar.gz...")
        subprocess.run(["tar", "-xzf", "/data/clevr_npz.tar.gz", "-C", "/data"], check=True)
        print("Done extracting.")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    timeout=3600,
)
def threshold_sweep(
    run_name: str,
    kind: str,                # "sa" or "ode"
    num_samples: int = 5000,
    out_name: str = "threshold_data.npz",
):
    """Inference + threshold sweep. Saves masks, GT segs, images, sweep results to npz."""
    import os
    import sys
    import pickle
    import numpy as np
    from pathlib import Path
    from PIL import Image

    sys.path.insert(0, "/root/slotode")
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import equinox as eqx
    from scipy.optimize import linear_sum_assignment

    from model import SlotODEModel
    from model_baseline import SlotAttentionModel

    _ensure_data()

    ckpt_path = f"/ckpts/{run_name}/best.eqx"
    meta_path = f"/ckpts/{run_name}/best_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    args = meta["args"]
    print(f"Loaded meta: step={meta['step']}  num_iter={args['num_iter']}  dt={args.get('dt')}")

    RESOLUTION = (64, 64)
    NUM_SLOTS = args["num_slots"]
    key = jax.random.key(0)

    if kind == "sa":
        model = SlotAttentionModel(
            resolution=RESOLUTION,
            num_slots=NUM_SLOTS,
            slot_dim=args["slot_dim"],
            enc_hidden_dim=args.get("enc_hidden_dim", 64),
            num_iter=args["num_iter"],
            key=key,
        )
    elif kind == "ode":
        model = SlotODEModel(
            resolution=RESOLUTION,
            num_slots=NUM_SLOTS,
            slot_dim=args["slot_dim"],
            enc_hidden_dim=args.get("enc_hidden_dim", 64),
            num_iter=args["num_iter"],
            dt0=args.get("dt") or 1.0,
            key=key,
        )
    else:
        raise ValueError(f"unknown kind: {kind}")
    model = eqx.tree_deserialise_leaves(ckpt_path, model)

    # load val
    data_dir = Path("/data/CLEVR_64")
    img_dir = data_dir / "images" / "val"
    mask_dir = data_dir / "masks" / "val"
    vis_dir = data_dir / "visibility" / "val"
    images, gt_segs = [], []
    for ip in sorted(img_dir.glob("*.png"))[:num_samples]:
        stem = ip.stem
        mp = mask_dir / f"{stem}.npy"
        vp = vis_dir / f"{stem}.npy"
        if not mp.exists():
            continue
        img = np.array(Image.open(ip).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        images.append(img)
        masks_gt = np.load(mp)
        visibility = np.load(vp) if vp.exists() else np.ones(11, dtype=np.float32)
        seg = np.zeros((RESOLUTION[0], RESOLUTION[1]), dtype=np.int32)
        for obj_idx in range(1, 11):
            if visibility[obj_idx] > 0.5:
                seg[masks_gt[obj_idx] > 127] = obj_idx
        gt_segs.append(seg)
    images_np = np.stack(images)
    gt_segs_np = np.stack(gt_segs)
    print(f"Loaded {len(gt_segs)} val images")

    @eqx.filter_jit
    def predict(model, images, key):
        recon, masks, slots = model(images, key=key)
        return masks

    BS = 64
    all_masks = []
    key = jax.random.key(42)
    for i in range(0, len(images_np), BS):
        batch = jnp.array(images_np[i:i + BS])
        key, sk = jax.random.split(key)
        m = predict(model, batch, sk)
        all_masks.append(np.array(m))
        if (i // BS) % 10 == 0:
            print(f"  batch {i // BS + 1}/{(len(images_np) + BS - 1) // BS}")
    all_masks_np = np.concatenate(all_masks, axis=0).astype(np.float32)
    print(f"Inference done: masks {all_masks_np.shape}")

    def miou_thresholded(pred_probs, gt_mask, num_pred_slots, tau):
        pred_mask = pred_probs.argmax(axis=0)
        max_prob = pred_probs.max(axis=0)
        confident = max_prob >= tau
        gt_ids = np.unique(gt_mask)
        gt_ids = gt_ids[gt_ids > 0]
        n_gt = len(gt_ids)
        if n_gt == 0:
            return 1.0
        iou = np.zeros((n_gt, num_pred_slots))
        for i, gid in enumerate(gt_ids):
            gb = (gt_mask == gid)
            for j in range(num_pred_slots):
                pb = (pred_mask == j) & confident
                inter = (gb & pb).sum()
                union = (gb | pb).sum()
                iou[i, j] = inter / max(union, 1)
        ri, ci = linear_sum_assignment(-iou)
        return iou[ri, ci].mean()

    thresholds = np.arange(0.0, 0.96, 0.025)
    means, stds = [], []
    for tau in thresholds:
        scores = [miou_thresholded(all_masks_np[k], gt_segs_np[k], NUM_SLOTS, tau)
                  for k in range(len(gt_segs_np))]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        print(f"  tau={tau:.3f}  mIoU={means[-1]:.4f}")
    means = np.array(means)
    stds = np.array(stds)
    best = int(means.argmax())
    print(f"\nbaseline (tau=0): {means[0]:.4f}    best: {means[best]:.4f} at tau={thresholds[best]:.3f}")

    out_path = f"/ckpts/{run_name}/{out_name}"
    np.savez_compressed(
        out_path,
        run_name=run_name,
        kind=kind,
        num_slots=NUM_SLOTS,
        num_samples=len(gt_segs_np),
        images=images_np,
        gt_segs=gt_segs_np,
        masks=all_masks_np,
        thresholds=thresholds,
        miou_mean=means,
        miou_std=stds,
        best_tau=thresholds[best],
        best_miou=means[best],
        baseline_miou=means[0],
    )
    ckpt_vol.commit()
    print(f"Wrote {out_path} (committed). Pull with:")
    print(f"  modal volume get slotode-ckpts {run_name}/{out_name} ./threshold_data_{run_name}.npz")
    return {
        "run_name": run_name,
        "best_tau": float(thresholds[best]),
        "best_miou": float(means[best]),
        "baseline_miou": float(means[0]),
    }


@app.local_entrypoint()
def main():
    print(threshold_sweep.remote("baseline_sa_s11_T5", "sa", 5000))
