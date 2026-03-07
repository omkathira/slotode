"""
evaluate.py — Compute ARI-FG and mIoU for trained Slot Attention models on CLEVR-with-masks.

================================================================================
METRICS
================================================================================

ARI-FG (Adjusted Rand Index, foreground only):
  Measures clustering quality of predicted slot masks vs ground-truth object
  masks, ignoring the background. Primary metric in slot attention literature.
  Range: [-1, 1], 1 = perfect clustering.

mIoU (Mean Intersection over Union):
  For each ground-truth object, find the predicted slot with highest IoU
  (Hungarian matching). Average over all objects and images.

MSE (Mean Squared Error):
  Reconstruction quality, averaged over all pixels.

================================================================================
DATA SOURCES
================================================================================

Supports two modes:

1. TFRecords (default, --tfrecords flag):
   Reads directly from the CLEVR-with-masks TFRecords file. No conversion
   needed. This is the recommended mode — avoids disk I/O overhead of PNGs.

2. PNG directory (--data_root flag without --tfrecords):
   Reads from converted PNGs (output of convert_tfrecords.py).

================================================================================

Usage:
    # Direct from TFRecords (recommended):
    python evaluate.py --ckpt checkpoints/best.eqx --tfrecords clevr_with_masks_clevr_with_masks_train.tfrecords

    # From converted PNGs:
    python evaluate.py --ckpt checkpoints/best.eqx --data_root CLEVR_masks

    # Baseline model:
    python evaluate.py --ckpt checkpoints/best.eqx --model baseline --tfrecords clevr_with_masks_clevr_with_masks_train.tfrecords

    # Limit evaluation size:
    python evaluate.py --ckpt checkpoints/best.eqx --tfrecords clevr_with_masks_clevr_with_masks_train.tfrecords --num_samples 1000
"""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from PIL import Image
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

from model import SlotODEModel
from model_baseline import SlotAttentionModel


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt_mask(mask_path, resolution):
    """Load a ground-truth segmentation mask PNG and resize to model resolution."""
    mask = Image.open(mask_path)
    mask = mask.resize((resolution[1], resolution[0]), Image.NEAREST)
    return np.array(mask, dtype=np.int32)


def masks_to_segmentation(masks, visibility):
    """Convert [11, H, W] binary masks to [H, W] segmentation map.

    Args:
        masks: [11, H, W] uint8 binary masks (0 or 255)
        visibility: [11] float

    Returns:
        seg: [H, W] int32, 0=background, 1..10=objects
    """
    H, W = masks.shape[1], masks.shape[2]
    seg = np.zeros((H, W), dtype=np.int32)
    for obj_idx in range(1, 11):
        if visibility[obj_idx] > 0.5:
            seg[masks[obj_idx] > 127] = obj_idx
    return seg


def iter_tfrecords(tfrecords_path, num_samples, val_offset=None):
    """Iterate over TFRecords yielding (image_np, gt_seg) pairs.

    Args:
        tfrecords_path: path to .tfrecords file
        num_samples: max number of samples to yield
        val_offset: if set, skip this many records first (to get val split)

    Yields:
        (image, seg) where image is [H, W, 3] uint8, seg is [H, W] int32
    """
    import tensorflow as tf

    dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type='GZIP')

    if val_offset is not None:
        dataset = dataset.skip(val_offset)

    count = 0
    for raw_record in dataset:
        if count >= num_samples:
            break

        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        feat = example.features.feature

        img_bytes = feat['image'].bytes_list.value
        image = np.frombuffer(b''.join(img_bytes), dtype=np.uint8).reshape(240, 320, 3)

        mask_bytes = feat['mask'].bytes_list.value
        masks = np.frombuffer(b''.join(mask_bytes), dtype=np.uint8).reshape(11, 240, 320)

        visibility = np.array(feat['visibility'].float_list.value, dtype=np.float32)

        seg = masks_to_segmentation(masks, visibility)

        yield image, seg
        count += 1


def iter_pngs(data_root, split, num_samples, resolution):
    """Iterate over PNG image/mask pairs.

    Yields:
        (image, seg) where image is [H, W, 3] uint8, seg is [H, W] int32
    """
    img_dir = Path(data_root) / "images" / split
    mask_dir = Path(data_root) / "masks" / split

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {img_dir}\n"
            f"Run convert_tfrecords.py first, or use --tfrecords to read directly."
        )

    img_paths = sorted(img_dir.glob("*.png"))[:num_samples]
    for ip in img_paths:
        mp = mask_dir / ip.name
        if not mp.exists():
            continue

        img = np.array(Image.open(ip).convert('RGB'))
        seg = np.array(Image.open(mp), dtype=np.int32)
        yield img, seg


def preprocess_image(img_np, resolution):
    """Resize image to model resolution and convert to [3, H, W] float32 in [0, 1]."""
    img = Image.fromarray(img_np)
    img = img.resize((resolution[1], resolution[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return np.transpose(arr, (2, 0, 1))  # [3, H, W]


def preprocess_mask(seg_np, resolution):
    """Resize segmentation mask to model resolution with nearest-neighbor."""
    seg = Image.fromarray(seg_np.astype(np.uint8))
    seg = seg.resize((resolution[1], resolution[0]), Image.NEAREST)
    return np.array(seg, dtype=np.int32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ari_fg(pred_mask, gt_mask):
    """Compute foreground Adjusted Rand Index.

    Args:
        pred_mask: [H, W] predicted segmentation (slot assignments)
        gt_mask: [H, W] ground-truth labels (0 = background)

    Returns:
        ARI score (float), considering only foreground pixels
    """
    fg = gt_mask > 0
    if fg.sum() == 0:
        return 1.0

    pred_flat = pred_mask[fg].flatten()
    gt_flat = gt_mask[fg].flatten()

    return adjusted_rand_score(gt_flat, pred_flat)


def compute_miou(pred_mask, gt_mask, num_pred_slots):
    """Compute mean IoU with Hungarian matching.

    Args:
        pred_mask: [H, W] predicted slot assignments
        gt_mask: [H, W] ground-truth labels (0 = background)
        num_pred_slots: number of predicted slots

    Returns:
        mIoU score (float)
    """
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    n_gt = len(gt_ids)

    if n_gt == 0:
        return 1.0

    iou_matrix = np.zeros((n_gt, num_pred_slots))
    for i, gt_id in enumerate(gt_ids):
        gt_binary = (gt_mask == gt_id)
        for j in range(num_pred_slots):
            pred_binary = (pred_mask == j)
            intersection = (gt_binary & pred_binary).sum()
            union = (gt_binary | pred_binary).sum()
            iou_matrix[i, j] = intersection / max(union, 1)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]

    return matched_ious.mean()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    print(f"JAX devices: {jax.devices()}")

    res = args.resolution
    resolution = (res, res)

    # ---- load model -------------------------------------------------------
    meta_path = args.ckpt.replace('.eqx', '_meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    ckpt_args = meta['args']

    key = jax.random.key(0)
    if args.model == "slot_ode":
        model = SlotODEModel(
            resolution=resolution,
            num_slots=ckpt_args['num_slots'],
            slot_dim=ckpt_args['slot_dim'],
            enc_hidden_dim=ckpt_args['enc_hidden_dim'],
            num_iter=ckpt_args['num_iter'],
            solver=ckpt_args.get('solver', 'tsit5'),
            key=key,
        )
    else:
        model = SlotAttentionModel(
            resolution=resolution,
            num_slots=ckpt_args['num_slots'],
            slot_dim=ckpt_args['slot_dim'],
            enc_hidden_dim=ckpt_args['enc_hidden_dim'],
            num_iter=ckpt_args['num_iter'],
            key=key,
        )

    model = eqx.tree_deserialise_leaves(args.ckpt, model)
    n_params = sum(p.size for p in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {args.model} ({n_params:,} params)")
    print(f"Checkpoint: {args.ckpt} (step {meta['step']})")

    num_slots = ckpt_args['num_slots']

    # ---- set up data iterator ---------------------------------------------
    if args.tfrecords:
        # read val split = last 5000 records (matching convert_tfrecords.py default)
        # for train split, start from beginning
        if args.split == "val":
            # We need total count to compute offset. Count once.
            import tensorflow as tf
            print("Counting TFRecords to locate val split...")
            ds = tf.data.TFRecordDataset(args.tfrecords, compression_type='GZIP')
            total = sum(1 for _ in ds)
            val_offset = total - args.val_size
            print(f"Total: {total}, val offset: {val_offset}, val size: {args.val_size}")
            data_iter = iter_tfrecords(args.tfrecords, args.num_samples, val_offset=val_offset)
        else:
            data_iter = iter_tfrecords(args.tfrecords, args.num_samples)
        source_desc = f"TFRecords: {args.tfrecords} ({args.split} split)"
    else:
        data_iter = iter_pngs(args.data_root, args.split, args.num_samples, resolution)
        source_desc = f"PNGs: {args.data_root}/{args.split}"

    print(f"Data source: {source_desc}")
    print(f"Evaluating up to {args.num_samples} images...")

    # ---- JIT-compile forward pass -----------------------------------------
    @eqx.filter_jit
    def predict(model, images, key):
        recon, masks, slots = model(images, key=key)
        return recon, masks

    ari_scores = []
    miou_scores = []
    mse_scores = []
    key = jax.random.key(args.seed)

    # accumulate a batch, then run forward pass
    batch_imgs = []
    batch_segs = []

    def process_batch(batch_imgs, batch_segs, key):
        imgs_jax = jnp.array(np.stack(batch_imgs))
        key, subkey = jax.random.split(key)
        recon, masks = predict(model, imgs_jax, subkey)
        masks_np = np.array(masks)
        recon_np = np.array(recon)
        imgs_np = np.array(imgs_jax)

        for i in range(len(batch_segs)):
            pred_mask = masks_np[i].argmax(axis=0)
            gt_mask = batch_segs[i]

            ari_scores.append(compute_ari_fg(pred_mask, gt_mask))
            miou_scores.append(compute_miou(pred_mask, gt_mask, num_slots))
            mse_scores.append(float(((recon_np[i] - imgs_np[i]) ** 2).mean()))

        return key

    n_processed = 0
    for img_np, seg_np in data_iter:
        batch_imgs.append(preprocess_image(img_np, resolution))
        batch_segs.append(preprocess_mask(seg_np, resolution))

        if len(batch_imgs) == args.batch_size:
            key = process_batch(batch_imgs, batch_segs, key)
            n_processed += len(batch_imgs)
            batch_imgs = []
            batch_segs = []

            if (n_processed // args.batch_size) % 10 == 0:
                print(f"  [{n_processed:>5d}]  "
                      f"ARI-FG={np.mean(ari_scores):.4f}  "
                      f"mIoU={np.mean(miou_scores):.4f}  "
                      f"MSE={np.mean(mse_scores):.6f}")

    # process remaining
    if batch_imgs:
        key = process_batch(batch_imgs, batch_segs, key)
        n_processed += len(batch_imgs)

    # ---- results ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Results on {len(ari_scores)} images ({args.split} split)")
    print(f"{'='*60}")
    print(f"  ARI-FG:  {np.mean(ari_scores):.4f} +/- {np.std(ari_scores):.4f}")
    print(f"  mIoU:    {np.mean(miou_scores):.4f} +/- {np.std(miou_scores):.4f}")
    print(f"  MSE:     {np.mean(mse_scores):.6f} +/- {np.std(mse_scores):.6f}")
    print(f"{'='*60}")

    return {
        'ari_fg': float(np.mean(ari_scores)),
        'ari_fg_std': float(np.std(ari_scores)),
        'miou': float(np.mean(miou_scores)),
        'miou_std': float(np.std(miou_scores)),
        'mse': float(np.mean(mse_scores)),
        'mse_std': float(np.std(mse_scores)),
        'num_images': len(ari_scores),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Slot Attention models on CLEVR-with-masks"
    )
    p.add_argument("--ckpt", required=True, help="Path to .eqx checkpoint")
    p.add_argument("--model", default="slot_ode", choices=["slot_ode", "baseline"])

    # data source (pick one)
    p.add_argument("--tfrecords", default=None,
                   help="Path to CLEVR-with-masks .tfrecords file (reads directly, no conversion needed)")
    p.add_argument("--data_root", default="CLEVR_masks",
                   help="Root directory with images/ and masks/ from convert_tfrecords.py")

    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--val_size", type=int, default=5000,
                   help="Size of val split when reading from TFRecords (last N records)")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=5000,
                   help="Max number of images to evaluate on")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
