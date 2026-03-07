"""
convert_tfrecords.py — Convert CLEVR-with-masks TFRecords to PNG images + masks.

Reads the DeepMind Multi-Object Datasets CLEVR-with-masks TFRecords
(GZIP compressed) and saves individual images and per-object segmentation
masks as PNGs.

TFRecord schema (per example):
  - image:      bytes_list, 230400 entries (240*320*3 individual bytes) -> [240, 320, 3] uint8
  - mask:       bytes_list, 844800 entries (11*240*320 individual bytes) -> [11, 240, 320] uint8 binary {0, 255}
  - visibility: float_list, 11 values (1.0 = present, 0.0 = absent)
  - color, material, shape, size: bytes_list, 11 entries (per-object attributes)
  - x, y, z, rotation: float_list, 11 values (per-object 3D pose)
  - pixel_coords: float_list, 33 values (11*3, per-object pixel coords + depth)

Masks: channel 0 = background, channels 1-10 = foreground objects.
Each channel is binary (0 or 255). We combine into a single segmentation map
where pixel value = object ID (0=bg, 1..N=objects).

Output structure:
    CLEVR_masks/
    ├── images/
    │   ├── train/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── val/
    │       ├── 000000.png
    │       └── ...
    └── masks/
        ├── train/
        │   ├── 000000.png   (pixel values = object IDs: 0=bg, 1..N=objects)
        │   └── ...
        └── val/
            ├── 000000.png
            └── ...

Usage:
    python convert_tfrecords.py
    python convert_tfrecords.py --input path/to/file.tfrecords --output CLEVR_masks
    python convert_tfrecords.py --val_size 5000
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_example_proto(raw_record_bytes):
    """Parse a single TFRecord example from raw bytes using protobuf directly.

    This avoids needing tf.io.parse_single_example and handles the
    bytes_list format used by Multi-Object Datasets.
    """
    import tensorflow as tf

    example = tf.train.Example()
    example.ParseFromString(raw_record_bytes)
    feat = example.features.feature

    # image: 230400 single-byte entries -> [240, 320, 3]
    img_bytes = feat['image'].bytes_list.value
    image = np.frombuffer(b''.join(img_bytes), dtype=np.uint8).reshape(240, 320, 3)

    # mask: 844800 single-byte entries -> [11, 240, 320]
    mask_bytes = feat['mask'].bytes_list.value
    masks = np.frombuffer(b''.join(mask_bytes), dtype=np.uint8).reshape(11, 240, 320)

    # visibility: [11]
    visibility = np.array(feat['visibility'].float_list.value, dtype=np.float32)

    return image, masks, visibility


def masks_to_segmentation(masks, visibility):
    """Convert per-object binary masks to a single segmentation map.

    Args:
        masks: [11, H, W] uint8 binary masks (0 or 255)
        visibility: [11] float, 1.0 if object present

    Returns:
        seg: [H, W] uint8, pixel values = object ID (0=background, 1..10=objects)
    """
    H, W = masks.shape[1], masks.shape[2]
    seg = np.zeros((H, W), dtype=np.uint8)

    # masks[0] = background, masks[1..10] = objects
    # later objects overwrite earlier ones at overlapping pixels (front-to-back)
    for obj_idx in range(1, 11):
        if visibility[obj_idx] > 0.5:
            obj_mask = masks[obj_idx] > 127
            seg[obj_mask] = obj_idx

    return seg


def convert(args):
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    input_path = args.input
    output_dir = Path(args.output)
    val_size = args.val_size

    print(f"Reading: {input_path}")
    print(f"Output:  {output_dir}")

    # create output directories
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'masks' / split).mkdir(parents=True, exist_ok=True)

    # read TFRecords (GZIP compressed)
    dataset = tf.data.TFRecordDataset(input_path, compression_type='GZIP')

    # count total records
    print("Counting records...")
    total = 0
    for _ in dataset:
        total += 1
    print(f"Total records: {total}")

    # split: last val_size go to val
    train_size = total - val_size
    print(f"Train: {train_size}, Val: {val_size}")

    # second pass: convert
    dataset = tf.data.TFRecordDataset(input_path, compression_type='GZIP')
    train_idx = 0
    val_idx = 0

    for i, raw_record in enumerate(dataset):
        image, masks, visibility = parse_example_proto(raw_record.numpy())

        # create segmentation map
        seg = masks_to_segmentation(masks, visibility)

        # determine split
        if i < train_size:
            split = 'train'
            idx = train_idx
            train_idx += 1
        else:
            split = 'val'
            idx = val_idx
            val_idx += 1

        # save image
        img_path = output_dir / 'images' / split / f'{idx:06d}.png'
        Image.fromarray(image).save(img_path)

        # save segmentation mask
        mask_path = output_dir / 'masks' / split / f'{idx:06d}.png'
        Image.fromarray(seg).save(mask_path)

        if (i + 1) % 5000 == 0:
            print(f"  [{i + 1:>6d}/{total}] processed")

    print(f"\nDone! Train: {train_idx}, Val: {val_idx}")
    print(f"Images: {output_dir / 'images'}")
    print(f"Masks:  {output_dir / 'masks'}")


def parse_args():
    p = argparse.ArgumentParser(description="Convert CLEVR-with-masks TFRecords to PNGs")
    p.add_argument("--input", default="clevr_with_masks_clevr_with_masks_train.tfrecords",
                   help="Path to .tfrecords file")
    p.add_argument("--output", default="CLEVR_masks",
                   help="Output directory")
    p.add_argument("--val_size", type=int, default=5000,
                   help="Number of images for validation split")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args)
