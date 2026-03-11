"""
convert_tfrecords.py — Convert CLEVR-with-masks TFRecords to 64x64 PNGs.

Reads the DeepMind Multi-Object Datasets CLEVR-with-masks TFRecords
(GZIP compressed) and saves resized images and per-object binary masks.

Images are resized from 240x320 to 64x64 (bilinear).
Masks are resized from 240x320 to 64x64 (nearest-neighbor) and saved as
numpy arrays to preserve all 11 binary channels.

Output structure:
    CLEVR_64/
    ├── images/
    │   ├── train/
    │   │   ├── 000000.png      [64, 64, 3] uint8
    │   │   └── ...
    │   └── val/
    │       └── ...
    ├── masks/
    │   ├── train/
    │   │   ├── 000000.npy      [11, 64, 64] uint8 binary {0, 255}
    │   │   └── ...
    │   └── val/
    │       └── ...
    └── visibility/
        ├── train/
        │   ├── 000000.npy      [11] float32
        │   └── ...
        └── val/
            └── ...

Usage:
    python convert_tfrecords.py
    python convert_tfrecords.py --input path/to/file.tfrecords --output CLEVR_64
    python convert_tfrecords.py --resolution 128
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_example_proto(raw_record_bytes):
    """Parse a single TFRecord example from raw bytes using protobuf directly."""
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


def resize_image(image, resolution):
    """Resize [240, 320, 3] uint8 -> [res, res, 3] uint8 (bilinear)."""
    return np.array(Image.fromarray(image).resize((resolution, resolution), Image.BILINEAR))


def resize_masks(masks, resolution):
    """Resize [11, 240, 320] uint8 -> [11, res, res] uint8 (nearest)."""
    out = np.zeros((11, resolution, resolution), dtype=np.uint8)
    for i in range(11):
        out[i] = np.array(Image.fromarray(masks[i]).resize(
            (resolution, resolution), Image.NEAREST))
    return out


def convert(args):
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    input_path = args.input
    output_dir = Path(args.output)
    val_size = args.val_size
    res = args.resolution

    print(f"Reading: {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Resolution: {res}x{res}")

    # create output directories
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'masks' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'visibility' / split).mkdir(parents=True, exist_ok=True)

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

        # resize to target resolution
        image = resize_image(image, res)
        masks = resize_masks(masks, res)

        # determine split
        if i < train_size:
            split = 'train'
            idx = train_idx
            train_idx += 1
        else:
            split = 'val'
            idx = val_idx
            val_idx += 1

        # save image as PNG
        img_path = output_dir / 'images' / split / f'{idx:06d}.png'
        Image.fromarray(image).save(img_path)

        # save masks as numpy (preserves all 11 binary channels)
        mask_path = output_dir / 'masks' / split / f'{idx:06d}.npy'
        np.save(mask_path, masks)

        # save visibility as numpy
        vis_path = output_dir / 'visibility' / split / f'{idx:06d}.npy'
        np.save(vis_path, visibility)

        if (i + 1) % 5000 == 0:
            print(f"  [{i + 1:>6d}/{total}] processed")

    print(f"\nDone! Train: {train_idx}, Val: {val_idx}")
    print(f"Output: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Convert CLEVR-with-masks TFRecords to resized PNGs + masks")
    p.add_argument("--input", default="clevr_with_masks_clevr_with_masks_train.tfrecords",
                   help="Path to .tfrecords file")
    p.add_argument("--output", default="CLEVR_64",
                   help="Output directory")
    p.add_argument("--resolution", type=int, default=64,
                   help="Target resolution (default: 64)")
    p.add_argument("--val_size", type=int, default=5000,
                   help="Number of images for validation split")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args)
