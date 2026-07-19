"""Pack CLEVR_64 PNG+npy files into single .npz files for fast loading."""

import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def pack_split(data_dir, split):
    img_dir = Path(data_dir) / "images" / split
    mask_dir = Path(data_dir) / "masks" / split
    vis_dir = Path(data_dir) / "visibility" / split

    img_files = sorted(img_dir.glob("*.png"))
    n = len(img_files)
    print(f"{split}: {n} images")

    sample = np.array(Image.open(img_files[0]))
    H, W = sample.shape[:2]

    images = np.zeros((n, 3, H, W), dtype=np.float32)
    masks = np.zeros((n, 11, H, W), dtype=np.uint8)
    visibility = np.zeros((n, 11), dtype=np.float32)

    for i, img_file in enumerate(img_files):
        stem = img_file.stem
        img = np.array(Image.open(img_file))
        images[i] = img.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
        masks[i] = np.load(mask_dir / f"{stem}.npy")
        visibility[i] = np.load(vis_dir / f"{stem}.npy")
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n}")

    out_path = Path(data_dir) / f"{split}.npz"
    print(f"Saving {out_path}...")
    np.savez(out_path, images=images, masks=masks, visibility=visibility)
    print(f"Done. {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="CLEVR_64")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    args = p.parse_args()
    for split in args.splits:
        pack_split(args.data_dir, split)
