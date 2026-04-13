#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGGT on a directory of frames/images."
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing input frames, e.g. stream/",
    )
    parser.add_argument(
        "--output",
        default="vggt_output",
        help="Output directory to save predictions.npz",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/VGGT-1B",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help='Glob pattern inside image_dir, e.g. "*.jpg" or "*.png"',
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of images to use",
    )
    parser.add_argument(
        "--save-fp32",
        action="store_true",
        help="Convert outputs to float32 before saving to reduce downstream dtype issues",
    )
    return parser.parse_args()


def find_images(image_dir: str, pattern: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    paths = [p for p in paths if Path(p).suffix.lower() in exts]
    return paths


def to_numpy_dict(predictions: dict) -> dict:
    out = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            if value.shape[0] == 1:
                value = value.squeeze(0)
        out[key] = value
    return out


def maybe_float32_dict(predictions: dict) -> dict:
    out = {}
    for key, value in predictions.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            out[key] = value.astype(np.float32)
        else:
            out[key] = value
    return out


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this script, but no CUDA device was found.")

    image_paths = find_images(args.image_dir, args.pattern)
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir!r} with pattern {args.pattern!r}")

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    print(f"Found {len(image_paths)} images")
    for i, p in enumerate(image_paths[:5]):
        print(f"  [{i}] {p}")
    if len(image_paths) > 5:
        print("  ...")

    device = "cuda"
    major_cc = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if major_cc >= 8 else torch.float16

    print(f"Loading model: {args.model_name}")
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()

    print("Loading and preprocessing images...")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Image tensor shape: {tuple(images.shape)}")

    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.amp.autocast(dtype=dtype, device_type='cuda'):
            predictions = model(images)

    print("Converting pose encoding -> extrinsic/intrinsic...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Converting tensors to numpy...")
    predictions = to_numpy_dict(predictions)

    # Optional cleanup of fields you may not need
    if "pose_enc_list" in predictions:
        predictions["pose_enc_list"] = None

    print("Computing world points from depth...")
    # depth: usually (S, H, W, 1)
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"],
        predictions["extrinsic"],
        predictions["intrinsic"],
    )

    predictions["image_paths"] = np.array(image_paths, dtype=object)

    if args.save_fp32:
        predictions = maybe_float32_dict(predictions)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.npz"

    print(f"Saving predictions to {out_file}")
    np.savez_compressed(out_file, **predictions)

    print("Done.")
    print(f"Saved: {out_file}")
    print("Saved keys:")
    for k, v in predictions.items():
        shape = getattr(v, "shape", None)
        print(f"  {k}: {shape}")


if __name__ == "__main__":
    main()