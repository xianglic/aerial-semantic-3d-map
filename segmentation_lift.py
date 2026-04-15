#!/usr/bin/env python3
import argparse
import glob
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lift SAM3 masks into 3D using VGGT world points."
    )
    parser.add_argument("--predictions", required=True, help="Path to VGGT predictions.npz")
    parser.add_argument("--text-prompt", required=True, help='Text prompt, e.g. "tree", "building"')
    parser.add_argument("--sam-model", default="facebook/sam3", help="SAM3 model id")
    parser.add_argument("--frame-glob", help="Optional glob for original images. If omitted, uses image_paths stored in predictions.npz.")
    parser.add_argument("--use-depth-world-points", action="store_true", help="Use world_points_from_depth instead of world_points.")
    parser.add_argument("--sam-threshold", type=float, default=0.7, help="Instance segmentation confidence threshold.")
    parser.add_argument("--point-conf-threshold", type=float, default=0.3, help="VGGT point confidence threshold (applied to normalized [0,1] conf scores; 0.3 ≈ top 70%%).")
    parser.add_argument("--out-dir", default="seg3d_output", help="Output directory.")
    parser.add_argument("--save-per-frame-overlay", action="store_true", help="Save 2D mask overlay for each frame.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def load_predictions(predictions_path: str, use_depth_world_points: bool) -> dict:
    data = np.load(predictions_path, allow_pickle=True)
    point_key = "world_points_from_depth" if use_depth_world_points else "world_points"

    for key in [point_key, "world_points_conf", "image_paths"]:
        if key not in data:
            raise KeyError(f"Missing key {key!r} in {predictions_path}")

    conf = np.array(data["world_points_conf"])
    conf_min, conf_max = conf.min(), conf.max()
    conf = (conf - conf_min) / (conf_max - conf_min + 1e-8)

    return {
        "point_key": point_key,
        "world_points": data[point_key],
        "world_points_conf": conf,
        "image_paths": [str(p) for p in data["image_paths"].tolist()],
    }


def resolve_frame_paths(stored_paths: List[str], frame_glob: Optional[str]) -> List[str]:
    if frame_glob is None:
        return stored_paths
    frame_paths = sorted(glob.glob(frame_glob))
    if len(frame_paths) != len(stored_paths):
        raise ValueError(
            f"frame_glob matched {len(frame_paths)} files, but predictions has {len(stored_paths)} frames."
        )
    return frame_paths


def load_sam3(model_name: str, device: str):
    logger.info("Loading SAM3 model: %s", model_name)
    processor = Sam3Processor.from_pretrained(model_name)
    model = Sam3Model.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def run_sam3_on_image(
    image_pil: Image.Image,
    text_prompt: str,
    processor: Sam3Processor,
    model: Sam3Model,
    device: str,
    threshold: float,
) -> List[dict]:
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = result.get("masks", [])
    scores = result.get("scores")

    items = []
    for i, mask in enumerate(masks):
        m = np.asarray(mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask).astype(bool)
        score = float(scores[i]) if scores is not None else 0.0
        items.append({"index": i, "mask": m, "score": score, "area": int(m.sum())})

    return items


def make_overlay(image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_np.astype(np.float32).copy()
    overlay[mask] = 0.65 * overlay[mask] + 0.35 * np.array([255, 0, 0], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def lift_mask_to_3d_points(
    mask: np.ndarray,
    point_map: np.ndarray,
    point_conf: np.ndarray,
    conf_thresh: float,
) -> tuple[np.ndarray, int, int]:
    """Returns (points, n_raw_masked, n_conf_passing)."""
    pts = point_map[mask]
    conf = point_conf[mask]
    valid = np.isfinite(pts).all(axis=1) & (conf >= conf_thresh)
    return pts[valid], len(pts), int(valid.sum())


def build_scene_cloud(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    frame_paths: List[str],
    point_conf_threshold: float,
) -> o3d.geometry.PointCloud:
    num_frames, H, W, _ = world_points.shape
    pts_list, colors_list = [], []

    for k in range(num_frames):
        img = np.array(Image.open(frame_paths[k]).convert("RGB"))
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        valid = np.isfinite(world_points[k]).all(axis=-1) & (world_points_conf[k] >= point_conf_threshold)
        pts_list.append(world_points[k][valid])
        colors_list.append(img[valid] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_list))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors_list))
    return pcd


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    preds = load_predictions(args.predictions, args.use_depth_world_points)
    world_points = preds["world_points"]
    world_points_conf = preds["world_points_conf"]
    frame_paths = resolve_frame_paths(preds["image_paths"], args.frame_glob)

    num_frames, H, W, _ = world_points.shape

    # Squeeze trailing dim if VGGT emitted (S, H, W, 1)
    if world_points_conf.ndim == 4 and world_points_conf.shape[-1] == 1:
        world_points_conf = world_points_conf.squeeze(-1)
    if world_points_conf.shape != (num_frames, H, W):
        raise ValueError(
            f"Unexpected world_points_conf shape {world_points_conf.shape}, "
            f"expected ({num_frames}, {H}, {W})"
        )

    conf_flat = world_points_conf.ravel()
    print(
        f"world_points_conf: shape={world_points_conf.shape} | "
        f"min={conf_flat.min():.4f} max={conf_flat.max():.4f} mean={conf_flat.mean():.4f} | "
        f"p50={np.percentile(conf_flat, 50):.4f} p75={np.percentile(conf_flat, 75):.4f} "
        f"p90={np.percentile(conf_flat, 90):.4f} p95={np.percentile(conf_flat, 95):.4f}"
    )

    vggt_pixels_per_frame = H * W
    print(f"VGGT grid: {H}x{W} = {vggt_pixels_per_frame:,} points/frame | "
          f"{num_frames} frames | {vggt_pixels_per_frame * num_frames:,} total raw points")

    # Global confidence stats across all frames (before any masking)
    total_raw = vggt_pixels_per_frame * num_frames
    conf_pass_global = int((np.isfinite(world_points).all(axis=-1) &
                            (world_points_conf >= args.point_conf_threshold)).sum())
    print(f"Global conf>={args.point_conf_threshold}: {conf_pass_global:,} / {total_raw:,} "
          f"({100*conf_pass_global/total_raw:.1f}%)")

    processor, model = load_sam3(args.sam_model, device)
    merged_object_points = []

    total_image_pixels = 0
    total_mask_pixels_orig = 0
    total_mask_pixels_vggt = 0
    total_raw_masked = 0
    total_conf_passing = 0

    for k in range(num_frames):
        image = Image.open(frame_paths[k]).convert("RGB")
        image_np = np.array(image)
        H_orig, W_orig = image_np.shape[:2]
        image_pixels = H_orig * W_orig
        total_image_pixels += image_pixels

        mask_items = run_sam3_on_image(
            image_pil=image,
            text_prompt=args.text_prompt,
            processor=processor,
            model=model,
            device=device,
            threshold=args.sam_threshold,
        )

        valid_masks = [item for item in mask_items if item["area"] > 0]
        if not valid_masks:
            print(f"[{k:03d}] image={H_orig}x{W_orig} ({image_pixels:,} px) | no valid mask")
            continue

        frame_pts = []
        union_mask_orig = np.zeros((H_orig, W_orig), dtype=bool)
        union_mask_vggt = np.zeros((H, W), dtype=bool)
        frame_raw_masked = 0
        frame_conf_passing = 0

        for item in valid_masks:
            mask = item["mask"]
            mask_for_vggt = (
                cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                if mask.shape != (H, W) else mask
            )
            pts, n_raw, n_conf = lift_mask_to_3d_points(
                mask=mask_for_vggt,
                point_map=world_points[k],
                point_conf=world_points_conf[k],
                conf_thresh=args.point_conf_threshold,
            )
            frame_raw_masked += n_raw
            frame_conf_passing += n_conf
            if len(pts) > 0:
                frame_pts.append(pts)
            union_mask_orig |= mask
            union_mask_vggt |= mask_for_vggt

        mask_px_orig = int(union_mask_orig.sum())
        mask_px_vggt = int(union_mask_vggt.sum())
        total_mask_pixels_orig += mask_px_orig
        total_mask_pixels_vggt += mask_px_vggt
        total_raw_masked += frame_raw_masked
        total_conf_passing += frame_conf_passing

        pct_img = 100 * mask_px_orig / image_pixels if image_pixels else 0
        pct_conf = 100 * frame_conf_passing / frame_raw_masked if frame_raw_masked else 0
        print(
            f"[{k:03d}] image={H_orig}x{W_orig} ({image_pixels:,} px) | "
            f"mask_orig={mask_px_orig:,} ({pct_img:.1f}%) | "
            f"mask_vggt={mask_px_vggt:,} | "
            f"raw_masked={frame_raw_masked:,} | "
            f"conf_pass={frame_conf_passing:,} ({pct_conf:.1f}%) | "
            f"3d_pts={sum(len(p) for p in frame_pts):,}"
        )
        merged_object_points.extend(frame_pts)

        if args.save_per_frame_overlay:
            Image.fromarray(make_overlay(image_np, union_mask_orig)).save(out_dir / f"overlay_{k:03d}.png")

    print(
        f"\n=== Summary ===\n"
        f"  Image pixels total   : {total_image_pixels:,} ({total_image_pixels // num_frames:,}/frame)\n"
        f"  VGGT raw points total: {total_raw:,} ({vggt_pixels_per_frame:,}/frame)\n"
        f"  Global conf passing  : {conf_pass_global:,} ({100*conf_pass_global/total_raw:.1f}% of raw)\n"
        f"  Union mask (orig res): {total_mask_pixels_orig:,} px across all frames\n"
        f"  Union mask (vggt res): {total_mask_pixels_vggt:,} px across all frames\n"
        f"  Masked raw points    : {total_raw_masked:,}\n"
        f"  Masked conf passing  : {total_conf_passing:,} "
        f"({100*total_conf_passing/total_raw_masked:.1f}% of masked raw)"
        if total_raw_masked else "  Masked conf passing  : 0"
    )

    if not merged_object_points:
        raise RuntimeError("No 3D points extracted from any frame.")

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np.concatenate(merged_object_points))
    obj_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    scene_pcd = build_scene_cloud(world_points, world_points_conf, frame_paths, args.point_conf_threshold)
    combined_pcd = scene_pcd + obj_pcd

    obj_path = out_dir / "segmented_object.ply"
    scene_path = out_dir / "scene_context.ply"
    combined_path = out_dir / "combined_overlay.ply"

    o3d.io.write_point_cloud(str(obj_path), obj_pcd)
    o3d.io.write_point_cloud(str(scene_path), scene_pcd)
    o3d.io.write_point_cloud(str(combined_path), combined_pcd)

    print(f"Saved segmented object cloud: {obj_path}")
    print(f"Saved scene context cloud:    {scene_path}")
    print(f"Saved combined overlay cloud: {combined_path}")


if __name__ == "__main__":
    main()
