#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List

from segmentation_lift import (
    build_scene_cloud,
    lift_all_frames,
    load_predictions,
    load_sam3,
    run_vggt_inference,
)
from hole_filling import build_scene_index, expand_segmentation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lift SAM3 masks into 3D using VGGT world points, with optional BFS hole filling."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--load", help="Path to saved VGGT predictions.npz")
    mode.add_argument("--inference", help="Path to folder of input images; runs VGGT inference")
    parser.add_argument("--save-predictions", help="(Inference mode) Save VGGT predictions to this .npz path for reuse.")
    parser.add_argument("--vggt-model-url", default="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
    parser.add_argument("--text-prompt", required=True, help='Text prompt e.g. "tree", "building"')
    parser.add_argument("--sam-model", default="facebook/sam3")
    parser.add_argument("--use-depth-world-points", action="store_true",
                        help="Use world_points_from_depth instead of world_points.")
    parser.add_argument("--sam-threshold", type=float, default=0.7)
    parser.add_argument("--point-conf-threshold", type=float, default=0.3,
                        help="VGGT point confidence threshold on normalized [0,1] scores.")
    parser.add_argument("--out-dir", default="seg3d_output")
    parser.add_argument("--save-per-frame-overlay", action="store_true")
    parser.add_argument("--expand-radius", type=float, default=0.0,
                        help="3D radius for BFS hole filling (0 = disabled). Units match scene scale.")
    parser.add_argument("--expand-color-thresh", type=float, default=0.15,
                        help="Max color distance (L2 in [0,1] RGB) from seed mean to accept a neighbor.")
    parser.add_argument("--expand-iters", type=int, default=20,
                        help="Max BFS iterations.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log_conf_stats(world_points: np.ndarray, world_points_conf: np.ndarray, conf_threshold: float) -> int:
    num_frames, H, W, _ = world_points.shape
    conf_flat = world_points_conf.ravel()
    logger.info(
        "world_points_conf: shape=%s | min=%.4f max=%.4f mean=%.4f | "
        "p50=%.4f p75=%.4f p90=%.4f p95=%.4f",
        world_points_conf.shape,
        conf_flat.min(), conf_flat.max(), conf_flat.mean(),
        np.percentile(conf_flat, 50), np.percentile(conf_flat, 75),
        np.percentile(conf_flat, 90), np.percentile(conf_flat, 95),
    )
    total_raw = H * W * num_frames
    conf_pass = int((np.isfinite(world_points).all(axis=-1) & (world_points_conf >= conf_threshold)).sum())
    logger.info(
        "VGGT grid: %dx%d | %d frames | %d total raw points | conf>=%.2f: %d (%.1f%%)",
        H, W, num_frames, total_raw, conf_threshold, conf_pass, 100 * conf_pass / total_raw,
    )
    return total_raw


def log_summary(stats: dict, num_frames: int, total_raw: int, vggt_pixels_per_frame: int, conf_pass_global: int) -> None:
    logger.info(
        "\n=== Summary ===\n"
        "  Image pixels total   : %d (%d/frame)\n"
        "  VGGT raw points total: %d (%d/frame)\n"
        "  Global conf passing  : %d (%.1f%% of raw)\n"
        "  Union mask (orig res): %d px across all frames\n"
        "  Union mask (vggt res): %d px across all frames\n"
        "  Masked raw points    : %d\n"
        "  Masked conf passing  : %s",
        stats["image_pixels"], stats["image_pixels"] // num_frames,
        total_raw, vggt_pixels_per_frame,
        conf_pass_global, 100 * conf_pass_global / total_raw,
        stats["mask_px_orig"],
        stats["mask_px_vggt"],
        stats["raw_masked"],
        f"{stats['conf_passing']:,} ({100 * stats['conf_passing'] / stats['raw_masked']:.1f}% of masked raw)"
        if stats["raw_masked"] else "0",
    )


def save_point_clouds(
    merged_object_points: List[np.ndarray],
    merged_object_colors: List[np.ndarray],
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    images: np.ndarray,
    point_conf_threshold: float,
    out_dir: Path,
) -> None:
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np.concatenate(merged_object_points))
    if merged_object_colors:
        obj_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(merged_object_colors))
    else:
        obj_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    scene_pcd = build_scene_cloud(world_points, world_points_conf, images, point_conf_threshold)
    combined_pcd = scene_pcd + obj_pcd

    o3d.io.write_point_cloud(str(out_dir / "segmented_object.ply"), obj_pcd)
    o3d.io.write_point_cloud(str(out_dir / "scene_context.ply"), scene_pcd)
    o3d.io.write_point_cloud(str(out_dir / "combined_overlay.ply"), combined_pcd)
    logger.info("Saved point clouds to %s", out_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- VGGT ---
    if args.load is not None:
        preds = load_predictions(args.load, args.use_depth_world_points)
    else:
        preds = run_vggt_inference(
            image_folder=args.inference,
            use_depth_world_points=args.use_depth_world_points,
            model_url=args.vggt_model_url,
            save_path=args.save_predictions,
        )

    world_points = preds["world_points"]
    world_points_conf = preds["world_points_conf"]
    images = preds["images"]

    num_frames, H, W, _ = world_points.shape
    assert world_points_conf.shape == (num_frames, H, W)

    total_raw = log_conf_stats(world_points, world_points_conf, args.point_conf_threshold)
    conf_pass_global = int(
        (np.isfinite(world_points).all(axis=-1) & (world_points_conf >= args.point_conf_threshold)).sum()
    )

    # --- Segmentation lift ---
    processor, sam_model = load_sam3(args.sam_model, device)
    merged_object_points, merged_object_colors, stats = lift_all_frames(
        images=images,
        world_points=world_points,
        world_points_conf=world_points_conf,
        processor=processor,
        sam_model=sam_model,
        text_prompt=args.text_prompt,
        sam_threshold=args.sam_threshold,
        point_conf_threshold=args.point_conf_threshold,
        out_dir=out_dir,
        device=device,
        save_per_frame_overlay=args.save_per_frame_overlay,
    )

    log_summary(stats, num_frames, total_raw, H * W, conf_pass_global)

    if not merged_object_points:
        raise RuntimeError("No 3D points extracted from any frame.")

    # --- Hole filling ---
    if args.expand_radius > 0:
        logger.info("Running BFS hole filling: radius=%.3f color_thresh=%.3f max_iters=%d",
                    args.expand_radius, args.expand_color_thresh, args.expand_iters)
        seed_pts = np.concatenate(merged_object_points)
        seed_colors = np.concatenate(merged_object_colors)
        scene_tree, scene_pts, scene_colors = build_scene_index(
            world_points, world_points_conf, images, args.point_conf_threshold
        )
        exp_pts, exp_colors = expand_segmentation(
            seed_pts, seed_colors, scene_tree, scene_pts, scene_colors,
            args.expand_radius, args.expand_color_thresh, args.expand_iters,
        )
        logger.info("Hole filling: %d seed pts → %d pts", len(seed_pts), len(exp_pts))
        merged_object_points = [exp_pts]
        merged_object_colors = [exp_colors]

    save_point_clouds(
        merged_object_points, merged_object_colors,
        world_points, world_points_conf, images,
        args.point_conf_threshold, out_dir,
    )


if __name__ == "__main__":
    main()
