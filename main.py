#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

from constructor import Constructor
from segmenter import Segmenter
from filler import Filler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_construction(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    for key in ["world_points", "world_points_conf", "images"]:
        if key not in data:
            raise KeyError(f"Missing key {key!r} in {path}")
    return {
        "world_points": np.array(data["world_points"]),
        "world_points_conf": np.array(data["world_points_conf"]),
        "images": np.array(data["images"]),
    }


def make_pcd(pts: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_scene_cloud(construction: dict) -> o3d.geometry.PointCloud:
    world_points = construction["world_points"]
    colors_hwc = construction["images"].transpose(0, 2, 3, 1)
    pts_list, colors_list = [], []
    for k in range(world_points.shape[0]):
        valid = np.isfinite(world_points[k]).all(axis=-1)
        pts_list.append(world_points[k][valid])
        colors_list.append(colors_hwc[k][valid])
    return make_pcd(np.concatenate(pts_list), np.concatenate(colors_list))


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["output"]["out_dir"]
    def _sub(obj):
        if isinstance(obj, str):  return obj.replace("{out_dir}", out_dir)
        if isinstance(obj, dict): return {k: _sub(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_sub(v) for v in obj]
        return obj
    return _sub(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aerial semantic 3D mapping. Edit config.yaml to configure the run."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO if cfg["output"].get("verbose") else logging.WARNING,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    out_dir          = Path(cfg["output"]["out_dir"])
    construction_dir = out_dir / "construction"
    segmentation_dir = out_dir / "segmentation"
    hole_filling_dir = out_dir / "hole_filling"

    input_config = cfg.get("input", {})
    filler_config = cfg.get("filling", {})
    construction_config = cfg.get("construction", {})
    segmentation_config = cfg.get("segmentation", {})
    output_config = cfg.get("output", {})
    mode           = input_config.get("mode", "seg_fill")
    conf_threshold = construction_config.get("point_conf_threshold", 0.05)
    device         = output_config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    

    if mode not in ("full", "seg_fill", "fill_only"):
        raise ValueError(f"input.mode must be 'full', 'seg_fill', or 'fill_only', got {mode!r}")

    # ── Step 1: get construction ───────────────────────────────────────────
    if mode == "full":
        constructor    = Constructor(device, point_conf_threshold=conf_threshold)
        construction_dir.mkdir(parents=True, exist_ok=True)
        construction = constructor.run(input_config["path"])
        np.savez(str(construction_dir / "construction.npz"),
                 world_points=construction["world_points"],
                 world_points_conf=construction["world_points_conf"],
                 images=construction["images"])
        logger.info("3D construction saved to %s", construction_dir)
    else:  # seg_fill | fill_only
        construction = load_construction(input_config["path"])

    # ── Step 2: get segmentation ───────────────────────────────────────────
    if mode == "fill_only":
        seg_ply = segmentation_dir / "segmented_object.ply"
        if not seg_ply.exists():
            raise FileNotFoundError(f"{seg_ply} not found — run mode: full or seg_fill first")
        obj_pcd = o3d.io.read_point_cloud(str(seg_ply))
        merged_pts = [np.asarray(obj_pcd.points)]
        merged_colors = [np.asarray(obj_pcd.colors)]
    else:  # full | seg_fill
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        segmenter = Segmenter(
            device=device,
            sam_threshold=segmentation_config.get("sam_threshold", 0.7),
            save_per_frame_overlay=True,
        )
        merged_pts, merged_colors = segmenter.run(construction, segmentation_config["text_prompt"], segmentation_dir)
        scene_pcd = build_scene_cloud(construction)
        obj_pcd = make_pcd(np.concatenate(merged_pts), np.concatenate(merged_colors))
        o3d.io.write_point_cloud(str(segmentation_dir / "segmented_object.ply"), obj_pcd)
        o3d.io.write_point_cloud(str(segmentation_dir / "scene_context.ply"), scene_pcd)
        o3d.io.write_point_cloud(str(segmentation_dir / "combined_overlay.ply"), scene_pcd + obj_pcd)
        logger.info("Saved segmentation PLYs to %s", segmentation_dir)

    # ── Step 3: fill ──────────────────────────────────────────────────────
    filler = Filler(
        radius=filler_config["radius"],
        color_thresh=filler_config.get("color_thresh", 0.15),
        max_iters=filler_config.get("max_iters", 20),
        backend=filler_config.get("backend", "cpp"),
        downsample=filler_config.get("downsample", True),
    ) if filler_config.get("enabled", False) else None

    if filler is not None:
        hole_filling_dir.mkdir(parents=True, exist_ok=True)
        exp_pts, exp_colors = filler.run(
            merged_object_points=merged_pts,
            merged_object_colors=merged_colors,
            world_points=construction["world_points"],
            images=construction["images"],
        )
        scene_pcd = build_scene_cloud(construction)
        exp_pcd = make_pcd(exp_pts, exp_colors)
        o3d.io.write_point_cloud(str(hole_filling_dir / "segmented_object_filled.ply"), exp_pcd)
        o3d.io.write_point_cloud(str(hole_filling_dir / "combined_overlay_filled.ply"), scene_pcd + exp_pcd)
        logger.info("Saved hole-filled PLYs to %s", hole_filling_dir)


if __name__ == "__main__":
    main()
