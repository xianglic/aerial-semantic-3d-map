#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import open3d as o3d
import yaml

from segmenter import Segmenter, build_scene_cloud, load_predictions, run_vggt_inference
from filler import Filler

logger = logging.getLogger(__name__)

VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
SAM_MODEL = "facebook/sam3"
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aerial semantic 3D mapping. Edit config.yaml to configure the run."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"Path to YAML config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="Enable verbose logging (overrides output.verbose in config)")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    if args.verbose:
        cfg["output"]["verbose"] = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)

    logging.basicConfig(
        level=logging.INFO if cfg["output"].get("verbose") else logging.WARNING,
        format="%(levelname)s [%(name)s]: %(message)s",
    )
    logger.info("Config: %s", cfg)

    out_dir = Path(cfg["output"]["out_dir"])
    predictions_dir = out_dir / "predictions"
    segmentation_dir = out_dir / "segmentation"
    hole_filling_dir = out_dir / "hole_filling"

    for d in [predictions_dir, segmentation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inp = cfg["input"]

    # --- VGGT ---
    mode = inp.get("mode", "load")
    path = inp["path"]

    if mode == "load":
        preds = load_predictions(path)
    elif mode == "inference":
        save_path = predictions_dir / "preds.npz"
        preds = run_vggt_inference(image_folder=path, model_url=VGGT_MODEL_URL, save_path=str(save_path))
        logger.info("Saved VGGT predictions to %s", save_path)
    else:
        raise ValueError(f"input.mode must be 'load' or 'inference', got {mode!r}")

    # --- Segmentation ---
    seg = cfg["segmentation"]
    segmenter = Segmenter(
        sam_model_name=SAM_MODEL,
        device=device,
        sam_threshold=seg.get("sam_threshold", 0.7),
        point_conf_threshold=inp.get("point_conf_threshold", 0.05),
        save_per_frame_overlay=True,
    )
    merged_object_points, merged_object_colors = segmenter.run(preds, seg["text_prompt"], segmentation_dir)

    # --- Hole filling ---
    fil = cfg.get("filling", {})
    if fil.get("enabled", False):
        hole_filling_dir.mkdir(parents=True, exist_ok=True)

        filler = Filler(
            radius=fil["radius"],
            color_thresh=fil.get("color_thresh", 0.15),
            max_iters=fil.get("max_iters", 20),
            backend=fil.get("backend", "cpp"),
        )
        exp_pts, exp_colors = filler.run(
            merged_object_points=merged_object_points,
            merged_object_colors=merged_object_colors,
            world_points=preds["world_points"],
            world_points_conf=preds["world_points_conf"],
            images=preds["images"],
            conf_threshold=inp.get("point_conf_threshold", 0.05),
        )

        exp_pcd = o3d.geometry.PointCloud()
        exp_pcd.points = o3d.utility.Vector3dVector(exp_pts)
        exp_pcd.colors = o3d.utility.Vector3dVector(exp_colors)
        scene_pcd = build_scene_cloud(
            preds["world_points"], preds["world_points_conf"], preds["images"],
            inp.get("point_conf_threshold", 0.05),
        )
        o3d.io.write_point_cloud(str(hole_filling_dir / "segmented_object_filled.ply"), exp_pcd)
        o3d.io.write_point_cloud(str(hole_filling_dir / "combined_overlay_filled.ply"), scene_pcd + exp_pcd)
        logger.info("Saved hole-filled PLYs to %s", hole_filling_dir)


if __name__ == "__main__":
    main()
