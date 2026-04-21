import glob
import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

logger = logging.getLogger(__name__)


# ========== VGGT (static, no model state) ==========


def load_predictions(predictions_path: str) -> dict:
    data = np.load(predictions_path, allow_pickle=True)
    for key in ["world_points", "world_points_conf", "images"]:
        if key not in data:
            raise KeyError(f"Missing key {key!r} in {predictions_path}")
    conf = np.array(data["world_points_conf"])
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    return {
        "world_points": np.array(data["world_points"]),
        "world_points_conf": conf,
        "images": np.array(data["images"]),
    }


def run_vggt_inference(
    image_folder: str,
    model_url: str,
    device: str,
    save_path: str = None,
) -> dict:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    logger.info("Loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
    model.eval().to(device)

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    logger.info("Found %d images in %s", len(image_names), image_folder)
    images_tensor = load_and_preprocess_images(image_names).to(device)

    logger.info("Running VGGT inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            predictions = model(images_tensor)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    images_np = images_tensor.cpu().numpy()

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    raw_conf = np.array(predictions["world_points_conf"])

    if save_path is not None:
        np.savez(save_path, world_points=predictions["world_points"], world_points_conf=raw_conf, images=images_np)
        logger.info("Predictions saved to %s", save_path)

    conf = (raw_conf - raw_conf.min()) / (raw_conf.max() - raw_conf.min() + 1e-8)
    return {"world_points": predictions["world_points"], "world_points_conf": conf, "images": images_np}


# ========== Point cloud helpers ==========


def build_scene_cloud(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    images: np.ndarray,
    point_conf_threshold: float,
) -> o3d.geometry.PointCloud:
    colors_hwc = images.transpose(0, 2, 3, 1)
    pts_list, colors_list = [], []
    for k in range(world_points.shape[0]):
        valid = np.isfinite(world_points[k]).all(axis=-1) & (world_points_conf[k] >= point_conf_threshold)
        pts_list.append(world_points[k][valid])
        colors_list.append(colors_hwc[k][valid])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_list))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors_list))
    return pcd


# ========== Segmenter class ==========


class Segmenter:
    """Lifts SAM3 2D masks into 3D using VGGT world points.

    The SAM3 model is loaded once on construction and reused across calls.
    """

    def __init__(
        self,
        sam_model_name: str,
        device: str,
        sam_threshold: float = 0.7,
        point_conf_threshold: float = 0.05,
        save_per_frame_overlay: bool = False,
    ) -> None:
        self.sam_model_name = sam_model_name
        self.device = device
        self.sam_threshold = sam_threshold
        self.point_conf_threshold = point_conf_threshold
        self.save_per_frame_overlay = save_per_frame_overlay

        logger.info("Loading SAM3 model: %s on %s", sam_model_name, device)
        self.processor = Sam3Processor.from_pretrained(sam_model_name)
        self.sam_model = Sam3Model.from_pretrained(sam_model_name).to(device)
        self.sam_model.eval()

    def run(
        self,
        preds: dict,
        text_prompt: str,
        out_dir: Path,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Lift masks to 3D for all frames, log stats, save original PLYs.

        Returns (merged_object_points, merged_object_colors).
        """
        world_points = preds["world_points"]
        world_points_conf = preds["world_points_conf"]
        images = preds["images"]
        num_frames, H, W, _ = world_points.shape

        conf_flat = world_points_conf.ravel()
        total_raw = H * W * num_frames
        conf_pass_global = int(
            (np.isfinite(world_points).all(axis=-1) & (world_points_conf >= self.point_conf_threshold)).sum()
        )
        logger.info(
            "VGGT: %d frames %dx%d | conf p50=%.3f p95=%.3f | passing=%.1f%%",
            num_frames, H, W,
            np.percentile(conf_flat, 50), np.percentile(conf_flat, 95),
            100 * conf_pass_global / total_raw,
        )

        merged_object_points, merged_object_colors, stats = self._lift_all_frames(
            images, world_points, world_points_conf, text_prompt, out_dir
        )

        logger.info(
            "Summary | image_px=%d (%d/frame) | conf_pass=%d (%.1f%%) | mask_px=%d | masked_conf=%s",
            stats["image_pixels"], stats["image_pixels"] // num_frames,
            conf_pass_global, 100 * conf_pass_global / total_raw,
            stats["mask_px_orig"],
            f"{stats['conf_passing']:,} ({100 * stats['conf_passing'] / stats['raw_masked']:.1f}% of masked)"
            if stats["raw_masked"] else "0",
        )

        if not merged_object_points:
            raise RuntimeError("No 3D points extracted from any frame.")

        self._save_results(merged_object_points, merged_object_colors, world_points, world_points_conf, images, out_dir)
        return merged_object_points, merged_object_colors

    def _lift_all_frames(
        self,
        images: np.ndarray,
        world_points: np.ndarray,
        world_points_conf: np.ndarray,
        text_prompt: str,
        out_dir: Path,
    ) -> tuple[List[np.ndarray], List[np.ndarray], dict]:
        num_frames, H, W, _ = world_points.shape
        merged_pts, merged_colors = [], []
        stats = dict(image_pixels=0, mask_px_orig=0, mask_px_vggt=0, raw_masked=0, conf_passing=0)

        for k in range(num_frames):
            image_np = (images[k].transpose(1, 2, 0) * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            H_orig, W_orig = image_np.shape[:2]
            stats["image_pixels"] += H_orig * W_orig

            mask_items = self._run_sam3(image_pil, text_prompt)
            valid_masks = [item for item in mask_items if item["area"] > 0]
            if not valid_masks:
                logger.info("[frame %03d] no valid mask", k)
                continue

            color_map = images[k].transpose(1, 2, 0)
            union_mask_orig = np.zeros((H_orig, W_orig), dtype=bool)
            union_mask_vggt = np.zeros((H, W), dtype=bool)
            frame_pts, frame_colors = [], []
            frame_raw, frame_conf = 0, 0

            for item in valid_masks:
                mask = item["mask"]
                mask_vggt = (
                    cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if mask.shape != (H, W) else mask
                )
                pts, colors, n_raw, n_conf = self._lift_mask(
                    mask_vggt, world_points[k], world_points_conf[k], color_map
                )
                frame_raw += n_raw
                frame_conf += n_conf
                if len(pts) > 0:
                    frame_pts.append(pts)
                    frame_colors.append(colors)
                union_mask_orig |= mask
                union_mask_vggt |= mask_vggt

            stats["mask_px_orig"] += int(union_mask_orig.sum())
            stats["mask_px_vggt"] += int(union_mask_vggt.sum())
            stats["raw_masked"] += frame_raw
            stats["conf_passing"] += frame_conf

            logger.info(
                "[frame %03d] mask=%d raw=%d conf=%d 3d_pts=%d",
                k, int(union_mask_orig.sum()), frame_raw, frame_conf,
                sum(len(p) for p in frame_pts),
            )
            merged_pts.extend(frame_pts)
            merged_colors.extend(frame_colors)

            if self.save_per_frame_overlay:
                overlay = image_np.astype(np.float32).copy()
                overlay[union_mask_orig] = 0.65 * overlay[union_mask_orig] + 0.35 * np.array([255, 0, 0])
                Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_dir / f"overlay_{k:03d}.png")

        return merged_pts, merged_colors, stats

    def _run_sam3(self, image_pil: Image.Image, text_prompt: str) -> List[dict]:
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.sam_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        masks = result.get("masks", [])
        scores = result.get("scores")
        return [
            {"index": i,
             "mask": np.asarray(m.detach().cpu() if isinstance(m, torch.Tensor) else m).astype(bool),
             "score": float(scores[i]) if scores is not None else 0.0,
             "area": int(m.detach().cpu().sum() if isinstance(m, torch.Tensor) else np.asarray(m).sum())}
            for i, m in enumerate(masks)
        ]

    def _lift_mask(self, mask, point_map, point_conf, color_map):
        pts = point_map[mask]
        conf = point_conf[mask]
        colors = color_map[mask]
        valid = np.isfinite(pts).all(axis=1) & (conf >= self.point_conf_threshold)
        return pts[valid], colors[valid], len(pts), int(valid.sum())

    def _save_results(self, merged_pts, merged_colors, world_points, world_points_conf, images, out_dir):
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(np.concatenate(merged_pts))
        if merged_colors:
            obj_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(merged_colors))
        else:
            obj_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        scene_pcd = build_scene_cloud(world_points, world_points_conf, images, self.point_conf_threshold)
        o3d.io.write_point_cloud(str(out_dir / "segmented_object.ply"), obj_pcd)
        o3d.io.write_point_cloud(str(out_dir / "scene_context.ply"), scene_pcd)
        o3d.io.write_point_cloud(str(out_dir / "combined_overlay.ply"), scene_pcd + obj_pcd)
        logger.info("Saved segmentation PLYs to %s", out_dir)
