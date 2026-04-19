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


def make_overlay(image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_np.astype(np.float32).copy()
    overlay[mask] = 0.65 * overlay[mask] + 0.35 * np.array([255, 0, 0], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def lift_mask_to_3d_points(
    mask: np.ndarray,
    point_map: np.ndarray,
    point_conf: np.ndarray,
    color_map: np.ndarray,
    conf_thresh: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Returns (points, colors, n_raw_masked, n_conf_passing)."""
    pts = point_map[mask]
    conf = point_conf[mask]
    colors = color_map[mask]
    valid = np.isfinite(pts).all(axis=1) & (conf >= conf_thresh)
    return pts[valid], colors[valid], len(pts), int(valid.sum())


def build_scene_cloud(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    images: np.ndarray,
    point_conf_threshold: float,
) -> o3d.geometry.PointCloud:
    """Build a colored Open3D point cloud from all valid VGGT world points."""
    colors_hwc = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    pts_list, colors_list = [], []
    for k in range(world_points.shape[0]):
        valid = np.isfinite(world_points[k]).all(axis=-1) & (world_points_conf[k] >= point_conf_threshold)
        pts_list.append(world_points[k][valid])
        colors_list.append(colors_hwc[k][valid])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_list))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors_list))
    return pcd


# ========== VGGT ==========


def load_predictions(predictions_path: str, use_depth_world_points: bool) -> dict:
    data = np.load(predictions_path, allow_pickle=True)
    point_key = "world_points_from_depth" if use_depth_world_points else "world_points"

    for key in [point_key, "world_points_conf", "images"]:
        if key not in data:
            raise KeyError(f"Missing key {key!r} in {predictions_path}")

    conf = np.array(data["world_points_conf"])
    conf_min, conf_max = conf.min(), conf.max()
    conf = (conf - conf_min) / (conf_max - conf_min + 1e-8)

    return {
        "world_points": np.array(data[point_key]),
        "world_points_conf": conf,
        "images": np.array(data["images"]),
    }


def run_vggt_inference(
    image_folder: str,
    use_depth_world_points: bool,
    model_url: str,
    save_path: str = None,
) -> dict:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
    model.eval()
    model = model.to(device)

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    logger.info("Found %d images in %s", len(image_names), image_folder)

    images_tensor = load_and_preprocess_images(image_names).to(device)
    logger.info("Preprocessed images shape: %s", images_tensor.shape)

    logger.info("Running VGGT inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_tensor)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    images_np = images_tensor.cpu().numpy()

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # squeeze batch dim added by model()

    if use_depth_world_points:
        predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
            predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
        )

    point_key = "world_points_from_depth" if use_depth_world_points else "world_points"
    raw_conf = np.array(predictions["world_points_conf"])

    if save_path is not None:
        logger.info("Saving predictions to %s...", save_path)
        save_dict = {
            "world_points": predictions["world_points"],
            "world_points_conf": raw_conf,
            "images": images_np,
        }
        if "world_points_from_depth" in predictions:
            save_dict["world_points_from_depth"] = predictions["world_points_from_depth"]
        np.savez(save_path, **save_dict)
        logger.info("Predictions saved to %s", save_path)

    conf_min, conf_max = raw_conf.min(), raw_conf.max()
    conf_normalized = (raw_conf - conf_min) / (conf_max - conf_min + 1e-8)

    return {
        "world_points": predictions[point_key],
        "world_points_conf": conf_normalized,
        "images": images_np,
    }


# ========== SAM3 ==========


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


# ========== LIFT ==========


def lift_all_frames(
    images: np.ndarray,
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    processor: Sam3Processor,
    sam_model: Sam3Model,
    text_prompt: str,
    sam_threshold: float,
    point_conf_threshold: float,
    out_dir: Path,
    device: str,
    save_per_frame_overlay: bool = False,
) -> tuple[List[np.ndarray], List[np.ndarray], dict]:
    """Run SAM3 on each frame and lift masked pixels to 3D.

    Returns (merged_object_points, merged_object_colors, stats).
    """
    num_frames, H, W, _ = world_points.shape
    merged_object_points = []
    merged_object_colors = []
    stats = dict(image_pixels=0, mask_px_orig=0, mask_px_vggt=0, raw_masked=0, conf_passing=0)

    for k in range(num_frames):
        image_np = (images[k].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        H_orig, W_orig = image_np.shape[:2]
        image_pixels = H_orig * W_orig
        stats["image_pixels"] += image_pixels

        mask_items = run_sam3_on_image(
            image_pil=image_pil,
            text_prompt=text_prompt,
            processor=processor,
            model=sam_model,
            device=device,
            threshold=sam_threshold,
        )

        valid_masks = [item for item in mask_items if item["area"] > 0]
        if not valid_masks:
            logger.info("[%03d] image=%dx%d (%d px) | no valid mask", k, H_orig, W_orig, image_pixels)
            continue

        frame_pts = []
        frame_colors = []
        color_map = images[k].transpose(1, 2, 0)  # (H, W, 3) float [0, 1]
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
            pts, colors, n_raw, n_conf = lift_mask_to_3d_points(
                mask=mask_for_vggt,
                point_map=world_points[k],
                point_conf=world_points_conf[k],
                color_map=color_map,
                conf_thresh=point_conf_threshold,
            )
            frame_raw_masked += n_raw
            frame_conf_passing += n_conf
            if len(pts) > 0:
                frame_pts.append(pts)
                frame_colors.append(colors)
            union_mask_orig |= mask
            union_mask_vggt |= mask_for_vggt

        mask_px_orig = int(union_mask_orig.sum())
        mask_px_vggt = int(union_mask_vggt.sum())
        stats["mask_px_orig"] += mask_px_orig
        stats["mask_px_vggt"] += mask_px_vggt
        stats["raw_masked"] += frame_raw_masked
        stats["conf_passing"] += frame_conf_passing

        pct_img = 100 * mask_px_orig / image_pixels if image_pixels else 0
        pct_conf = 100 * frame_conf_passing / frame_raw_masked if frame_raw_masked else 0
        logger.info(
            "[%03d] image=%dx%d (%d px) | mask_orig=%d (%.1f%%) | mask_vggt=%d | "
            "raw_masked=%d | conf_pass=%d (%.1f%%) | 3d_pts=%d",
            k, H_orig, W_orig, image_pixels,
            mask_px_orig, pct_img, mask_px_vggt,
            frame_raw_masked, frame_conf_passing, pct_conf,
            sum(len(p) for p in frame_pts),
        )
        merged_object_points.extend(frame_pts)
        merged_object_colors.extend(frame_colors)

        if save_per_frame_overlay:
            Image.fromarray(make_overlay(image_np, union_mask_orig)).save(out_dir / f"overlay_{k:03d}.png")

    return merged_object_points, merged_object_colors, stats
