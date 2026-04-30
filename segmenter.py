import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

logger = logging.getLogger(__name__)

SAM_MODEL = "facebook/sam3"



class Segmenter:
    def __init__(
        self,
        device: str,
        sam_threshold: float = 0.7,
        save_per_frame_overlay: bool = False,
    ) -> None:
        self.device = device
        self.sam_threshold = sam_threshold
        self.save_per_frame_overlay = save_per_frame_overlay

        logger.info("Loading SAM3 model: %s on %s", SAM_MODEL, device)
        self.processor = Sam3Processor.from_pretrained(SAM_MODEL)
        self.sam_model = Sam3Model.from_pretrained(SAM_MODEL).to(device)
        self.sam_model.eval()

    def run(
        self,
        preds: dict,
        text_prompt: str,
        out_dir: Path,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        world_points = preds["world_points"]
        world_points_conf = preds["world_points_conf"]
        images = preds["images"]
        num_frames, H, W, _ = world_points.shape

        total_raw = H * W * num_frames
        valid_global = int(np.isfinite(world_points).all(axis=-1).sum())
        logger.info(
            "VGGT: %d frames %dx%d | valid pts=%d (%.1f%%)",
            num_frames, H, W, valid_global, 100 * valid_global / total_raw,
        )

        merged_object_points, merged_object_colors, stats = self._lift_all_frames(
            images, world_points, world_points_conf, text_prompt, out_dir
        )

        logger.info(
            "Summary | image_px=%d (%d/frame) | mask_px=%d | lifted=%s",
            stats["image_pixels"], stats["image_pixels"] // num_frames,
            stats["mask_px_orig"],
            f"{stats['conf_passing']:,} ({100 * stats['conf_passing'] / stats['raw_masked']:.1f}% of masked)"
            if stats["raw_masked"] else "0",
        )

        if not merged_object_points:
            raise RuntimeError("No 3D points extracted from any frame.")

        return merged_object_points, merged_object_colors

    def _lift_all_frames(self, images, world_points, world_points_conf, text_prompt, out_dir):
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
                    mask_vggt, world_points[k], color_map
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

            logger.info("[frame %03d] mask=%d raw=%d conf=%d 3d_pts=%d",
                        k, int(union_mask_orig.sum()), frame_raw, frame_conf,
                        sum(len(p) for p in frame_pts))
            merged_pts.extend(frame_pts)
            merged_colors.extend(frame_colors)

            if self.save_per_frame_overlay:
                overlay = image_np.astype(np.float32).copy()
                overlay[union_mask_orig] = 0.65 * overlay[union_mask_orig] + 0.35 * np.array([255, 0, 0])
                Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_dir / f"overlay_{k:03d}.png")

        return merged_pts, merged_colors, stats

    def _run_sam3(self, image_pil, text_prompt):
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

    def _lift_mask(self, mask, point_map, color_map):
        pts = point_map[mask]
        colors = color_map[mask]
        valid = np.isfinite(pts).all(axis=1)
        return pts[valid], colors[valid], len(pts), int(valid.sum())

