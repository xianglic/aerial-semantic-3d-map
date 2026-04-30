import glob
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


class Constructor:
    """Builds a 3D point cloud from images using VGGT."""

    def __init__(self, device: str, point_conf_threshold: float = 0.05) -> None:
        self.device = device
        self.point_conf_threshold = point_conf_threshold

    def run(self, image_folder: str) -> dict:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        logger.info("Loading VGGT model...")
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_MODEL_URL))
        model.eval().to(self.device)

        image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
        logger.info("Found %d images in %s", len(image_names), image_folder)
        images_tensor = load_and_preprocess_images(image_names).to(self.device)

        logger.info("Running VGGT inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                predictions = model(images_tensor)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        images_np = images_tensor.cpu().numpy()

        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        conf = np.array(predictions["world_points_conf"])
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        world_points = predictions["world_points"].copy()
        world_points[conf < self.point_conf_threshold] = np.nan
        return {
            "world_points": world_points,
            "world_points_conf": conf,
            "images": images_np,
        }
