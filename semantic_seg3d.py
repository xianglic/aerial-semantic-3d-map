"""
3D Point Cloud Semantic Segmentation using OpenCLIP + VGGT

Pipeline:
1. Run OpenCLIP on input images → dense per-pixel semantic feature maps
2. Run VGGT on the same images → 3D point cloud + camera poses
3. Project 2D semantic features onto 3D points via camera geometry
4. Use VGGT's tracking head to enforce cross-view feature consistency
5. (Optional) Assign semantic labels via text queries

Usage:
    uv run semantic_seg3d.py --image_folder path/to/images
    uv run semantic_seg3d.py --image_folder path/to/images \\
        --text_labels "chair" "table" "floor" "wall" \\
        --save_ply output/scene.ply
"""

import os
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: OpenCLIP dense feature extraction
# ---------------------------------------------------------------------------

def load_clip_model(model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda"):
    """Load OpenCLIP model and preprocessing transform."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def extract_clip_dense_features(
    clip_model,
    preprocess,
    image_paths: list[str],
    device: str = "cuda",
    target_size: int = 518,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Extract dense (spatial) feature maps from OpenCLIP's ViT image encoder by
    hooking into the final transformer block to collect patch tokens.

    Images are resized/cropped to target_size x target_size to match VGGT's
    preprocessing, so patch-grid coordinates are directly comparable.

    For ViT-L/14 on 518×518: grid is 37×37, feature dim = 1024.

    Returns:
        (S, G, G, D) float32 numpy array, L2-normalised per patch
    """
    from PIL import Image

    patch_tokens: dict = {}

    def _hook(_module, _input, output):
        # output: (B, N_tokens, D) — token 0 is CLS, rest are patch tokens
        patch_tokens["feats"] = output[:, 1:, :]

    handle = clip_model.visual.transformer.resblocks[-1].register_forward_hook(_hook)

    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            # Resize so the shorter side equals target_size, then centre-crop to square
            scale = target_size / min(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            nw, nh = img.size
            left = (nw - target_size) // 2
            top = (nh - target_size) // 2
            img = img.crop((left, top, left + target_size, top + target_size))
            imgs.append(preprocess(img))

        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            clip_model.encode_image(batch)

        feats = patch_tokens["feats"]  # (B, N, D)
        B, N, D = feats.shape
        grid_size = round(N ** 0.5)
        assert grid_size * grid_size == N, f"Non-square patch grid: {N} patches"

        feats = feats.reshape(B, grid_size, grid_size, D)
        feats = F.normalize(feats, dim=-1)
        all_features.append(feats.cpu().float().numpy())

    handle.remove()
    return np.concatenate(all_features, axis=0)  # (S, G, G, D)


def upsample_clip_features(clip_features: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Bilinearly upsample (S, G, G, D) patch features to (S, H, W, D).
    """
    t = torch.from_numpy(clip_features).permute(0, 3, 1, 2).float()  # (S, D, G, G)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy()  # (S, H, W, D)


# ---------------------------------------------------------------------------
# Step 2: VGGT reconstruction
# ---------------------------------------------------------------------------

def run_vggt(image_paths: list[str], device: str = "cuda"):
    """
    Load VGGT-1B from HuggingFace Hub, run inference on the images.

    Returns:
        preds:         dict of numpy arrays (batch dim squeezed)
        model:         the loaded VGGT nn.Module (still on device, for tracking)
        images_tensor: (S, 3, H, W) float tensor on device
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    print("Loading VGGT-1B...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.eval().to(device)

    print(f"Preprocessing {len(image_paths)} images for VGGT...")
    images = load_and_preprocess_images(image_paths).to(device)  # (S, 3, H, W)

    try:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    except RuntimeError:
        dtype = torch.float16
    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert all tensors to float32 numpy, drop batch dim
    preds = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            preds[k] = v.cpu().float().numpy().squeeze(0)
        else:
            preds[k] = v

    return preds, model, images


# ---------------------------------------------------------------------------
# Step 3: Project 2D features → 3D
# ---------------------------------------------------------------------------

def project_features_to_3d(
    clip_features_dense: np.ndarray,  # (S, H, W, D)
    world_points: np.ndarray,         # (S, H, W, 3)
    conf: np.ndarray,                 # (S, H, W)
    conf_threshold_pct: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten all frames into a point cloud, discard low-confidence points.

    Returns:
        points3d:   (N, 3)
        features3d: (N, D) — L2-normalised CLIP embeddings
        mask:       (S*H*W,) bool — which pixels survived the confidence cut
    """
    flat_points = world_points.reshape(-1, 3)
    flat_feats = clip_features_dense.reshape(-1, clip_features_dense.shape[-1])
    flat_conf = conf.reshape(-1)

    threshold = np.percentile(flat_conf, conf_threshold_pct)
    mask = (flat_conf >= threshold) & (flat_conf > 1e-5)

    return flat_points[mask], flat_feats[mask], mask


# ---------------------------------------------------------------------------
# Step 4: Tracking head — cross-view feature consistency
# ---------------------------------------------------------------------------

def refine_features_with_tracking(
    vggt_model,
    images_batched: torch.Tensor,  # (1, S, 3, H, W) on device
    clip_features_patch: np.ndarray,  # (S, G, G, D)
    device: str = "cuda",
    num_query_points: int = 2048,
    track_iters: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use VGGT's tracking head to propagate features across frames:

    1. Sample query points on the patch grid of frame 0.
    2. Track each query point across all S frames → coordinates + visibility.
    3. For each query, aggregate CLIP features from all visible frames weighted
       by tracking confidence, producing a view-consistent feature embedding.

    Returns:
        refined_features: (N, D) — visibility-weighted average CLIP features
        query_pts:        (N, 2) — pixel coords in frame 0
        track_coords:     (S, N, 2) — pixel coords across all frames
        vis_scores:       (S, N) — visibility in [0, 1]
    """
    S, G, _, D = clip_features_patch.shape
    patch_size = 14  # ViT-L/14

    # Build query points on the patch grid (patch centres in pixel space)
    ys = np.arange(G) * patch_size + patch_size // 2
    xs = np.arange(G) * patch_size + patch_size // 2
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    query_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)  # (G*G, 2)

    if len(query_pts) > num_query_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(query_pts), num_query_points, replace=False)
        query_pts = query_pts[idx]

    query_tensor = torch.from_numpy(query_pts).unsqueeze(0).to(device)  # (1, N, 2)

    try:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    except RuntimeError:
        dtype = torch.float16
    print(f"Running tracking head on {len(query_pts)} query points across {S} frames...")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(images_batched)
            track_list, vis_scores, conf_scores = vggt_model.track_head(
                aggregated_tokens_list,
                images=images_batched,
                patch_start_idx=patch_start_idx,
                query_points=query_tensor,
                iters=track_iters,
            )

    # Last iteration: (1, S, N, 2) → (S, N, 2)
    track_coords = track_list[-1].squeeze(0).cpu().float().numpy()
    vis = vis_scores.squeeze(0).cpu().float().numpy()  # (S, N)

    N = track_coords.shape[1]
    refined_features = np.zeros((N, D), dtype=np.float32)
    weight_sum = np.zeros(N, dtype=np.float32)

    for s in range(S):
        px = track_coords[s, :, 0]
        py = track_coords[s, :, 1]
        pi = np.clip((py / patch_size).astype(int), 0, G - 1)
        pj = np.clip((px / patch_size).astype(int), 0, G - 1)
        w = vis[s]  # (N,)
        refined_features += w[:, None] * clip_features_patch[s, pi, pj, :]
        weight_sum += w

    valid = weight_sum > 1e-6
    refined_features[valid] /= weight_sum[valid, None]

    # Fallback for never-visible points: use frame-0 patch feature
    pi0 = np.clip((query_pts[:, 1] / patch_size).astype(int), 0, G - 1)
    pj0 = np.clip((query_pts[:, 0] / patch_size).astype(int), 0, G - 1)
    refined_features[~valid] = clip_features_patch[0, pi0[~valid], pj0[~valid], :]

    # Re-normalise
    norms = np.linalg.norm(refined_features, axis=-1, keepdims=True)
    refined_features /= np.maximum(norms, 1e-8)

    return refined_features, query_pts, track_coords, vis


# ---------------------------------------------------------------------------
# Step 5 (optional): Semantic label assignment via text queries
# ---------------------------------------------------------------------------

def assign_labels(
    features3d: np.ndarray,  # (N, D)
    text_labels: list[str],
    clip_model,
    tokenizer,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine-similarity nearest-neighbour label assignment.

    Returns:
        label_ids: (N,) int indices into text_labels
        scores:    (N, L) float32 similarity matrix
    """
    texts = tokenizer(text_labels).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(texts)
        text_feats = F.normalize(text_feats.float(), dim=-1).cpu().numpy()

    scores_parts = []
    chunk = 8192
    for i in range(0, len(features3d), chunk):
        scores_parts.append(features3d[i : i + chunk] @ text_feats.T)
    scores = np.concatenate(scores_parts, axis=0)
    return scores.argmax(axis=-1).astype(np.int32), scores.astype(np.float32)


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------

# 20-colour qualitative palette
LABEL_PALETTE = np.array([
    [230, 25,  75 ], [60,  180, 75 ], [255, 225, 25 ], [0,   130, 200],
    [245, 130, 48 ], [145, 30,  180], [70,  240, 240], [240, 50,  230],
    [210, 245, 60 ], [250, 190, 212], [0,   128, 128], [220, 190, 255],
    [170, 110, 40 ], [255, 250, 200], [128, 0,   0  ], [170, 255, 195],
    [128, 128, 0  ], [255, 215, 180], [0,   0,   128], [128, 128, 128],
], dtype=np.uint8)


def save_ply(
    path: str,
    points: np.ndarray,           # (N, 3)
    rgb: np.ndarray,              # (N, 3) float [0,1] or uint8
    label_ids: np.ndarray = None, # (N,) optional
    label_names: list[str] = None,
):
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    N = len(points)
    has_labels = label_ids is not None

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if has_labels:
            f.write("property int label\n")
        f.write("end_header\n")
        for i in range(N):
            line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}"
            if has_labels:
                line += f" {label_ids[i]}"
            f.write(line + "\n")

    print(f"Saved {N} points → {path}")
    if has_labels and label_names:
        unique, counts = np.unique(label_ids, return_counts=True)
        for uid, cnt in zip(unique, counts):
            print(f"  [{uid:2d}] {label_names[uid]:20s}: {cnt:,} pts")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Collect images
    image_paths = sorted(
        glob.glob(os.path.join(args.image_folder, "*.jpg"))
        + glob.glob(os.path.join(args.image_folder, "*.jpeg"))
        + glob.glob(os.path.join(args.image_folder, "*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_folder}")
    print(f"Found {len(image_paths)} images")

    # ------------------------------------------------------------------
    # Step 1: CLIP dense features
    # ------------------------------------------------------------------
    print("\n=== Step 1: OpenCLIP dense feature extraction ===")
    clip_model, clip_preprocess, tokenizer = load_clip_model(
        model_name=args.clip_model, pretrained=args.clip_pretrained, device=device
    )
    clip_patch_feats = extract_clip_dense_features(
        clip_model, clip_preprocess, image_paths,
        device=device, batch_size=args.clip_batch_size,
    )
    S, G, _, D = clip_patch_feats.shape
    print(f"  → (S={S}, G={G}×{G}, D={D})")

    # ------------------------------------------------------------------
    # Step 2: VGGT reconstruction
    # ------------------------------------------------------------------
    print("\n=== Step 2: VGGT reconstruction ===")
    vggt_preds, vggt_model, images_tensor = run_vggt(image_paths, device=device)
    H_img, W_img = images_tensor.shape[-2:]

    world_points = unproject_depth_map_to_point_map(
        vggt_preds["depth"],
        vggt_preds["extrinsic"],
        vggt_preds["intrinsic"],
    )  # (S, H, W, 3)
    conf = vggt_preds["depth_conf"]  # (S, H, W)
    print(f"  → point cloud shape: {world_points.shape}")

    # ------------------------------------------------------------------
    # Step 3: Project features to 3D
    # ------------------------------------------------------------------
    print("\n=== Step 3: Projecting CLIP features to 3D ===")
    clip_dense = upsample_clip_features(clip_patch_feats, H_img, W_img)  # (S, H, W, D)
    points3d, features3d, conf_mask = project_features_to_3d(
        clip_dense, world_points, conf,
        conf_threshold_pct=args.conf_threshold,
    )
    # RGB colours from VGGT's stored images (S, 3, H, W)
    colors_flat = vggt_preds["images"].transpose(0, 2, 3, 1).reshape(-1, 3)
    colors3d = colors_flat[conf_mask]
    print(f"  → {len(points3d):,} 3D points retained")

    # ------------------------------------------------------------------
    # Step 4: Tracking-head consistency refinement
    # ------------------------------------------------------------------
    print("\n=== Step 4: Tracking head — cross-view consistency ===")
    images_batched = images_tensor.unsqueeze(0)  # (1, S, 3, H, W)
    refined_feats, query_pts, track_coords, vis_scores = refine_features_with_tracking(
        vggt_model, images_batched, clip_patch_feats,
        device=device,
        num_query_points=args.num_query_points,
        track_iters=args.track_iters,
    )
    print(f"  → {len(refined_feats):,} view-consistent query features")

    # ------------------------------------------------------------------
    # Step 5 (optional): Text-driven label assignment
    # ------------------------------------------------------------------
    label_ids = None
    if args.text_labels:
        print(f"\n=== Step 5: Label assignment {args.text_labels} ===")
        label_ids, scores = assign_labels(features3d, args.text_labels, clip_model, tokenizer, device)
        palette = np.tile(LABEL_PALETTE, (len(args.text_labels) // len(LABEL_PALETTE) + 1, 1))
        colors3d = (palette[label_ids % len(palette)] / 255.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    ply_path = os.path.join(args.output_dir, args.save_ply)
    save_ply(ply_path, points3d, colors3d, label_ids, args.text_labels)

    npz_path = os.path.join(args.output_dir, "semantic_pointcloud.npz")
    save_data = dict(
        points3d=points3d,
        features3d=features3d,
        colors3d=colors3d,
        refined_features=refined_feats,
        query_pts_frame0=query_pts,
        track_coords=track_coords,
        vis_scores=vis_scores,
        extrinsic=vggt_preds["extrinsic"],
        intrinsic=vggt_preds["intrinsic"],
    )
    if label_ids is not None:
        save_data["label_ids"] = label_ids
        save_data["label_names"] = np.array(args.text_labels)
    np.savez_compressed(npz_path, **save_data)
    print(f"Saved arrays → {npz_path}")

    return save_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="3D Point Cloud Semantic Segmentation (OpenCLIP + VGGT)")
    p.add_argument("--image_folder", required=True,
                   help="Path to folder containing input images")

    g = p.add_argument_group("CLIP")
    g.add_argument("--clip_model", default="ViT-L-14",
                   help="OpenCLIP model name (default: ViT-L-14)")
    g.add_argument("--clip_pretrained", default="openai",
                   help="OpenCLIP pretrained weights (default: openai)")
    g.add_argument("--clip_batch_size", type=int, default=4,
                   help="Batch size for CLIP feature extraction (default: 4)")

    g = p.add_argument_group("Reconstruction")
    g.add_argument("--conf_threshold", type=float, default=25.0,
                   help="Bottom-%%  of depth confidence points to discard (default: 25)")

    g = p.add_argument_group("Tracking")
    g.add_argument("--num_query_points", type=int, default=2048,
                   help="Number of patch-grid query points for the tracking head (default: 2048)")
    g.add_argument("--track_iters", type=int, default=4,
                   help="Tracking refinement iterations (default: 4)")

    g = p.add_argument_group("Labelling")
    g.add_argument("--text_labels", nargs="+", default=None,
                   help="Text labels for semantic segmentation, e.g. 'chair' 'table' 'floor'")

    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="output",
                   help="Directory for output files (default: output/)")
    g.add_argument("--save_ply", default="semantic_cloud.ply",
                   help="PLY filename inside output_dir (default: semantic_cloud.ply)")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)
