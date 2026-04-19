import logging

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def build_scene_index(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    images: np.ndarray,
    conf_threshold: float,
) -> tuple[KDTree, np.ndarray, np.ndarray]:
    """Flatten all valid world points and colors from all frames into a KDTree."""
    colors_hwc = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    pts_list, colors_list = [], []
    for k in range(world_points.shape[0]):
        valid = np.isfinite(world_points[k]).all(axis=-1) & (world_points_conf[k] >= conf_threshold)
        pts_list.append(world_points[k][valid])
        colors_list.append(colors_hwc[k][valid])
    all_pts = np.concatenate(pts_list)
    all_colors = np.concatenate(colors_list)
    return KDTree(all_pts), all_pts, all_colors


def expand_segmentation(
    seed_points: np.ndarray,
    seed_colors: np.ndarray,
    scene_tree: KDTree,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    radius: float,
    color_thresh: float,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel BFS region growing from seed points using 3D radius + color similarity.

    Each iteration queries the entire frontier batch at once, then accepts neighbors
    whose color is within color_thresh (L2 in [0,1] RGB) of the seed mean color.

    Returns (expanded_points, expanded_colors).
    """
    in_set = np.zeros(len(scene_points), dtype=bool)

    _, seed_indices = scene_tree.query(seed_points, k=1)
    in_set[seed_indices] = True

    ref_color = seed_colors.mean(axis=0)
    frontier = seed_indices

    for iteration in range(max_iters):
        if len(frontier) == 0:
            break
        neighbor_lists = scene_tree.query_ball_point(scene_points[frontier], r=radius)
        candidates = np.unique(np.concatenate(neighbor_lists).astype(np.intp))
        candidates = candidates[~in_set[candidates]]
        if len(candidates) == 0:
            break
        color_dists = np.linalg.norm(scene_colors[candidates] - ref_color, axis=1)
        accepted = candidates[color_dists < color_thresh]
        in_set[accepted] = True
        frontier = accepted
        logger.info("BFS iter %d: frontier=%d accepted=%d total_in_set=%d",
                    iteration + 1, len(frontier), len(accepted), int(in_set.sum()))

    result = np.where(in_set)[0]
    return scene_points[result], scene_colors[result]
