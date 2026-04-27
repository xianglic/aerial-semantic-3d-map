import logging
import subprocess
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

BFS_BINARY = Path(__file__).parent / "bfs_expand"


# ========== Scene index ==========


def build_scene_index(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    images: np.ndarray,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten all valid world points and colors from all frames into flat arrays."""
    colors_hwc = images.transpose(0, 2, 3, 1)
    pts_list, colors_list = [], []
    for k in range(world_points.shape[0]):
        valid = np.isfinite(world_points[k]).all(axis=-1) & (world_points_conf[k] >= conf_threshold)
        pts_list.append(world_points[k][valid])
        colors_list.append(colors_hwc[k][valid])
    return np.concatenate(pts_list), np.concatenate(colors_list)


# ========== Filler class ==========


class Filler:
    """BFS hole filling on a 3D point cloud using radius + color similarity.

    Supports two backends:
      - "cpp": Parlay parallel C++ binary (fast). Build with: bash build_bfs.sh
      - "python": pure-Python BFS on a voxel-downsampled scene (no build needed).
    """

    def __init__(
        self,
        radius: float,
        color_thresh: float,
        max_iters: int = 20,
        backend: str = "cpp",
    ) -> None:
        self.radius = radius
        self.color_thresh = color_thresh
        self.max_iters = max_iters
        self.backend = backend
        logger.info(
            "Filler: radius=%.4f color_thresh=%.4f max_iters=%d backend=%s",
            radius, color_thresh, max_iters, backend,
        )

    def run(
        self,
        merged_object_points: List[np.ndarray],
        merged_object_colors: List[np.ndarray],
        world_points: np.ndarray,
        world_points_conf: np.ndarray,
        images: np.ndarray,
        conf_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run BFS hole filling. Returns (expanded_points, expanded_colors)."""
        seed_pts = np.concatenate(merged_object_points)
        seed_colors = np.concatenate(merged_object_colors)
        scene_pts, scene_colors = build_scene_index(world_points, world_points_conf, images, conf_threshold)

        logger.info("Building scene index: %d pts, %d seeds", len(scene_pts), len(seed_pts))

        if self.backend == "cpp":
            exp_pts, exp_colors = self._run_cpp(seed_pts, seed_colors, scene_pts, scene_colors)
        else:
            exp_pts, exp_colors = self._run_python(seed_pts, seed_colors, scene_pts, scene_colors)

        logger.info("%d seed pts → %d pts after filling", len(seed_pts), len(exp_pts))
        return exp_pts, exp_colors

    # ---- C++ backend ----

    def _run_cpp(self, seed_pts, seed_colors, scene_pts, scene_colors):
        if not BFS_BINARY.exists():
            raise FileNotFoundError(
                f"bfs_expand binary not found at {BFS_BINARY}. Build it with: bash build_bfs.sh"
            )

        N = len(scene_pts)
        ref_color = seed_colors.mean(axis=0)
        pts = np.concatenate([scene_pts, scene_colors], axis=1).astype(np.float32)

        seed_tree = KDTree(scene_pts)
        _, seed_indices = seed_tree.query(seed_pts, k=1)
        seed_indices = np.unique(seed_indices).astype(np.int32)

        header = (
            np.int64(N).tobytes() +
            np.int64(len(seed_indices)).tobytes() +
            np.int64(self.max_iters).tobytes() +
            np.float64(self.radius).tobytes() +
            np.float64(self.color_thresh).tobytes() +
            np.float64(ref_color[0]).tobytes() +
            np.float64(ref_color[1]).tobytes() +
            np.float64(ref_color[2]).tobytes()
        )
        payload = header + pts.tobytes() + seed_indices.tobytes()

        logger.info("cpp BFS: N=%d seeds=%d", N, len(seed_indices))
        proc = subprocess.run([str(BFS_BINARY)], input=payload, stdout=subprocess.PIPE, stderr=None)
        if proc.returncode != 0:
            raise RuntimeError(f"bfs_expand exited with code {proc.returncode}")

        out = proc.stdout
        n_result = np.frombuffer(out[:8], dtype=np.int64)[0]
        result_indices = np.frombuffer(out[8:8 + n_result * 4], dtype=np.int32)
        logger.info("cpp BFS: %d result pts", n_result)
        return scene_pts[result_indices], scene_colors[result_indices]

    # ---- Python backend ----

    def _run_python(self, seed_pts, seed_colors, scene_pts, scene_colors):
        bbox = scene_pts.max(axis=0) - scene_pts.min(axis=0)
        bbox_volume = max(float(np.prod(bbox)), 1e-12)
        voxel_size = (bbox_volume / 50_000) ** (1 / 3)
        logger.info("python BFS: bbox=%.3f x %.3f x %.3f voxel=%.4f", bbox[0], bbox[1], bbox[2], voxel_size)

        ds_pts, ds_colors, full_to_ds = self._voxel_downsample(scene_pts, scene_colors, voxel_size)
        logger.info("python BFS: scene %d → %d downsampled pts", len(scene_pts), len(ds_pts))

        ds_tree = KDTree(ds_pts)
        in_set = np.zeros(len(ds_pts), dtype=bool)

        _, seed_ds = ds_tree.query(seed_pts, k=1)
        seed_ds = np.unique(seed_ds)
        in_set[seed_ds] = True
        ref_color = seed_colors.mean(axis=0)
        logger.info("python BFS: %d seeds → %d downsampled voxels", len(seed_pts), len(seed_ds))

        # Initial frontier: only boundary voxels (those with out-of-set neighbors)
        neighbor_lists = ds_tree.query_ball_point(ds_pts[seed_ds], r=self.radius)
        lengths = np.array([len(n) for n in neighbor_lists], dtype=np.intp)
        if lengths.sum() > 0:
            flat_neighbors = np.concatenate([np.asarray(n, dtype=np.intp) for n in neighbor_lists])
            flat_outside = ~in_set[flat_neighbors]
            seed_has_outside = np.zeros(len(seed_ds), dtype=bool)
            starts = np.concatenate([[0], np.cumsum(lengths[:-1])])
            valid = lengths > 0
            seed_has_outside[valid] = np.logical_or.reduceat(flat_outside, starts[valid])
            frontier = seed_ds[seed_has_outside]
        else:
            frontier = seed_ds
        logger.info("python BFS: initial frontier %d boundary voxels", len(frontier))

        for iteration in range(self.max_iters):
            if len(frontier) == 0:
                break
            neighbor_lists = ds_tree.query_ball_point(ds_pts[frontier], r=self.radius)
            candidates = np.unique(np.concatenate(neighbor_lists).astype(np.intp))
            candidates = candidates[~in_set[candidates]]
            if len(candidates) == 0:
                break
            accepted = candidates[np.linalg.norm(ds_colors[candidates] - ref_color, axis=1) < self.color_thresh]
            in_set[accepted] = True
            frontier = accepted
            logger.info("python BFS iter %d: frontier=%d accepted=%d total=%d",
                        iteration + 1, len(frontier), len(accepted), int(in_set.sum()))

        ds_in_set = np.where(in_set)[0]
        full_mask = np.isin(full_to_ds, ds_in_set)
        logger.info("python BFS done: %d voxels → %d full-density pts", len(ds_in_set), int(full_mask.sum()))
        return scene_pts[full_mask], scene_colors[full_mask]

    @staticmethod
    def _voxel_downsample(points, colors, voxel_size):
        keys = np.floor(points / voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        n = counts.shape[0]
        ds_pts = np.zeros((n, 3), dtype=points.dtype)
        ds_colors = np.zeros((n, 3), dtype=colors.dtype)
        np.add.at(ds_pts, inverse, points)
        np.add.at(ds_colors, inverse, colors)
        ds_pts /= counts[:, None]
        ds_colors /= counts[:, None]
        return ds_pts, ds_colors, inverse
