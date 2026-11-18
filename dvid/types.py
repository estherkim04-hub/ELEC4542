from dataclasses import dataclass

import numpy as np

from .contracts import check_K


@dataclass(frozen=True)
class Scene:
    """Coloured point cloud in world coordinates (= source camera frame), metres."""

    points: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 3) uint8, RGB
    K: np.ndarray       # (3, 3) float64, source-view intrinsics in pixels

    def __post_init__(self):
        assert self.points.ndim == 2 and self.points.shape[1] == 3, f"points: expected (N,3), got {self.points.shape}"
        assert self.points.dtype == np.float32, f"points: expected float32, got {self.points.dtype}"
        assert self.colors.shape == self.points.shape, f"colors {self.colors.shape} must match points {self.points.shape}"
        assert self.colors.dtype == np.uint8, f"colors: expected uint8, got {self.colors.dtype}"
        check_K(self.K)

    def __len__(self):
        return len(self.points)
