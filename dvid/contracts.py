"""Shape/dtype checks for the shared interface. See CONVENTIONS.md.

Call these at the start and end of every interface function so that a
convention mismatch fails immediately and with a clear message.
"""
import numpy as np

W, H = 1024, 576


def check_rgb(img, name="rgb"):
    assert isinstance(img, np.ndarray), f"{name}: expected np.ndarray, got {type(img)}"
    assert img.shape == (H, W, 3), f"{name}: expected shape {(H, W, 3)}, got {img.shape}"
    assert img.dtype == np.uint8, f"{name}: expected uint8, got {img.dtype}"


def check_mask(mask, name="mask"):
    assert isinstance(mask, np.ndarray), f"{name}: expected np.ndarray, got {type(mask)}"
    assert mask.shape == (H, W), f"{name}: expected shape {(H, W)}, got {mask.shape}"
    assert mask.dtype == bool, f"{name}: expected bool, got {mask.dtype}"


def check_depth(depth, name="depth"):
    assert isinstance(depth, np.ndarray), f"{name}: expected np.ndarray, got {type(depth)}"
    assert depth.shape == (H, W), f"{name}: expected shape {(H, W)}, got {depth.shape}"
    assert depth.dtype == np.float32, f"{name}: expected float32, got {depth.dtype}"
    assert np.all(depth >= 0), f"{name}: negative depth found (0 = invalid)"


def check_K(K, name="K"):
    assert isinstance(K, np.ndarray) and K.shape == (3, 3), f"{name}: expected (3,3), got {getattr(K, 'shape', None)}"
    assert K.dtype == np.float64, f"{name}: expected float64, got {K.dtype}"
    assert K[2, 2] == 1 and K[0, 1] == 0, f"{name}: not a pinhole intrinsics matrix"


def check_pose(T, name="pose"):
    assert isinstance(T, np.ndarray) and T.shape == (4, 4), f"{name}: expected (4,4), got {getattr(T, 'shape', None)}"
    assert T.dtype == np.float64, f"{name}: expected float64, got {T.dtype}"
    assert np.allclose(T[3], [0, 0, 0, 1]), f"{name}: last row must be [0,0,0,1]"
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-4), f"{name}: rotation part is not orthonormal"
