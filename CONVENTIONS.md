# Conventions

Every module follows these rules. If you need to break one, change this file first and tell your teammate.
`dvid/contracts.py` enforces them in code. Call its `check_*` functions at the start and end of every interface function.

## Resolution
- Working resolution: **W = 1024, H = 576** (16:9). Both are multiples of 8, and of 64, so SDXL inpainting works without padding.
- Resize every input (your own photos, RealEstate10K frames) to this size **once, when loading**. Scale the intrinsics by the same factor.
- Constants: `dvid.contracts.W`, `dvid.contracts.H`. Never hard-code the numbers.

## Images
- `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, **RGB** channel order, values 0–255.
- OpenCV loads images as BGR. Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` right after `cv2.imread`.

## Masks
- `np.ndarray`, shape `(H, W)`, dtype `bool`.
- `hole_mask`: `True` = no valid pixel (needs inpainting). `False` = valid rendered pixel.
- `visible_mask = ~hole_mask` (pixels reprojected from the source image or the scene).
- Cracks (small gaps from magnification) are filled by the renderer and are **not** holes.

## Depth
- `np.ndarray`, shape `(H, W)`, dtype `float32`, **metres**.
- **z-depth** (distance along the camera's z axis), *not* distance along the ray.
- `0.0` = invalid / no depth (e.g. inside holes, sky).

## Cameras
- Intrinsics `K`: `(3, 3)` `float64`, **in pixels** at the working resolution:
  ```
  [[fx, 0, cx],
   [0, fy, cy],
   [0,  0,  1]]
  ```
- Pose `T`: `(4, 4)` `float64`, **world-to-camera** (`x_cam = T @ x_world`).
- Camera axes: **OpenCV**, i.e. x right, y down, z forward (into the scene).
- World frame: the **source camera's frame**. The source pose is the identity.
- Relative pose from camera a to camera b: `T_b @ np.linalg.inv(T_a)`.
- RealEstate10K stores intrinsics as fractions of image size and poses as 3×4 world-to-camera. Convert both on load (Member 1's loader).

## Scene (point cloud)
- `dvid.types.Scene`: `points (N,3) float32` (world, metres), `colors (N,3) uint8` (RGB), `K` (intrinsics of the source view).
- The scene is only ever changed by `add_to_scene()`, which returns a **new** Scene and never edits the old one in place.

## Randomness
- Every stochastic call (diffusion inpainting) takes a `seed: int`. The default seed is `0`. Experiments use seeds `0, 1, 2`.

## Output layout
```
runs/<clip_id>/<condition>/seed<k>/
    frame_0000.png        # final RGB frame
    hole_0000.png         # hole mask (255 = hole)
    depth_0000.npy        # float32 depth in metres
    poses.json            # {"K": 3x3, "poses": [4x4, ...]} as nested lists
    timing.json           # seconds per stage per frame
```
- `<condition>` is one of `A_warp`, `B_full`, `C_selective`, `D_reuse`, `wan`.
- `<clip_id>` = the RealEstate10K clip id, or `photo_<name>` for your own photos.

## Who owns what
| Function | Owner | File |
|---|---|---|
| `render(scene, T)` | Member 1 | `dvid/geometry.py` |
| `add_to_scene(scene, rgb, hole_mask, T)` | Member 1 | `dvid/geometry.py` |
| `inpaint(rgb, hole_mask, seed)` | Member 2 | `dvid/inpainting.py` |
| `selective_composite(warped, raw, hole_mask)` | Member 2 | `dvid/inpainting.py` |

Work on your own branch (`m1-geometry`, `m2-inpainting`) and merge to `main` only when `pytest` passes.
