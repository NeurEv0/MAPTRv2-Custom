"""
patched_dataset.py
──────────────────
Converts RawSampleData → SampleData by slicing the full BEV map and its
layered annotations into per-frame, yaw-aligned patches ready for model
training.
 
Coordinate-system conventions used throughout
──────────────────────────────────────────────
  Image pixel space   (u, v)   – origin top-left, x→right, y→down
  World space         (X, Y)   – from img2world (a 4×4 homogeneous matrix)
  Patch space         (p, q)   – origin top-left of the rectified patch,
                                  p→right (≈ world lateral), q→down (≈ world forward)
 
Key transforms built for every frame
──────────────────────────────────────
  img2world        (4×4)  – given; maps image pixels  →  world XY
  world2img        (4×4)  – inverse; maps world XY   →  image pixels
  patch_M          (3×3)  – perspective warp used to extract the patch
                            (maps image pixels → patch pixels)
  patch_M_inv      (3×3)  – inverse perspective warp
                            (maps patch pixels → image pixels)
  patch2world      (4×4)  – maps patch pixels → world XY
                            built as  img2world @ patch_M_inv_hom
  world2patch      (4×4)  – inverse of patch2world; maps world XY → patch pixels
 
Annotation format produced (FrameData.annotations)
────────────────────────────────────────────────────
  "line"  : {
        class_name1: [np.ndarray(N1, 2), np.ndarray(N2, 2), ...],
        class_name2: [np.ndarray(M1, 2), np.ndarray(M2, 2), ...],
        class_name3: [np.ndarray(K1, 2), np.ndarray(K2, 2), ...],
    }
 
  "way"   : {
        class_name1: [np.ndarray(N1, 2), np.ndarray(N2, 2), ...],
        class_name2: [np.ndarray(M1, 2), np.ndarray(M2, 2), ...],
        class_name3: [np.ndarray(K1, 2), np.ndarray(K2, 2), ...],
    }
 
  "other" : {
        class_name1: [np.ndarray(N1, 2), np.ndarray(N2, 2), ...],
        class_name2: [np.ndarray(M1, 2), np.ndarray(M2, 2), ...],
        class_name3: [np.ndarray(K1, 2), np.ndarray(K2, 2), ...],
    }
Each element in a list is one instance of that map element — a numpy array of shape (N, 2) where N is the number of vertices and 2 corresponds to (x, y) in the ego-centric LiDAR BEV coordinate frame.
──────────────────────────────────────────────────
"""
 
from __future__ import annotations
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
 
import cv2
import numpy as np
from tqdm import tqdm
 
from my_dataset import MyDataset, RawFrameData, RawSampleData

OTHER_TYPES = MyDataset.other_types
LINE_TYPES = MyDataset.line_types
WAY_TYPES = MyDataset.way_types

@dataclass
class FrameData:
    frame_index: int
    patch_img: np.ndarray           # rectified BEV map patch  (H, W, 3)  BGR
 
    # ── Annotations in patch pixel space ─────────────────────────────────────
    annotations: Dict[str, dict]
 
    # ── Per-camera sensor info ────────────────────────────────────────────────
    cameras: Dict[str, dict]        # {
                                    #   cam_name: {
                                    #     "img_path":                str
                                    #     "camera2world":            list (3×4 world-to-image proj)
                                    #     "patch2world":             np.ndarray (4×4)
                                    #     "world2patch":             np.ndarray (4×4)
                                    #   }
                                    # }
 
    frame_point: List[float]        # [u, v] collection-point in image pixel space
 
    # ── Coordinate transforms ─────────────────────────────────────────────────
    patch_M:     np.ndarray         # (3,3) perspective warp: image pixel → patch pixel
    patch_M_inv: np.ndarray         # (3,3) inverse warp:     patch pixel → image pixel
    patch2world: np.ndarray         # (4,4) patch pixel → world
    world2patch: np.ndarray         # (4,4) world       → patch pixel
    img2world:   np.ndarray         # (4,4) image pixel → world (inherited from sample)
 
    # ── OBB geometry ──────────────────────────────────────────────────────────
    corners_world: np.ndarray       # (4,2) OBB corners in world space
    corners_img:   np.ndarray       # (4,2) OBB corners in image pixel space
    center_world:  Tuple[float, float]
    yaw_rad:       float            # vehicle heading used for this patch
 
 
@dataclass
class SampleData:
    sample_folder: str
    sample_id: str
    frames: Dict[int, FrameData]

class PatchedDataset:
    def __init__(
        self,
        raw_dataset: Dict[str, RawSampleData],
        patch_x: float = 40.0,
        patch_y: float = 80.0,
    ):
        """
        Args:
            raw_dataset : dict  sample_id → RawSampleData
            patch_x     : patch width  perpendicular to vehicle heading (metres)
            patch_y     : patch height along vehicle heading (metres)
        """
        self.raw_dataset = raw_dataset

        self.patch_x = patch_x
        self.patch_y = patch_y
 
        self._samples: Dict[str, SampleData] = {}
        self._process_all_samples()

    def _process_single_samples(self, sample: RawSampleData) -> Optional[SampleData]:
        """
        Convert one RawSampleData into a SampleData by:
 
        1. Computing a yaw angle per frame from the ordered collection-point
           trajectory (see v_rotated_patch.py for rationale).
        2. For each frame:
           a. Building all coordinate transforms (patch_M, patch2world, …).
           b. Warping the BEV map image into a rectified patch.
           c. Building training-ready annotation targets:
                - line masks  : {type_str → (H,W) uint8}
                - way  masks  : {type_str → (H,W) uint8}
                - other       : {"rectangular"    → List[AnchorBox],
                                 "non_rectangular" → {type_str → (H,W) uint8}}
           d. Attaching per-camera sensor info with the composite transforms.
        """
        if not sample.frames:
            return None
 
        img2world   = sample.img2world
        map_img     = sample.map
        annotations = sample.annotations
 
        per_frame_yaws = _build_per_frame_yaws(sample.frame_points, img2world)
 
        raw_node    = annotations.get("node",    {})
        raw_line    = annotations.get("line",    {})
        raw_way     = annotations.get("way",     {})
        raw_other   = annotations.get("other",   {})
 
        processed_frames: Dict[int, FrameData] = {}
 
        frames_list = sample.frames if isinstance(sample.frames, list) \
                      else list(sample.frames.values())
 
        for raw_frame in frames_list:
            fidx = raw_frame.frame_index
            yaw  = per_frame_yaws.get(fidx, 0.0)
 
            # ── 1. Coordinate transforms ─────────────────────────────────────
            tf = _build_frame_transforms(
                raw_frame.frame_point, yaw, img2world,
                self.patch_x, self.patch_y,
            )
            patch_M     = tf["patch_M"]
            patch_M_inv = tf["patch_M_inv"]
            patch2world = tf["patch2world"]
            world2patch = tf["world2patch"]
            out_w       = tf["out_w"]
            out_h       = tf["out_h"]
 
            # ── 2. Warp BEV map image ─────────────────────────────────────────
            patch_img = _extract_patch_image(map_img, patch_M, out_w, out_h)
 
            # ── 3. Annotation targets ─────────────────────────────────────────
            # Nodes are an intermediate needed by line/way builders; they are
            # not stored in the final annotations dict.
            patch_nodes = _process_nodes_for_patch(raw_node, patch_M, out_w, out_h)
 
            line_masks = _build_line_masks(raw_line, patch_nodes, img2world)
            way_masks  = _build_way_masks(raw_way, raw_line, patch_nodes, img2world)
            other_anno = _build_other_annotations(raw_other, img2world)
 
            patch_annotations = {
                "line":  line_masks,
                "way":   way_masks,
                "other": other_anno,
            }
 
            # ── 4. Camera info with composite transforms ──────────────────────
            cameras: Dict[str, dict] = {}
            for cam_name, cam_info in raw_frame.cameras.items():
                cameras[cam_name] = {
                    "img_path":     cam_info.get("img_path", ""),
                    "camera2world": cam_info.get("camera2world", None),
                    "patch2world":  patch2world,
                    "world2patch":  world2patch,
                }
 
            processed_frames[fidx] = FrameData(
                frame_index=fidx,
                patch_img=patch_img,
                annotations=patch_annotations,
                cameras=cameras,
                frame_point=raw_frame.frame_point,
                patch_M=patch_M,
                patch_M_inv=patch_M_inv,
                patch2world=patch2world,
                world2patch=world2patch,
                img2world=img2world,
                corners_world=tf["corners_world"],
                corners_img=tf["corners_img"],
                center_world=tf["center_world"],
                yaw_rad=yaw,
            )
 
        return SampleData(
            sample_folder=sample.sample_folder,
            sample_id=sample.sample_id,
            frames=processed_frames,
        )
    
    def _process_all_samples(self):
        sids = self.raw_dataset.sample_ids
        process_bar = tqdm(sids, desc="Patch all samples")
        for sid in process_bar:
            sample = self.raw_dataset.get_sample(sid)
            s = self._process_single_samples(sample)
            if s is not None:
                self._samples[sid] = s
        print(f"Patch complete — {len(self._samples)} samples ready.")
        
    @property
    def sample_ids(self) -> List[str]:
        return list(self._samples.keys())
 
    def get_sample(self, sid: str) -> Optional[SampleData]:
        return self._samples.get(sid, None)
 
    def get_frame(self, sid: str, frame_idx: int) -> Optional[FrameData]:
        sample = self.get_sample(sid)
        if sample is None:
            return None
        return sample.frames.get(frame_idx, None)
 
    def __len__(self) -> int:
        return len(self._samples)
    
    @staticmethod
    def patch_to_world(
        frame: FrameData, px: float, py: float
    ) -> Tuple[float, float]:
        """Convert a patch-pixel coordinate to world (X, Y)."""
        pt = frame.patch2world @ np.array([px, py, 0.0, 1.0], dtype=np.float64)
        return float(pt[0]), float(pt[1])
 
    @staticmethod
    def world_to_patch(
        frame: FrameData, x_world: float, y_world: float
    ) -> Tuple[float, float]:
        """Convert a world (X, Y) coordinate to patch-pixel space."""
        pt = frame.world2patch @ np.array([x_world, y_world, 0.0, 1.0], dtype=np.float64)
        return float(pt[0]), float(pt[1])
 
    @staticmethod
    def patch_to_image(
        frame: FrameData, px: float, py: float
    ) -> Tuple[float, float]:
        """Convert a patch-pixel coordinate to image-pixel (u, v)."""
        src = np.array([[[px, py]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, frame.patch_M_inv.astype(np.float32))
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])
 
    @staticmethod
    def image_to_patch(
        frame: FrameData, u: float, v: float
    ) -> Tuple[float, float]:
        """Convert an image-pixel (u, v) coordinate to patch-pixel space."""
        src = np.array([[[u, v]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, frame.patch_M.astype(np.float32))
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

def _build_per_frame_yaws(
    frame_points: Dict[int, List[float]],
    img2world: np.ndarray,
) -> Dict[int, float]:
    """Forward-difference yaw from the ordered collection-point trajectory."""
    sorted_items = sorted(frame_points.items())
    n = len(sorted_items)
    if n == 0:
        return {}
 
    def px_to_world_xy(uv):
        w = img2world @ np.array([uv[0], uv[1], 0.0, 1.0], dtype=np.float64)
        return w[0], w[1]
 
    world_xy = [px_to_world_xy(uv) for _, uv in sorted_items]
 
    yaws: Dict[int, float] = {}
    for i, (frame_idx, _) in enumerate(sorted_items):
        src = world_xy[i]
        dst = world_xy[i + 1] if i < n - 1 else world_xy[i - 1]
        dx, dy = dst[0] - src[0], dst[1] - src[1]
        if np.hypot(dx, dy) < 1e-6:
            yaws[frame_idx] = yaws.get(sorted_items[i - 1][0], 0.0) if i > 0 else 0.0
        else:
            if i == n - 1:
                dx, dy = -dx, -dy
            yaws[frame_idx] = float(np.arctan2(dy, dx))
    return yaws
 
 
def _rotated_patch_corners_world(
    center_world: Tuple[float, float],
    patch_w_world: float,
    patch_h_world: float,
    yaw_rad: float,
) -> np.ndarray:
    """Four corners of the yaw-aligned OBB in world space (front-left → CW)."""
    cx, cy = center_world
    hw, hh = patch_w_world / 2.0, patch_h_world / 2.0
    local = np.array([[ hh,  hw], [ hh, -hw], [-hh, -hw], [-hh,  hw]], dtype=np.float64)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    return (R @ local.T).T + np.array([cx, cy])
 
 
def _world_corners_to_image(
    corners_world: np.ndarray,
    img2world: np.ndarray,
) -> np.ndarray:
    """Project world (x,y) corners into image pixel space."""
    world2img = np.linalg.inv(img2world)
    N = len(corners_world)
    pts = np.ones((4, N), dtype=np.float64)
    pts[0] = corners_world[:, 0]
    pts[1] = corners_world[:, 1]
    pts[2] = 0.0
    return (world2img @ pts)[:2].T
 
 
def _build_perspective_matrix(
    corners_img: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Build the 3×3 perspective matrix that maps the four OBB corners
    (front-left, front-right, rear-right, rear-left) to the axis-aligned
    patch corners (TL, TR, BR, BL).
 
    The *forward* direction (vehicle heading) maps to the top of the patch
    so that the patch is ego-centric and navigation-aligned.
    """
    src = corners_img.astype(np.float32)        # (4, 2)
    dst = np.array([
        [0,       0      ],
        [out_w-1, 0      ],
        [out_w-1, out_h-1],
        [0,       out_h-1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)  # (3, 3)
 
 
def _patch_size_pixels(
    corners_img: np.ndarray,
    patch_w_world: float,
    patch_h_world: float,
) -> Tuple[int, int]:
    """
    Derive a sensible output resolution for the patch.
 
    We compute the pixel-space distance between adjacent OBB corners to
    estimate the pixels-per-metre scale of the source image, then apply it to
    the desired world extent.  This preserves the local map resolution.
    """
    # Width direction: front-left (0) to front-right (1)
    w_px = float(np.linalg.norm(corners_img[0] - corners_img[1]))
    # Height direction: front-left (0) to rear-left (3)
    h_px = float(np.linalg.norm(corners_img[0] - corners_img[3]))
 
    # Guard against degenerate patches
    out_w = max(1, int(round(w_px)))
    out_h = max(1, int(round(h_px)))
    return out_w, out_h
 
 
def _hom44_from_3x3(M3: np.ndarray) -> np.ndarray:
    """Embed a 3×3 perspective matrix into a 4×4 homogeneous form."""
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 1], M[0, 3] = M3[0, 0], M3[0, 1], M3[0, 2]
    M[1, 0], M[1, 1], M[1, 3] = M3[1, 0], M3[1, 1], M3[1, 2]
    M[2, 2]                    = 1.0
    M[3, 0], M[3, 1], M[3, 3] = M3[2, 0], M3[2, 1], M3[2, 2]
    return M
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Transform helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_frame_transforms(
    frame_point: List[float],
    yaw_rad: float,
    img2world: np.ndarray,
    patch_w_world: float,
    patch_h_world: float,
) -> Dict:
    """
    Compute all transforms and the rectified patch for one frame.
 
    Returns a dict with:
        corners_world   (4,2) world corners of the OBB
        corners_img     (4,2) image-pixel corners
        patch_M         (3,3) perspective warp  image pixel → patch pixel
        patch_M_inv     (3,3) inverse warp       patch pixel → image pixel
        patch2world     (4,4) patch pixel → world
        world2patch     (4,4) world       → patch pixel
        out_w, out_h    output resolution
        center_world    (X, Y)
    """
    u_c = float(frame_point[0])
    v_c = float(frame_point[1])
 
    # World center
    world_pt = img2world @ np.array([u_c, v_c, 0.0, 1.0], dtype=np.float64)
    cx, cy = float(world_pt[0]), float(world_pt[1])
 
    # OBB in world and image space
    corners_w = _rotated_patch_corners_world((cx, cy), patch_w_world, patch_h_world, yaw_rad)
    corners_i = _world_corners_to_image(corners_w, img2world)
 
    out_w, out_h = _patch_size_pixels(corners_i, patch_w_world, patch_h_world)
 
    # Perspective matrix (image → patch)
    patch_M     = _build_perspective_matrix(corners_i, out_w, out_h)
    patch_M_inv = np.linalg.inv(patch_M)
 
    # Composite transforms
    # patch2world:  patch px → image px → world
    #   Use _hom44_from_3x3 so the result is (4,4) for consistent downstream use.
    patch2world = img2world @ _hom44_from_3x3(patch_M_inv)
    world2patch = np.linalg.inv(patch2world)
 
    return dict(
        corners_world=corners_w,
        corners_img=corners_i,
        patch_M=patch_M,
        patch_M_inv=patch_M_inv,
        patch2world=patch2world,
        world2patch=world2patch,
        out_w=out_w,
        out_h=out_h,
        center_world=(cx, cy),
    )
 
 
def _extract_patch_image(
    map_img: np.ndarray,
    patch_M: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Warp the map image into the rectified patch."""
    return cv2.warpPerspective(
        map_img, patch_M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities for annotation clipping
# ─────────────────────────────────────────────────────────────────────────────
 
def _img_pt_to_patch(
    x_img: float, y_img: float, patch_M: np.ndarray
) -> Tuple[float, float]:
    """Apply the 3×3 perspective transform to a single image-space point."""
    src = np.array([[[x_img, y_img]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, patch_M.astype(np.float32))
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])

# ─────────────────────────────────────────────────────────────────────────────
# Per-layer annotation processing
# ─────────────────────────────────────────────────────────────────────────────
 
def _process_nodes_for_patch(
    node_dict: Dict,
    patch_M: np.ndarray,
    out_w: int,
    out_h: int,
) -> Dict:
    """
    Transform every node from image pixel space into patch pixel space.
 
    Each output node gets:
        px, py   – coordinates in patch pixel space (float)
        in_patch – bool: True iff the point lies within the patch boundary
        (original x, y, z, type fields are preserved)
 
    Nodes are an internal intermediate; they are not part of the final
    FrameData.annotations but are used as inputs by the mask builders.
    """
    result = {}
    for nid, node in node_dict.items():
        x_img = float(node.get("x", 0.0))
        y_img = float(node.get("y", 0.0))
        px, py = _img_pt_to_patch(x_img, y_img, patch_M)
        in_patch = (0 <= px <= out_w - 1) and (0 <= py <= out_h - 1)
        result[nid] = {
            **node,
            "px": px,
            "py": py,
            "in_patch": in_patch,
        }
    return result

# ── line layer ────────────────────────────────────────────────────────────────
 
def _build_line_masks(
    line_dict: Dict,
    node_dict_patch: Dict,
    img2world: np.ndarray,
) -> Dict[str, List[np.ndarray]]:
    """
    Build per-class vertex lists for the *line* layer.
 
    Each line's node (x, y) coordinates are stored in image pixel space (as
    written by _process_node).  They are converted to world (ego-centric LiDAR
    BEV) coordinates via img2world and accumulated into lists keyed by the
    line's type string.
 
    Lines whose type is None / empty are skipped.
 
    Returns:
        {type_str: [np.ndarray(N, 2), …]}
        Each element is one line instance; columns are (x, y) in world space.
    """
    class_instances: Dict[str, List[np.ndarray]] = {}
 
    for lid, line in line_dict.items():
        class_name = line.get("type")
        if not class_name:
            continue
 
        assert class_name in LINE_TYPES, \
            f"[warning] {class_name} not in standard line types"
 
        # Collect image-space node coordinates in token order
        pts_img: List[List[float]] = []
        for tok in line.get("node_tokens", []):
            nd = node_dict_patch.get(tok)
            if nd is not None:
                pts_img.append([float(nd["x"]), float(nd["y"])])
 
        if len(pts_img) < 2:
            continue
 
        # Convert image pixel → world (ego-centric LiDAR BEV)
        pts_img_arr = np.array(pts_img, dtype=np.float64)          # (N, 2)
        ones        = np.ones((len(pts_img_arr), 1), dtype=np.float64)
        zeros       = np.zeros((len(pts_img_arr), 1), dtype=np.float64)
        hom         = np.hstack([pts_img_arr, zeros, ones])        # (N, 4)
        world_pts   = (img2world @ hom.T).T[:, :2]                 # (N, 2)
 
        if class_name not in class_instances:
            class_instances[class_name] = []
        class_instances[class_name].append(world_pts)
 
    return class_instances
 
 
 
# ── way layer ─────────────────────────────────────────────────────────────────
def _build_way_masks(
    way_dict: Dict,
    line_dict: Dict,
    node_dict_patch: Dict,
    img2world: np.ndarray,
) -> Dict[str, List[np.ndarray]]:
    """
    Build per-class vertex lists for the *way* layer.
 
    A way references one or more lines (via its `ways` field); the node tokens
    from those lines are concatenated in order to form the way polyline.  Node
    (x, y) coordinates are in image pixel space and are converted to world
    (ego-centric LiDAR BEV) coordinates via img2world.
 
    Each way's `type` field (a WAY_CN2EN English value, e.g. "chedaoxian",
    "stopline") is used as the class key.
 
    Returns:
        {type_str: [np.ndarray(N, 2), …]}
        Each element is one way instance; columns are (x, y) in world space.
    """
    class_instances: Dict[str, List[np.ndarray]] = {}
 
    for wid, way in way_dict.items():
        class_name = way.get("type")
        assert class_name in WAY_TYPES, \
            f"[warning] {class_name} not in standard way types"
 
        # Collect image-space node coordinates across all referenced lines
        pts_img: List[List[float]] = []
        for lid in way.get("ways", []):
            line = line_dict.get(lid)
            if line is None:
                continue
            for tok in line.get("node_tokens", []):
                nd = node_dict_patch.get(tok)
                if nd is not None:
                    pts_img.append([float(nd["x"]), float(nd["y"])])
 
        if len(pts_img) < 2:
            continue
 
        # Convert image pixel → world (ego-centric LiDAR BEV)
        pts_img_arr = np.array(pts_img, dtype=np.float64)          # (N, 2)
        ones        = np.ones((len(pts_img_arr), 1), dtype=np.float64)
        zeros       = np.zeros((len(pts_img_arr), 1), dtype=np.float64)
        hom         = np.hstack([pts_img_arr, zeros, ones])        # (N, 4)
        world_pts   = (img2world @ hom.T).T[:, :2]                 # (N, 2)
 
        if class_name not in class_instances:
            class_instances[class_name] = []
        class_instances[class_name].append(world_pts)
 
    return class_instances
 
 
 
# ── other layer ───────────────────────────────────────────────────────────────
 
def _build_other_annotations(
    other_dict: Dict,
    img2world: np.ndarray,
) -> Dict[str, List[np.ndarray]]:
    """
    Process the 'other' layer (landmarks, intersection areas, …) into
    per-class lists of vertex arrays in the ego-centric LiDAR BEV
    (world) coordinate frame.
 
    Source geometry
    ───────────────
    Both rectangular and non-rectangular instances carry a `points` field —
    a list of [x, y, z] (or [x, y]) triples/pairs in image pixel space, as
    written by _process_other.  Each instance is mapped to world space via
    img2world and stored as one np.ndarray(N, 2).
 
      rectangular     – 4 corner points → (4, 2) world array.
      non-rectangular – arbitrary polygon vertices → (K, 2) world array
                        (K ≥ 3 required).
 
    Returns:
        {type_str: [np.ndarray(N, 2), …]}
        Each element is one instance; columns are (x, y) in world space.
    """
    class_instances: Dict[str, List[np.ndarray]] = {}
 
    for oid, other in other_dict.items():
        type_str = str(other.get("type", None))
 
        raw_pts  = other.get("points", [])
        is_rect  = other.get("rectangular", False)
        min_pts  = 4 if is_rect else 3
 
        if len(raw_pts) < min_pts:
            continue
 
        # raw points are [x, y, z] or [x, y] in image pixel space
        pts_img = np.array(
            [[float(p[0]), float(p[1])] for p in raw_pts], dtype=np.float64
        )                                                               # (N, 2)
 
        # Convert image pixel → world (ego-centric LiDAR BEV)
        ones      = np.ones((len(pts_img), 1), dtype=np.float64)
        zeros     = np.zeros((len(pts_img), 1), dtype=np.float64)
        hom       = np.hstack([pts_img, zeros, ones])                  # (N, 4)
        world_pts = (img2world @ hom.T).T[:, :2]                       # (N, 2)
 
        if type_str not in class_instances:
            class_instances[type_str] = []
        class_instances[type_str].append(world_pts)
 
    return class_instances
 
if __name__=="__main__":
    my_dataset = MyDataset(root_folder="samples")
    patched_dataset = PatchedDataset(my_dataset)
 
    sids = patched_dataset.sample_ids
 
    from collections import Counter
    line_types = []
    way_types = []
    other_types = []
 
    for sid in sids:
        sample = patched_dataset.get_sample(sid)
        for idx, f in sample.frames.items():
            a = f.annotations
            print(a)
            line_types.extend(a['line'].keys())
            way_types.extend(a['way'].keys())
            other_types.extend(a['other'].keys())
 
    print("line\n")
    print(Counter(line_types))
    print("way\n")
    print(Counter(way_types))
    print("other\n")
    print(Counter(other_types))