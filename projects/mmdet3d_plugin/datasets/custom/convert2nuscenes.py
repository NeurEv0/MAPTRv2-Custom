"""
convert2nuscenes.py
───────────────────────
Converts a PatchedDataset into the NuScenes-style BEV map segmentation
annotation format.
"""

from __future__ import annotations

import uuid
import os
import random
import pickle
import cv2
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from scipy.linalg import rq

# ── local imports ─────────────────────────────────────────────────────────────
from my_dataset import MyDataset
from patched_dataset import PatchedDataset, FrameData, SampleData

OTHER_TYPES = MyDataset.other_types
LINE_TYPES = MyDataset.line_types
WAY_TYPES = MyDataset.way_types

def _make_token(sample_id: str, frame_idx: int) -> str:
    """Unique string token for a (sample, frame) pair."""
    return f"{sample_id}_{frame_idx:06d}"

def decompose_projection(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a 3×4 projection matrix P = K @ [R | t]
    Returns K (3×3 upper-triangular intrinsic), R (3×3 rotation), t (3,).
    """
    M = P[:, :3]
    K, R = rq(M)

    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1

    if K[2, 2] != 0:
        K = K / K[2, 2]
    t = np.linalg.inv(K) @ P[:, 3]
    return K, R, t

def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [w, x, y, z]

def _extract_cam2ego(
    camera2world: np.ndarray,
    patch2world: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """
    Derive sensor2ego_translation and sensor2ego_rotation.
 
    camera2world is the 3×4 projection matrix P stored in the dataset
    (named "camera2world" but it encodes world→image, i.e. P = K[R|t]).
    We decompose P to recover the camera centre in world space, then
    express it relative to the patch (ego) frame via world2patch.
 
    Returns (translation [3], quaternion [w,x,y,z] [4]).
    """
    P = np.asarray(camera2world, dtype=np.float64)
    if P.shape == (3, 4):
        K, R_cw, t_cw = decompose_projection(P)
        # Camera centre in world space: C_world = -R_cw^T @ t_cw
        C_world = -R_cw.T @ t_cw                       # (3,)
    else:
        # Fallback: treat as identity offset
        C_world = np.zeros(3, dtype=np.float64)
        R_cw    = np.eye(3, dtype=np.float64)
        K       = np.eye(3, dtype=np.float64)
 
    # Express camera centre in ego (patch) frame
    world2patch = np.linalg.inv(patch2world)
    C_hom       = np.array([C_world[0], C_world[1], C_world[2], 1.0])
    C_patch     = world2patch @ C_hom                  # (4,)
    t_ego       = C_patch[:3].tolist()
 
    # Rotation: camera-in-world orientation, then into patch frame
    # R_ego_cam = R_patch_world @ R_world_cam  (R_world_cam = R_cw^T)
    R_patch_world = world2patch[:3, :3]
    R_ego_cam     = R_patch_world @ R_cw.T
    q_ego         = rotation_matrix_to_quaternion(R_ego_cam)
 
    return t_ego, q_ego, K

def _extract_ego2global(patch2world: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Derive ego2global_translation and ego2global_rotation from patch2world.

    patch2world maps patch-pixel (0,0) → world (X,Y,Z).  We treat the
    patch origin as the "ego" position for this frame, so:
        translation = patch2world[:3, 3]   (the world-space origin of the patch)
        rotation    = quaternion of the upper-left 3×3 block
    """
    t = patch2world[:3, 3].tolist()
    R = patch2world[:3, :3]
    # Normalise columns to remove any scale baked into the perspective warp
    for i in range(3):
        col_norm = np.linalg.norm(R[:, i])
        if col_norm > 1e-9:
            R[:, i] /= col_norm
    q = rotation_matrix_to_quaternion(R)
    return t, q


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame info builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_frame_info(
    sample: SampleData,
    frame: FrameData,
    prev_token: str,
    next_token: str,
    field: str,
    cam_img_folder: str
) -> dict:
    """
    Build one info dict from a single FrameData, mirroring the structure of
    the NuScenes-style info.
    """
    token = _make_token(sample.sample_id, frame.frame_index)
    time_stamp = frame.frame_index * 100000

    # ── Ego pose ─────────────────────────────────────────────────────────────
    ego2global_t, ego2global_r = _extract_ego2global(frame.patch2world)

    # ── Camera info ───────────────────────────────────────────────────────────
    cams: Dict[str, dict] = {}
    for cam_name, cam_info in frame.cameras.items():
        # Derive per-camera extrinsics and intrinsics from the 3×4 projection
        # matrix stored as "camera2world" (world→image projection P = K[R|t]).
        camera2world = cam_info.get("camera2world")
        cam2ego_t, cam2ego_r, cam_intrinsic = _extract_cam2ego(
            camera2world, frame.patch2world
        )
 
        # Copy the source image to the output folder.
        old_cam_path = cam_info.get("img_path", "")
        cam_path = os.path.join(cam_img_folder, f"{token}_{cam_name}.jpg")
        src_img = cv2.imread(old_cam_path)
        if src_img is not None:
            cv2.imwrite(cam_path, src_img)
 
        cam_token = str(uuid.uuid4())
        cams[cam_name] = {
            "data_path": cam_path,
            "type": cam_name,
            "sample_data_token": cam_token,
            "sensor2ego_translation": cam2ego_t,
            "sensor2ego_rotation": cam2ego_r,
            "ego2global_translation": ego2global_t,
            "ego2global_rotation": ego2global_r,
            "sensor2lidar_rotation": None,
            "sensor2lidar_translation": None,
            "cam_intrinsic": cam_intrinsic,
            "timestamp": time_stamp
        }

    # ── Annotation ──────────────────────────
    annotation = frame.annotations.get(field,  {})

    return {
        "token":          token,
        "prev":           prev_token,
        "next":           next_token,
        "scene_token":    "puyang",     # not real
        "frame_idx":      frame.frame_index,
        "timestamp":      time_stamp,
        "map_location":   "puyang",     # not real
        
        "lidar_path": None,         # No lidar here
        "lidar2ego_translation": None,
        "lidar2ego_rotation": None,
        "ego2global_translation": ego2global_t,
        "ego2global_rotation":    ego2global_r,

        "can_bus": np.zeros(18),
        "cams": cams,
        "annotation": annotation,
        "sweeps": []   # no sweep data available
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion entry point
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    root_path: str,
    out_dir_prefix: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Convert all samples in `root_path` to BEV map segmentation info files.

    Args:
        root_path       : root folder passed to MyDataset
        out_dir         : directory where .pkl (and optionally patch PNGs) are written
        val_ratio       : fraction of samples assigned to the validation split
        save_patch_imgs : if True, write each FrameData.patch_img as a PNG
        seed            : random seed for train/val split
    """
    fields = ["other", "line", "way"]
    field_out_dirs = [f"{out_dir_prefix}_{f}" for f in fields]
    field2classes = {"other": OTHER_TYPES, "line": LINE_TYPES, "way": WAY_TYPES}

    # ── 1. Load raw → patched dataset ────────────────────────────────────────
    print("[1/3] Loading raw dataset …")
    raw_dataset = MyDataset(root_folder=root_path)

    print("[2/3] Building patched dataset …")
    patched = PatchedDataset(raw_dataset)

    # ── 2. Train / val split (by sample, not by frame) ───────────────────────
    all_sids = patched.sample_ids
    random.seed(seed)
    random.shuffle(all_sids)
    n_val   = max(1, int(len(all_sids) * val_ratio))
    val_sids   = set(all_sids[:n_val])
    train_sids = set(all_sids[n_val:])
    print(f"    Samples → train: {len(train_sids)}  val: {len(val_sids)}")

    # ── 3. Build info lists ───────────────────────────────────────────────────
    print("[3/3] Building info dicts …")
    for field, field_out_dir in zip(fields, field_out_dirs):
        os.makedirs(field_out_dir, exist_ok=True)
        train_infos: List[dict] = []
        val_infos:   List[dict] = []

        patch_img_dir = os.path.join(field_out_dir, "patch_imgs")
        os.makedirs(patch_img_dir, exist_ok=True)

        cam_imgs_dir = os.path.join(field_out_dir, "view_imgs")
        os.makedirs(cam_imgs_dir, exist_ok=True)

        for sid in tqdm(patched.sample_ids, desc="Converting samples"):
            sample = patched.get_sample(sid)
            if sample is None:
                continue

            # Sort frames by index so prev/next tokens are correct
            sorted_frame_idxs = sorted(sample.frames.keys())
            n_frames = len(sorted_frame_idxs)

            for pos, fidx in enumerate(sorted_frame_idxs):
                frame = sample.frames[fidx]

                prev_token = (
                    _make_token(sid, sorted_frame_idxs[pos - 1]) if pos > 0 else ""
                )
                next_token = (
                    _make_token(sid, sorted_frame_idxs[pos + 1])
                    if pos < n_frames - 1
                    else ""
                )

                patch_img_path = ""
                fname = f"{sid}_{fidx:06d}.png"
                patch_img_path = os.path.join(patch_img_dir, fname)
                cv2.imwrite(patch_img_path, frame.patch_img)

                info = _build_frame_info(
                    sample=sample, frame=frame, prev_token=prev_token, next_token=next_token, field=field, cam_img_folder=cam_imgs_dir
                )

                if sid in train_sids:
                    train_infos.append(info)
                else:
                    val_infos.append(info)

        metadata = {"version": "custom_v1.0"}
        classes_path = os.path.join(field_out_dir, f"{field}_classes.txt")
        with open(classes_path, "w", encoding="utf-8") as f:
            f.write("\n".join(field2classes[field]))

        train_pkl = os.path.join(field_out_dir, f"{field}_map_infos_temporal_train.pkl")
        with open(train_pkl, "wb") as f:
            pickle.dump({"infos": train_infos, "metadata": metadata}, f)
        print(f"Saved train infos ({len(train_infos)} frames) → {train_pkl}")

        val_pkl = os.path.join(field_out_dir, f"{field}_map_infos_temporal_val.pkl")
        with open(val_pkl, "wb") as f:
            pickle.dump({"infos": val_infos, "metadata": metadata}, f)
        print(f"Saved val infos   ({len(val_infos)} frames) → {val_pkl}")

    print("Done.")

if __name__ == "__main__":
    convert(
        root_path="./space_samples",
        out_dir_prefix="./my_dataset",
        val_ratio=0.2,
        seed=42,
    )