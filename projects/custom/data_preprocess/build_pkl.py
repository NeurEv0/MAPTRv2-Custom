"""
Build train.pkl / val.pkl from data/space_samples/ for the pre-annotation model.

Choices baked in (per design discussion):
  - World-axis-aligned lidar/ego frame (translation only; no yaw needed).
  - BEV perception range: 50 m x 50 m centered on ego.
  - num_points = 40 per polyline (uniform arc-length resample).
  - Drop frames where any of the 10 expected cameras is missing.
  - Sample-level 80/20 train/val split, seed=0.
  - Drop z everywhere from GT; keep z=0 plane for ego.
  - 5 classes: car-no-go, other-no-go, column-no-go, wall-no-go, outside-no-go.
  - Skip `segment` and `other` annotations.

Usage (from repo root):
    python projects/custom/data_preprocess/build_pkl.py \
        --root data/space_samples \
        --out_dir data/space_samples_processed
"""

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED_CAMS = [
    "ofilm_around_front_190_3M",
    "ofilm_around_left_190_3M",
    "ofilm_around_rear_190_3M",
    "ofilm_around_right_190_3M",
    "ofilm_surround_front_120_8M",
    "ofilm_surround_front_left_100_2M",
    "ofilm_surround_front_right_100_2M",
    "ofilm_surround_rear_100_2M",
    "ofilm_surround_rear_left_100_2M",
    "ofilm_surround_rear_right_100_2M",
]

CLASS_NAMES = [
    "car-no-go",
    "other-no-go",
    "column-no-go",
    "wall-no-go",
    "outside-no-go",
]
CLASS2ID = {c: i for i, c in enumerate(CLASS_NAMES)}

DEFAULT_BEV_X = (-25.0, 25.0)
DEFAULT_BEV_Y = (-25.0, 25.0)
DEFAULT_NUM_POINTS = 40


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def way_class_id(way_dict):
    type_dict = way_dict.get("type") or {}
    for v in type_dict.values():
        if v in CLASS2ID:
            return CLASS2ID[v]
    return -1


def pixel_to_world(pts_xyz_px, img_to_world):
    """pts_xyz_px: (N, 3); z column is passed through unchanged because
    img_to_world's z row is identity."""
    pts_h = np.hstack([pts_xyz_px, np.ones((len(pts_xyz_px), 1))])
    return (img_to_world @ pts_h.T).T[:, :3]


def resample_polyline(pts_2d, num_points):
    """Uniform arc-length resample of an (N, 2) polyline to (num_points, 2)."""
    if len(pts_2d) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(pts_2d) == 1:
        return np.tile(pts_2d[0], (num_points, 1)).astype(np.float32)
    diffs = np.diff(pts_2d, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_lens[-1]
    if total < 1e-6:
        return np.tile(pts_2d[0], (num_points, 1)).astype(np.float32)
    targets = np.linspace(0.0, total, num_points)
    out = np.zeros((num_points, 2), dtype=np.float32)
    seg_idx = np.searchsorted(cum_lens, targets, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(pts_2d) - 2)
    for i, (t, idx) in enumerate(zip(targets, seg_idx)):
        denom = max(seg_lens[idx], 1e-9)
        u = (t - cum_lens[idx]) / denom
        u = float(np.clip(u, 0.0, 1.0))
        out[i] = pts_2d[idx] + u * (pts_2d[idx + 1] - pts_2d[idx])
    return out


def _segment_rect_entry_exit(p1, p2, x_range, y_range):
    """Return parameters (t_in, t_out) of the portion of segment p1->p2 that
    lies inside the rect, using Liang-Barsky. None if no overlap."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t_in, t_out = 0.0, 1.0
    for p, d, lo, hi in (
        (p1[0], dx, x_range[0], x_range[1]),
        (p1[1], dy, y_range[0], y_range[1]),
    ):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return None
            continue
        t1 = (lo - p) / d
        t2 = (hi - p) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_in = max(t_in, t1)
        t_out = min(t_out, t2)
        if t_in > t_out:
            return None
    return t_in, t_out


def clip_polyline(pts_2d, x_range, y_range):
    """Clip a polyline against an axis-aligned rect. Returns a list of sub-
    polylines (each (M, 2) with M >= 2). Preserves vertex order."""
    out_subs = []
    current = []
    in_range = (
        (pts_2d[:, 0] >= x_range[0]) & (pts_2d[:, 0] <= x_range[1]) &
        (pts_2d[:, 1] >= y_range[0]) & (pts_2d[:, 1] <= y_range[1])
    )
    for i in range(len(pts_2d) - 1):
        p1, p2 = pts_2d[i], pts_2d[i + 1]
        seg = _segment_rect_entry_exit(p1, p2, x_range, y_range)
        if seg is None:
            if len(current) >= 2:
                out_subs.append(np.array(current))
            current = []
            continue
        t_in, t_out = seg
        entry = p1 + t_in * (p2 - p1)
        exit_ = p1 + t_out * (p2 - p1)
        if not current:
            current.append(entry)
        else:
            # Continue from previous segment's exit point if p1 is inside,
            # otherwise restart at the entry.
            if not in_range[i]:
                if len(current) >= 2:
                    out_subs.append(np.array(current))
                current = [entry]
        current.append(exit_)
        if t_out < 1.0 - 1e-9:
            # segment exits the rect before reaching p2 -> close current sub
            if len(current) >= 2:
                out_subs.append(np.array(current))
            current = []
    if len(current) >= 2:
        out_subs.append(np.array(current))
    # de-dup consecutive identical points
    cleaned = []
    for sp in out_subs:
        keep = [sp[0]]
        for q in sp[1:]:
            if np.linalg.norm(q - keep[-1]) > 1e-6:
                keep.append(q)
        if len(keep) >= 2:
            cleaned.append(np.array(keep))
    return cleaned


def find_available_frames(sample_dir, frame_to_point):
    """Return list of (frame_id_str, frame_int) where the subdir exists and
    all 10 cameras are present."""
    out = []
    for fid_str in frame_to_point.keys():
        fid = int(fid_str)
        fdir = sample_dir / str(fid)
        if not fdir.is_dir():
            continue
        ok = all((fdir / f"{c}.jpg").exists() for c in EXPECTED_CAMS)
        if ok:
            out.append((fid_str, fid))
    return out


def process_sample(sample_dir, bev_x, bev_y, num_points, stats):
    """Returns (records, num_total_frames, num_kept_frames)."""
    try:
        ann = load_json(sample_dir / "annotation.json")
        coord = load_json(sample_dir / "coord_distribution.json")
    except FileNotFoundError as e:
        stats["sample_missing_files"] += 1
        return [], 0, 0

    img_to_world = np.array(coord["img_to_world"], dtype=np.float64)
    frame_to_point = coord.get("frame_to_point", {})

    nodes_by_id = {n["id"]: n for n in ann.get("node", [])}

    polylines_world = []
    line_to_way = {}
    for w in ann.get("way", []):
        cls = way_class_id(w)
        if cls < 0:
            stats["way_unknown_class"] += 1
            continue
        for lid in w.get("ways", []):
            line_to_way[lid] = (cls, w["id"])

    for ln in ann.get("line", []):
        info = line_to_way.get(ln["id"])
        if info is None:
            stats["line_no_way"] += 1
            continue
        cls, way_id = info
        pts_px = []
        for nid in ln.get("node_tokens", []):
            n = nodes_by_id.get(nid)
            if n is None:
                continue
            try:
                pts_px.append([float(n["x"]), float(n["y"]), float(n["z"])])
            except (KeyError, ValueError, TypeError):
                continue
        if len(pts_px) < 2:
            stats["line_too_few_nodes"] += 1
            continue
        pts_px = np.array(pts_px, dtype=np.float64)
        pts_world = pixel_to_world(pts_px, img_to_world)
        polylines_world.append({
            "class_id": cls,
            "way_id": way_id,
            "line_id": ln["id"],
            "pts_world_xy": pts_world[:, :2],
        })
        stats["lines_raw_by_class"][CLASS_NAMES[cls]] += 1

    available_frames = find_available_frames(sample_dir, frame_to_point)
    n_total = len(frame_to_point)
    stats["frames_total"] += n_total
    stats["frames_dropped_missing_cam"] += (n_total - len(available_frames))

    sample_name = sample_dir.name
    records = []
    for fid_str, fid in available_frames:
        ego_px_xy = frame_to_point[fid_str]
        ego_world = pixel_to_world(
            np.array([[ego_px_xy[0], ego_px_xy[1], 0.0]], dtype=np.float64),
            img_to_world,
        )[0]
        ego_world_xy = ego_world[:2].copy()

        lidar_to_world = np.eye(4, dtype=np.float64)
        lidar_to_world[:3, 3] = [ego_world_xy[0], ego_world_xy[1], 0.0]

        lidar2img = []
        for cam in EXPECTED_CAMS:
            w2i_3x4 = np.array(coord[cam]["world_to_image"][fid_str], dtype=np.float64)
            w2i_4x4 = np.eye(4, dtype=np.float64)
            w2i_4x4[:3, :] = w2i_3x4
            lidar2img.append((w2i_4x4 @ lidar_to_world).astype(np.float32))
        lidar2img = np.stack(lidar2img, axis=0)

        img_filenames = [
            f"{sample_name}/{fid}/{cam}.jpg" for cam in EXPECTED_CAMS
        ]

        gt_polylines, gt_labels, gt_meta = [], [], []
        for poly in polylines_world:
            pts_lidar = (poly["pts_world_xy"] - ego_world_xy).astype(np.float64)
            subs = clip_polyline(pts_lidar, bev_x, bev_y)
            for sub in subs:
                if len(sub) < 2:
                    continue
                gt_polylines.append(resample_polyline(sub, num_points))
                gt_labels.append(poly["class_id"])
                gt_meta.append({
                    "way_id": poly["way_id"],
                    "line_id": poly["line_id"],
                    "n_src_points": int(len(sub)),
                })
                stats["lines_kept_by_class"][CLASS_NAMES[poly["class_id"]]] += 1
        if gt_polylines:
            gt_polylines_arr = np.stack(gt_polylines, axis=0)
        else:
            gt_polylines_arr = np.zeros((0, num_points, 2), dtype=np.float32)
        gt_labels_arr = np.array(gt_labels, dtype=np.int64)

        records.append({
            "sample_name": sample_name,
            "frame_id": fid,
            "img_filenames": img_filenames,
            "cam_names": list(EXPECTED_CAMS),
            "lidar2img": lidar2img,                  # (10, 4, 4) float32
            "ego_world_xy": ego_world_xy.astype(np.float32),
            "gt_polylines": gt_polylines_arr,        # (M, num_points, 2) float32, lidar frame
            "gt_labels": gt_labels_arr,              # (M,) int64
            "gt_meta": gt_meta,
            "image_type": ann.get("image_type", {}),
        })
        if len(gt_polylines) == 0:
            stats["frames_kept_empty_gt"] += 1
        stats["frames_kept"] += 1

    return records, n_total, len(available_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/space_samples")
    ap.add_argument("--out_dir", default="data/space_samples_processed")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_points", type=int, default=DEFAULT_NUM_POINTS)
    ap.add_argument("--bev_x", nargs=2, type=float, default=list(DEFAULT_BEV_X))
    ap.add_argument("--bev_y", nargs=2, type=float, default=list(DEFAULT_BEV_Y))
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bev_x = tuple(args.bev_x)
    bev_y = tuple(args.bev_y)

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()],
                         key=lambda p: p.name)
    rng = random.Random(args.seed)
    shuffled = list(sample_dirs)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.val_ratio)))
    val_set = {p.name for p in shuffled[:n_val]}
    train_set = {p.name for p in shuffled[n_val:]}

    stats = {
        "sample_missing_files": 0,
        "frames_total": 0,
        "frames_kept": 0,
        "frames_dropped_missing_cam": 0,
        "frames_kept_empty_gt": 0,
        "way_unknown_class": 0,
        "line_no_way": 0,
        "line_too_few_nodes": 0,
        "lines_raw_by_class": Counter(),
        "lines_kept_by_class": Counter(),
    }

    train_records, val_records = [], []
    for s in sample_dirs:
        records, n_total, n_kept = process_sample(
            s, bev_x, bev_y, args.num_points, stats,
        )
        bucket = val_records if s.name in val_set else train_records
        bucket.extend(records)

    metainfo = {
        "data_root": str(root.resolve()),
        "class_names": CLASS_NAMES,
        "num_points": args.num_points,
        "bev_x": list(bev_x),
        "bev_y": list(bev_y),
        "cam_names": list(EXPECTED_CAMS),
    }

    for name, recs in (("train", train_records), ("val", val_records)):
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump({"metainfo": metainfo, "samples": recs}, f, protocol=4)

    print("======== build_pkl summary ========")
    print(f"  total samples scanned : {len(sample_dirs)}")
    print(f"  train samples         : {len(train_set)}  (frames: {len(train_records)})")
    print(f"  val   samples         : {len(val_set)}    (frames: {len(val_records)})")
    print(f"  frames total          : {stats['frames_total']}")
    print(f"  frames kept           : {stats['frames_kept']}")
    print(f"  frames dropped (cam)  : {stats['frames_dropped_missing_cam']}")
    print(f"  frames kept w/ empty GT: {stats['frames_kept_empty_gt']}")
    print(f"  way unknown class     : {stats['way_unknown_class']}")
    print(f"  lines (raw, by class) :")
    for cls in CLASS_NAMES:
        print(f"      {cls:20s} raw={stats['lines_raw_by_class'][cls]:6d}  "
              f"kept={stats['lines_kept_by_class'][cls]:6d}")
    print(f"  output dir            : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
