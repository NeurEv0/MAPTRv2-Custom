"""Visualize raw `segment` and `other` annotations for one parking sample.

Unlike `visualize_gt.py`, this script reads the raw sample directory because the
training PKL intentionally drops `segment` and `other`. It renders:
  - all 10 camera views for a selected frame
  - an ego-centered BEV panel
  - the original top-down annotation map (`rgb_lane_map.png`)
  - a compact summary panel

Usage (from repo root):
    python projects/custom/data_preprocess/visualize_segment_other.py \
        --root data/space_samples \
        --sample 车道线132 \
        --frame-id 0

    python projects/custom/data_preprocess/visualize_segment_other.py \
        --sample-dir data/space_samples/车道线132 \
        --all-frames \
        --out data/space_samples_processed/vis_segment_other
"""

import argparse
import json
import os
import os.path as osp
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Glyph .* missing from current font")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from PIL import Image


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

DEFAULT_BEV_X = (-25.0, 25.0)
DEFAULT_BEV_Y = (-25.0, 25.0)
DEFAULT_SEG_COLOR = "#5414ED"
DEFAULT_OTHER_COLOR = "#60D024"
PANEL_BG = "#202020"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pixel_to_world(pts_xyz_px, img_to_world):
    pts_h = np.hstack([pts_xyz_px, np.ones((len(pts_xyz_px), 1), dtype=np.float64)])
    return (img_to_world @ pts_h.T).T[:, :3]


def project_points(pts_xyz, proj_mat, eps=0.1):
    pts_xyz = np.asarray(pts_xyz, dtype=np.float64)
    if pts_xyz.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=bool)
    if pts_xyz.ndim != 2 or pts_xyz.shape[1] < 2:
        raise ValueError(f"Expected (N, 2+) points, got shape {pts_xyz.shape}")

    if pts_xyz.shape[1] == 2:
        pts_xyz = np.concatenate(
            [pts_xyz, np.zeros((len(pts_xyz), 1), dtype=np.float64)],
            axis=1,
        )
    proj_mat = np.asarray(proj_mat, dtype=np.float64)
    proj_mat = proj_mat[:3, :4]

    homo = np.concatenate(
        [pts_xyz[:, :3], np.ones((len(pts_xyz), 1), dtype=np.float64)],
        axis=1,
    )
    proj = (proj_mat @ homo.T).T
    depth = proj[:, 2]
    valid = depth > eps
    denom = np.where(np.abs(depth) > 1e-6, depth, 1e-6)
    uv = proj[:, :2] / denom[:, None]
    return uv[:, 0], uv[:, 1], valid


def draw_projected_path(ax, pts_xyz, proj_mat, color, linewidth, closed=False, alpha=0.95):
    pts_xyz = np.asarray(pts_xyz, dtype=np.float64)
    if len(pts_xyz) < 2:
        return False
    if closed:
        pts_xyz = np.concatenate([pts_xyz, pts_xyz[:1]], axis=0)
    u, v, in_front = project_points(pts_xyz, proj_mat)
    drew_any = False
    starts = np.where(in_front[:-1] & in_front[1:])[0]
    for idx in starts:
        ax.plot(
            [u[idx], u[idx + 1]],
            [v[idx], v[idx + 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        drew_any = True
    return drew_any


def safe_color(raw_color, fallback):
    if not raw_color:
        return fallback
    try:
        matplotlib.colors.to_rgba(raw_color)
    except ValueError:
        return fallback
    return raw_color


def format_type_dict(type_dict):
    if not type_dict:
        return ""
    pairs = [f"{k}={v}" for k, v in type_dict.items()]
    return ", ".join(pairs)


def label_anchor(points_xy):
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if len(points_xy) == 0:
        return np.zeros((2,), dtype=np.float64)
    return points_xy[len(points_xy) // 2]


def load_image_if_exists(path):
    if not path.exists():
        return None
    return np.array(Image.open(path))


def build_way_lookup(ann):
    line_to_way = {}
    for way in ann.get("way", []):
        class_name = ""
        for value in (way.get("type") or {}).values():
            class_name = str(value)
            break
        for line_id in way.get("ways", []):
            line_to_way[line_id] = {
                "way_id": way.get("id", ""),
                "class_name": class_name,
                "type_str": way.get("typeStr", ""),
            }
    return line_to_way


def parse_segment_entries(ann, img_to_world):
    nodes_by_id = {node.get("id"): node for node in ann.get("node", [])}
    lines_by_id = {line.get("id"): line for line in ann.get("line", [])}
    line_to_way = build_way_lookup(ann)

    segments = []
    stats = Counter()
    for seg_idx, segment in enumerate(ann.get("segment", []), start=1):
        seg_lines = []
        for line_id in segment.get("segs", []):
            line = lines_by_id.get(line_id)
            if line is None:
                stats["missing_line_ref"] += 1
                continue

            pts_px = []
            for node_id in line.get("node_tokens", []):
                node = nodes_by_id.get(node_id)
                if node is None:
                    stats["missing_node_ref"] += 1
                    continue
                try:
                    pts_px.append([
                        float(node["x"]),
                        float(node["y"]),
                        float(node.get("z", 0.0)),
                    ])
                except (KeyError, TypeError, ValueError):
                    stats["bad_node_value"] += 1

            if len(pts_px) < 2:
                stats["segment_line_too_short"] += 1
                continue

            pts_px = np.asarray(pts_px, dtype=np.float64)
            pts_world = pixel_to_world(pts_px, img_to_world)
            seg_lines.append({
                "line_id": line_id,
                "way_id": line_to_way.get(line_id, {}).get("way_id", ""),
                "class_name": line_to_way.get(line_id, {}).get("class_name", ""),
                "pts_px": pts_px,
                "pts_world": pts_world,
            })

        segments.append({
            "id": segment.get("id", f"seg-{seg_idx}"),
            "line_refs": list(segment.get("segs", [])),
            "lines": seg_lines,
            "color": safe_color(segment.get("color"), DEFAULT_SEG_COLOR),
            "type_str": segment.get("typeStr", ""),
        })
    return segments, stats


def parse_other_entries(ann, img_to_world):
    others = []
    stats = Counter()
    for other_idx, other in enumerate(ann.get("other", []), start=1):
        pts_px = []
        for point in other.get("points", []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                z_val = float(point[2]) if len(point) >= 3 else 0.0
                pts_px.append([float(point[0]), float(point[1]), z_val])
            except (TypeError, ValueError):
                stats["bad_other_point"] += 1

        if len(pts_px) < 3:
            stats["other_polygon_too_short"] += 1
            continue

        pts_px = np.asarray(pts_px, dtype=np.float64)
        pts_world = pixel_to_world(pts_px, img_to_world)
        others.append({
            "id": other.get("id", f"other-{other_idx}"),
            "region": other.get("region", ""),
            "type_desc": format_type_dict(other.get("type") or {}),
            "color": safe_color(other.get("color"), DEFAULT_OTHER_COLOR),
            "pts_px": pts_px,
            "pts_world": pts_world,
        })
    return others, stats


def frame_projection_by_cam(coord, frame_id_str):
    out = {}
    for cam in EXPECTED_CAMS:
        cam_dict = coord.get(cam, {})
        world_to_image = cam_dict.get("world_to_image", {})
        if frame_id_str in world_to_image:
            out[cam] = np.asarray(world_to_image[frame_id_str], dtype=np.float64)
    return out


def summarize_counter(counter):
    if not counter:
        return "none"
    return ", ".join(f"{k}:{v}" for k, v in counter.items())


def render_camera_panel(ax, image, cam_name, proj_mat, segments, others):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(cam_name.replace("ofilm_", ""), fontsize=8)
    ax.axis("off")

    if image is None:
        ax.text(
            0.5, 0.5, "missing image",
            transform=ax.transAxes,
            color="white",
            ha="center",
            va="center",
            fontsize=9,
        )
        return

    height, width = image.shape[:2]
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    if proj_mat is None:
        ax.text(
            0.5, 0.5, "missing projection",
            transform=ax.transAxes,
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.55, pad=3),
        )
        return

    visible_segments = 0
    for segment in segments:
        seg_visible = False
        for line in segment["lines"]:
            seg_visible |= draw_projected_path(
                ax,
                line["pts_world"],
                proj_mat,
                color=segment["color"],
                linewidth=1.2,
                closed=False,
            )
        visible_segments += int(seg_visible)

    visible_others = 0
    for other in others:
        other_visible = draw_projected_path(
            ax,
            other["pts_world"],
            proj_mat,
            color=other["color"],
            linewidth=1.6,
            closed=True,
        )
        visible_others += int(other_visible)

    drawable_segments = sum(bool(segment["lines"]) for segment in segments)
    ax.text(
        8, 28,
        f"seg {visible_segments}/{drawable_segments}  other {visible_others}/{len(others)}",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.55, pad=2),
    )


def render_bev_panel(ax, segments, others, ego_world_xy, bev_x, bev_y, annotate_ids):
    ax.set_facecolor(PANEL_BG)
    for segment in segments:
        for line in segment["lines"]:
            pts_xy = line["pts_world"][:, :2] - ego_world_xy[None, :]
            ax.plot(
                pts_xy[:, 0],
                pts_xy[:, 1],
                color=segment["color"],
                linewidth=1.4,
                alpha=0.95,
            )
        if annotate_ids and segment["lines"]:
            anchor = label_anchor(segment["lines"][0]["pts_world"][:, :2] - ego_world_xy[None, :])
            ax.text(
                anchor[0],
                anchor[1],
                segment["id"],
                color="white",
                fontsize=6,
                bbox=dict(facecolor="black", alpha=0.45, pad=1),
            )

    for other in others:
        pts_xy = other["pts_world"][:, :2] - ego_world_xy[None, :]
        patch = MplPolygon(
            pts_xy,
            closed=True,
            facecolor=other["color"],
            edgecolor=other["color"],
            alpha=0.22,
            linewidth=1.8,
        )
        ax.add_patch(patch)
        if annotate_ids:
            anchor = np.mean(pts_xy, axis=0)
            ax.text(
                anchor[0],
                anchor[1],
                other["id"],
                color="white",
                fontsize=6,
                bbox=dict(facecolor="black", alpha=0.45, pad=1),
            )

    ax.plot(0.0, 0.0, marker="o", color="white", markersize=8, markeredgecolor="black")
    ax.set_xlim(bev_x[0], bev_x[1])
    ax.set_ylim(bev_y[0], bev_y[1])
    ax.set_aspect("equal")
    ax.set_title("BEV (ego frame)", fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)


def render_global_panel(ax, map_image, segments, others, ego_px_xy, annotate_ids):
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Annotation Map (pixel space)", fontsize=9)
    ax.axis("off")

    if map_image is not None:
        height, width = map_image.shape[:2]
        ax.imshow(map_image)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    for segment in segments:
        for line in segment["lines"]:
            pts_xy = line["pts_px"][:, :2]
            ax.plot(
                pts_xy[:, 0],
                pts_xy[:, 1],
                color=segment["color"],
                linewidth=1.3,
                alpha=0.95,
            )
        if annotate_ids and segment["lines"]:
            anchor = label_anchor(segment["lines"][0]["pts_px"][:, :2])
            ax.text(
                anchor[0],
                anchor[1],
                segment["id"],
                color="white",
                fontsize=6,
                bbox=dict(facecolor="black", alpha=0.45, pad=1),
            )

    for other in others:
        pts_xy = other["pts_px"][:, :2]
        patch = MplPolygon(
            pts_xy,
            closed=True,
            facecolor=other["color"],
            edgecolor=other["color"],
            alpha=0.22,
            linewidth=1.6,
        )
        ax.add_patch(patch)
        if annotate_ids:
            anchor = np.mean(pts_xy, axis=0)
            ax.text(
                anchor[0],
                anchor[1],
                other["id"],
                color="white",
                fontsize=6,
                bbox=dict(facecolor="black", alpha=0.45, pad=1),
            )

    ax.plot(
        ego_px_xy[0],
        ego_px_xy[1],
        marker="o",
        color="white",
        markersize=8,
        markeredgecolor="black",
    )


def render_summary_panel(ax, sample_name, frame_id, ann, frame_stats, segment_parse_stats, other_parse_stats):
    ax.axis("off")
    ax.set_facecolor("white")

    segments = frame_stats["segments"]
    others = frame_stats["others"]

    seg_line_ref_card = Counter(len(segment["line_refs"]) for segment in segments)
    seg_way_classes = Counter()
    for segment in segments:
        for line in segment["lines"]:
            seg_way_classes[line["class_name"] or "<no-way-class>"] += 1

    other_types = Counter()
    for other in others:
        label = other["type_desc"] or other["region"] or "<untyped>"
        other_types[label] += 1

    y = 0.96
    lines = [
        f"{sample_name}  frame {frame_id}",
        f"scene: {ann.get('image_type', {}).get('scene', '')}",
        f"segment total: {len(ann.get('segment', []))}  drawable: {sum(bool(seg['lines']) for seg in segments)}",
        f"segment refs/segment: {summarize_counter(seg_line_ref_card)}",
        f"segment way classes: {summarize_counter(seg_way_classes)}",
        f"other total: {len(ann.get('other', []))}  drawable: {len(others)}",
        f"other types: {summarize_counter(other_types)}",
        f"ego world xy: ({frame_stats['ego_world_xy'][0]:.2f}, {frame_stats['ego_world_xy'][1]:.2f}) m",
    ]
    if segment_parse_stats:
        lines.append(f"segment parse notes: {summarize_counter(segment_parse_stats)}")
    if other_parse_stats:
        lines.append(f"other parse notes: {summarize_counter(other_parse_stats)}")

    ax.text(0.0, y, lines[0], transform=ax.transAxes, fontsize=11, weight="bold")
    y -= 0.08
    ax.text(0.0, y, "Purple: segment", transform=ax.transAxes, fontsize=9, color=DEFAULT_SEG_COLOR)
    y -= 0.05
    ax.text(0.0, y, "Green: other", transform=ax.transAxes, fontsize=9, color=DEFAULT_OTHER_COLOR)
    y -= 0.08
    for line in lines[1:]:
        ax.text(0.0, y, line, transform=ax.transAxes, fontsize=9)
        y -= 0.07


def resolve_sample_dir(root, sample, sample_dir):
    if sample_dir:
        return Path(sample_dir)
    if not sample:
        raise ValueError("Provide either --sample or --sample-dir")
    return Path(root) / sample


def resolve_frame_ids(frame_to_point, frame_id, all_frames):
    frame_ids = sorted(frame_to_point.keys(), key=int)
    if not frame_ids:
        raise ValueError("No frames found in coord_distribution.frame_to_point")
    if all_frames:
        return frame_ids
    if frame_id is None:
        return [frame_ids[0]]
    matches = [fid for fid in frame_ids if int(fid) == int(frame_id)]
    if not matches:
        raise ValueError(f"Frame {frame_id} not found. Available frames start with {frame_ids[:10]}")
    return [matches[0]]


def render_frame(
    sample_dir,
    sample_name,
    frame_id_str,
    ann,
    coord,
    segments,
    others,
    segment_parse_stats,
    other_parse_stats,
    out_dir,
    dpi,
    bev_x,
    bev_y,
    annotate_ids,
):
    ego_px_xy = np.asarray(coord["frame_to_point"][frame_id_str], dtype=np.float64)[:2]
    img_to_world = np.asarray(coord["img_to_world"], dtype=np.float64)
    ego_world_xy = pixel_to_world(
        np.asarray([[ego_px_xy[0], ego_px_xy[1], 0.0]], dtype=np.float64),
        img_to_world,
    )[0, :2]

    projections = frame_projection_by_cam(coord, frame_id_str)
    map_image = load_image_if_exists(sample_dir / "rgb_lane_map.png")
    frame_dir = sample_dir / str(int(frame_id_str))

    frame_id_int = int(frame_id_str)
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    axes = axes.ravel()

    for ax in axes:
        ax.set_facecolor(PANEL_BG)

    for idx, cam_name in enumerate(EXPECTED_CAMS):
        image_path = frame_dir / f"{cam_name}.jpg"
        image = load_image_if_exists(image_path)
        render_camera_panel(
            axes[idx],
            image=image,
            cam_name=cam_name,
            proj_mat=projections.get(cam_name),
            segments=segments,
            others=others,
        )

    frame_stats = {
        "segments": segments,
        "others": others,
        "ego_world_xy": ego_world_xy,
    }
    render_bev_panel(axes[10], segments, others, ego_world_xy, bev_x, bev_y, annotate_ids)
    render_global_panel(axes[11], map_image, segments, others, ego_px_xy, annotate_ids)

    render_summary_panel(
        axes[12],
        sample_name=sample_name,
        frame_id=frame_id_int,
        ann=ann,
        frame_stats=frame_stats,
        segment_parse_stats=segment_parse_stats,
        other_parse_stats=other_parse_stats,
    )

    for ax in axes[13:]:
        ax.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"{sample_name}_frame{frame_id_int:03d}_segment_other.png")
    fig.suptitle(f"Raw segment/other visualization - {sample_name} / frame {frame_id_int}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/space_samples")
    ap.add_argument("--sample", help="sample name under --root, e.g. 车道线132")
    ap.add_argument("--sample-dir", help="explicit path to one sample directory")
    ap.add_argument("--frame-id", type=int, help="frame id to render")
    ap.add_argument("--all-frames", action="store_true", help="render every frame listed in coord_distribution.json")
    ap.add_argument("--out", default="data/space_samples_processed/vis_segment_other")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--bev-x", nargs=2, type=float, default=list(DEFAULT_BEV_X))
    ap.add_argument("--bev-y", nargs=2, type=float, default=list(DEFAULT_BEV_Y))
    ap.add_argument("--annotate-ids", action="store_true", help="label segment/other ids on the BEV and global panels")
    args = ap.parse_args()

    sample_dir = resolve_sample_dir(args.root, args.sample, args.sample_dir)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    ann = load_json(sample_dir / "annotation.json")
    coord = load_json(sample_dir / "coord_distribution.json")
    img_to_world = np.asarray(coord["img_to_world"], dtype=np.float64)

    segments, segment_parse_stats = parse_segment_entries(ann, img_to_world)
    others, other_parse_stats = parse_other_entries(ann, img_to_world)

    frame_ids = resolve_frame_ids(coord.get("frame_to_point", {}), args.frame_id, args.all_frames)
    sample_name = sample_dir.name

    print(
        f"Sample {sample_name}: {len(segments)} segment entries, "
        f"{len(others)} other entries, rendering {len(frame_ids)} frame(s)."
    )
    if segment_parse_stats:
        print(f"segment parse notes: {dict(segment_parse_stats)}")
    if other_parse_stats:
        print(f"other parse notes: {dict(other_parse_stats)}")

    written = []
    for frame_id_str in frame_ids:
        out_path = render_frame(
            sample_dir=sample_dir,
            sample_name=sample_name,
            frame_id_str=frame_id_str,
            ann=ann,
            coord=coord,
            segments=segments,
            others=others,
            segment_parse_stats=segment_parse_stats,
            other_parse_stats=other_parse_stats,
            out_dir=args.out,
            dpi=args.dpi,
            bev_x=tuple(args.bev_x),
            bev_y=tuple(args.bev_y),
            annotate_ids=args.annotate_ids,
        )
        written.append(out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
