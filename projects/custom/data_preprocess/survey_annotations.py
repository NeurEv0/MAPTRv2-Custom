"""
Survey the schema and content distribution across all samples in
data/space_samples/ before committing to a model architecture.

Run from repo root:
    python projects/custom/data_preprocess/survey_annotations.py \
        --root data/space_samples \
        [--out survey_report.json]
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def survey_files(sample_dir: Path):
    issues = []
    expected_files = {"annotation.json", "coord_distribution.json",
                      "global.npy", "rgb_lane_map.png"}
    present = {p.name for p in sample_dir.iterdir() if p.is_file()}
    for f in expected_files:
        if f not in present:
            issues.append(f"missing file: {f}")

    frame_dirs = sorted(
        [p for p in sample_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    frame_indices = [int(p.name) for p in frame_dirs]

    per_frame_cams = {}
    for fd in frame_dirs:
        jpgs = sorted([p.stem for p in fd.iterdir() if p.suffix.lower() == ".jpg"])
        per_frame_cams[fd.name] = jpgs
        missing = set(EXPECTED_CAMS) - set(jpgs)
        extra = set(jpgs) - set(EXPECTED_CAMS)
        if missing:
            issues.append(f"frame {fd.name} missing cams: {sorted(missing)}")
        if extra:
            issues.append(f"frame {fd.name} extra cams: {sorted(extra)}")

    return {
        "n_frames": len(frame_dirs),
        "frame_indices": frame_indices,
        "issues": issues,
    }


def survey_annotation(ann: dict):
    """Pulls everything we need from one annotation.json."""
    out = {"top_level_keys": sorted(ann.keys())}

    ways = ann.get("way", [])
    lines = ann.get("line", [])
    nodes = ann.get("node", [])
    segments = ann.get("segment", [])
    others = ann.get("other", [])

    out["counts"] = {
        "way": len(ways), "line": len(lines), "node": len(nodes),
        "segment": len(segments), "other": len(others),
    }

    # ---- way ----
    way_type_keys = Counter()  # outer keys, e.g. "no-go"
    way_type_vals = Counter()  # inner values, e.g. "outside-no-go"
    way_type_str = Counter()
    way_ways_card = Counter()  # how many lines a single way references
    way_extra_fields = set()
    for w in ways:
        for k, v in (w.get("type") or {}).items():
            way_type_keys[k] += 1
            way_type_vals[f"{k}={v}"] += 1
        way_type_str[w.get("typeStr", "")] += 1
        way_ways_card[len(w.get("ways", []))] += 1
        way_extra_fields.update(w.keys())
    out["way"] = {
        "type_outer_keys": dict(way_type_keys),
        "type_kv": dict(way_type_vals),
        "typeStr": dict(way_type_str),
        "ways_cardinality": dict(way_ways_card),
        "fields_present": sorted(way_extra_fields),
    }

    # ---- line ----
    line_regions = Counter()
    line_type_keys = Counter()  # if non-empty, captures schema surprises
    line_node_lens = []
    line_fields = set()
    for ln in lines:
        line_regions[ln.get("region", "")] += 1
        for k in (ln.get("type") or {}).keys():
            line_type_keys[k] += 1
        line_node_lens.append(len(ln.get("node_tokens", [])))
        line_fields.update(ln.keys())
    out["line"] = {
        "region": dict(line_regions),
        "type_outer_keys": dict(line_type_keys),
        "node_tokens_len_minmax": (
            (min(line_node_lens), max(line_node_lens)) if line_node_lens else None
        ),
        "fields_present": sorted(line_fields),
    }

    # ---- node ----
    node_type_counts = Counter()
    custom_type_counts = Counter()
    xs, ys, zs = [], [], []
    node_fields = set()
    for n in nodes:
        node_type_counts[n.get("node_type", "")] += 1
        for ct in n.get("custom_type", []) or []:
            custom_type_counts[str(ct)] += 1
        try:
            xs.append(float(n["x"]))
            ys.append(float(n["y"]))
            zs.append(float(n["z"]))
        except (KeyError, ValueError, TypeError):
            pass
        node_fields.update(n.keys())
    out["node"] = {
        "node_type": dict(node_type_counts),
        "custom_type": dict(custom_type_counts),
        "x_range": (min(xs), max(xs)) if xs else None,
        "y_range": (min(ys), max(ys)) if ys else None,
        "z_range": (min(zs), max(zs)) if zs else None,
        "fields_present": sorted(node_fields),
    }

    # ---- segment ----
    seg_card = Counter()
    seg_type_keys = Counter()
    seg_typestr = Counter()
    seg_fields = set()
    for s in segments:
        seg_card[len(s.get("segs", []))] += 1
        for k in (s.get("type") or {}).keys():
            seg_type_keys[k] += 1
        seg_typestr[s.get("typeStr", "")] += 1
        seg_fields.update(s.keys())
    out["segment"] = {
        "segs_cardinality": dict(seg_card),
        "type_outer_keys": dict(seg_type_keys),
        "typeStr": dict(seg_typestr),
        "fields_present": sorted(seg_fields),
    }

    # ---- other ----
    other_kinds = Counter()
    for o in others:
        if isinstance(o, dict):
            kind = o.get("region") or o.get("category") or "<dict>"
            other_kinds[str(kind)] += 1
        else:
            other_kinds[type(o).__name__] += 1
    out["other"] = {
        "non_empty": len(others) > 0,
        "sample_count": len(others),
        "kinds": dict(other_kinds),
    }

    # ---- image_type / flagList ----
    out["image_type"] = ann.get("image_type", {})
    out["flagList_len"] = len(ann.get("flagList", []))

    # Cross-check: every line referenced by exactly one way? Every node by exactly one line?
    line_ids = {ln["id"] for ln in lines}
    referenced_by_way = Counter()
    for w in ways:
        for lid in w.get("ways", []):
            referenced_by_way[lid] += 1
    out["consistency"] = {
        "lines_total": len(line_ids),
        "lines_referenced_by_way": len(referenced_by_way),
        "lines_referenced_multiply": sum(1 for c in referenced_by_way.values() if c > 1),
        "lines_unreferenced": len(line_ids - set(referenced_by_way.keys())),
    }
    node_ids = {n["id"] for n in nodes}
    ref_by_line = Counter()
    for ln in lines:
        for nid in ln.get("node_tokens", []):
            ref_by_line[nid] += 1
    out["consistency"].update({
        "nodes_total": len(node_ids),
        "nodes_referenced_by_line": len(ref_by_line),
        "nodes_referenced_multiply": sum(1 for c in ref_by_line.values() if c > 1),
        "nodes_unreferenced": len(node_ids - set(ref_by_line.keys())),
    })

    return out


def survey_coord(coord: dict):
    out = {"top_level_keys": sorted(coord.keys())}
    bc = coord.get("bev_config", {})
    out["bev_config"] = {k: bc.get(k) for k in
                         ("scale_x", "scale_y", "scale_z",
                          "offset_x", "offset_y", "offset_z",
                          "width", "height", "depth")}
    out["region"] = coord.get("region")
    out["version"] = coord.get("version")
    out["n_frames_in_frame_to_point"] = len(coord.get("frame_to_point", {}))
    cam_keys = [k for k in coord.keys() if k.startswith("ofilm_")]
    out["cams_present"] = sorted(cam_keys)
    out["missing_cams"] = sorted(set(EXPECTED_CAMS) - set(cam_keys))
    out["extra_cams"] = sorted(set(cam_keys) - set(EXPECTED_CAMS))
    per_cam_frames = {}
    for c in cam_keys:
        per_cam_frames[c] = len(coord[c].get("world_to_image", {}))
    out["n_frames_per_cam"] = per_cam_frames
    return out


def merge_counters(global_, local_, prefix=""):
    """Merge a local counter-of-dicts into the global aggregate."""
    for k, v in local_.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            global_.setdefault(key, Counter()).update(v)
        elif isinstance(v, (int, float)):
            global_.setdefault(key, Counter())[v] += 1
        else:
            global_.setdefault(key, Counter())[str(v)] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/space_samples",
                    help="Root containing 车道线N/ subdirs.")
    ap.add_argument("--out", default=None,
                    help="Optional path to write the full per-sample report as JSON.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root dir not found: {root}")

    samples = sorted([p for p in root.iterdir() if p.is_dir()])
    print(f"Found {len(samples)} sample dirs under {root}")

    per_sample = {}
    file_issues = []  # samples with file-layout problems
    n_frames_dist = Counter()
    way_type_kv_global = Counter()
    way_typeStr_global = Counter()
    way_ways_card_global = Counter()
    line_region_global = Counter()
    line_type_keys_global = Counter()
    node_type_global = Counter()
    custom_type_global = Counter()
    seg_card_global = Counter()
    seg_typeStr_global = Counter()
    other_nonempty_samples = []
    image_type_flag = Counter()
    image_type_scene = Counter()
    image_type_specialtype = Counter()
    bev_scale_x = Counter()
    bev_scale_y = Counter()
    bev_width = Counter()
    bev_height = Counter()
    versions = Counter()
    n_frames_in_coord = Counter()
    consistency_warnings = []  # samples where line/node referencing is off

    x_min_all, x_max_all = float("inf"), float("-inf")
    y_min_all, y_max_all = float("inf"), float("-inf")
    z_min_all, z_max_all = float("inf"), float("-inf")

    line_lens_all = []  # vertices-per-polyline, sample-wide

    for s in samples:
        rec = {}
        try:
            files_rec = survey_files(s)
            rec["files"] = files_rec
            n_frames_dist[files_rec["n_frames"]] += 1
            if files_rec["issues"]:
                file_issues.append((s.name, files_rec["issues"]))
        except Exception as e:
            file_issues.append((s.name, [f"file scan crashed: {e!r}"]))

        ann_path = s / "annotation.json"
        if ann_path.is_file():
            try:
                ann = load_json(ann_path)
                arec = survey_annotation(ann)
                rec["annotation"] = arec

                for kv, c in arec["way"]["type_kv"].items():
                    way_type_kv_global[kv] += c
                for k, c in arec["way"]["typeStr"].items():
                    way_typeStr_global[k] += c
                for k, c in arec["way"]["ways_cardinality"].items():
                    way_ways_card_global[k] += c
                for k, c in arec["line"]["region"].items():
                    line_region_global[k] += c
                for k, c in arec["line"]["type_outer_keys"].items():
                    line_type_keys_global[k] += c
                for k, c in arec["node"]["node_type"].items():
                    node_type_global[k] += c
                for k, c in arec["node"]["custom_type"].items():
                    custom_type_global[k] += c
                for k, c in arec["segment"]["segs_cardinality"].items():
                    seg_card_global[k] += c
                for k, c in arec["segment"]["typeStr"].items():
                    seg_typeStr_global[k] += c
                if arec["other"]["non_empty"]:
                    other_nonempty_samples.append((s.name, arec["other"]))

                it = arec.get("image_type") or {}
                image_type_flag[it.get("flag", "")] += 1
                image_type_scene[it.get("scene", "")] += 1
                image_type_specialtype[str(it.get("specialtype", ""))] += 1

                rng = arec["node"]
                if rng.get("x_range"):
                    x_min_all = min(x_min_all, rng["x_range"][0])
                    x_max_all = max(x_max_all, rng["x_range"][1])
                if rng.get("y_range"):
                    y_min_all = min(y_min_all, rng["y_range"][0])
                    y_max_all = max(y_max_all, rng["y_range"][1])
                if rng.get("z_range"):
                    z_min_all = min(z_min_all, rng["z_range"][0])
                    z_max_all = max(z_max_all, rng["z_range"][1])

                lens = arec["line"].get("node_tokens_len_minmax")
                if lens:
                    line_lens_all.extend(lens)

                con = arec["consistency"]
                if (con["lines_unreferenced"] or con["lines_referenced_multiply"]
                        or con["nodes_unreferenced"] or con["nodes_referenced_multiply"]):
                    consistency_warnings.append((s.name, con))
            except Exception as e:
                rec["annotation_error"] = repr(e)

        coord_path = s / "coord_distribution.json"
        if coord_path.is_file():
            try:
                coord = load_json(coord_path)
                crec = survey_coord(coord)
                rec["coord"] = crec
                bev_scale_x[crec["bev_config"].get("scale_x")] += 1
                bev_scale_y[crec["bev_config"].get("scale_y")] += 1
                bev_width[crec["bev_config"].get("width")] += 1
                bev_height[crec["bev_config"].get("height")] += 1
                versions[str(crec.get("version"))] += 1
                n_frames_in_coord[crec.get("n_frames_in_frame_to_point")] += 1
            except Exception as e:
                rec["coord_error"] = repr(e)

        per_sample[s.name] = rec

    # ============= print summary =============
    def show(title, counter, top=None):
        items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
        if top:
            items = items[:top]
        print(f"\n[{title}]  ({len(counter)} unique)")
        for k, v in items:
            print(f"  {v:5d}  {k}")

    print("\n=========== SUMMARY ===========")
    print(f"total samples scanned: {len(samples)}")
    print(f"samples with file-layout issues: {len(file_issues)}")
    for name, iss in file_issues[:20]:
        print(f"  - {name}: {iss}")
    if len(file_issues) > 20:
        print(f"  ... and {len(file_issues)-20} more")

    show("frames per sample (subdir count)", n_frames_dist)
    show("frames per sample (coord_distribution.frame_to_point)", n_frames_in_coord)
    show("coord_distribution.version", versions)
    show("bev_config.scale_x", bev_scale_x)
    show("bev_config.scale_y", bev_scale_y)
    show("bev_config.width", bev_width)
    show("bev_config.height", bev_height)

    show("way.type k=v", way_type_kv_global)
    show("way.typeStr", way_typeStr_global)
    show("way.ways[] cardinality (lines per way)", way_ways_card_global)
    show("line.region", line_region_global)
    show("line.type outer keys (expect empty)", line_type_keys_global)
    show("node.node_type", node_type_global)
    show("node.custom_type (expect empty)", custom_type_global)
    show("segment.segs cardinality", seg_card_global)
    show("segment.typeStr", seg_typeStr_global)
    show("image_type.flag", image_type_flag)
    show("image_type.scene", image_type_scene)
    show("image_type.specialtype", image_type_specialtype)

    print(f"\n[coord ranges across all nodes]")
    print(f"  x: [{x_min_all:.2f}, {x_max_all:.2f}]")
    print(f"  y: [{y_min_all:.2f}, {y_max_all:.2f}]")
    print(f"  z: [{z_min_all:.4f}, {z_max_all:.4f}]")

    if line_lens_all:
        print(f"\n[polyline vertex counts]")
        print(f"  min={min(line_lens_all)}  max={max(line_lens_all)}  "
              f"mean={sum(line_lens_all)/len(line_lens_all):.1f}")

    print(f"\n[samples where 'other' is non-empty]: {len(other_nonempty_samples)}")
    for name, o in other_nonempty_samples[:20]:
        print(f"  - {name}: {o}")

    print(f"\n[samples with ref-consistency warnings]: {len(consistency_warnings)}")
    for name, c in consistency_warnings[:20]:
        print(f"  - {name}: {c}")

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "per_sample": per_sample,
                "globals": {
                    "n_frames_dist": dict(n_frames_dist),
                    "way_type_kv": dict(way_type_kv_global),
                    "way_typeStr": dict(way_typeStr_global),
                    "way_ways_cardinality": dict(way_ways_card_global),
                    "line_region": dict(line_region_global),
                    "node_type": dict(node_type_global),
                    "segment_segs_cardinality": dict(seg_card_global),
                    "image_type_flag": dict(image_type_flag),
                    "image_type_scene": dict(image_type_scene),
                    "image_type_specialtype": dict(image_type_specialtype),
                    "x_range": [x_min_all, x_max_all],
                    "y_range": [y_min_all, y_max_all],
                    "z_range": [z_min_all, z_max_all],
                    "bev_scale_x": dict(bev_scale_x),
                    "bev_scale_y": dict(bev_scale_y),
                    "bev_width": dict(bev_width),
                    "bev_height": dict(bev_height),
                    "versions": dict(versions),
                    "n_frames_in_coord": dict(n_frames_in_coord),
                },
                "other_nonempty_samples": [n for n, _ in other_nonempty_samples],
                "consistency_warnings": [n for n, _ in consistency_warnings],
                "file_issues": file_issues,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nFull per-sample report written to {out_path}")


if __name__ == "__main__":
    main()
