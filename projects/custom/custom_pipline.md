# SpaceLane (parking-scene no-go boundary) pipeline

End-to-end pipeline for training a MapTR-based pre-annotation model on the in-house
parking/garage dataset. This document is the entry point for picking the work up
later or handing it to someone new.

## Goal

Train a BEV polyline detector that pre-annotates "no-go" boundaries (parked cars,
columns, walls, curbs, other obstacles) from 10-camera surround imagery, so the
labeling team gets a strong starting point instead of drawing every line by hand.
Output schema matches the existing `annotation.json` format the labeling tool
consumes (`way`, `line`, `node`, `segment`, `other`).

## Domain

160 web-crawled samples in [data/space_samples/](../../data/space_samples/), 94%
parking / garage (`jianzhuwuneicheku`, `lumiantingchequyu`, `dishangcheku`) and
~6% road (`daoluchangjing`). Each sample is a short multi-frame trajectory
(2–15 frames; mode = 6) with 10 cameras per frame and a single global-frame GT
annotation spanning the whole trajectory.

## End-to-end flow

```
raw samples (annotation.json, coord_distribution.json, 10×JPEG per frame)
        │
        │  build_pkl.py  (one-time conversion)
        ▼
data/space_samples_processed/{train,val}.pkl   ←  per-frame records
        │
        │  SpaceLaneDataset.__getitem__   (per training step)
        ▼
{ img: (10,3,480,800) tensor,
  img_metas: dict with lidar2img (10×4×4), scene_token, can_bus, ... ,
  gt_bboxes_3d: LiDARInstanceLines,
  gt_labels_3d: tensor }
        │
        ▼
MapTR (ResNet-50 backbone → FPN → BEVFormer encoder → MapTR decoder → polyline head)
```

## Key design choices (decisions already made)

| Choice | Value | Why |
|---|---|---|
| Coordinate frame | World-axis-aligned ego BEV (translation only, no yaw) | Avoids the missing-yaw problem — `frame_to_point` only gives 2D position. For parking, "ego forward" has no natural meaning anyway. |
| Z handling | Dropped from GT (2D polylines only) | User-requested deferral; can revisit with a z-regression head later. |
| Skipped fields | `segment`, `other`, `node_type` | Per design discussion. `segment` is 1-to-1 with `line` across all 160 samples. `other` (rects in 62/160) needs a separate BBox head; ignored for now. `node_type` is post-process from polyline order. |
| BEV perception range | ±25 m × ±25 m square | Garage scenes are tight; symmetric because we have no canonical forward direction. |
| `num_points` per polyline | 40 (uniform arc-length resample) | Survey: min=2, max=162, mean=22. Stock MapTR uses 20. |
| Classes | 5 (`car-no-go`, `other-no-go`, `column-no-go`, `wall-no-go`, `outside-no-go`) | Survey confirmed these are the only `way.type` values. |
| Camera count | All 10 cams | 360° coverage critical for parking. See "Camera rationale". |
| Image input size | 480 × 800 (stretch, not letterbox) | Standard MapTR nuScenes size. Aspect distortion absorbed by per-cam `lidar2img` scaling. |
| Missing-camera frames | Drop the frame, keep the sample | 21 of 160 samples have at least one frame missing a camera (most often `surround_front_120_8M`). |
| Train/val split | 128 / 32 samples → 846 / 207 frames | Sample-level split (no trajectory leak), seed=0. |
| Base model | **MapTR v1** (BEVFormer encoder), **not** MapTRv2 | v2's LSS encoder needs lidar depth supervision; we have no lidar. |
| Eval metric | Per-class chamfer-distance AP @ thresholds {0.5, 1.0, 1.5} m | Matches MapTR/MapTRv2 nuScenes evaluation. |

## File layout

```
projects/custom/
├── __init__.py                          # registers datasets + pipelines into mmdet
├── data_preprocess/
│   ├── survey_annotations.py            # schema audit across all 160 samples
│   ├── build_pkl.py                     # raw → train.pkl / val.pkl
│   └── visualize_gt.py                  # project GT polylines back onto 10 cams + BEV (sanity check)
├── datasets/
│   ├── __init__.py
│   └── space_lane_dataset.py            # SpaceLaneDataset(Custom3DDataset)
├── pipelines/
│   ├── __init__.py
│   └── transforms.py                    # LoadMultiViewImageFromFilesHeterogeneous + ResizeMultiViewImageToFixed
└── configs/
    └── maptr_space_lane.py              # the training config
```

Inputs / outputs:
- `data/space_samples/车道线{1..160}/` — raw
- `data/space_samples_processed/{train,val}.pkl` — converted, ready for the dataset
- `data/space_samples_processed/vis/*.png` — projection sanity-check images
- `work_dirs/maptr_space_lane/` — training run output (logs, checkpoints, tensorboard)

## Data conversion ([build_pkl.py](data_preprocess/build_pkl.py))

For each sample × each frame (that has all 10 cams):
1. Read `annotation.json`. Convert each polyline's pixel coords → world meters via
   `coord_distribution.json::img_to_world` (4×4 affine, scale ≈ 0.025 m/px).
2. Compute ego position in world meters from `frame_to_point[frame_id]` (also via
   `img_to_world`).
3. Translate polylines to ego frame: `polyline_ego_xy = polyline_world_xy − ego_world_xy`.
   Drop z.
4. Clip to ±25 m square (Liang–Barsky segment clip; preserve vertex order; sub-
   polylines emitted separately if a line crosses the boundary multiple times).
5. Uniform arc-length resample to exactly 40 points per polyline.
6. Build per-camera `lidar2img = world_to_image[cam] @ T_translate(ego_world_xy)`
   for all 10 cams, padded to 4×4.

Output record (one per frame, in either `train.pkl` or `val.pkl`):
```python
{
  "sample_name": "车道线1",
  "frame_id": 0,
  "img_filenames": [<10 relative paths>],
  "cam_names":     [<10 cam names>],
  "lidar2img":     np.float32 (10, 4, 4),
  "ego_world_xy":  np.float32 (2,),
  "gt_polylines":  np.float32 (M, 40, 2),   # ego-frame meters
  "gt_labels":     np.int64   (M,),
  "gt_meta":       [{way_id, line_id, n_src_points}, ...],
  "image_type":    {flag, scene, specialtype}
}
```

Run again any time the source data or design parameters change:
```bash
python projects/custom/data_preprocess/build_pkl.py \
    --root data/space_samples \
    --out_dir data/space_samples_processed
```

## Dataset class ([space_lane_dataset.py](datasets/space_lane_dataset.py))

`SpaceLaneDataset` inherits `mmdet3d.datasets.Custom3DDataset`. Registered as
`@DATASETS.register_module()` so the config can refer to it as
`type='SpaceLaneDataset'`.

Key methods:
- `load_annotations(ann_file)` — `pickle.load` our `{metainfo, samples}` pkl.
- `get_data_info(index)` — returns the dict the pipeline expects:
  - `sample_idx`, `img_filename` (10 absolute paths), `lidar2img`, `cam_intrinsic`
    (identity 4×4 placeholders), `scene_token` (= `sample_name`), `can_bus` (18-zero
    array — required defensively because the MapTR detector reads it unconditionally),
    `ann_info` (only in train mode).
- `get_ann_info(index)` — wraps the (M, 40, 2) polylines in MapTR's `LiDARInstanceLines`
  shapely-backed container.
- `prepare_train_data` → runs pipeline, calls `vectormap_pipeline` to attach
  `gt_bboxes_3d` and `gt_labels_3d` as DataContainers, then `_wrap_queue_one` to
  add the queue dim (MapTR's `forward_train` expects `(B, queue, N_cam, C, H, W)`
  with `queue_length=1` here).
- `prepare_test_data` → runs pipeline and returns as-is. **No queue wrap** for
  test — `MultiScaleFlipAug3D` already produces the list-of-DCs that
  `forward_test` consumes directly. (This was the bug that caused the epoch-2
  validation crash; fixed.)
- `evaluate(results, metric='chamfer', ...)` — formats predictions and GT into
  the JSON shape MapTRv2's `mean_ap.eval_map` expects, calls it at thresholds
  {0.5, 1.0, 1.5} m, returns per-class AP and mAP under the
  `SpaceLane_chamfer/*` namespace.

## Custom pipeline transforms ([transforms.py](pipelines/transforms.py))

The 10 cameras have three native resolutions
(`around_*_190_3M` 1920×1536, `surround_front_120_8M` 3840×2160,
`surround_*_100_2M` 1920×1280). MapTR's standard
`LoadMultiViewImageFromFiles` `np.stack`s them on load — which fails for
heterogeneous sizes. Two new transforms:

1. **`LoadMultiViewImageFromFilesHeterogeneous`** — loads each image as a list
   entry; no stacking.
2. **`ResizeMultiViewImageToFixed`** — resizes every image to a common (H, W)
   target. For each image, computes its own per-cam scale `(sx, sy)` and updates
   its `lidar2img` with `diag(sx, sy, 1, 1) @ lidar2img`. Stretching (not letter-
   boxing) so the projection math stays exact.

Train pipeline:
```python
[
  LoadMultiViewImageFromFilesHeterogeneous,
  ResizeMultiViewImageToFixed(size=(480, 800)),
  PhotoMetricDistortionMultiViewImage,
  NormalizeMultiviewImage(**img_norm_cfg),
  PadMultiViewImage(size_divisor=32),
  DefaultFormatBundle3D(with_gt=False, with_label=False),
  CustomCollect3D(keys=['img']),
]
```
Test pipeline replaces aug with `MultiScaleFlipAug3D` wrapper.

## Training config ([maptr_space_lane.py](configs/maptr_space_lane.py))

Key model bits, all derived from MapTR v1 `maptr_tiny_r50_24e.py`:
- `model.type = 'MapTR'`
- ResNet-50 backbone, pretrained `ckpts/resnet50-19c8e357.pth`
- BEV grid 200 × 200 over the 50 × 50 m perception range (0.25 m / cell)
- `MapTRPerceptionTransformer`: `num_cams=10`, `rotate_prev_bev=False`,
  `use_shift=False`, `use_can_bus=False`. No temporal queue.
- `GeometrySptialCrossAttention`: `num_cams=10`.
- `MapTRHead`: `num_vec=100`, `num_pts_per_vec=40`, `num_classes=5`.
- Losses: `FocalLoss(weight=2)`, `PtsL1Loss(weight=5)`, `PtsDirCosLoss(weight=0.005)`,
  bbox + iou loss weighted to 0 (we don't supervise the per-instance bbox).
- Optim: AdamW, lr=6e-4 (backbone lr_mult=0.1), CosineAnnealing schedule,
  200-iter warmup, 24 epochs, batch=1, fp16 enabled.
- Eval interval = 2 epochs, `save_best='SpaceLane_chamfer/mAP'`.

## Patches to upstream MapTRv2 plugin code

Three small fixes were needed (each clearly commented at the patch site):

1. **`projects/mmdet3d_plugin/maptr/modules/transformer.py:204–222`** —
   `attn_bev_encode` was returning the raw BEVFormer encoder tensor, but
   `get_bev_features` indexed it as `dict['bev']`. Wrapped the return as
   `dict(bev=bev_embed, depth=None)` so the v1 path matches the v2 path.

2. **`projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py:270`** — the v1
   head unpacked the transformer return as a 4-tuple; the (now-shared)
   transformer returns a 5-tuple including `depth`. Changed to
   `bev_embed, _depth, hs, init_reference, inter_references = outputs`.

3. **`mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py:5`** — modern
   numba moved `numba.errors` → `numba.core.errors`. Added a try/except fallback.

These are all in the upstream/vendored code, not in `projects/custom/`. If you
ever re-clone the MapTRv2 plugin from upstream, re-apply them.

## Environment

Python 3.8 + the OpenMMLab 1.x stack (per `docs/install.md`). The current
machine has it in conda env `maptr`:
```
torch 1.10.0+cu113   torchvision 0.11.0+cu113
mmcv-full 1.4.0      mmdet 2.14.0       mmdet3d 0.17.2 (editable from in-repo)
mmsegmentation 0.14.1  numpy 1.23.5  numba 0.57.1  shapely 1.8.5.post1
yapf 0.32.0  av2 0.2.0  nuscenes-devkit 1.1.11
```

Two infra-side gotchas patched once:
- `yapf >= 0.40` removed the `verify` kwarg that mmcv 1.4 still passes — pinned
  yapf to 0.32.0.
- `torch.utils.tensorboard/__init__.py` does `LooseVersion = distutils.version.LooseVersion`
  which fails under modern setuptools — added an explicit
  `import distutils.version` (one-line edit in the site-packages file).

## Smoke checks (re-run before each long training run)

```bash
# 1. Survey raw schema, confirm no surprises
python projects/custom/data_preprocess/survey_annotations.py --root data/space_samples

# 2. Convert (regenerates train.pkl / val.pkl)
python projects/custom/data_preprocess/build_pkl.py

# 3. Project GT onto 10 cams + BEV (visual sanity)
python projects/custom/data_preprocess/visualize_gt.py \
    --pkl data/space_samples_processed/val.pkl --idx 0

# 4. Inspect one batch through the dataset + pipeline
PYTHONPATH=. python -c "
import projects.custom, projects.mmdet3d_plugin
from mmcv import Config
from mmdet.datasets import build_dataset
cfg = Config.fromfile('projects/custom/configs/maptr_space_lane.py')
ds = build_dataset(cfg.data.train)
print(ds[0]['img'].data.shape)         # expect (10, 3, 480, 800)
print(len(ds[0]['gt_bboxes_3d'].data.instance_list))
"
```

## Training

**Always launch detached + with stdout captured.** Bare foreground launches die
when the terminal closes, and any traceback goes with them.

```bash
cd /home/puyang/MAPTRv2-Custom
PYTHONPATH=. setsid nohup \
  /home/puyang/miniconda3/envs/maptr/bin/python tools/train.py \
    projects/custom/configs/maptr_space_lane.py \
    --work-dir work_dirs/maptr_space_lane \
    --gpu-ids 0 \
    > work_dirs/maptr_space_lane/launch.log 2>&1 < /dev/null &
disown
echo "train pid: $!"
```

Tail the log:
```bash
tail -F work_dirs/maptr_space_lane/launch.log | \
  grep --line-buffered -E "Epoch \[|Saving|SpaceLane_chamfer|Traceback|Error|OOM|Killed"
```

Tensorboard:
```bash
tensorboard --logdir work_dirs/maptr_space_lane/tf_logs --port 6006
```

Expected throughput on a single RTX 4090: ~0.4 s/iter, 846 iters/epoch ≈ 6 min/epoch,
24 epochs ≈ 2.5 h total. Peak GPU mem ~15 GB.

## Evaluation

```bash
PYTHONPATH=. python tools/test.py \
    projects/custom/configs/maptr_space_lane.py \
    work_dirs/maptr_space_lane/latest.pth \
    --eval chamfer
```

Output is per-class chamfer AP at thresholds {0.5, 1.0, 1.5} m, plus the mean,
under the `SpaceLane_chamfer/*` namespace.

## Known issues / things deferred

- **`--resume-from` from a mid-training checkpoint sometimes produces NaN on the
  first iter** — AdamW second-moment estimates appear to interact badly with the
  restored LR. Workaround: start fresh, or use `--cfg-options load_from=…`
  (model weights only, fresh optimizer).
- **`other` rects** (62 / 160 samples have 1–4 bounding rectangles) are not
  predicted. If labelers need them, easiest fix is to convert each rect to a
  4-vertex polyline at conversion time and label it as a sixth class.
- **`z` coordinate** is dropped. If z becomes required, MapTR v1 has a 3D
  variant (`code_size=3`); MapTRv2 has it natively via LSS but would need lidar.
- **190° around-view cameras are fisheye**, but `lidar2img` is a 3×4 pinhole-
  equivalent matrix. Projection in visualize_gt looked reasonable, but for
  near-range objects (<5 m) there could be systematic offset. Worth a focused
  check if final-stage AP plateaus.
- **Eval reports two AP tables in a row in the log** — the mmcv `EvalHook`
  internally calls `format_results` twice; the chamfer code prints each time.
  Cosmetic, not a real bug.
