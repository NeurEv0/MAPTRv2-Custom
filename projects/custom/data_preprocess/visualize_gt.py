"""Project GT polylines from one record onto all 10 cameras and BEV.

Sanity check that `lidar2img` and the ego-translation are consistent with the
images. Use this before training to catch projection bugs that would otherwise
look like loss being stuck.

Usage (from repo root):
    python projects/custom/data_preprocess/visualize_gt.py \\
        --pkl data/space_samples_processed/val.pkl \\
        --idx 0 \\
        --out data/space_samples_processed/vis
"""

import argparse
import os
import os.path as osp
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_COLOR = {
    'car-no-go':     '#FF3333',  # red
    'other-no-go':   '#D3DA17',  # yellow-green (matches annotation.json)
    'column-no-go':  '#33FF33',  # lime
    'wall-no-go':    '#3399FF',  # blue
    'outside-no-go': '#7EEAF4',  # cyan (matches annotation.json)
}


def project_polyline(pts_2d, lidar2img, z=0.0, eps=0.1):
    """pts_2d: (N, 2) in lidar (ego) frame meters.
    Returns (u, v, in_front) with shape (N,)."""
    N = pts_2d.shape[0]
    homo = np.concatenate([
        pts_2d,
        np.full((N, 1), z, dtype=np.float64),
        np.ones((N, 1), dtype=np.float64),
    ], axis=1)
    proj = (lidar2img @ homo.T).T  # (N, 4)
    depth = proj[:, 2]
    in_front = depth > eps
    uv = proj[:, :2] / np.where(np.abs(depth[:, None]) > 1e-6,
                                depth[:, None], 1e-6)
    return uv[:, 0], uv[:, 1], in_front


def draw_polyline_on_ax(ax, u, v, in_front, color, W, H):
    """Plot only segments whose BOTH endpoints are in-front. Don't filter by
    image bounds — matplotlib will clip naturally and partially-visible lines
    are still informative."""
    starts = np.where(in_front[:-1] & in_front[1:])[0]
    for i in starts:
        ax.plot([u[i], u[i + 1]], [v[i], v[i + 1]],
                color=color, linewidth=1.0, alpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True,
                    help='train.pkl or val.pkl produced by build_pkl.py')
    ap.add_argument('--idx', type=int, default=0, help='record index in pkl')
    ap.add_argument('--out', default='data/space_samples_processed/vis',
                    help='output dir')
    ap.add_argument('--dpi', type=int, default=110)
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        payload = pickle.load(f)
    metainfo = payload['metainfo']
    samples = payload['samples']
    data_root = metainfo['data_root']
    class_names = metainfo['class_names']
    bev_x = metainfo['bev_x']
    bev_y = metainfo['bev_y']

    rec = samples[args.idx]
    sample_name = rec['sample_name']
    frame_id = rec['frame_id']
    img_paths = [osp.join(data_root, p) for p in rec['img_filenames']]
    cam_names = rec['cam_names']
    lidar2img = np.asarray(rec['lidar2img'])  # (10, 4, 4)
    gt = np.asarray(rec['gt_polylines'])      # (M, num_points, 2)
    labels = np.asarray(rec['gt_labels'])

    print(f'Sample {sample_name} frame {frame_id}: '
          f'{len(gt)} polylines, scene={rec.get("image_type", {})}')

    # 3 rows x 4 cols: 10 cameras + 1 BEV + 1 legend
    fig, axes = plt.subplots(3, 4, figsize=(22, 12))
    axes = axes.ravel()

    for i, (cam, img_path) in enumerate(zip(cam_names, img_paths)):
        ax = axes[i]
        img = np.array(Image.open(img_path))
        H, W = img.shape[:2]
        ax.imshow(img)
        ax.set_title(cam.replace('ofilm_', ''), fontsize=8)
        ax.axis('off')
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        n_drawn = 0
        for poly, lab in zip(gt, labels):
            u, v, in_front = project_polyline(poly, lidar2img[i])
            color = CLASS_COLOR[class_names[int(lab)]]
            if in_front.sum() >= 2:
                draw_polyline_on_ax(ax, u, v, in_front, color, W, H)
                n_drawn += 1
        ax.text(8, 28, f'in-FOV: {n_drawn}/{len(gt)}',
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

    # BEV panel
    ax = axes[10]
    for poly, lab in zip(gt, labels):
        color = CLASS_COLOR[class_names[int(lab)]]
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1.2, alpha=0.9)
    # ego marker
    ax.plot(0, 0, marker='o', color='white', markersize=8,
            markeredgecolor='black')
    ax.arrow(0, 0, 2.0, 0, head_width=0.6, color='white',
             length_includes_head=True)
    ax.set_xlim(bev_x[0], bev_x[1])
    ax.set_ylim(bev_y[0], bev_y[1])
    ax.set_aspect('equal')
    ax.set_title('BEV (ego frame; world-axis-aligned)', fontsize=9)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_facecolor('#202020')
    ax.grid(True, alpha=0.2)

    # Legend panel
    ax = axes[11]
    ax.axis('off')
    counts = {n: 0 for n in class_names}
    for lab in labels:
        counts[class_names[int(lab)]] += 1
    y = 0.95
    ax.text(0.0, y, f'{sample_name}  frame {frame_id}',
            transform=ax.transAxes, fontsize=11, weight='bold')
    y -= 0.08
    ax.text(0.0, y, f'scene: {rec.get("image_type", {}).get("scene", "")}',
            transform=ax.transAxes, fontsize=9)
    y -= 0.06
    ax.text(0.0, y, f'total polylines: {len(gt)}',
            transform=ax.transAxes, fontsize=9)
    y -= 0.12
    for name in class_names:
        ax.plot([0.0, 0.12], [y, y], color=CLASS_COLOR[name],
                linewidth=3.0, transform=ax.transAxes)
        ax.text(0.16, y - 0.005, f'{name}: {counts[name]}',
                transform=ax.transAxes, fontsize=10, va='center')
        y -= 0.08

    os.makedirs(args.out, exist_ok=True)
    out_path = osp.join(
        args.out, f'{sample_name}_frame{frame_id}.png')
    fig.suptitle(f'GT projection sanity check — {sample_name} / frame {frame_id}',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
