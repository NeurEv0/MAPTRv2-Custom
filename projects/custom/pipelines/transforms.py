"""Custom pipeline transforms for the SpaceLane dataset.

The 10 cameras in this rig come at three different native resolutions
(around-view 1920x1536, surround-front 3840x2160, surround-narrow 1920x1280).
MapTR's standard `RandomScaleImageMultiViewImage` scales each camera by the
same ratio, which leaves the output sizes still mismatched and breaks the
stack into one tensor (N_cam, C, H, W).

`ResizeMultiViewImageToFixed` resizes every camera to a common target
(H, W), updating each camera's `lidar2img` with its own per-camera scale
factors. Aspect distortion is absorbed by the projection matrix.
"""

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFilesHeterogeneous(object):
    """Load N multi-view images as a list (no stacking).

    Mmdet3d's `LoadMultiViewImageFromFiles` calls `np.stack` immediately, which
    fails when cameras have different native resolutions. This loader keeps
    them as a list so `ResizeMultiViewImageToFixed` can normalize sizes first.
    """

    def __init__(self, to_float32=True, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filenames = results['img_filename']
        imgs = [mmcv.imread(p, self.color_type) for p in filenames]
        if self.to_float32:
            imgs = [im.astype(np.float32) for im in imgs]
        results['filename'] = filenames
        results['img'] = imgs
        results['img_shape'] = [im.shape for im in imgs]
        results['ori_shape'] = [im.shape for im in imgs]
        results['pad_shape'] = [im.shape for im in imgs]
        results['scale_factor'] = 1.0
        num_channels = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(to_float32={self.to_float32}, '
                f"color_type='{self.color_type}')")


@PIPELINES.register_module()
class ResizeMultiViewImageToFixed(object):
    """Resize each multi-view image to the same (H, W); update lidar2img.

    Args:
        size: (H, W) target after resize. The width axis runs along image
            columns (the x of `lidar2img`).
    """

    def __init__(self, size):
        assert isinstance(size, (tuple, list)) and len(size) == 2
        self.target_h, self.target_w = int(size[0]), int(size[1])

    def __call__(self, results):
        imgs = results['img']
        new_imgs = []
        new_l2i = []
        new_aug = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            sx = self.target_w / float(w)
            sy = self.target_h / float(h)
            resized = mmcv.imresize(
                img, (self.target_w, self.target_h), return_scale=False)
            new_imgs.append(resized)

            scale_mat = np.eye(4, dtype=np.float32)
            scale_mat[0, 0] = sx
            scale_mat[1, 1] = sy

            l2i = np.asarray(results['lidar2img'][i], dtype=np.float32)
            new_l2i.append(scale_mat @ l2i)
            new_aug.append(scale_mat)

        results['img'] = new_imgs
        results['lidar2img'] = new_l2i
        results['img_aug_matrix'] = new_aug
        results['img_shape'] = [im.shape for im in new_imgs]
        results['ori_shape'] = [im.shape for im in new_imgs]
        results['pad_shape'] = [im.shape for im in new_imgs]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(size=({self.target_h}, {self.target_w}))'
