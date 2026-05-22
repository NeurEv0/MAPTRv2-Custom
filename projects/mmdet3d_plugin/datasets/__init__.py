from numpy_compat import ensure_numpy_typing
from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

from .nuscenes_map_dataset import CustomNuScenesLocalMapDataset
from .nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalMapDataset

# AV2 support depends on the external ``av2`` package and a newer NumPy API
# than some MapTR environments provide. Keep these imports optional so
# non-AV2 training remains usable.
ensure_numpy_typing()
try:
    from .av2_map_dataset import CustomAV2LocalMapDataset
    from .av2_offlinemap_dataset import CustomAV2OfflineLocalMapDataset
except ImportError:
    CustomAV2LocalMapDataset = None
    CustomAV2OfflineLocalMapDataset = None

__all__ = [
    'CustomNuScenesDataset','CustomNuScenesLocalMapDataset'
]
