import random

import numpy as np
from omegaconf import OmegaConf
from torch.utils import data as data

from basicsr.utils import get_root_logger

# keys shared across all sub-sources (must be consistent so batches stack)
_SHARED_KEYS = ['num_frame', 'gt_size', 'scale', 'use_hflip', 'use_rot']


def _to_dict(opt):
    if OmegaConf.is_config(opt):
        return OmegaConf.to_container(opt, resolve=True)
    return dict(opt)


def build_recurrent_dataset(opt):
    """Factory: construct a recurrent VSR dataset from an ``opt`` dict.

    Dispatch on ``opt['type']`` (default 'reds'):
        reds            -> REDSRecurrentDataset
        vimeo           -> Vimeo90KRecurrentDataset
        video / youhq   -> VideoClipRecurrentDataset
        mixed           -> MixedRecurrentDataset (weighted blend of sub-sources)
    """
    opt = _to_dict(opt)
    dtype = str(opt.get('type', 'reds')).lower()

    if 'mixed' in dtype:
        return MixedRecurrentDataset(opt)
    if 'vimeo' in dtype:
        from dataset.vimeo_dataset import Vimeo90KRecurrentDataset
        return Vimeo90KRecurrentDataset(opt)
    if dtype in ('video', 'youhq', 'mp4'):
        from dataset.video_dataset import VideoClipRecurrentDataset
        return VideoClipRecurrentDataset(opt)
    from dataset.reds_dataset import REDSRecurrentDataset
    return REDSRecurrentDataset(opt)


class MixedRecurrentDataset(data.Dataset):
    """Weighted blend of several recurrent VSR datasets.

    Each ``__getitem__`` first picks a sub-source in proportion to its ``ratio``
    (so a huge dataset like YouHQ does not drown a small one like REDS), then
    draws a random sample from it. This decouples the effective mixing ratio
    from the datasets' sizes -- which is exactly what we want, since there is no
    standard REDS:Vimeo:YouHQ ratio in the literature and it must be tuned.

    Config (``opt``)::

        type: mixed
        num_frame: 3        # shared by all sources
        gt_size: 256
        scale: 4
        use_hflip: true
        use_rot: false
        sources:
          - {type: reds,  ratio: 2, dataroot_gt: ..., ...}
          - {type: youhq, ratio: 2, dataroot: ...}
          - {type: vimeo, ratio: 1, dataroot_gt: ..., ...}
    """

    def __init__(self, opt):
        super(MixedRecurrentDataset, self).__init__()
        opt = _to_dict(opt)
        shared = {k: opt[k] for k in _SHARED_KEYS if k in opt}

        self.subsets = []
        self.names = []
        ratios = []
        for src in opt['sources']:
            src = _to_dict(src)
            ratio = float(src.pop('ratio', 1.0))
            name = src.pop('name', src.get('type', 'reds'))
            # shared keys are defaults; per-source values win if present
            sub_opt = {**shared, **src}
            self.subsets.append(build_recurrent_dataset(sub_opt))
            self.names.append(name)
            ratios.append(ratio)

        ratios = np.asarray(ratios, dtype=np.float64)
        self.ratios = ratios / ratios.sum()
        # nominal epoch length: configurable, else sum of sub-dataset sizes
        self._len = int(opt.get('nominal_length') or sum(len(s) for s in self.subsets))

        logger = get_root_logger()
        summary = ', '.join(f'{n}({len(s)})={r:.2f}'
                            for n, s, r in zip(self.names, self.subsets, self.ratios))
        logger.info(f'[Mixed] sampling ratios -> {summary}; nominal length {self._len}')

    def __getitem__(self, index):
        si = int(np.random.choice(len(self.subsets), p=self.ratios))
        sub = self.subsets[si]
        return sub[random.randrange(len(sub))]

    def __len__(self):
        return self._len
