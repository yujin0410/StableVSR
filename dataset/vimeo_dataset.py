import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

try:
    # MATLAB-style bicubic, matches the convention used to create most SR LR sets
    from basicsr.utils.matlab_functions import imresize as _matlab_imresize
except Exception:  # pragma: no cover - optional dependency path
    _matlab_imresize = None


def _bicubic_downsample(img_gt, scale):
    """Downsample a HWC float32 [0, 1] image by ``scale`` using bicubic.

    Prefers BasicSR's MATLAB-style bicubic (to match how REDS bicubic LR are
    generated); falls back to cv2 INTER_CUBIC if it is unavailable.
    """
    if _matlab_imresize is not None:
        return _matlab_imresize(img_gt, 1.0 / scale)
    h, w = img_gt.shape[:2]
    return cv2.resize(img_gt, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


class Vimeo90KRecurrentDataset(data.Dataset):
    """Vimeo-90K (septuplet) dataset for training recurrent VSR networks.

    Vimeo-90K is organised as::

        <dataroot_gt>/00001/0001/im1.png ... im7.png
        <dataroot_gt>/00001/0002/im1.png ... im7.png
        ...

    and the split files (``sep_trainlist.txt`` / ``sep_testlist.txt``) list one
    sequence per line, e.g. ``00001/0001``.

    This loader returns the same dict layout as ``REDSRecurrentDataset`` so it is
    a drop-in replacement for StableVSR training:

        - ``lq``  : (t, c, h, w) low-quality (LR) frames
        - ``gt``  : (t, c, h, w) ground-truth frames
        - ``key`` : sequence key, e.g. ``00001/0001``

    Args (via ``opt`` dict):
        dataroot_gt (str): Root of the GT sequences folder.
        dataroot_lq (str | None): Root of precomputed LR sequences (same tree as
            GT). If ``None`` / missing, LR frames are generated on-the-fly by
            bicubic downsampling the GT by ``scale``.
        meta_info_file (str): Path to ``sep_trainlist.txt`` (or test list).
        io_backend (dict): IO backend, e.g. ``{type: disk}``.
        num_frame (int): Number of consecutive frames per sample (<= 7).
        gt_size (int): GT crop size.
        scale (int): SR scale factor (e.g. 4).
        interval_list (list): Temporal sampling intervals. Default ``[1]``.
        random_reverse (bool): Randomly reverse the frame order.
        use_hflip (bool): Horizontal flip augmentation.
        use_rot (bool): Rotation augmentation.
        num_clip_frames (int): Frames available per clip. Default 7 (septuplet).
    """

    def __init__(self, opt):
        super(Vimeo90KRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])
        self.lq_root = Path(opt['dataroot_lq']) if opt.get('dataroot_lq') else None
        self.num_frame = opt['num_frame']
        self.num_clip_frames = opt.get('num_clip_frames', 7)
        assert self.num_frame <= self.num_clip_frames, (
            f"num_frame ({self.num_frame}) cannot exceed num_clip_frames "
            f"({self.num_clip_frames}).")

        # build keys from the split file: one sequence per line (e.g. 00001/0001)
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                line = line.strip()
                if line:
                    self.keys.append(line)

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = self.io_backend_opt['type'] == 'lmdb'
        if self.is_lmdb:
            raise NotImplementedError(
                'lmdb backend is not supported by Vimeo90KRecurrentDataset; '
                'use io_backend.type: disk.')

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'[Vimeo90K] Temporal augmentation interval list: '
                    f'[{interval_str}]; random reverse is {self.random_reverse}. '
                    f"LR source: {'precomputed' if self.lq_root else 'on-the-fly bicubic'}.")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]  # e.g. 00001/0001

        # pick the temporal interval and a valid starting frame (1-indexed)
        interval = random.choice(self.interval_list)
        span = (self.num_frame - 1) * interval
        max_start = self.num_clip_frames - span  # inclusive, 1-indexed
        start_frame_idx = random.randint(1, max_start)
        neighbor_list = list(range(start_frame_idx, start_frame_idx + span + 1, interval))

        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (
            f'Wrong length of neighbor list: {len(neighbor_list)}')

        img_lqs = []
        img_gts = []
        img_gt_path = None
        for neighbor in neighbor_list:
            img_gt_path = self.gt_root / key / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

            if self.lq_root is not None:
                img_lq_path = self.lq_root / key / f'im{neighbor}.png'
                img_bytes = self.file_client.get(img_lq_path, 'lq')
                img_lq = imfrombytes(img_bytes, float32=True)
            else:
                img_lq = _bicubic_downsample(img_gt, scale)
                img_lq = np.clip(img_lq, 0, 1).astype(np.float32)
            img_lqs.append(img_lq)

        # randomly crop (operates on lists of HWC arrays)
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        # contiguous float32 so basicsr's in-place cv2.flip accepts the arrays
        img_lqs = [np.ascontiguousarray(v, dtype=np.float32) for v in img_lqs]
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_results) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_results) // 2], dim=0)

        # img_lqs: (t, c, h, w); img_gts: (t, c, h, w); key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
