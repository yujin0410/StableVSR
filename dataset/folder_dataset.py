import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

from dataset.vimeo_dataset import _bicubic_downsample

_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def _is_clip_dir(path, min_frames):
    try:
        n = sum(1 for f in os.listdir(path) if f.lower().endswith(_IMG_EXTS))
    except OSError:
        return False
    return n >= min_frames


class FolderRecurrentDataset(data.Dataset):
    """Recurrent VSR dataset over clip folders of extracted frames.

    Use this when video clips have been extracted to frames on disk, e.g.::

        <dataroot>/<clip_key>/00000000.png
        <dataroot>/<clip_key>/00000001.png ...

    Clip folders may be nested (any depth); a folder is treated as a clip if it
    directly contains at least ``num_frame`` image files. Frame naming is
    arbitrary as long as a lexicographic sort gives temporal order. LR frames
    are read from ``dataroot_lq`` (same tree) if given, else generated on-the-fly
    by bicubic downsampling so the degradation matches the bicubic x4 benchmarks.

    Output dict matches the REDS/Vimeo/Video loaders:
        ``lq`` (t,c,h,w), ``gt`` (t,c,h,w), ``key`` (clip path relative to root).

    Args (via ``opt``): dataroot, dataroot_lq (optional), meta_info_file
        (optional list of clip keys, one per line), num_frame, gt_size, scale,
        interval_list, random_reverse, use_hflip, use_rot, io_backend, max_clips.
    """

    def __init__(self, opt):
        super(FolderRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root = opt['dataroot']
        self.lq_root = opt.get('dataroot_lq') or None
        self.num_frame = opt['num_frame']
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)

        meta = opt.get('meta_info_file')
        if meta:
            with open(meta, 'r') as fin:
                self.clips = [ln.strip() for ln in fin if ln.strip()]
        else:
            self.clips = []
            for root, dirs, files in os.walk(self.gt_root):
                if _is_clip_dir(root, self.num_frame):
                    self.clips.append(os.path.relpath(root, self.gt_root))
            self.clips.sort()

        if opt.get('max_clips'):
            self.clips = self.clips[:int(opt['max_clips'])]
        if len(self.clips) == 0:
            raise FileNotFoundError(f'No clip folders with >= {self.num_frame} frames under {self.gt_root}')

        # cache the sorted frame list per clip (paths relative to the clip dir)
        self._frames = {}

        self.file_client = None
        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})

        logger = get_root_logger()
        logger.info(f'[Folder] {len(self.clips)} clips under {self.gt_root}; '
                    f"interval_list={self.interval_list}, random_reverse={self.random_reverse}, "
                    f"LR source: {'precomputed' if self.lq_root else 'on-the-fly bicubic'}.")

    def _frame_list(self, clip):
        if clip not in self._frames:
            d = os.path.join(self.gt_root, clip)
            self._frames[clip] = sorted(f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS))
        return self._frames[clip]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt['type'])

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        clip = self.clips[index]
        names = self._frame_list(clip)
        n = len(names)

        interval = random.choice(self.interval_list)
        required = (self.num_frame - 1) * interval + 1
        if n < required:
            interval = 1
            required = self.num_frame
        start = random.randint(0, max(n - required, 0))
        idxs = [min(start + i * interval, n - 1) for i in range(self.num_frame)]
        if self.random_reverse and random.random() < 0.5:
            idxs.reverse()

        img_gts, img_lqs = [], []
        for i in idxs:
            gt_path = os.path.join(self.gt_root, clip, names[i])
            img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
            img_gts.append(img_gt)
            if self.lq_root is not None:
                lq_path = os.path.join(self.lq_root, clip, names[i])
                img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)
            else:
                img_lq = np.clip(_bicubic_downsample(img_gt, scale), 0, 1).astype(np.float32)
            img_lqs.append(img_lq)

        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, gt_path)
        img_lqs.extend(img_gts)
        # contiguous float32 so basicsr's in-place cv2.flip accepts the arrays
        # (on-the-fly bicubic / cropped views can have layouts new OpenCV rejects)
        img_lqs = [np.ascontiguousarray(v, dtype=np.float32) for v in img_lqs]
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_results) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_results) // 2], dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': clip}

    def __len__(self):
        return len(self.clips)
