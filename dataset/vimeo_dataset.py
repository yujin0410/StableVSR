"""Vimeo-90K Septuplet dataset for StableVSR fine-tuning.

Each clip has 7 frames (im1.png ... im7.png) at 256x448 HR.
LR is generated on-the-fly via bicubic 4x downsample to keep disk usage low
and to match the BIx4 protocol used during evaluation.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import img2tensor


class VimeoSeptupletDataset(data.Dataset):
    """Vimeo-90K Septuplet dataset.

    Args:
        opt (dict):
            dataroot_gt (str): Path to .../vimeo_septuplet/sequences/
            meta_info_file (str): Path to sep_trainlist.txt (or sep_testlist.txt)
            num_frame (int): Number of consecutive frames to sample (must be <= 7).
            gt_size (int): HR patch size after random crop.
            scale (int): SR scale factor.
            use_hflip (bool): Random horizontal flip.
            use_rot (bool): Random rotation/transpose.
            random_reverse (bool): Randomly reverse temporal order.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])
        self.num_frame = int(opt['num_frame'])
        self.scale = int(opt['scale'])
        self.gt_size = int(opt['gt_size'])
        self.random_reverse = bool(opt.get('random_reverse', False))
        self.total_frames = 7

        if self.num_frame > self.total_frames:
            raise ValueError(
                f"num_frame ({self.num_frame}) must be <= 7 for Vimeo Septuplet"
            )

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                clip = line.strip()
                if clip:
                    self.keys.append(clip)

    def _load_hr(self, key, frame_idx):
        path = self.gt_root / key / f'im{frame_idx + 1}.png'
        img = Image.open(path).convert('RGB')
        return np.asarray(img, dtype=np.float32) / 255.0

    def _hr_to_lr(self, img_hr):
        t = torch.from_numpy(img_hr).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(
            t, scale_factor=1.0 / self.scale, mode='bicubic', antialias=True
        )
        return t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()

    def __getitem__(self, index):
        key = self.keys[index]

        max_start = self.total_frames - self.num_frame
        start_idx = random.randint(0, max_start)
        frame_indices = list(range(start_idx, start_idx + self.num_frame))
        if self.random_reverse and random.random() < 0.5:
            frame_indices.reverse()

        img_gts = [self._load_hr(key, i) for i in frame_indices]
        img_lqs = [self._hr_to_lr(g) for g in img_gts]

        gt_path_for_error = str(self.gt_root / key)
        img_gts, img_lqs = paired_random_crop(
            img_gts, img_lqs, self.gt_size, self.scale, gt_path_for_error
        )

        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = img2tensor(img_results)

        half = len(img_results) // 2
        img_lqs = torch.stack(img_results[:half], dim=0)
        img_gts = torch.stack(img_results[half:], dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
