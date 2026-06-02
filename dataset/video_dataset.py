import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import get_root_logger, img2tensor

from dataset.vimeo_dataset import _bicubic_downsample


class VideoClipRecurrentDataset(data.Dataset):
    """Recurrent VSR dataset that reads frames directly from video clips (.mp4).

    Designed for YouHQ-Train, organised as::

        <dataroot>/<category>/<youtube_id>/<frame_range>.mp4

    but works for any tree of video files. Frames are decoded on-the-fly with
    decord (random access, no pre-extraction needed); LR frames are produced by
    bicubic downsampling so the degradation matches the bicubic x4 benchmarks
    (REDS4 / Vid4 / UDM10 / SPMCS). The output dict matches the REDS / Vimeo
    loaders so it is a drop-in for StableVSR training:

        - ``lq``  : (t, c, h, w) low-quality (LR) frames
        - ``gt``  : (t, c, h, w) ground-truth frames
        - ``key`` : clip path relative to dataroot

    Args (via ``opt`` dict):
        dataroot (str): Root containing the video clips.
        meta_info_file (str | None): Optional text file listing clip paths
            (relative to dataroot), one per line. If omitted, the tree is
            globbed for ``*.mp4`` at init.
        video_ext (str): Clip extension to scan for. Default ``mp4``.
        max_clips (int | None): Cap the number of clips (handy for quick runs).
        num_frame (int): Consecutive frames per sample.
        gt_size (int): GT crop size.
        scale (int): SR scale factor.
        interval_list (list): Temporal sampling intervals. Default ``[1]``.
        random_reverse (bool): Randomly reverse frame order.
        use_hflip (bool): Horizontal flip augmentation.
        use_rot (bool): Rotation augmentation.
    """

    def __init__(self, opt):
        super(VideoClipRecurrentDataset, self).__init__()
        self.opt = opt
        self.dataroot = opt['dataroot']
        self.num_frame = opt['num_frame']
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)

        meta = opt.get('meta_info_file')
        if meta:
            with open(meta, 'r') as fin:
                self.clips = [ln.strip() for ln in fin if ln.strip()]
        else:
            ext = opt.get('video_ext', 'mp4')
            paths = glob(os.path.join(self.dataroot, '**', f'*.{ext}'), recursive=True)
            self.clips = sorted(os.path.relpath(p, self.dataroot) for p in paths)

        if opt.get('max_clips'):
            self.clips = self.clips[:int(opt['max_clips'])]

        if len(self.clips) == 0:
            raise FileNotFoundError(f'No video clips found under {self.dataroot}')

        # fail early with a helpful message if decord is missing
        try:
            import decord  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                'VideoClipRecurrentDataset requires `decord` (pip install decord). '
                f'Import failed: {e}')

        logger = get_root_logger()
        logger.info(f'[Video] {len(self.clips)} clips under {self.dataroot}; '
                    f"interval_list={self.interval_list}, random_reverse={self.random_reverse}, "
                    f'LR source: on-the-fly bicubic.')

    def _read_frames(self, clip_path, indices):
        from decord import VideoReader, cpu
        # Open per call: decord readers do not survive DataLoader worker forks.
        vr = VideoReader(clip_path, ctx=cpu(0))
        n = len(vr)
        indices = [min(max(i, 0), n - 1) for i in indices]
        frames = vr.get_batch(indices).asnumpy()  # (t, H, W, 3) uint8 RGB
        del vr
        return frames, n

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.clips[index]
        clip_path = os.path.join(self.dataroot, key)

        interval = random.choice(self.interval_list)
        required = (self.num_frame - 1) * interval + 1

        # peek length cheaply by reading the first frame's reader length
        from decord import VideoReader, cpu
        n = len(VideoReader(clip_path, ctx=cpu(0)))
        if n < required:
            interval = 1
            required = self.num_frame
        max_start = max(n - required, 0)
        start = random.randint(0, max_start)
        indices = [start + i * interval for i in range(self.num_frame)]
        if self.random_reverse and random.random() < 0.5:
            indices.reverse()

        frames, _ = self._read_frames(clip_path, indices)

        # RGB uint8 -> BGR float32 [0,1] so the basicsr pipeline (img2tensor with
        # bgr2rgb=True) yields RGB tensors, identical to the REDS/Vimeo loaders.
        img_gts = [f[..., ::-1].astype(np.float32) / 255.0 for f in frames]
        img_lqs = [np.clip(_bicubic_downsample(g, scale), 0, 1).astype(np.float32) for g in img_gts]

        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, clip_path)

        img_lqs.extend(img_gts)
        # contiguous float32 so basicsr's in-place cv2.flip accepts the arrays
        img_lqs = [np.ascontiguousarray(v, dtype=np.float32) for v in img_lqs]
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_results) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_results) // 2], dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.clips)
