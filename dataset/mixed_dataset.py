"""Mixed REDS + Vimeo dataset for StableVSR fine-tuning.

Each __getitem__ probabilistically samples from REDS or Vimeo according to
`reds_ratio`. The configured length is the sum of the two datasets so the
DataLoader sees enough items per epoch.
"""

import random

from torch.utils import data as data

from dataset.reds_dataset import REDSRecurrentDataset
from dataset.vimeo_dataset import VimeoSeptupletDataset


class MixedDataset(data.Dataset):
    """REDS + Vimeo Septuplet probabilistic mix.

    Args:
        opt (dict):
            reds_ratio (float): Probability of drawing from REDS (0-1).
            reds (dict): Config passed to REDSRecurrentDataset.
            vimeo (dict): Config passed to VimeoSeptupletDataset.
    """

    def __init__(self, opt):
        super().__init__()
        self.reds = REDSRecurrentDataset(opt['reds'])
        self.vimeo = VimeoSeptupletDataset(opt['vimeo'])
        self.reds_ratio = float(opt.get('reds_ratio', 0.5))
        if not 0.0 <= self.reds_ratio <= 1.0:
            raise ValueError(f"reds_ratio must be in [0, 1], got {self.reds_ratio}")
        self._len = len(self.reds) + len(self.vimeo)

    def __getitem__(self, index):
        if random.random() < self.reds_ratio:
            return self.reds[random.randint(0, len(self.reds) - 1)]
        return self.vimeo[random.randint(0, len(self.vimeo) - 1)]

    def __len__(self):
        return self._len
