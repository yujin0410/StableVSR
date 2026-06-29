"""Seed-variance of generated high-frequency detail (anchoring evidence).

Mechanism test for the "spatial conditioning anchors detail" claim, on an axis
independent of temporal metrics (so it is NOT circular with tLPIPS).

Given K runs of the SAME frames generated with different seeds, measure the
per-pixel standard deviation of the high-frequency (detail) content across the
K seeds, averaged over pixels/frames. Low std = the conditioning anchors the
detail (output is seed-invariant); high std = the seed freely moves detail
around. Compare:  full (bands on)  vs  no_both (bands off).
If full has much lower seed-variance -> bands anchor detail placement.

Pass the K seed output dirs (same <seq>/<frame> layout) as a comma list.

Usage:
    python ablation_tools/eval_seed_variance.py \
        --paths /path/seed0,/path/seed1,/path/seed2,/path/seed3,/path/seed4
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

p = argparse.ArgumentParser()
p.add_argument("--paths", required=True, help="comma-separated K seed output dirs")
p.add_argument("--ksize", type=int, default=7, help="box-blur kernel for high-pass")
args = p.parse_args()

device = torch.device("cuda")
tt = ToTensor()
paths = [x for x in args.paths.split(",") if x]
assert len(paths) >= 2, "need >=2 seed dirs"


def highpass(x):
    k = args.ksize
    pad = k // 2
    blur = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"),
                        kernel_size=k, stride=1)
    return x - blur


# sequences common to all seed dirs
seqs = sorted(set.intersection(*[
    {d for d in os.listdir(pp) if os.path.isdir(os.path.join(pp, d))}
    for pp in paths
]))

seq_means = []
for seq in seqs:
    frame_lists = [sorted(os.listdir(os.path.join(pp, seq))) for pp in paths]
    n = min(len(fl) for fl in frame_lists)
    vals = []
    for i in range(n):
        stack = []
        for pp, fl in zip(paths, frame_lists):
            img = tt(Image.open(os.path.join(pp, seq, fl[i])).convert("RGB"))
            stack.append(highpass(img.unsqueeze(0).to(device)))
        stack = torch.cat(stack, dim=0)          # [K,3,H,W] high-freq
        std = stack.std(dim=0, unbiased=False)   # per-pixel std across seeds
        vals.append(std.mean().item())
    if vals:
        m = float(np.mean(vals))
        seq_means.append(m)
        print(f"  {seq:20s} seed-std={m * 1e3:8.4f} (x1e3, n={n})")

print(f"--> mean seed-std = {np.mean(seq_means) * 1e3:.4f} (x1e3)   "
      f"[lower = output more anchored / seed-invariant]")
