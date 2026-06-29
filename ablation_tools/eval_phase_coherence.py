"""Flow-aligned high-frequency PHASE COHERENCE metric (placement consistency).

Mechanism evidence for the "anchoring" claim: spatial conditioning makes the
generated high-frequency detail appear at CONSISTENT spatial locations across
frames. In the wavelet domain, phase = spatial position of structure; so
flow-aligned high-band phase agreement directly measures placement consistency.

For consecutive OUTPUT frames (t-1, t):
    flow = RAFT(t -> t-1 alignment);  warp out_{t-1} -> aligned
    PC   = magnitude-weighted mean of (1 - cos(phase_t - phase_aligned))
           over the high DT-CWT bands (j=0,1).
Lower PC = detail placed consistently (less per-frame flicker).

Operates on output frames only (no GT). Compare variants:
    full vs no_both vs StableVSR.  If full < no_both -> bands improve placement.

Usage:
    python ablation_tools/eval_phase_coherence.py --out_path <SR frames dir>
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.flow_utils import get_flow, flow_warp
from util.frequency_utils import DTCWTForward

p = argparse.ArgumentParser()
p.add_argument("--out_path", required=True, help="folder of <seq>/<frames> SR outputs")
p.add_argument("--high_levels", default="0,1", help="DT-CWT high levels to use")
args = p.parse_args()

device = torch.device("cuda")
of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
dtcwt = DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device)
high = [int(x) for x in args.high_levels.split(",")]
tt = ToTensor()


def phase_coherence(cur, warped_prev):
    _, Yh_c = dtcwt(cur.float())
    _, Yh_w = dtcwt(warped_prev.float())
    num, den = 0.0, 0.0
    for j in high:
        rc, ic = Yh_c[j][..., 0], Yh_c[j][..., 1]
        rw, iw = Yh_w[j][..., 0], Yh_w[j][..., 1]
        m = torch.sqrt(rc * rc + ic * ic + 1e-8)
        phi_c = torch.atan2(ic, rc)
        phi_w = torch.atan2(iw, rw)
        num += (m * (1.0 - torch.cos(phi_c - phi_w))).sum().item()
        den += m.sum().item()
    return num / max(den, 1e-8)


seqs = sorted(d for d in os.listdir(args.out_path)
              if os.path.isdir(os.path.join(args.out_path, d)))
seq_means = []
for seq in seqs:
    sd = os.path.join(args.out_path, seq)
    ims = sorted(os.listdir(sd))
    prev = None
    vals = []
    for im in ims:
        cur = tt(Image.open(os.path.join(sd, im)).convert("RGB")).unsqueeze(0).to(device)
        if prev is not None:
            with torch.no_grad():
                f = get_flow(of_model, cur, prev)      # warps prev -> cur
                pw = flow_warp(prev, f)
                vals.append(phase_coherence(cur, pw))
        prev = cur
    if vals:
        m = float(np.mean(vals))
        seq_means.append(m)
        print(f"  {seq:20s} PC={m * 1e3:8.3f} (x1e3, n={len(vals)})")

print(f"--> mean PC = {np.mean(seq_means) * 1e3:.3f} (x1e3)   "
      f"[lower = more consistent high-freq placement]")
