"""Output-only diagnosis of the magnitude/phase <-> spatial/temporal duality.

NO retraining. Runs on already-generated SR frames (+ GT). It computes, on the
DT-CWT of the *outputs* themselves:

  (1) phase-incoherence  vs  flicker         -- does inter-frame phase instability
                                                track perceptual flicker?
  (2) phase-swap reconstruction              -- do magnitude/phase carry
                                                content/position respectively?
  (3) magnitude-flicker vs phase-flicker     -- which one drives flicker?

These are *signal-level* (correlational) evidence for the duality
"magnitude = content (spatial), phase = position (temporal)". They do NOT prove
the *method* caused it (that needs the cross-ablation), but they are cheap and
need no GPU training.

Requirements:
    pip install pytorch_wavelets lpips scipy opencv-python

Usage (each dataset = a folder of sequence subfolders of frames):
    python scripts/analyze_magphase.py \
        --out_path /path/to/SR_outputs \
        --gt_path  /path/to/GT \
        --tag dual_sft_reds

Run it for several models/datasets and compare the printed numbers, e.g.
    dual_sft on REDS (good temporal)  vs  dual_sft on UDM10 (bad temporal)
    dual_sft  vs  tft  vs  StableVSR
If phase-incoherence is high exactly where flicker is high, (1) supports
"flicker = phase instability". (2)/(3) test what each component carries.

NOTE: for a quick check this uses *raw* consecutive frames (no optical-flow
warping). The real tLPIPS/loss warp the previous frame; adding RAFT warping
here would refine (1)/(3) but needs the flow model. The relative comparison
across models/datasets is still informative.
"""

import os
import sys
import glob
import argparse

# make repo root importable (so `util.flow_utils` works when run as scripts/...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from scipy.stats import pearsonr, spearmanr

try:
    import lpips as lpips_lib
except Exception:  # pragma: no cover
    lpips_lib = None

# optional optical-flow warping (fair, motion-compensated test)
try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from util.flow_utils import flow_warp, get_flow_forward_backward, detect_occlusion
    _WARP_OK, _WARP_ERR = True, None
except Exception as _e:  # pragma: no cover
    _WARP_OK, _WARP_ERR = False, repr(_e)

_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
EPS = 1e-8


def list_seqs(root):
    seqs = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and any(f.lower().endswith(_IMG_EXTS) for f in os.listdir(p)):
            seqs.append(d)
    return seqs


def frame_paths(d):
    fs = [f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS)]
    return [os.path.join(d, f) for f in sorted(fs)]


def load_rgb(path, device):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    return t


def to_luma(rgb):
    # BT.601 luma, (1,1,H,W)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def dtcwt_bands(xfm, luma):
    """Return list over levels of (mag, re, im), each (1,1,6,Hj,Wj)."""
    Yl, Yh = xfm(luma)
    out = []
    for yh in Yh:                     # yh: (1,1,6,Hj,Wj,2)
        re, im = yh[..., 0], yh[..., 1]
        mag = torch.sqrt(re * re + im * im + EPS)
        out.append((mag, re, im))
    return Yl, out


def pair_phase_incoherence(bands_t, bands_p, high_levels, masks=None):
    """magnitude-weighted mean(1 - cos(dphi)) over high bands (scalar)."""
    num, den = 0.0, 0.0
    for j in high_levels:
        mt, ret, imt = bands_t[j]
        mp, rep, imp = bands_p[j]
        cosd = (ret * rep + imt * imp) / (mt * mp + EPS)   # cos(phi_t - phi_p)
        w = mt * mp
        if masks is not None:
            w = w * masks[j]
        num += (w * (1.0 - cosd)).sum().item()
        den += w.sum().item()
    return num / (den + EPS)


def pair_mag_flicker(bands_t, bands_p, high_levels, masks=None):
    """scale-free magnitude change: weighted mean |mt-mp| / mean(mt,mp)."""
    num, den = 0.0, 0.0
    for j in high_levels:
        mt = bands_t[j][0]
        mp = bands_p[j][0]
        avg = 0.5 * (mt + mp)
        rel = (mt - mp).abs() / (avg + EPS)
        w = avg
        if masks is not None:
            w = w * masks[j]
        num += (w * rel).sum().item()
        den += w.sum().item()
    return num / (den + EPS)


def phase_swap(xfm, ifm, luma_out, luma_gt):
    """Reconstruct: (out-mag + GT-phase) and (GT-mag + out-phase). Return both,
    each (1,1,H,W)."""
    Yl_o, Yh_o = xfm(luma_out)
    Yl_g, Yh_g = xfm(luma_gt)
    yh_magOut, yh_phaseOut = [], []
    for yo, yg in zip(Yh_o, Yh_g):
        reo, imo = yo[..., 0], yo[..., 1]
        reg, img = yg[..., 0], yg[..., 1]
        mo = torch.sqrt(reo * reo + imo * imo + EPS)
        mg = torch.sqrt(reg * reg + img * img + EPS)
        # out magnitude + GT phase
        cos_g, sin_g = reg / (mg + EPS), img / (mg + EPS)
        yh_magOut.append(torch.stack([mo * cos_g, mo * sin_g], dim=-1))
        # GT magnitude + out phase
        cos_o, sin_o = reo / (mo + EPS), imo / (mo + EPS)
        yh_phaseOut.append(torch.stack([mg * cos_o, mg * sin_o], dim=-1))
    rec_magOut = ifm((Yl_o, yh_magOut))     # keeps out lowpass
    rec_phaseOut = ifm((Yl_g, yh_phaseOut))  # keeps GT lowpass
    return rec_magOut, rec_phaseOut


def psnr(a, b):
    mse = torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item()
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def _pad8(x):
    _, _, h, w = x.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), (h, w)


def warp_prev(of_model, cur_rgb, prev_rgb):
    """Motion-compensate prev->cur. Returns (prev_warped_rgb, valid_mask (1,1,H,W))."""
    h, w = cur_rgb.shape[-2:]
    cur_p, _ = _pad8(cur_rgb * 2 - 1)
    prev_p, _ = _pad8(prev_rgb * 2 - 1)
    fw, bw = get_flow_forward_backward(of_model, cur_p, prev_p)  # (1,Hp,Wp,2)
    fw, bw = fw[:, :h, :w, :], bw[:, :h, :w, :]
    prev_warped = flow_warp(prev_rgb, fw, padding_mode='border')
    occ = detect_occlusion(fw, bw)               # (1,H,W), 1=valid
    mask = occ.unsqueeze(1)                       # (1,1,H,W)
    return prev_warped, mask


def _band_mask(mask, like):
    """Downsample full-res valid mask to a band's spatial size."""
    return F.adaptive_avg_pool2d(mask, like.shape[-2:]).unsqueeze(2)  # (1,1,1,Hj,Wj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_path', required=True, help='SR output frames (seq subfolders)')
    ap.add_argument('--gt_path', default=None, help='GT frames (needed for phase-swap)')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--high_levels', default='0,1', help='DT-CWT levels for HF analysis')
    ap.add_argument('--max_seqs', type=int, default=0, help='cap sequences (0=all)')
    ap.add_argument('--swap_examples', type=int, default=2, help='save N phase-swap images')
    ap.add_argument('--save_dir', default='magphase_analysis')
    ap.add_argument('--warp', action='store_true',
                    help='motion-compensate prev via RAFT before measuring (FAIR test)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    high_levels = [int(x) for x in args.high_levels.split(',')]
    os.makedirs(args.save_dir, exist_ok=True)

    xfm = DTCWTForward(J=4, biort='near_sym_b', qshift='qshift_b').to(device)
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(device)
    lp = lpips_lib.LPIPS(net='alex').to(device) if lpips_lib is not None else None

    of_model = None
    if args.warp:
        assert _WARP_OK, f'warp import failed: {_WARP_ERR}'
        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        of_model.requires_grad_(False)

    seqs = list_seqs(args.out_path)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]

    incoh_list, magflick_list, flick_list = [], [], []  # per consecutive pair
    swap_magOut_psnr, swap_phaseOut_psnr = [], []
    n_saved = 0

    for si, seq in enumerate(seqs):
        ofiles = frame_paths(os.path.join(args.out_path, seq))
        if len(ofiles) < 2:
            continue
        prev_luma = None
        prev_bands = None
        prev_rgb = None
        for fi, of in enumerate(ofiles):
            rgb = load_rgb(of, device)
            luma = to_luma(rgb)
            _, bands = dtcwt_bands(xfm, luma)
            if prev_rgb is not None:
                with torch.no_grad():
                    if args.warp:
                        # motion-compensate previous frame to current (FAIR test)
                        prev_w_rgb, mask = warp_prev(of_model, rgb, prev_rgb)
                        _, ref_bands = dtcwt_bands(xfm, to_luma(prev_w_rgb))
                        masks = {j: _band_mask(mask, bands[j][0][0, 0, 0]) for j in high_levels}
                        ref_rgb = prev_w_rgb
                    else:
                        ref_bands, masks, ref_rgb = prev_bands, None, prev_rgb
                    # (1) phase incoherence  &  (3) magnitude flicker
                    incoh = pair_phase_incoherence(bands, ref_bands, high_levels, masks)
                    magf = pair_mag_flicker(bands, ref_bands, high_levels, masks)
                    # perceptual flicker (warped if --warp)
                    if lp is not None:
                        a, b = rgb, ref_rgb
                        if args.warp:
                            a, b = rgb * mask, ref_rgb * mask
                        fl = lp(a * 2 - 1, b * 2 - 1).item()
                    else:
                        fl = torch.mean((rgb - ref_rgb) ** 2).item()
                incoh_list.append(incoh)
                magflick_list.append(magf)
                flick_list.append(fl)
            prev_bands, prev_luma, prev_rgb = bands, luma, rgb

        # (2) phase-swap (needs GT)
        if args.gt_path is not None:
            gseq = os.path.join(args.gt_path, seq)
            if os.path.isdir(gseq):
                gfiles = frame_paths(gseq)
                k = min(len(ofiles), len(gfiles))
                for fi in range(0, k, max(1, k // 3)):  # a few frames per seq
                    lo = to_luma(load_rgb(ofiles[fi], device))
                    lg = to_luma(load_rgb(gfiles[fi], device))
                    if lo.shape != lg.shape:
                        continue
                    rec_m, rec_p = phase_swap(xfm, ifm, lo, lg)
                    swap_magOut_psnr.append(psnr(rec_m, lg))     # out-mag + GT-phase
                    swap_phaseOut_psnr.append(psnr(rec_p, lg))   # GT-mag + out-phase
                    if n_saved < args.swap_examples:
                        def w(name, t):
                            img = (t.clamp(0, 1)[0, 0].cpu().numpy() * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(args.save_dir, name), img)
                        w(f'{args.tag}_{seq}_{fi}_out.png', lo)
                        w(f'{args.tag}_{seq}_{fi}_gt.png', lg)
                        w(f'{args.tag}_{seq}_{fi}_outMag_gtPhase.png', rec_m)
                        w(f'{args.tag}_{seq}_{fi}_gtMag_outPhase.png', rec_p)
                        n_saved += 1

    def corr(a, b):
        if len(a) < 3:
            return float('nan'), float('nan')
        return pearsonr(a, b)[0], spearmanr(a, b)[0]

    print('\n==================  magnitude/phase duality diagnosis  ==================')
    print(f'tag={args.tag}  pairs={len(flick_list)}  high_levels={high_levels}  '
          f'device={device}  warp={"ON (motion-compensated)" if args.warp else "OFF (raw consecutive)"}')
    print(f'mean phase-incoherence : {np.mean(incoh_list):.4f}' if incoh_list else 'no pairs')
    print(f'mean magnitude-flicker : {np.mean(magflick_list):.4f}' if magflick_list else '')
    print(f'mean perceptual-flicker: {np.mean(flick_list):.4f}' if flick_list else '')

    print('\n--- (1)/(3) correlation with perceptual flicker (higher = drives flicker) ---')
    pr, sr = corr(incoh_list, flick_list)
    print(f'  phase-incoherence  vs flicker :  pearson={pr:+.3f}  spearman={sr:+.3f}')
    pr, sr = corr(magflick_list, flick_list)
    print(f'  magnitude-flicker  vs flicker :  pearson={pr:+.3f}  spearman={sr:+.3f}')
    print('  -> if PHASE corr >> MAGNITUDE corr: flicker is phase-driven (supports phase->temporal)')

    if swap_magOut_psnr:
        print('\n--- (2) phase-swap reconstruction vs GT (luma PSNR) ---')
        print(f'  out-magnitude + GT-phase :  PSNR={np.mean(swap_magOut_psnr):.2f}')
        print(f'  GT-magnitude + out-phase :  PSNR={np.mean(swap_phaseOut_psnr):.2f}')
        print('  -> if (out-mag+GT-phase) >> (GT-mag+out-phase): structure/position lives in PHASE,')
        print('     i.e. the output error is mostly in phase -> phase carries position.')
        print(f'  example images saved to: {args.save_dir}/')
    print('========================================================================\n')


if __name__ == '__main__':
    main()
