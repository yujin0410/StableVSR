import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward


def compute_phase_diff(Yh_cur, Yh_prev_w):
    """
    Per-scale phase difference between two DT-CWT highpass tensor lists.
    Each element shape: [B, C, 6, H, W, 2] (last dim: real, imag).
    Returns list of tensors shape [B, C, 6, H, W] wrapped to [-pi, pi].
    """
    phase_diffs = []
    for yh_c, yh_p in zip(Yh_cur, Yh_prev_w):
        real_c, imag_c = yh_c[..., 0], yh_c[..., 1]
        real_p, imag_p = yh_p[..., 0], yh_p[..., 1]

        phase_c = torch.atan2(imag_c, real_c)
        phase_p = torch.atan2(imag_p, real_p)

        diff = phase_c - phase_p
        # Wrap to [-pi, pi] to avoid discontinuities (e.g. 170° - (-170°))
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        phase_diffs.append(diff)

    return phase_diffs


def get_dtcwt_magnitude(Yh):
    """
    Complex magnitude per scale. Yh element shape: [B, C, 6, H, W, 2].
    Returns list of tensors shape [B, C, 6, H, W].
    """
    mags = []
    for yh in Yh:
        mag = torch.sqrt(yh[..., 0] ** 2 + yh[..., 1] ** 2 + 1e-8)
        mags.append(mag)
    return mags


def process_freq_cond(Yh_cur, Yh_prev_w, target_level=0):
    """
    Build SFT conditioning tensor at the chosen wavelet scale.
    Output shape: [B, 2 * C * 6, H, W]  (mag concat with phase_diff).
    """
    mags = get_dtcwt_magnitude(Yh_cur)
    phase_diffs = compute_phase_diff(Yh_cur, Yh_prev_w)

    mag_target = mags[target_level]
    phase_diff_target = phase_diffs[target_level]

    B, C, D, H, W = mag_target.shape
    mag_reshaped = mag_target.reshape(B, C * D, H, W)
    phase_reshaped = phase_diff_target.reshape(B, C * D, H, W)

    return torch.cat([mag_reshaped, phase_reshaped], dim=1)


def wavelet_magnitude_loss(Yh_pred, Yh_gt):
    """
    L1 loss on per-scale complex magnitudes. Preferred over elementwise
    torch.abs on (real, imag) in diffusion training since the latter
    over-regularizes and encourages blurry predictions.
    """
    loss = 0.0
    for pred_h, gt_h in zip(Yh_pred, Yh_gt):
        pred_mag = torch.sqrt(pred_h[..., 0] ** 2 + pred_h[..., 1] ** 2 + 1e-8)
        gt_mag = torch.sqrt(gt_h[..., 0] ** 2 + gt_h[..., 1] ** 2 + 1e-8)
        loss = loss + torch.nn.functional.l1_loss(pred_mag, gt_mag)
    return loss
