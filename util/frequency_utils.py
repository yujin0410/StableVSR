import torch
import torch.nn as nn

from pytorch_wavelets import DTCWTForward


def compute_phase_diff(Yh_cur, Yh_prev_w):
    """Per-scale phase difference between two DTCWT high-pass pyramids.

    Each element of Yh is a real tensor of shape [B, C, 6, H, W, 2]
    where the last dim is (real, imag). Returns a list of phase-difference
    tensors of shape [B, C, 6, H, W] wrapped to [-pi, pi].
    """
    phase_diffs = []
    for yh_c, yh_p in zip(Yh_cur, Yh_prev_w):
        real_c, imag_c = yh_c[..., 0], yh_c[..., 1]
        real_p, imag_p = yh_p[..., 0], yh_p[..., 1]
        phase_c = torch.atan2(imag_c, real_c)
        phase_p = torch.atan2(imag_p, real_p)
        diff = phase_c - phase_p
        # Wrap to [-pi, pi] so phase discontinuities do not dominate the signal.
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        phase_diffs.append(diff)
    return phase_diffs


def get_dtcwt_magnitude(Yh):
    """Magnitude sqrt(real^2 + imag^2) of a DTCWT high-pass pyramid.

    Input Yh: list of tensors with last dim = 2 (real, imag).
    Output: list of tensors with the last dim squeezed out.
    The small epsilon keeps gradients finite at zero magnitude.
    """
    mags = []
    for yh in Yh:
        mag = torch.sqrt(yh[..., 0] ** 2 + yh[..., 1] ** 2 + 1e-8)
        mags.append(mag)
    return mags


def process_freq_cond(Yh_cur, Yh_prev_w, target_level=0):
    """Assemble the SFT conditioning tensor from DTCWT pyramids.

    Returns [B, 2 * C * 6, H, W]: magnitude (C*6 channels) concatenated with
    wrapped phase-difference (C*6 channels) at the requested pyramid level.
    For 3-channel RGB LR input this yields 36 channels.
    """
    mags = get_dtcwt_magnitude(Yh_cur)
    phase_diffs = compute_phase_diff(Yh_cur, Yh_prev_w)

    mag_target = mags[target_level]              # [B, C, 6, H, W]
    phase_diff_target = phase_diffs[target_level]  # [B, C, 6, H, W]

    B, C, D, H, W = mag_target.shape
    mag_reshaped = mag_target.reshape(B, C * D, H, W)
    phase_reshaped = phase_diff_target.reshape(B, C * D, H, W)
    return torch.cat([mag_reshaped, phase_reshaped], dim=1)
