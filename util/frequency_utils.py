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


def process_freq_cond(Yh_cur, Yh_prev_w):
    """Build per-scale [magnitude, phase_diff] features for SFT conditioning.

    Returns:
        List of tensors, one per DT-CWT scale (finest first), each of shape
        [B, 2 * C * 6, H_j, W_j] where 2 = (mag, phase_diff), C = RGB,
        and 6 = directional subbands.
    """
    mags = get_dtcwt_magnitude(Yh_cur)
    phase_diffs = compute_phase_diff(Yh_cur, Yh_prev_w)

    per_scale = []
    for mag, pd in zip(mags, phase_diffs):
        B, C, D, H, W = mag.shape
        mag_r = mag.reshape(B, C * D, H, W)
        pd_r = pd.reshape(B, C * D, H, W)
        per_scale.append(torch.cat([mag_r, pd_r], dim=1))
    return per_scale


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


# ---------------------------------------------------------------------------
# Frequency-separated loss for the dual-SFT design.
# HIGH (j=1, j=2): magnitude-only L1 (phase free, allows hallucination).
# LOW  (j=3, j=4): magnitude L1 + magnitude-weighted phase mismatch.
# Lowpass: simple L1.
# ---------------------------------------------------------------------------


def _magnitude(yh_j):
    return torch.sqrt(yh_j[..., 0] ** 2 + yh_j[..., 1] ** 2 + 1e-8)


def _phase(yh_j):
    return torch.atan2(yh_j[..., 1], yh_j[..., 0])


def frequency_separated_loss(yh_pred, yh_gt, yl_pred, yl_gt,
                             high_levels=(0, 1), low_levels=(2, 3),
                             lambda_high=0.1, lambda_low=1.0, lambda_lp=1.0):
    """Frequency-separated loss aligned with the dual-SFT design.

    Args:
        yh_pred, yh_gt: lists of DT-CWT highpass tensors (J=4).
        yl_pred, yl_gt: lowpass tensors.
        high_levels: indices of high-frequency levels (mag-only loss).
        low_levels: indices of low-frequency levels (mag + phase loss).
        lambda_high / lambda_low / lambda_lp: weights.
    """
    loss_high = 0.0
    for j in high_levels:
        m_pred = _magnitude(yh_pred[j])
        m_gt = _magnitude(yh_gt[j])
        loss_high = loss_high + torch.nn.functional.l1_loss(m_pred, m_gt)

    loss_low = 0.0
    for j in low_levels:
        m_pred = _magnitude(yh_pred[j])
        m_gt = _magnitude(yh_gt[j])
        phi_pred = _phase(yh_pred[j])
        phi_gt = _phase(yh_gt[j])
        loss_low = loss_low + torch.nn.functional.l1_loss(m_pred, m_gt)
        # Magnitude-weighted phase mismatch (1 - cos(d)) is in [0, 2]
        loss_low = loss_low + (m_gt * (1.0 - torch.cos(phi_pred - phi_gt))).mean()

    loss_lp = torch.nn.functional.l1_loss(yl_pred, yl_gt)

    return lambda_low * loss_low + lambda_high * loss_high + lambda_lp * loss_lp
