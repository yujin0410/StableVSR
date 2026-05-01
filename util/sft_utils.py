import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention (Woo et al., 2018)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_p = x.max(dim=1, keepdim=True)[0]
        attn = torch.cat([avg, max_p], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class SFTHead(nn.Module):
    """Two-conv head (3x3 -> SiLU -> 3x3) producing gamma or beta.

    The last conv is zero-initialized; bias is set to ``init`` so that the
    head outputs the constant ``init`` everywhere on initialization. This
    lets gamma start at 1.0 (identity scale) and beta at 0.0 (no shift),
    so ``x * gamma + beta == x`` at the very first step.
    """

    def __init__(self, in_channels, out_channels, init=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, init)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class PerLevelProcessor(nn.Module):
    """Per-DT-CWT-level processor with magnitude/phase pathway separation.

    Input shape: ``[B, 3, 6, H, W, 2]`` (DT-CWT highpass output, last dim
    holds real and imaginary components).

    Magnitude pathway: ``M -> DW conv -> SiLU -> SA -> 1x1 -> e_M``.
    Phase pathway: stacked ``[sin(phi), cos(phi)] -> DW conv -> SiLU
    -> SA -> 1x1 -> [e_sin, e_cos]``.
    Recombination: ``e_real = e_M * e_cos``, ``e_imag = e_M * e_sin``.
    SFT heads consume the concatenation ``[e_M, e_real, e_imag]`` and
    produce ``gamma`` (init 1.0) and ``beta`` (init 0.0).

    A residual feature ``e_M`` is also returned alongside a learnable
    scalar ``alpha`` (zero-initialized). The residual is currently not
    applied to the U-Net feature; it is reserved for future extensions.
    """

    def __init__(self, in_channels=18, mid_channels=64, sft_out_channels=320,
                 target_size=None):
        super().__init__()
        self.target_size = tuple(target_size) if target_size is not None else None
        self.mid_channels = mid_channels

        self.M_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.M_sa = SpatialAttention(7)
        self.M_pw = nn.Conv2d(in_channels, mid_channels, 1)

        self.phase_dw = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1,
                                  groups=in_channels * 2)
        self.phase_sa = SpatialAttention(7)
        self.phase_pw = nn.Conv2d(in_channels * 2, mid_channels * 2, 1)

        self.sft_gamma = SFTHead(mid_channels * 3, sft_out_channels, init=1.0)
        self.sft_beta = SFTHead(mid_channels * 3, sft_out_channels, init=0.0)

        self.res_alpha = nn.Parameter(torch.zeros(1))

    def _maybe_resize(self, x):
        if self.target_size is None:
            return x
        if x.shape[-2:] != self.target_size:
            return F.interpolate(x, size=self.target_size,
                                 mode='bilinear', align_corners=False)
        return x

    def forward(self, yh_j):
        # yh_j: [B, 3, 6, H, W, 2]
        B, C, D, H, W, _ = yh_j.shape

        real = yh_j[..., 0]
        imag = yh_j[..., 1]

        M = torch.sqrt(real * real + imag * imag + 1e-8)
        sin_phi = imag / (M + 1e-8)
        cos_phi = real / (M + 1e-8)

        M_flat = M.reshape(B, C * D, H, W)
        sin_flat = sin_phi.reshape(B, C * D, H, W)
        cos_flat = cos_phi.reshape(B, C * D, H, W)

        M_flat = self._maybe_resize(M_flat)
        sin_flat = self._maybe_resize(sin_flat)
        cos_flat = self._maybe_resize(cos_flat)

        e_M = F.silu(self.M_dw(M_flat))
        e_M = self.M_sa(e_M)
        e_M = self.M_pw(e_M)

        phase_in = torch.cat([sin_flat, cos_flat], dim=1)
        e_phase = F.silu(self.phase_dw(phase_in))
        e_phase = self.phase_sa(e_phase)
        e_phase = self.phase_pw(e_phase)
        e_sin, e_cos = e_phase.chunk(2, dim=1)

        e_real = e_M * e_cos
        e_imag = e_M * e_sin

        combined = torch.cat([e_M, e_real, e_imag], dim=1)
        gamma = self.sft_gamma(combined)
        beta = self.sft_beta(combined)

        return gamma, beta, e_M, self.res_alpha


class FrequencyConditioningEncoder(nn.Module):
    """Frequency-separated multi-level conditioning encoder.

    Splits a 4-level DT-CWT decomposition into HIGH (j=1, j=2 = yh[0],
    yh[1]) and LOW (j=3, j=4 = yh[2], yh[3]) groups. Each level is
    processed by its own ``PerLevelProcessor``; the resulting per-level
    gamma/beta tensors are concatenated channel-wise within each group.

    Args:
        sft_out_channels_high: per-level SFT output channels for HIGH.
            The final HIGH gamma/beta channel count is
            ``2 * sft_out_channels_high`` (must equal the U-Net inject
            channels at ``up_blocks[1]``).
        sft_out_channels_low: per-level SFT output channels for LOW.
            The final LOW gamma/beta channel count is
            ``2 * sft_out_channels_low`` (must equal the U-Net inject
            channels at ``down_blocks[1]``).
        high_target / low_target: spatial size each per-level processor
            should resize to. ``None`` keeps the input's native spatial
            size (only safe if all levels in a group already match).
    """

    def __init__(self, in_channels=18, mid_channels=64,
                 sft_out_channels_high=320, sft_out_channels_low=320,
                 high_target=None, low_target=None):
        super().__init__()
        self.proc_j1 = PerLevelProcessor(in_channels, mid_channels,
                                         sft_out_channels_high, high_target)
        self.proc_j2 = PerLevelProcessor(in_channels, mid_channels,
                                         sft_out_channels_high, high_target)
        self.proc_j3 = PerLevelProcessor(in_channels, mid_channels,
                                         sft_out_channels_low, low_target)
        self.proc_j4 = PerLevelProcessor(in_channels, mid_channels,
                                         sft_out_channels_low, low_target)

    def forward(self, yh, yl=None):
        gamma_j1, beta_j1, res_j1, a1 = self.proc_j1(yh[0])
        gamma_j2, beta_j2, res_j2, a2 = self.proc_j2(yh[1])
        gamma_j3, beta_j3, res_j3, a3 = self.proc_j3(yh[2])
        gamma_j4, beta_j4, res_j4, a4 = self.proc_j4(yh[3])

        high_gamma = torch.cat([gamma_j1, gamma_j2], dim=1)
        high_beta = torch.cat([beta_j1, beta_j2], dim=1)
        high_residual = a1 * res_j1 + a2 * res_j2

        low_gamma = torch.cat([gamma_j3, gamma_j4], dim=1)
        low_beta = torch.cat([beta_j3, beta_j4], dim=1)
        low_residual = a3 * res_j3 + a4 * res_j4

        return {
            'high_gamma': high_gamma,
            'high_beta': high_beta,
            'high_residual': high_residual,
            'low_gamma': low_gamma,
            'low_beta': low_beta,
            'low_residual': low_residual,
        }


class UNetWithDualSFT(nn.Module):
    """Wrap a U-Net with dual SFT injection.

    LOW SFT modulates the output of ``down_blocks[1]`` (encoder mid-low
    layer, used for structural / low-frequency guidance).
    HIGH SFT modulates the output of ``up_blocks[1]`` (decoder
    mid-high layer, used for textural / high-frequency guidance).

    Modulation is ``x * gamma + beta``. With ``gamma`` initialized to 1
    and ``beta`` to 0, the wrapped U-Net starts as the identity for
    safe fine-tuning.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.current_low = None      # tuple (gamma, beta) or None
        self.current_high = None

        self.unet.down_blocks[1].register_forward_hook(self._low_hook)
        self.unet.up_blocks[1].register_forward_hook(self._high_hook)

    @staticmethod
    def _resize_to(t, target_hw):
        if t.shape[-2:] != target_hw:
            return F.interpolate(t.float(), size=target_hw,
                                 mode='bilinear', align_corners=False)
        return t

    def _modulate(self, output, params):
        if params is None:
            return output
        gamma, beta = params
        if isinstance(output, tuple):
            ref = output[0]
            g = self._resize_to(gamma, ref.shape[-2:]).to(ref.dtype)
            b = self._resize_to(beta, ref.shape[-2:]).to(ref.dtype)
            return (ref * g + b,) + output[1:]
        ref = output
        g = self._resize_to(gamma, ref.shape[-2:]).to(ref.dtype)
        b = self._resize_to(beta, ref.shape[-2:]).to(ref.dtype)
        return ref * g + b

    def _low_hook(self, module, input, output):
        return self._modulate(output, self.current_low)

    def _high_hook(self, module, input, output):
        return self._modulate(output, self.current_high)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states,
                low_sft=None, high_sft=None, **kwargs):
        self.current_low = low_sft
        self.current_high = high_sft
        try:
            return self.unet(noisy_latents, timesteps, encoder_hidden_states, **kwargs)
        finally:
            self.current_low = None
            self.current_high = None


# ---------------------------------------------------------------------------
# Backwards-compatible alias for the old single-SFT design used by older
# checkpoints. Kept so test.py / train.py do not need to import-guard.
# Prefer ``FrequencyConditioningEncoder`` + ``UNetWithDualSFT`` for new
# training runs.
# ---------------------------------------------------------------------------


class SFTAdapter(nn.Module):
    """Legacy single-SFT adapter (multi-scale, single up_blocks[3] inject).

    Kept for compatibility with checkpoints trained before the dual-SFT
    redesign. New runs should use ``FrequencyConditioningEncoder`` and
    ``UNetWithDualSFT`` instead.
    """

    def __init__(self, cond_channels=36, feature_channels=256, num_levels=3):
        super().__init__()
        self.num_levels = num_levels

        self.scale_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cond_channels, cond_channels, 3, padding=1, groups=cond_channels),
                nn.SiLU(),
                SpatialAttention(7),
                nn.Conv2d(cond_channels, cond_channels, 1),
            )
            for _ in range(num_levels)
        ])
        self.scale_logits = nn.Parameter(torch.zeros(num_levels))

        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, feature_channels, 3, padding=1),
        )
        self.gamma = nn.Conv2d(feature_channels, feature_channels, 1)
        self.beta = nn.Conv2d(feature_channels, feature_channels, 1)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def _aggregate(self, multi_scale_feats):
        target_h, target_w = multi_scale_feats[0].shape[-2:]
        enhanced = []
        for j, feat in enumerate(multi_scale_feats):
            e = self.scale_enhancers[j](feat) if j < self.num_levels else feat
            if e.shape[-2:] != (target_h, target_w):
                e = F.interpolate(e, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            enhanced.append(e)
        weights = F.softmax(self.scale_logits[: len(enhanced)], dim=0)
        return sum(w * e for w, e in zip(weights, enhanced))

    def forward(self, x, multi_scale_cond):
        if not isinstance(multi_scale_cond, (list, tuple)):
            multi_scale_cond = [multi_scale_cond]
        cond = self._aggregate(multi_scale_cond)
        if cond.shape[-2:] != x.shape[-2:]:
            orig_dtype = cond.dtype
            cond = F.interpolate(cond.float(), size=x.shape[-2:],
                                 mode='bilinear', align_corners=False).to(orig_dtype)
        c = self.cond_conv(cond)
        gamma = self.gamma(c)
        beta = self.beta(c)
        return x * (1 + gamma) + beta


class UNetWithSFT(nn.Module):
    """Legacy single-injection wrapper kept for compatibility."""

    def __init__(self, unet, sft_adapter):
        super().__init__()
        self.unet = unet
        self.sft_adapter = sft_adapter
        self.current_cond = None
        self.unet.up_blocks[3].register_forward_hook(self._sft_hook)

    def _sft_hook(self, module, input, output):
        if self.current_cond is None:
            return output
        if isinstance(output, tuple):
            ref = output[0]
            return (self.sft_adapter(ref, self.current_cond).to(ref.dtype),) + output[1:]
        return self.sft_adapter(output, self.current_cond).to(output.dtype)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, sft_cond=None, **kwargs):
        self.current_cond = sft_cond
        try:
            return self.unet(noisy_latents, timesteps, encoder_hidden_states, **kwargs)
        finally:
            self.current_cond = None
