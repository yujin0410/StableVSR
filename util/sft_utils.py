import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention (Woo et al., 2018)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        attn = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class PerScaleEnhancer(nn.Module):
    """Per-scale wavelet feature enhancement.

    Pipeline: depthwise 3x3 conv -> SiLU -> spatial attention -> pointwise 1x1 conv.
    Channel count is preserved across the block.
    """

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.act = nn.SiLU()
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        x = self.spatial_attn(x)
        x = self.pw(x)
        return x


class SFTAdapter(nn.Module):
    """Multi-scale frequency conditioning + spatial feature transform.

    Three-stage design:
      1. Per-scale processing: each DT-CWT level (mag + phase_diff,
         `cond_channels` channels) is enhanced independently via
         depthwise conv -> spatial attention -> 1x1 conv.
      2. Multi-scale aggregation: enhanced features are spatially aligned
         to the finest level and combined via softmax-weighted sum,
         keeping the channel count at `cond_channels`.
      3. SFT injection: the aggregated feature drives gamma/beta heads
         that modulate the target U-Net feature as x*(1+gamma) + beta.
         Gamma and beta are zero-initialized so the adapter starts as
         identity.
    """

    def __init__(self, cond_channels=36, feature_channels=256, num_levels=3):
        super().__init__()
        self.num_levels = num_levels

        self.scale_enhancers = nn.ModuleList(
            [PerScaleEnhancer(cond_channels) for _ in range(num_levels)]
        )
        self.scale_logits = nn.Parameter(torch.zeros(num_levels))

        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, feature_channels, kernel_size=3, padding=1),
        )
        self.gamma = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.beta = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def _aggregate(self, multi_scale_feats):
        target_h, target_w = multi_scale_feats[0].shape[-2:]

        enhanced = []
        for j, feat in enumerate(multi_scale_feats):
            if j < self.num_levels:
                e = self.scale_enhancers[j](feat)
            else:
                e = feat
            if e.shape[-2:] != (target_h, target_w):
                e = F.interpolate(
                    e, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
            enhanced.append(e)

        weights = F.softmax(self.scale_logits[: len(enhanced)], dim=0)
        out = sum(w * e for w, e in zip(weights, enhanced))
        return out

    def forward(self, x, multi_scale_cond):
        if not isinstance(multi_scale_cond, (list, tuple)):
            multi_scale_cond = [multi_scale_cond]

        cond = self._aggregate(multi_scale_cond)

        if cond.shape[-2:] != x.shape[-2:]:
            orig_dtype = cond.dtype
            cond = F.interpolate(
                cond.float(), size=x.shape[-2:], mode='bilinear', align_corners=False
            ).to(orig_dtype)

        c = self.cond_conv(cond)
        gamma = self.gamma(c)
        beta = self.beta(c)

        return x * (1 + gamma) + beta


class UNetWithSFT(nn.Module):
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
            h = self.sft_adapter(ref, self.current_cond).to(ref.dtype)
            return (h,) + output[1:]
        return self.sft_adapter(output, self.current_cond).to(output.dtype)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, sft_cond=None, **kwargs):
        self.current_cond = sft_cond
        try:
            return self.unet(noisy_latents, timesteps, encoder_hidden_states, **kwargs)
        finally:
            self.current_cond = None
