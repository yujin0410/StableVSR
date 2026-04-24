import torch
import torch.nn as nn
import torch.nn.functional as F


class SFTAdapter(nn.Module):
    """Spatial Feature Transform adapter.

    cond_channels: channels of the concatenated conditioning map
        (e.g. magnitude 6 + phase_diff 6 = 12 for single-level DTCWT with 6 orientations).
    feature_channels: channels of the UNet feature map to modulate
        (320 for the last UpBlock of SD 2.1).
    """

    def __init__(self, cond_channels, feature_channels):
        super().__init__()
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, feature_channels, kernel_size=3, padding=1),
        )
        self.gamma = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.beta = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        # Zero-init so the adapter starts as identity (x * 1 + 0) and does not
        # disturb the frozen UNet at the beginning of training.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if cond.dtype != x.dtype:
            cond = cond.to(dtype=x.dtype)
        c = self.cond_conv(cond)
        gamma = self.gamma(c)
        beta = self.beta(c)
        return x * (1 + gamma) + beta


class UNetWithSFT(nn.Module):
    """Wraps a frozen UNet and injects SFT modulation on the last up-block
    output via a forward hook. The conditioning is passed through ``sft_cond``
    kwarg to ``forward``; if None, the hook is a no-op.
    """

    def __init__(self, unet, sft_adapter):
        super().__init__()
        self.unet = unet
        self.sft_adapter = sft_adapter
        self.current_cond = None
        # Register hook on last up-block (highest resolution decoder stage).
        self.unet.up_blocks[3].register_forward_hook(self.sft_hook)

    def sft_hook(self, module, input, output):
        if self.current_cond is None:
            return output
        if isinstance(output, tuple):
            h = self.sft_adapter(output[0], self.current_cond)
            return (h,) + output[1:]
        return self.sft_adapter(output, self.current_cond)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, sft_cond=None, **kwargs):
        self.current_cond = sft_cond
        try:
            return self.unet(noisy_latents, timesteps, encoder_hidden_states, **kwargs)
        finally:
            # Always reset so a subsequent bare unet(...) call does not leak conditioning.
            self.current_cond = None
