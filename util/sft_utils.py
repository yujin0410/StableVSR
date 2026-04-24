import torch
import torch.nn as nn
import torch.nn.functional as F


class SFTAdapter(nn.Module):
    def __init__(self, cond_channels, feature_channels):
        super().__init__()

        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, feature_channels, kernel_size=3, padding=1),
        )

        self.gamma = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.beta = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        # Zero-init so SFT starts as identity (x * 1 + 0 == x)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        if cond.shape[-2:] != x.shape[-2:]:
            # torch 2.0 bilinear upsample CUDA kernel has no bf16 path; round-trip fp32.
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

        # accelerate's prepared-model wrapper returns fp32, but UNet's next layers
        # are bf16/fp16, so restore the original feature dtype after SFT.
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
