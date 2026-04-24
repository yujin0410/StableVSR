import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTAdapter(nn.Module):
    def __init__(self, cond_channels, feature_channels):
        super().__init__()
        # cond_channels: mag와 phase_diff를 합친 채널 수 (예: 6 + 6 = 12)
        # feature_channels: UNet 마지막 UpBlock의 피처 채널 수 (SD 2.1 기준 보통 320)
        
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, feature_channels, kernel_size=3, padding=1)
        )
        
        # 감마(Scale)와 베타(Shift)를 생성하는 레이어
        self.gamma = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.beta = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        # ������ 핵심: Zero-Initialization
        # 처음 학습을 시작할 때 가이드가 0이 되도록 하여 Frozen UNet이 충격받지 않게 함
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        # UNet 피처맵(x)과 해상도가 다를 경우를 대비한 안전장치
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        c = self.cond_conv(cond)
        gamma = self.gamma(c)
        beta = self.beta(c)
        
        # SFT 연산: x * (1 + gamma) + beta
        return x * (1 + gamma) + beta
        
        
class UNetWithSFT(nn.Module):
    def __init__(self, unet, sft_adapter):
        super().__init__()
        self.unet = unet
        self.sft_adapter = sft_adapter
        self.current_cond = None

        # UNet의 4번째 UpBlock (마지막 고해상도 디코더 블록)에 Hook 등록
        self.unet.up_blocks[3].register_forward_hook(self.sft_hook)

    def sft_hook(self, module, input, output):
        # 조건(cond)이 들어왔을 때만 SFT 적용
        if self.current_cond is not None:
            # diffusers UNet의 출력은 보통 tuple 형태 (hidden_states,)
            if isinstance(output, tuple):
                h = output[0]
                h = self.sft_adapter(h, self.current_cond)
                return (h,) + output[1:]
            else:
                return self.sft_adapter(output, self.current_cond)
        return output

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, sft_cond=None, **kwargs):
        # 이번 forward step에서 사용할 주파수 가이드 저장
        self.current_cond = sft_cond
        # 기존 UNet 실행 (내부에서 sft_hook이 자동으로 작동함)
        return self.unet(noisy_latents, timesteps, encoder_hidden_states, **kwargs)