import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class VSRMetrics:
    """
    PSNR / SSIM / LPIPS aggregator for validation.

    Inputs expected in [0, 1] range, shape [B, 3, H, W]. LPIPS internally
    expects [-1, 1] so we rescale.
    """

    def __init__(self, device, lpips_net="alex"):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=lpips_net, normalize=False
        ).to(device)
        self.device = device

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.to(self.device).clamp(0.0, 1.0)
        gt = gt.to(self.device).clamp(0.0, 1.0)
        self.psnr.update(pred, gt)
        self.ssim.update(pred, gt)
        self.lpips.update(pred * 2 - 1, gt * 2 - 1)

    def compute(self):
        return {
            "psnr": self.psnr.compute().item(),
            "ssim": self.ssim.compute().item(),
            "lpips": self.lpips.compute().item(),
        }

    def reset(self):
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()
