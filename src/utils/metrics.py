"""Perceptual quality metrics: PSNR, SSIM, LPIPS."""

import torch
import numpy as np


def compute_psnr(clean: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute PSNR between clean and adversarial images.

    Args:
        clean: [1, 3, H, W] tensor in [0, 1]
        adversarial: [1, 3, H, W] tensor in [0, 1]

    Returns:
        PSNR in dB (higher = less distortion, target > 30dB)
    """
    mse = ((clean - adversarial) ** 2).mean().item()
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim(clean: torch.Tensor, adversarial: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM between clean and adversarial images.

    Simplified implementation — single-scale SSIM with Gaussian window.
    Target > 0.95.

    Args:
        clean: [1, 3, H, W] tensor in [0, 1]
        adversarial: [1, 3, H, W] tensor in [0, 1]

    Returns:
        SSIM value (higher = more similar, target > 0.95)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # create gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)  # [ws, ws]
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
    window = window.expand(3, 1, -1, -1).to(clean.device)  # [3, 1, ws, ws]

    pad = window_size // 2

    mu1 = torch.nn.functional.conv2d(clean, window, padding=pad, groups=3)
    mu2 = torch.nn.functional.conv2d(adversarial, window, padding=pad, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(clean ** 2, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(adversarial ** 2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(clean * adversarial, window, padding=pad, groups=3) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


class LPIPSMetric:
    """LPIPS perceptual distance metric wrapper."""

    def __init__(self, device: str = "cuda"):
        import lpips
        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def compute(self, clean: torch.Tensor, adversarial: torch.Tensor) -> float:
        """Compute LPIPS distance.

        Args:
            clean: [1, 3, H, W] tensor in [0, 1]
            adversarial: [1, 3, H, W] tensor in [0, 1]

        Returns:
            LPIPS distance (lower = more similar, target < 0.05)
        """
        # LPIPS expects [-1, 1] range
        clean_scaled = clean.to(self.device) * 2 - 1
        adv_scaled = adversarial.to(self.device) * 2 - 1
        return self.model(clean_scaled, adv_scaled).item()


def compute_lpips(clean: torch.Tensor, adversarial: torch.Tensor, lpips_model=None) -> float:
    """Convenience function for LPIPS computation.

    Args:
        clean: [1, 3, H, W] tensor in [0, 1]
        adversarial: [1, 3, H, W] tensor in [0, 1]
        lpips_model: pre-initialized LPIPSMetric (creates one if None)

    Returns:
        LPIPS distance
    """
    if lpips_model is None:
        device = clean.device.type if clean.is_cuda else "cpu"
        lpips_model = LPIPSMetric(device=device)
    return lpips_model.compute(clean, adversarial)


def compute_all_metrics(clean: torch.Tensor, adversarial: torch.Tensor, lpips_model=None) -> dict:
    """Compute all quality metrics at once.

    Returns:
        Dict with psnr, ssim, lpips values and pass/fail status
    """
    psnr = compute_psnr(clean, adversarial)
    ssim = compute_ssim(clean, adversarial)
    lpips_val = compute_lpips(clean, adversarial, lpips_model)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_val,
        "psnr_pass": psnr > 30.0,
        "ssim_pass": ssim > 0.95,
        "lpips_pass": lpips_val < 0.05,
        "all_pass": psnr > 30.0 and ssim > 0.95 and lpips_val < 0.05,
    }
