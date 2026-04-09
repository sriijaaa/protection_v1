"""Input diversity augmentations for robust adversarial perturbation.

These augmentations are applied during the PGD optimization to prevent
overfitting to exact preprocessing. Not used in Phase 0 but ready for Phase 1+.
"""

import torch
import torch.nn.functional as F
import io
from PIL import Image
import torchvision.transforms.functional as TF


class JPEGSimulation(torch.nn.Module):
    """Differentiable JPEG simulation via compress-decompress round-trip.

    Not truly differentiable — uses straight-through estimator (STE)
    to pass gradients through the non-differentiable JPEG step.
    """

    def __init__(self, quality: int = 85):
        super().__init__()
        self.quality = quality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JPEG simulation with STE for gradient flow.

        Args:
            x: [B, 3, H, W] tensor in [0, 1]
        """
        with torch.no_grad():
            compressed = self._jpeg_round_trip(x)
        # straight-through estimator: forward uses JPEG, backward passes through
        return x + (compressed - x).detach()

    def _jpeg_round_trip(self, x: torch.Tensor) -> torch.Tensor:
        """Compress and decompress each image in batch via JPEG."""
        result = torch.zeros_like(x)
        for i in range(x.shape[0]):
            img = TF.to_pil_image(x[i].clamp(0, 1).cpu())
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            img_reloaded = Image.open(buf).convert("RGB")
            result[i] = TF.to_tensor(img_reloaded).to(x.device)
        return result


class RandomResizeCrop(torch.nn.Module):
    """Random resize and crop augmentation."""

    def __init__(self, target_size: int = 224, scale_range: tuple = (0.8, 1.2)):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random resize followed by crop/pad to target size."""
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        new_size = int(self.target_size * scale)

        x = F.interpolate(x, size=new_size, mode="bilinear", align_corners=False)

        if new_size > self.target_size:
            # random crop
            top = torch.randint(0, new_size - self.target_size + 1, (1,)).item()
            left = torch.randint(0, new_size - self.target_size + 1, (1,)).item()
            x = x[:, :, top:top + self.target_size, left:left + self.target_size]
        elif new_size < self.target_size:
            # pad
            pad_h = self.target_size - new_size
            pad_w = self.target_size - new_size
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return x


class GaussianBlurAug(torch.nn.Module):
    """Random Gaussian blur augmentation."""

    def __init__(self, kernel_size: int = 5, sigma_range: tuple = (0.1, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random Gaussian blur."""
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()
        return TF.gaussian_blur(x, kernel_size=self.kernel_size, sigma=sigma)


class BrightnessJitter(torch.nn.Module):
    """Random brightness adjustment."""

    def __init__(self, factor_range: tuple = (0.8, 1.2)):
        super().__init__()
        self.factor_range = factor_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random brightness scaling."""
        factor = torch.empty(1).uniform_(*self.factor_range).item()
        return (x * factor).clamp(0, 1)


class InputDiversityPipeline(torch.nn.Module):
    """Combines multiple augmentations applied randomly during PGD.

    Each augmentation is applied with a given probability.
    """

    def __init__(self, prob: float = 0.5, target_size: int = 224, jpeg_quality: int = 85):
        super().__init__()
        self.prob = prob
        self.augmentations = torch.nn.ModuleList([
            RandomResizeCrop(target_size=target_size),
            JPEGSimulation(quality=jpeg_quality),
            GaussianBlurAug(),
            BrightnessJitter(),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each augmentation with probability self.prob."""
        for aug in self.augmentations:
            if torch.rand(1).item() < self.prob:
                x = aug(x)
        return x
