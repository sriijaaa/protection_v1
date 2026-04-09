"""Image loading, saving, and conversion utilities."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF


def load_image(path: str, resolution: int = 224) -> torch.Tensor:
    """Load image, resize to target resolution, return [1, 3, H, W] float32 tensor in [0, 1].

    Args:
        path: path to image file
        resolution: target size (square crop/resize)

    Returns:
        [1, 3, resolution, resolution] float32 tensor in [0, 1]
    """
    img = Image.open(path).convert("RGB")

    # resize maintaining aspect ratio then center crop
    img = TF.resize(img, resolution, interpolation=TF.InterpolationMode.BICUBIC)
    img = TF.center_crop(img, resolution)

    tensor = TF.to_tensor(img).unsqueeze(0).float()  # [1, 3, H, W] in [0, 1]
    return tensor


def load_image_native(path: str) -> torch.Tensor:
    """Load image at native resolution without resizing.

    Returns:
        [1, 3, H, W] float32 tensor in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0).float()
    return tensor


def save_image(tensor: torch.Tensor, path: str, quality: int = 95):
    """Save [1, 3, H, W] or [3, H, W] tensor as PNG or JPEG.

    Args:
        tensor: image tensor in [0, 1] range
        path: output path (extension determines format)
        quality: JPEG quality if saving as JPEG
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = tensor_to_pil(tensor)

    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(str(path), "JPEG", quality=quality)
    else:
        img.save(str(path), "PNG")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [1, 3, H, W] or [3, H, W] tensor in [0, 1] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_comparison(clean: torch.Tensor, adversarial: torch.Tensor, path: str):
    """Save side-by-side comparison of clean and adversarial images.

    Also saves the amplified perturbation (delta * 10) for visualization.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean_np = clean.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = adversarial.squeeze(0).permute(1, 2, 0).cpu().numpy()
    delta = (adv_np - clean_np)
    delta_amplified = np.clip(delta * 10 + 0.5, 0, 1)  # amplify and center at 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean_np)
    axes[0].set_title("Clean")
    axes[0].axis("off")

    axes[1].imshow(adv_np)
    axes[1].set_title("Protected")
    axes[1].axis("off")

    axes[2].imshow(delta_amplified)
    axes[2].set_title("Perturbation (10x)")
    axes[2].axis("off")

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
