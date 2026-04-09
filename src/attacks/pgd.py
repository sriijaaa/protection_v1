"""PGD (Projected Gradient Descent) attack with momentum for adversarial image protection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Callable

from ..models.clip_encoder import CLIPEncoder
from .losses import CombinedLoss


def compute_texture_mask(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Compute a per-pixel texture/edge map to guide perturbation placement.

    High values = textured/edge regions where perturbation is less visible.
    Low values = smooth/flat regions where even small noise is obvious.

    Returns [1, 1, H, W] mask normalized to [0, 1].
    """
    # convert to grayscale
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    # Sobel edges
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

    # local variance (texture measure)
    pad = kernel_size // 2
    avg_kernel = torch.ones(1, 1, kernel_size, kernel_size,
                            dtype=image.dtype, device=image.device) / (kernel_size ** 2)
    local_mean = F.conv2d(gray, avg_kernel, padding=pad)
    local_var = F.conv2d(gray ** 2, avg_kernel, padding=pad) - local_mean ** 2
    local_var = local_var.clamp(min=0)

    # combine edges + texture variance
    mask = edges + local_var.sqrt()

    # normalize to [0, 1] with a minimum floor so no region gets zero budget
    mask = mask / (mask.max() + 1e-8)
    mask = 0.3 + 0.7 * mask  # floor at 0.3 so smooth regions still get some perturbation

    return mask


class PGDAttack:
    """Momentum PGD attack against CLIP encoder with perceptual quality preservation.

    Generates adversarial perturbations that maximize embedding distance
    while staying within L-inf epsilon ball and satisfying perceptual quality
    constraints (PSNR, SSIM, LPIPS).

    Quality-aware features:
    - SSIM preservation loss in the optimization objective
    - Texture-based masking: concentrates perturbation in edges/textures
    - Per-pixel adaptive epsilon based on local image complexity
    """

    def __init__(
        self,
        encoder: CLIPEncoder,
        epsilon: float = 8 / 255,
        step_size: float = 1 / 255,
        num_steps: int = 100,
        momentum: float = 0.9,
        loss_fn: Optional[CombinedLoss] = None,
        input_diversity: Optional[Callable] = None,
        use_texture_mask: bool = True,
    ):
        self.encoder = encoder
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.momentum = momentum
        self.loss_fn = loss_fn or CombinedLoss()
        self.input_diversity = input_diversity
        self.device = encoder.device
        self.use_texture_mask = use_texture_mask

    def attack(
        self,
        clean_image: torch.Tensor,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Run PGD attack to generate adversarial image.

        Args:
            clean_image: [1, 3, H, W] tensor in [0, 1] range, float32
            verbose: show progress bar

        Returns:
            (adversarial_image, attack_log) where attack_log contains per-step losses
        """
        assert clean_image.dim() == 4 and clean_image.shape[0] == 1
        assert clean_image.dtype == torch.float32, "Must use float32 for gradient precision"

        clean_image = clean_image.to(self.device)

        # compute clean embedding (no grad needed)
        with torch.no_grad():
            clean_embedding = self.encoder.encode_image(clean_image)

        # compute texture mask for adaptive epsilon
        if self.use_texture_mask:
            texture_mask = compute_texture_mask(clean_image)  # [1, 1, H, W]
            adaptive_epsilon = self.epsilon * texture_mask  # per-pixel epsilon
            adaptive_step = self.step_size * texture_mask
        else:
            adaptive_epsilon = self.epsilon
            adaptive_step = self.step_size

        # initialize perturbation with small random noise
        delta = torch.zeros_like(clean_image, requires_grad=True)
        delta.data.uniform_(-self.epsilon * 0.05, self.epsilon * 0.05)
        delta.data = torch.clamp(clean_image + delta.data, 0, 1) - clean_image

        # momentum buffer
        grad_momentum = torch.zeros_like(clean_image)

        attack_log = {
            "losses": [],
            "cosine_sims": [],
            "loss_components": [],
        }

        iterator = tqdm(range(self.num_steps), desc="PGD Attack", disable=not verbose)

        for step in iterator:
            delta.requires_grad_(True)
            adv_image = clean_image + delta

            # optionally apply input diversity (augmentations)
            if self.input_diversity is not None:
                adv_input = self.input_diversity(adv_image)
            else:
                adv_input = adv_image

            # forward pass
            adv_embedding = self.encoder.encode_image(adv_input)

            # compute loss with SSIM preservation
            loss, loss_components = self.loss_fn(
                clean_embedding.detach(),
                adv_embedding,
                clean_image=clean_image,
                adv_image=adv_image,
            )

            # backward pass
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.data

                # normalize gradient
                grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12)

                # apply momentum
                grad_momentum = self.momentum * grad_momentum + grad_norm

                # signed gradient step with adaptive (texture-aware) step size
                delta.data = delta.data - adaptive_step * grad_momentum.sign()

                # project back to adaptive epsilon ball
                delta.data = torch.clamp(delta.data, -adaptive_epsilon, adaptive_epsilon)

                # project to valid image range
                delta.data = torch.clamp(clean_image + delta.data, 0, 1) - clean_image

            # zero gradients
            delta.grad = None

            # logging
            cosine_sim = loss_components.get("embedding", loss.item())
            attack_log["losses"].append(loss.item())
            attack_log["cosine_sims"].append(cosine_sim)
            attack_log["loss_components"].append(loss_components)

            if verbose:
                iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "cos_sim": f"{cosine_sim:.4f}",
                    "ssim_l": f"{loss_components.get('ssim', 0):.4f}",
                })

        # final adversarial image
        adv_image = torch.clamp(clean_image + delta.data, 0, 1)

        attack_log["final_cosine_sim"] = attack_log["cosine_sims"][-1]
        attack_log["final_delta_linf"] = delta.data.abs().max().item()
        attack_log["epsilon"] = self.epsilon
        attack_log["num_steps"] = self.num_steps

        return adv_image.detach(), attack_log
