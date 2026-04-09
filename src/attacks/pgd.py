"""PGD (Projected Gradient Descent) attack with momentum for adversarial image protection."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Callable
from torchvision.transforms.functional import gaussian_blur

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
        num_attention_layers: int = 4,
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
        self.num_attention_layers = num_attention_layers

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
        delta = torch.zeros_like(clean_image)
        delta.uniform_(-self.epsilon * 0.05, self.epsilon * 0.05)
        delta = torch.clamp(clean_image + delta, 0, 1) - clean_image

        # momentum buffer
        grad_momentum = torch.zeros_like(clean_image)

        attack_log = {
            "losses": [],
            "cosine_sims": [],
            "loss_components": [],
        }

        # register attention hooks if the loss weight is active
        use_attention = self.loss_fn.weights.get("attention", 0) > 0
        if use_attention:
            self.encoder.hook_attention_layers(self.num_attention_layers)

        iterator = tqdm(range(self.num_steps), desc="PGD Attack", disable=not verbose)

        for _ in iterator:
            # === NI-FGSM: compute gradient at Nesterov lookahead point ===
            lookahead_delta = (delta + adaptive_step * grad_momentum.sign()).detach().requires_grad_(True)
            adv_image = clean_image + lookahead_delta

            # optionally apply input diversity (augmentations)
            if self.input_diversity is not None:
                adv_input = self.input_diversity(adv_image)
            else:
                adv_input = adv_image

            # forward pass
            adv_embedding = self.encoder.encode_image(adv_input)

            # collect attention maps captured by hooks (empty list if hooks not active)
            attention_maps = self.encoder.get_attention_maps() if use_attention else []

            # compute loss with SSIM preservation + attention entropy
            loss, loss_components = self.loss_fn(
                clean_embedding.detach(),
                adv_embedding,
                clean_image=clean_image,
                adv_image=adv_image,
                attention_maps=attention_maps,
            )

            # backward pass
            loss.backward()

            with torch.no_grad():
                grad = lookahead_delta.grad.data

                # normalize gradient
                grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12)

                # apply momentum (using gradient from lookahead point)
                grad_momentum = self.momentum * grad_momentum + grad_norm

                # === TI-PGD: smooth accumulated gradient before sign to gain
                #     spatial-shift robustness (helps transfer to cropping pipelines)
                smoothed_momentum = gaussian_blur(grad_momentum, kernel_size=[5, 5], sigma=[1.5, 1.5])

                # signed gradient step with adaptive (texture-aware) step size
                delta = delta - adaptive_step * smoothed_momentum.sign()

                # project back to adaptive epsilon ball
                delta = torch.clamp(delta, -adaptive_epsilon, adaptive_epsilon)

                # project to valid image range
                delta = torch.clamp(clean_image + delta, 0, 1) - clean_image

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
                    "attn_l": f"{loss_components.get('attention', 0):.4f}",
                })

        if use_attention:
            self.encoder.remove_hooks()

        # final adversarial image
        adv_image = torch.clamp(clean_image + delta, 0, 1)

        attack_log["final_cosine_sim"] = attack_log["cosine_sims"][-1]
        attack_log["final_delta_linf"] = delta.abs().max().item()
        attack_log["epsilon"] = self.epsilon
        attack_log["num_steps"] = self.num_steps

        return adv_image.detach(), attack_log
