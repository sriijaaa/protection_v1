"""Loss functions for adversarial perturbation optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingDistanceLoss(nn.Module):
    """Maximize cosine distance between clean and adversarial CLIP embeddings.

    This is the primary Phase 0 loss. The intuition: if the adversarial image's
    CLIP embedding is far from the clean image's embedding, downstream models
    that rely on CLIP-family encoders will misunderstand the image content.
    """

    def __init__(self):
        super().__init__()

    def forward(self, clean_embedding: torch.Tensor, adv_embedding: torch.Tensor) -> torch.Tensor:
        """Compute negative cosine similarity (we minimize this to maximize distance).

        Args:
            clean_embedding: [B, D] normalized CLIP embedding of clean image
            adv_embedding: [B, D] normalized CLIP embedding of adversarial image

        Returns:
            Scalar loss (negative cosine similarity, so minimizing = maximizing distance)
        """
        cosine_sim = F.cosine_similarity(clean_embedding, adv_embedding, dim=-1)
        return -cosine_sim.mean()  # negative so that minimizing → maximizes distance


class PatchDisruptionLoss(nn.Module):
    """Disrupt per-patch features to break local spatial understanding.

    Maximizes distance between clean and adversarial patch-level features
    at the second-to-last ViT layer. Not used in Phase 0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, clean_patches: torch.Tensor, adv_patches: torch.Tensor) -> torch.Tensor:
        """Compute negative cosine similarity across all patches.

        Args:
            clean_patches: [B, num_patches, D] clean patch features
            adv_patches: [B, num_patches, D] adversarial patch features

        Returns:
            Scalar loss
        """
        # normalize per-patch
        clean_norm = F.normalize(clean_patches, dim=-1)
        adv_norm = F.normalize(adv_patches, dim=-1)

        # cosine similarity per patch, then average
        cosine_sim = (clean_norm * adv_norm).sum(dim=-1)  # [B, num_patches]
        return cosine_sim.mean()


class AttentionEntropyLoss(nn.Module):
    """Push attention distributions toward maximum entropy (uniform).

    When attention is uniform, the model can't localize objects —
    it can't find "the shirt" or "the background" to edit them.
    Not used in Phase 0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, attention_maps: list) -> torch.Tensor:
        """Compute negative entropy of attention distributions.

        Args:
            attention_maps: list of attention weight tensors from hooked layers

        Returns:
            Scalar loss (negative entropy — minimizing pushes toward uniform)
        """
        total_loss = 0.0
        count = 0

        for attn in attention_maps:
            # attn shape: [B*num_heads, seq_len, seq_len]
            # compute entropy per row
            attn_clamped = attn.clamp(min=1e-8)
            entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [B*H, seq_len]
            max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=attn.dtype, device=attn.device))
            # negative normalized entropy: minimizing this maximizes entropy
            total_loss += -(entropy / max_entropy).mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0)
        return total_loss / count


class SSIMPreservationLoss(nn.Module):
    """Differentiable SSIM loss to keep perturbation structurally invisible.

    Penalizes the attack when SSIM drops below the target threshold,
    guiding perturbation into regions where structural impact is minimal.
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        # precompute gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1))

    def forward(self, clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
        """Compute 1 - SSIM (minimizing this maximizes SSIM)."""
        if self.window.device != clean.device:
            self.window = self.window.to(clean.device)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        mu1 = F.conv2d(clean, self.window, padding=pad, groups=3)
        mu2 = F.conv2d(adversarial, self.window, padding=pad, groups=3)

        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(clean ** 2, self.window, padding=pad, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(adversarial ** 2, self.window, padding=pad, groups=3) - mu2_sq
        sigma12 = F.conv2d(clean * adversarial, self.window, padding=pad, groups=3) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()  # minimize → SSIM goes up


class CombinedLoss(nn.Module):
    """Weighted combination of all loss components."""

    def __init__(self, weights: dict = None):
        super().__init__()
        self.weights = weights or {
            "embedding": 1.0,
            "patch": 0.0,       # disabled in Phase 0
            "attention": 0.0,   # disabled in Phase 0
            "ssim": 0.5,        # SSIM preservation
        }
        self.embedding_loss = EmbeddingDistanceLoss()
        self.patch_loss = PatchDisruptionLoss()
        self.attention_loss = AttentionEntropyLoss()
        self.ssim_loss = SSIMPreservationLoss()

    def forward(self, clean_emb, adv_emb, clean_image=None, adv_image=None,
                clean_patches=None, adv_patches=None, attention_maps=None):
        losses = {}
        total = torch.tensor(0.0, device=clean_emb.device)

        # embedding distance (always active)
        l_emb = self.embedding_loss(clean_emb, adv_emb)
        losses["embedding"] = l_emb.item()
        total = total + self.weights["embedding"] * l_emb

        # SSIM preservation
        if self.weights.get("ssim", 0) > 0 and clean_image is not None and adv_image is not None:
            l_ssim = self.ssim_loss(clean_image, adv_image)
            losses["ssim"] = l_ssim.item()
            total = total + self.weights["ssim"] * l_ssim

        # patch disruption
        if self.weights.get("patch", 0) > 0 and clean_patches is not None and adv_patches is not None:
            l_patch = self.patch_loss(clean_patches, adv_patches)
            losses["patch"] = l_patch.item()
            total = total + self.weights["patch"] * l_patch

        # attention entropy
        if self.weights.get("attention", 0) > 0 and attention_maps:
            l_attn = self.attention_loss(attention_maps)
            losses["attention"] = l_attn.item()
            total = total + self.weights["attention"] * l_attn

        losses["total"] = total.item()
        return total, losses
