"""Validation pipeline for evaluating adversarial protection quality."""

import torch
import json
from pathlib import Path
from typing import Optional

from ..models.clip_encoder import CLIPEncoder
from ..utils.image_io import load_image, save_image, save_comparison
from ..utils.metrics import compute_all_metrics, LPIPSMetric


class ValidationPipeline:
    """Validates adversarial images against perceptual quality thresholds
    and measures embedding disruption.

    Checks:
    1. Perceptual quality: PSNR > 30dB, SSIM > 0.95, LPIPS < 0.05
    2. Embedding disruption: cosine similarity between clean/adv embeddings
    3. Optional: JPEG/resize robustness (purification test)
    """

    def __init__(self, encoder: CLIPEncoder, device: str = "cuda"):
        self.encoder = encoder
        self.device = device
        self.lpips_model = LPIPSMetric(device=device)

    @torch.no_grad()
    def validate_single(
        self,
        clean_path: str,
        adversarial_path: str,
        output_dir: Optional[str] = None,
        resolution: int = 224,
    ) -> dict:
        """Validate a single clean/adversarial image pair.

        Args:
            clean_path: path to clean image
            adversarial_path: path to adversarial (protected) image
            output_dir: if provided, save comparison visualizations here
            resolution: input resolution for CLIP

        Returns:
            Dict with all metrics and pass/fail status
        """
        clean = load_image(clean_path, resolution).to(self.device)
        adv = load_image(adversarial_path, resolution).to(self.device)

        # perceptual quality metrics
        metrics = compute_all_metrics(clean, adv, self.lpips_model)

        # embedding disruption
        clean_emb = self.encoder.encode_image(clean)
        adv_emb = self.encoder.encode_image(adv)
        cosine_sim = torch.nn.functional.cosine_similarity(clean_emb, adv_emb, dim=-1).item()
        metrics["cosine_similarity"] = cosine_sim
        metrics["embedding_distance"] = 1.0 - cosine_sim

        # save comparison if output dir provided
        if output_dir:
            out_path = Path(output_dir) / f"comparison_{Path(clean_path).stem}.png"
            save_comparison(clean, adv, str(out_path))
            metrics["comparison_path"] = str(out_path)

        return metrics

    @torch.no_grad()
    def validate_robustness(
        self,
        adversarial: torch.Tensor,
        clean_embedding: torch.Tensor,
        jpeg_quality: int = 85,
        resize_factor: float = 0.5,
    ) -> dict:
        """Test if perturbation survives JPEG compression and resizing.

        Args:
            adversarial: [1, 3, H, W] adversarial image tensor
            clean_embedding: [1, D] clean CLIP embedding
            jpeg_quality: JPEG compression quality
            resize_factor: resize down then up by this factor

        Returns:
            Dict with post-purification cosine similarity
        """
        from ..utils.augmentations import JPEGSimulation
        import torch.nn.functional as F

        results = {}

        # test JPEG robustness
        jpeg_sim = JPEGSimulation(quality=jpeg_quality)
        adv_jpeg = jpeg_sim._jpeg_round_trip(adversarial)
        emb_jpeg = self.encoder.encode_image(adv_jpeg.to(self.device))
        cos_jpeg = torch.nn.functional.cosine_similarity(clean_embedding, emb_jpeg, dim=-1).item()
        results["post_jpeg_cosine_sim"] = cos_jpeg

        # test resize robustness
        h, w = adversarial.shape[2:]
        small = F.interpolate(adversarial, scale_factor=resize_factor, mode="bilinear", align_corners=False)
        restored = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
        emb_resize = self.encoder.encode_image(restored.to(self.device))
        cos_resize = torch.nn.functional.cosine_similarity(clean_embedding, emb_resize, dim=-1).item()
        results["post_resize_cosine_sim"] = cos_resize

        # test combined JPEG + resize
        combined = jpeg_sim._jpeg_round_trip(restored)
        emb_combined = self.encoder.encode_image(combined.to(self.device))
        cos_combined = torch.nn.functional.cosine_similarity(clean_embedding, emb_combined, dim=-1).item()
        results["post_combined_cosine_sim"] = cos_combined

        return results

    def validate_batch(
        self,
        image_dir: str,
        results_dir: str,
        output_dir: Optional[str] = None,
        resolution: int = 224,
    ) -> list:
        """Validate all clean/adversarial pairs in directories.

        Expects matching filenames: image_dir/foo.png -> results_dir/foo_protected.png

        Args:
            image_dir: directory with clean images
            results_dir: directory with protected images
            output_dir: directory for comparison outputs

        Returns:
            List of per-image metric dicts
        """
        image_dir = Path(image_dir)
        results_dir = Path(results_dir)
        all_results = []

        clean_images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

        for clean_path in clean_images:
            stem = clean_path.stem
            # look for protected version
            adv_path = results_dir / f"{stem}_protected.png"
            if not adv_path.exists():
                adv_path = results_dir / f"{stem}_protected.jpg"
            if not adv_path.exists():
                print(f"  Skipping {stem}: no protected version found")
                continue

            result = self.validate_single(
                str(clean_path), str(adv_path), output_dir, resolution
            )
            result["image"] = stem
            all_results.append(result)

            status = "PASS" if result["all_pass"] else "FAIL"
            print(f"  {stem}: {status} | PSNR={result['psnr']:.1f}dB "
                  f"SSIM={result['ssim']:.4f} LPIPS={result['lpips']:.4f} "
                  f"CosSim={result['cosine_similarity']:.4f}")

        # save summary
        if output_dir:
            summary_path = Path(output_dir) / "validation_summary.json"
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2)

        return all_results
