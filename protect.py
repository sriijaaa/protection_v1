"""
Main script for Phase 0: Adversarial Image Protection via CLIP Perturbation.

Usage:
    # Protect a single image
    python protect.py --input path/to/image.png --output experiments/results/

    # Protect all images in a directory
    python protect.py --input experiments/images/ --output experiments/results/

    # With custom parameters
    python protect.py --input image.png --output results/ --steps 200 --epsilon 0.0627

    # CPU-only mode
    python protect.py --input image.png --output results/ --device cpu
"""

import argparse
import json
import time
import torch
from pathlib import Path

from src.models.clip_encoder import CLIPEncoder
from src.attacks.pgd import PGDAttack
from src.attacks.losses import CombinedLoss
from src.utils.image_io import load_image, save_image, save_comparison
from src.utils.metrics import compute_all_metrics, LPIPSMetric
from src.config import Phase0Config


def protect_image(
    image_path: str,
    encoder: CLIPEncoder,
    attack: PGDAttack,
    lpips_model: LPIPSMetric,
    config: Phase0Config,
) -> dict:
    """Protect a single image with adversarial perturbation.

    Args:
        image_path: path to clean input image
        encoder: CLIP encoder
        attack: PGD attack instance
        lpips_model: LPIPS metric instance
        config: configuration

    Returns:
        Dict with output paths and metrics
    """
    image_name = Path(image_path).stem
    print(f"\n{'='*60}")
    print(f"Protecting: {image_name}")
    print(f"{'='*60}")

    # load image at CLIP resolution
    clean = load_image(image_path, config.input_resolution).to(config.device)
    print(f"  Resolution: {clean.shape[2]}x{clean.shape[3]}")

    # run attack at CLIP resolution
    start_time = time.time()
    adversarial, attack_log = attack.attack(clean, verbose=True)
    elapsed = time.time() - start_time
    print(f"  Attack completed in {elapsed:.1f}s")

    # compute quality metrics
    res = clean.shape[2]
    metrics = compute_all_metrics(clean, adversarial, lpips_model)
    print(f"\n  Quality Metrics ({res}x{res}):")
    print(f"    PSNR:  {metrics['psnr']:.2f} dB  {'PASS' if metrics['psnr_pass'] else 'FAIL'} (threshold: >{config.psnr_threshold})")
    print(f"    SSIM:  {metrics['ssim']:.4f}    {'PASS' if metrics['ssim_pass'] else 'FAIL'} (threshold: >{config.ssim_threshold})")
    print(f"    LPIPS: {metrics['lpips']:.4f}    {'PASS' if metrics['lpips_pass'] else 'FAIL'} (threshold: <{config.lpips_threshold})")

    # compute embedding disruption
    with torch.no_grad():
        clean_emb = encoder.encode_image(clean)
        adv_emb = encoder.encode_image(adversarial)
        cosine_sim = torch.nn.functional.cosine_similarity(clean_emb, adv_emb, dim=-1).item()
    print(f"\n  Embedding Disruption:")
    print(f"    Cosine similarity: {cosine_sim:.4f}")
    print(f"    Embedding distance: {1 - cosine_sim:.4f}")

    # save protected image
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adv_path = output_dir / f"{image_name}_protected.png"
    save_image(adversarial, str(adv_path))
    print(f"\n  Saved protected image ({res}x{res}): {adv_path}")

    # save comparison visualization
    comp_dir = Path(config.comparison_dir)
    comp_path = comp_dir / f"{image_name}_comparison.png"
    save_comparison(clean, adversarial, str(comp_path))
    print(f"  Saved comparison: {comp_path}")

    result = {
        "image": image_name,
        "input_path": str(image_path),
        "output_path": str(adv_path),
        "comparison_path": str(comp_path),
        "metrics": metrics,
        "cosine_similarity": cosine_sim,
        "embedding_distance": 1 - cosine_sim,
        "attack_time_seconds": elapsed,
        "final_loss": attack_log["losses"][-1],
        "num_steps": config.num_steps,
        "epsilon": config.epsilon,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Adversarial Image Protection (Phase 0)")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: from config)")
    parser.add_argument("--steps", type=int, default=None, help="Number of PGD steps (default: 100)")
    parser.add_argument("--epsilon", type=float, default=None, help="L-inf epsilon (default: 16/255)")
    parser.add_argument("--step-size", type=float, default=None, help="PGD step size (default: 2/255)")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    parser.add_argument("--resolution", type=int, default=None, help="Input resolution (default: 224)")
    args = parser.parse_args()

    # config
    config = Phase0Config()
    if args.output:
        config.output_dir = args.output
        config.comparison_dir = str(Path(args.output) / "comparisons")
    if args.steps:
        config.num_steps = args.steps
    if args.epsilon:
        config.epsilon = args.epsilon
    if args.step_size:
        config.step_size = args.step_size
    if args.device:
        config.device = args.device
    if args.resolution:
        config.input_resolution = args.resolution

    # auto-detect device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"

    print(f"Device: {config.device}")
    if config.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"\nPhase 0 Configuration:")
    print(f"  Epsilon: {config.epsilon:.4f} ({config.epsilon * 255:.0f}/255)")
    print(f"  Step size: {config.step_size:.4f} ({config.step_size * 255:.0f}/255)")
    print(f"  Steps: {config.num_steps}")
    print(f"  Momentum: {config.momentum}")
    print(f"  Resolution: {config.input_resolution}")

    # initialize model
    print(f"\nLoading CLIP {config.clip_model}...")
    encoder = CLIPEncoder(
        model_name=config.clip_model,
        pretrained=config.clip_pretrained,
        device=config.device,
    )
    print(f"  {encoder}")

    # initialize loss
    loss_fn = CombinedLoss(weights=config.loss_weights)

    # initialize attack
    attack = PGDAttack(
        encoder=encoder,
        epsilon=config.epsilon,
        step_size=config.step_size,
        num_steps=config.num_steps,
        momentum=config.momentum,
        loss_fn=loss_fn,
    )

    # initialize LPIPS
    lpips_model = LPIPSMetric(device=config.device)

    # collect input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            list(input_path.glob("*.png")) +
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.jpeg"))
        )
        if not image_paths:
            print(f"No images found in {input_path}")
            return
        print(f"\nFound {len(image_paths)} images to protect")
    else:
        print(f"Input not found: {input_path}")
        return

    # process each image
    all_results = []
    for img_path in image_paths:
        result = protect_image(
            str(img_path), encoder, attack, lpips_model, config
        )
        all_results.append(result)

    # save summary
    summary_path = Path(config.output_dir) / "protection_summary.json"
    with open(summary_path, "w") as f:
        # convert non-serializable values
        json.dump(all_results, f, indent=2, default=lambda x: bool(x) if isinstance(x, (bool, type(True))) else float(x))

    # print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        m = r["metrics"]
        status = "PASS" if m["all_pass"] else "FAIL"
        print(f"  {r['image']}: {status} | "
              f"PSNR={m['psnr']:.1f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} | "
              f"CosSim={r['cosine_similarity']:.4f} | "
              f"{r['attack_time_seconds']:.1f}s")

    passed = sum(1 for r in all_results if r["metrics"]["all_pass"])
    print(f"\n  {passed}/{len(all_results)} images passed quality thresholds")
    print(f"  Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
