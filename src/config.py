"""Configuration for adversarial image protection."""

from dataclasses import dataclass, field


@dataclass
class Phase0Config:
    """Phase 0: Minimal PGD against CLIP ViT-L/14, single loss, 100 iterations."""

    # encoder
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"

    # PGD parameters
    epsilon: float = 8 / 255        # L-inf constraint (8/255 to meet PSNR>30dB)
    step_size: float = 1 / 255      # per-step perturbation
    num_steps: int = 100            # PGD iterations
    momentum: float = 0.9           # gradient momentum

    # input
    input_resolution: int = 224     # CLIP ViT-L/14 native resolution

    # loss weights
    loss_weights: dict = field(default_factory=lambda: {
        "embedding": 1.0,
        "patch": 0.0,
        "attention": 0.7,
        "ssim": 0.5,
    })

    # quality thresholds
    psnr_threshold: float = 30.0    # dB
    ssim_threshold: float = 0.95
    lpips_threshold: float = 0.05

    # device
    device: str = "cuda"

    # paths
    input_dir: str = "experiments/images"
    output_dir: str = "experiments/results"
    comparison_dir: str = "experiments/comparisons"


@dataclass
class Phase1Config(Phase0Config):
    """Phase 1: Add frequency masking + input diversity.

    Extends Phase 0 with:
    - Frequency survival masking (JPEG q=85 + bilinear resize)
    - Input diversity augmentations during optimization
    """
    num_steps: int = 200
    use_frequency_mask: bool = True
    jpeg_quality: int = 85
    use_input_diversity: bool = True
    input_diversity_prob: float = 0.5


@dataclass
class Phase2Config(Phase1Config):
    """Phase 2: Add attention entropy + patch disruption.

    Extends Phase 1 with:
    - Attention entropy maximization on last 4 transformer blocks
    - Patch-level embedding disruption at second-to-last layer
    """
    num_steps: int = 300
    loss_weights: dict = field(default_factory=lambda: {
        "embedding": 1.0,
        "patch": 0.3,
        "attention": 0.2,
    })
    attention_hook_layers: int = 4


@dataclass
class Phase3Config(Phase2Config):
    """Phase 3: Multi-encoder ensemble + purification hardening.

    Extends Phase 2 with:
    - SigLIP-400M and DINOv2-ViT-L as additional surrogates
    - Purification hardening via iterative correction
    """
    num_steps: int = 500
    ensemble_models: list = field(default_factory=lambda: [
        {"name": "ViT-L-14", "pretrained": "openai", "weight": 0.5},
        {"name": "ViT-SO400M-14-SigLIP", "pretrained": "webli", "weight": 0.3},
        # DINOv2 handled separately (not open_clip)
    ])
    use_purification_hardening: bool = True
    purification_rounds: int = 3
