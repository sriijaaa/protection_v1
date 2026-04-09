"""CLIP ViT-L/14 encoder wrapper for adversarial perturbation generation."""

import torch
import torch.nn as nn
import open_clip


class CLIPEncoder(nn.Module):
    """Wraps open_clip CLIP ViT-L/14 for feature extraction with optional attention hooking."""

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).float()  # force float32
        self.model.eval()

        # freeze all parameters — we never train the encoder
        for param in self.model.parameters():
            param.requires_grad = False

        # storage for hooked attention maps
        self._attention_maps = []
        self._hooks = []

    def get_visual_encoder(self):
        """Return the visual transformer trunk."""
        return self.model.visual

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensor [B,3,H,W] in [0,1] range to normalized embedding."""
        # open_clip expects preprocessed input; we apply normalization manually
        # since we need gradients to flow through the input
        x = self._normalize(x)
        features = self.model.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_image_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-patch features from second-to-last transformer layer.

        Returns shape [B, num_patches, dim] — useful for patch-level disruption loss.
        Not used in Phase 0 but wired up for later phases.
        """
        x = self._normalize(x)
        visual = self.model.visual

        # get patch embeddings
        x = visual.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, patches, width]

        # prepend CLS token
        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.unsqueeze(0)
        x = visual.ln_pre(x)

        # run through all but last transformer block
        # ViT-L/14 has 24 blocks
        transformer = visual.transformer
        for i, block in enumerate(transformer.resblocks):
            if i == len(transformer.resblocks) - 1:
                break  # stop before last block
            x = block(x)

        # return patch tokens only (exclude CLS)
        return x[:, 1:, :]

    def hook_attention_layers(self, num_layers: int = 4):
        """Register forward hooks on the last N attention layers to capture attention maps.

        Used for attention entropy maximization in later phases.
        """
        self.remove_hooks()
        self._attention_maps = []

        visual = self.model.visual
        blocks = visual.transformer.resblocks
        target_blocks = list(blocks)[-num_layers:]

        for block in target_blocks:
            hook = block.attn.register_forward_hook(self._attention_hook_fn)
            self._hooks.append(hook)

    def _attention_hook_fn(self, module, input, output):
        """Capture attention weights from multi-head attention."""
        # open_clip's attention returns (attn_output, attn_weights) when needed
        # We'll compute attention weights manually from Q, K
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input

        # For open_clip ViT, we can access attention via the module
        # Store the input for attention computation in the loss function
        self._attention_maps.append(x)

    def get_attention_maps(self) -> list:
        """Return captured attention maps and clear the buffer."""
        maps = self._attention_maps.copy()
        self._attention_maps = []
        return maps

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_maps = []

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CLIP's ImageNet normalization to [0,1] tensor."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def get_input_resolution(self) -> int:
        """Return expected input resolution (224 for ViT-L/14)."""
        return self.model.visual.image_size

    def __repr__(self):
        res = self.get_input_resolution()
        return f"CLIPEncoder(ViT-L/14, resolution={res}, device={self.device})"
