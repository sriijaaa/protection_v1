"""CLIP ViT-L/14 encoder wrapper for adversarial perturbation generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        """Compute and capture softmax attention weights from Q, K projections.

        Recomputes attention from the MHA input rather than relying on
        need_weights=True (which open_clip disables for speed).
        Produces [B*num_heads, seq_len, seq_len] maps — what AttentionEntropyLoss expects.
        """
        try:
            x = input[0]  # [seq_len, B, embed_dim] — query == key for self-attention
            seq_len, B = x.shape[0], x.shape[1]
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = embed_dim // num_heads

            # packed QKV weight: [3*embed_dim, embed_dim]
            W_q = module.in_proj_weight[:embed_dim].detach()
            W_k = module.in_proj_weight[embed_dim:2 * embed_dim].detach()
            b_q = module.in_proj_bias[:embed_dim].detach() if module.in_proj_bias is not None else None
            b_k = module.in_proj_bias[embed_dim:2 * embed_dim].detach() if module.in_proj_bias is not None else None

            # F.linear on x (which carries grad from adversarial input) with detached weights
            # ensures gradients flow back through x to the perturbation
            q = F.linear(x, W_q, b_q)  # [seq_len, B, embed_dim]
            k = F.linear(x, W_k, b_k)

            # reshape to [B*num_heads, seq_len, head_dim]
            q = q.view(seq_len, B * num_heads, head_dim).transpose(0, 1)
            k = k.view(seq_len, B * num_heads, head_dim).transpose(0, 1)

            scale = head_dim ** -0.5
            attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
            self._attention_maps.append(attn_weights)
        except Exception:
            pass  # never break the forward pass

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
