"""DINOv2 backbone wrapper.

Wraps HuggingFace's DINOv2 implementation with:
- Configurable partial freeze (freeze layers [0, unfreeze_from_layer-1])
- Discriminative LR groups (frozen / unfrozen / heads)
- CLS-token feature extraction
- Optional return of attention maps for Grad-CAM-style visualization
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config


@dataclass
class BackboneOutput:
    cls_features: torch.Tensor          # [B, embed_dim]  CLS token features
    patch_features: torch.Tensor        # [B, num_patches, embed_dim]
    attentions: list[torch.Tensor] | None  # per-layer attention maps if requested


class DINOv2Backbone(nn.Module):
    """Wrapper around facebook/dinov2-{small,base,large,giant}.

    Parameters
    ----------
    name : str
        HF model name (e.g. "facebook/dinov2-small").
    unfreeze_from_layer : int
        Index of the first transformer layer to unfreeze. Set to 12 to keep
        everything frozen (linear probe); set to 0 for full fine-tune.
    output_attentions : bool
        If True, returns per-layer attention maps. Slower; only enable for
        analysis runs.
    """

    def __init__(
        self,
        name: str = "facebook/dinov2-small",
        unfreeze_from_layer: int = 9,
        output_attentions: bool = False,
    ):
        super().__init__()
        self.name = name
        self.unfreeze_from_layer = unfreeze_from_layer
        self.output_attentions = output_attentions

        config = Dinov2Config.from_pretrained(name)
        config.output_attentions = output_attentions
        self.model = Dinov2Model.from_pretrained(name, config=config)
        self.embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self._apply_freeze_schedule()

    # ------------------------------------------------------------
    # Freeze logic
    # ------------------------------------------------------------

    def _apply_freeze_schedule(self) -> None:
        """Freeze patch embedding + lower transformer blocks."""
        # always freeze patch embeddings (they're stable)
        for p in self.model.embeddings.parameters():
            p.requires_grad = False

        # freeze blocks below the threshold
        for i, block in enumerate(self.model.encoder.layer):
            requires_grad = (i >= self.unfreeze_from_layer)
            for p in block.parameters():
                p.requires_grad = requires_grad

        # final layernorm: keep trainable iff any block is trainable
        any_block_unfrozen = self.unfreeze_from_layer < self.num_layers
        for p in self.model.layernorm.parameters():
            p.requires_grad = any_block_unfrozen

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------
    # Param groups (for discriminative LR)
    # ------------------------------------------------------------

    def get_param_groups(self, base_lr: float):
        """Return list of param groups for the optimizer."""
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr}]

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=self.output_attentions,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state          # [B, 1+N, D]
        cls = last_hidden[:, 0]                          # CLS token
        patches = last_hidden[:, 1:]
        return BackboneOutput(
            cls_features=cls,
            patch_features=patches,
            attentions=list(outputs.attentions) if self.output_attentions else None,
        )

    def get_attention_for_cls(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get the last-layer attention from CLS to each patch (for Grad-CAM-like vis)."""
        old = self.output_attentions
        self.output_attentions = True
        self.model.config.output_attentions = True
        try:
            out = self.forward(pixel_values)
            # last layer attentions: [B, num_heads, 1+N, 1+N]
            last_attn = out.attentions[-1]
            # average over heads, take CLS→patches row, drop CLS-to-CLS column
            cls_to_patches = last_attn.mean(dim=1)[:, 0, 1:]   # [B, N]
            return cls_to_patches
        finally:
            self.output_attentions = old
            self.model.config.output_attentions = old
