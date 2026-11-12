from dataclasses import dataclass


@dataclass(frozen=True)
class VisionTransformerConfig:
    num_classes: int
    mlp_size: int
    patch_size: int
    channels: int
    sequence_length: int  # the max number of patches
    n_layers: int
    n_heads: int
    n_embed: int
    is_causal: bool
