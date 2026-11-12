import torch
from torch import nn

from .config import VisionTransformerConfig
from .tranformer import TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embed))


        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Conv2d(config.channels, config.n_embed, kernel_size=config.patch_size, stride=config.patch_size),
                "wpe": nn.Embedding(config.sequence_length, config.n_embed),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )

        self.head = nn.Sequential(
            nn.Linear(config.n_embed, config.mlp_size),
            nn.GELU(),
            nn.Linear(config.mlp_size, config.num_classes)
        )

    def forward(self, x):
        return self.head(self.forward_features(x))


    def forward_features(self, x):
        """x : (Batch, Channels, Height, Width) """
        _, C, H, W = x.size()
        if C != self.config.channels:
            raise ValueError(
                f"Cannot forward sequence of channels {C}, expected {self.config.channels}"
            )
        if H % self.config.patch_size != 0 or W % self.config.patch_size != 0:
            raise ValueError(
                f"Cannot forward sequence of length {H}x{W}, sequence length must be a multiple of {self.config.patch_size}x{self.config.patch_size}"
            )

        x = self.transformer["wte"](x) # (Batch, n_embed, H/patch, W/patch)
        x = x.flatten(start_dim=2).transpose(1, 2) # (Batch, N, n_embed)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        T = x.size(1)
        assert T <= self.config.sequence_length, (
            f"Cannot forward sequence of length {T}, sequence length is only {self.config.sequence_length}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer["wpe"](pos)
        x = x + pos_emb[None, :, :]  # broadcast to (B, T, C) and add

        for transformer_block in self.transformer["h"]:  # type: ignore
            x = transformer_block(x)

        x = self.transformer["ln_f"](x)
        return x[:, 0, :]
