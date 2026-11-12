from dataclasses import dataclass


@dataclass(frozen=True)
class ResNetConfig:
    num_classes: int
