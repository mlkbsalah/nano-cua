from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int
    train_batch_size: int
    val_batch_size: int
    lr: float
    validation_interval: int
    weight_decay: float
    max_grad_norm: float
    seed: int

@dataclass(frozen=True)
class Config[ModelConfigT]:
    model: ModelConfigT
    train: TrainConfig

## Nice little trick for the future, it's just for the type checker tho
# “Declare a type placeholder T. Accept a file path and a class for T. Return a Config whose model field contains an instance of T.”
def load_config[ModelConfigT](path: Path, model_config_type: type[ModelConfigT]):
    with path.open() as f:
        data = yaml.safe_load(f)

    return Config(
        model=model_config_type(**data["model"]),
        train=TrainConfig(**data["train"]),
    )


def save_config[ModelConfigT](config: Config[ModelConfigT], run_dir: Path):
    with (run_dir / "run_config.yaml").open("w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
