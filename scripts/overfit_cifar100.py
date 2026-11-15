from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.transforms import Compose, Normalize, ToTensor

from nano_cua.config import load_config, save_config
from nano_cua.ResNet import ResNet18, ResNetConfig
from nano_cua.seeding import seed_everything, seed_worker
from nano_cua.trainer import fit
from nano_cua.ViT import VisionTransformer, VisionTransformerConfig


def resolve_model(model: str) -> tuple[type, type, Path]:
    if model == "resnet":
        return ResNet18, ResNetConfig, Path("config/resnet_config.yaml")
    elif model == "vit":
        return VisionTransformer, VisionTransformerConfig, Path("config/vit_config.yaml")
    raise ValueError(f"Unknown model: {model}")


def create_transform():
    preprocess = Compose([
        ToTensor(),
        Normalize(
            mean=(0.5071, 0.4866, 0.4409),
            std=(0.2673, 0.2564, 0.2762)
        ),
    ])

    def transform(samples):
        samples['img'] = [preprocess(img) for img in samples['img']]
        return samples

    return transform


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(model, run_dir, resume=None, handle_signals=True):
    model_class, config_class, config_path = resolve_model(model)
    if resume is not None:
        run_dir = Path(resume).parent
    config_path = (Path(resume).parent / "run_config.yaml" if resume is not None else config_path)
    config = load_config(config_path, config_class)
    run_dir.mkdir(parents=True, exist_ok=resume is not None)
    save_config(config, run_dir)

    device = get_device()
    seed_everything(config.train.seed)

    # Evaluate on the training subset to check that the model can memorize it.
    dataset = load_dataset("uoft-cs/cifar100")["train"].select(range(256))
    dataset.set_transform(create_transform())
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train.train_batch_size, shuffle=True, # type: ignore
        generator=torch.Generator().manual_seed(config.train.seed),
        worker_init_fn=seed_worker,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train.val_batch_size, shuffle=False, # type: ignore
        generator=torch.Generator().manual_seed(config.train.seed),
        worker_init_fn=seed_worker,
    )

    model = model_class(config.model)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay)
    total_steps = config.train.max_epochs * len(train_loader)
    warmup_steps = config.train.warmup_epochs * len(train_loader)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
        ],
        milestones=[warmup_steps],
    )
    criterion = torch.nn.CrossEntropyLoss()

    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_config=config.train,
        run_dir=run_dir,
        device=device,
        resume=resume,
        handle_signals=handle_signals,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["resnet", "vit"],
        help="Model to overfit (resnet, vit)",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        help="Directory for a new run; inferred from --resume when resuming"
    )
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume training from a checkpoint")
    args = parser.parse_args()

    if args.resume is None and args.run_dir is None:
        parser.error("--run_dir is required when not resuming from a checkpoint")

    main(args.model, args.run_dir, resume=args.resume)
