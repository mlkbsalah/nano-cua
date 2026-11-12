
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from nano_cua.config import load_config, save_config
from nano_cua.ResNet import ResNet18, ResNetConfig
from nano_cua.seeding import seed_everything, seed_worker
from nano_cua.trainer import fit

CONFIG_PATH = Path("config/resnet_config.yaml")

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

def main(run_dir, resume=None):
    if resume is not None:
        run_dir = Path(resume).parent
    config_path = (Path(resume).parent / "run_config.yaml" if resume is not None else CONFIG_PATH)
    config = load_config(config_path, ResNetConfig)
    run_dir.mkdir(parents=True, exist_ok=resume is not None)
    save_config(config, run_dir)


    device = get_device()
    seed_everything(config.train.seed)



    dataset = load_dataset("uoft-cs/cifar100")

    dataset.set_transform(create_transform())
    train_loader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=config.train.train_batch_size, shuffle=True, # type: ignore
        generator=torch.Generator().manual_seed(config.train.seed),
        worker_init_fn=seed_worker,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=config.train.val_batch_size, shuffle=False, # type: ignore
        generator=torch.Generator().manual_seed(config.train.seed),
        worker_init_fn=seed_worker,
    )


    model = ResNet18(config.model)
    model = model.to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()


    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        train_config=config.train,
        run_dir=run_dir,
        device=device,
        resume=resume,
    )






if __name__ == "__main__":

    parser = ArgumentParser()
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

    main(args.run_dir, resume=args.resume)
