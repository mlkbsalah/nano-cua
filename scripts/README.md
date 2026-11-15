# Script description

## `overfit_cifar100.py`

Overfit ResNet18 or ViT on the first 256 training images from `uoft-cs/cifar100`. 

```bash
uv run scripts/overfit_cifar100.py --model resnet --run_dir runs/overfit_resnet
uv run scripts/overfit_cifar100.py --model vit --run_dir runs/overfit_vit
uv run scripts/overfit_cifar100.py --model vit --resume runs/overfit_vit/last.pt
```

## `train_cifar100.py`

This script is an interface for training ViT or ResNet18 model on CIFAR100. To run the script, use the following command:

```bash
uv run scripts/train_cifar100.py --model resnet --run_dir <run_directory>
```

The run directory is where the model checkpoints and logs will be saved.

The script also exposes the option to resume training from a checkpoint.

```bash
uv run scripts/train_cifar100.py --model vit --resume <checkpoint_path>
```

The checkpoint path should be the path to a saved model checkpoint. Its parent directory is used as the run directory, so `--run_dir` is only required for new runs. If both arguments are supplied, the checkpoint's parent directory takes precedence.
