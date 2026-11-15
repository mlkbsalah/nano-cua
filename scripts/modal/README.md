### Running the ResNet18 training script on CIFAR-100 using Modal.

```bash
modal run --detach scripts/modal/train_resnet18.py \
  --run-dir second_modal_run_resnet18
```

### Listing the folders in the volume.

```bash
modal volume ls nano-cua-volume
```

### Deleting a folder in the volume.

```bash
modal volume rm nano-cua-volume
```

### Downloading the folder from the volume.

```bash
modal volume get nano-cua-volume /runs
```
