import json
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .config import TrainConfig


def train_epoch(model, train_loader, device, optimizer, criterion, train_config):
    loss_sum = 0.0
    norm_sum = 0.0
    samples_seen = 0

    for batch in train_loader:
        img = batch['img'].to(device)
        label = batch['fine_label'].to(device)

        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, label)
        loss.backward()
        norm= nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        optimizer.step()

        loss_sum += loss.item() * img.size(0)
        norm_sum += norm.item() * img.size(0)
        samples_seen += img.size(0)
    return {"epoch_loss": loss_sum / samples_seen, "norm": norm_sum / samples_seen, "samples_seen": samples_seen}


def evaluate(model, val_loader, device, criterion):
    loss_sum = 0.0
    correct = 0
    samples_seen = 0
    for batch in val_loader:
        img = batch['img'].to(device)
        label = batch['fine_label'].to(device)

        logits = model(img)
        loss = criterion(logits, label).item()
        loss_sum += loss * img.size(0)
        correct += (logits.argmax(dim=1) == label).sum().item()
        samples_seen += img.size(0)

    val_loss = loss_sum / samples_seen
    accuracy = correct / samples_seen
    return {"val_loss": val_loss, "accuracy": accuracy}


def append_jsonl(filename, data):
    with open(filename, "a") as f:
        json.dump(data, f)
        f.write("\n")


def fit(model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    run_dir: Path,
    train_config: TrainConfig,
):

    model.train()
    for epoch in range(train_config.max_epochs):
        epoch_start = time()
        epoch_metrics = train_epoch(model, train_loader, device, optimizer, criterion, train_config)
        epoch_end = time()

        epoch_metrics["Img/s"] = epoch_metrics.pop("samples_seen") / (epoch_end - epoch_start)
        row = {
            "epoch": epoch + 1,
            "epoch_time": epoch_end - epoch_start,
            **epoch_metrics
        }
        print(row)
        append_jsonl(run_dir / "train_losses.jsonl", row)


        if (epoch + 1) % train_config.validation_interval == 0 or epoch == train_config.max_epochs - 1:
            with torch.inference_mode():
                model.eval()
                validation_metrics = evaluate(model, val_loader, device, criterion)
                model.train()

            row = {"epoch": epoch + 1, **validation_metrics}
            print(row)
            append_jsonl(run_dir / "val_losses.jsonl", row)

    return model
