import json
import signal
from contextlib import contextmanager
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



@contextmanager
def stop_after_epoch(enabled: bool = True):
    if not enabled:
        yield lambda: False
        return

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("Stopping after this epoch. Press Ctrl+C again to stop immediately.", flush=True)

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: stop_requested
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def fit(model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    run_dir: Path,
    train_config: TrainConfig,
    resume: Path | None = None,
    handle_signals: bool = True,
):
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = None
    path_best = Path(run_dir) / "checkpoint_best_tmp.pt"
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]
        print(f"Resuming after epoch {start_epoch}; target: {train_config.max_epochs} epochs")
    with stop_after_epoch(enabled=handle_signals) as stop_requested:
        model.train()
        for epoch in range(start_epoch, train_config.max_epochs):
            is_best = False

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

            if stop_requested() or (epoch + 1) % train_config.validation_interval == 0 or epoch == train_config.max_epochs - 1 or epoch == 0:
                with torch.inference_mode():
                    model.eval()
                    validation_metrics = evaluate(model, val_loader, device, criterion)
                    model.train()

                    if validation_metrics["val_loss"] < best_val_loss:
                        is_best = True
                        best_val_loss = validation_metrics["val_loss"]
                        best_epoch = epoch + 1

                row = {"epoch": epoch + 1, **validation_metrics}
                print(row)
                append_jsonl(run_dir / "val_losses.jsonl", row)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }

            torch.save(checkpoint, run_dir / "checkpoint_last.pt")
            if is_best:
                path_best.unlink(missing_ok=True)
                path_best = run_dir / f"checkpoint_best_epoch={best_epoch}_val={best_val_loss:.4f}.pt"
                torch.save(checkpoint, path_best)

            if epoch + 1 == train_config.max_epochs:
                torch.save(checkpoint, run_dir / f"checkpoint_{epoch + 1:04d}.pt")

            if stop_requested():
                print(f"Epoch {epoch + 1} saved. Stopping.")
                break

    return model
