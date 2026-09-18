import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import save_json
from .model import save_checkpoint

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    eval_loss = running_loss / total if total > 0 else 0.0
    eval_acc = correct / total if total > 0 else 0.0

    return {
        "loss": eval_loss,
        "accuracy": eval_acc,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    learning_rate: float,
    class_weights: torch.Tensor | None,
    device: torch.device,
    checkpoint_path: str,
    history_path: str,
    phase_name: str,
) -> dict:
    # 1. Khởi tạo Hàm tổn thất (Loss Function) có tính đến Class Weights
    if class_weights is not None:
        weights = class_weights.to(device=device, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 2. Lọc các tham số được phép huấn luyện (Trainable Parameters)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6,
    )

    history = {
        "phase": phase_name,
        "epochs": n_epochs,
        "learning_rate": learning_rate,
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_seconds": [],
        "best_val_accuracy": 0.0,
    }

    best_val_accuracy = 0.0

    print(f"\nTraining {phase_name}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}")

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        elapsed = time.time() - start_time

        history["train_loss"].append(round(train_metrics["loss"], 6))
        history["train_accuracy"].append(round(train_metrics["accuracy"], 6))
        history["val_loss"].append(round(val_metrics["loss"], 6))
        history["val_accuracy"].append(round(val_metrics["accuracy"], 6))
        history["epoch_time_seconds"].append(round(elapsed, 1))

        print(
            f"Epoch {epoch:02d}/{n_epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Lưu Checkpoint xuất sắc nhất
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            history["best_val_accuracy"] = round(best_val_accuracy, 6)

            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model,
                checkpoint_path,
                meta={
                    "phase": phase_name,
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                },
            )

            print(f"  --> Best checkpoint saved! (Val Acc: {best_val_accuracy:.4f})")

    save_json(history, history_path)

    return history