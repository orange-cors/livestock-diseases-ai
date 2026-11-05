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
    checkpoint_path: str | Path,
    history_path: str | Path,
    phase_name: str,
    class_names: list[str] | None = None,
) -> dict:
    """
    Quy trình huấn luyện mô hình theo từng phase (Phase 1: Feature Extraction, Phase 2: Full Fine-tuning).

    Tham số:
    - model: Mô hình ResNet-18 cần huấn luyện.
    - train_loader: DataLoader tập train.
    - val_loader: DataLoader tập validation.
    - n_epochs: Số epoch huấn luyện.
    - learning_rate: Tốc độ học (1e-3 cho Phase 1, 1e-4 cho Phase 2).
    - class_weights: Trọng số phân lớp (Tensor) nhằm giảm ảnh hưởng mất cân bằng dữ liệu.
    - device: Thiết bị tính toán (torch.device('cuda') hoặc torch.device('cpu')).
    - checkpoint_path: Đường dẫn lưu checkpoint xuất sắc nhất (best validation accuracy).
    - history_path: Đường dẫn file JSON lưu lịch sử Loss và Accuracy qua từng epoch.
    - phase_name: Tên giai đoạn ('phase1_frozen_backbone' hoặc 'phase2_full_finetuning').
    - class_names: Danh sách tên các lớp bệnh để lưu kèm vào metadata của checkpoint.
    """
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be at least 1, got {n_epochs}.")

    # 1. Khởi tạo Hàm tổn thất (Loss Function) có tính đến Class Weights
    if class_weights is not None:
        weights = class_weights.to(device=device, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 2. Lọc các tham số được phép huấn luyện (Trainable Parameters)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

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

    best_val_accuracy = float("-inf")

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

        # Lưu Checkpoint xuất sắc nhất theo validation accuracy
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
                    "class_names": class_names or [],
                },
            )

            print(f"  --> Best checkpoint saved! (Val Acc: {best_val_accuracy:.4f})")

    # Lưu file lịch sử học tập
    save_json(history, history_path)

    # Đồng bộ cả 2 định dạng tên file history (<animal>_history_<phase>.json và history_<animal>_<phase>.json)
    history_p = Path(history_path)
    name = history_p.name
    if name.startswith("history_"):
        alt_name = name.replace("history_", "", 1)
        # chèn _history_ vào giữa animal và phase: e.g. cow_phase1.json -> cow_history_phase1.json
        parts = alt_name.split("_")
        if len(parts) >= 2:
            alt_name = f"{parts[0]}_history_{'_'.join(parts[1:])}"
            save_json(history, history_p.parent / alt_name)
    elif "_history_" in name:
        parts = name.split("_history_")
        alt_name = f"history_{parts[0]}_{parts[1]}"
        save_json(history, history_p.parent / alt_name)

    return history
