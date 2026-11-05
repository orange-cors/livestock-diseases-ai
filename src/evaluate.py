from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import DISPLAY_NAMES, save_json


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Thu thập toàn bộ nhãn thực tế (y_true) và nhãn dự đoán (y_pred) từ DataLoader."""
    model.eval()

    all_labels = []
    all_predictions = []

    for images, labels in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
):
    """Tính toán các chỉ số Precision, Recall, F1-score và Accuracy."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report["overall_accuracy"] = accuracy_score(y_true, y_pred)
    return report


def save_classification_report(
    report: dict,
    json_path: str | Path,
    csv_path: str | Path,
):
    """Lưu báo cáo phân loại ra định dạng file JSON và CSV."""
    save_json(report, json_path)

    report_df = pd.DataFrame(report).transpose()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(csv_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str | Path,
    normalize: bool = True,
    display_names: list | None = None,
):
    """Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix)."""
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

    # Sử dụng tên hiển thị đẹp nếu truyền vào hoặc khớp số lượng
    labels = display_names if display_names and len(display_names) == len(class_names) else (
        DISPLAY_NAMES if len(DISPLAY_NAMES) == len(class_names) else class_names
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm, cmap="Blues")

    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    value_format = ".2f" if normalize else "d"

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            threshold = 0.5 if normalize else cm.max() / 2
            ax.text(
                j,
                i,
                format(val, value_format),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if val > threshold else "black",
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(title)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(
    report: dict,
    class_names: list,
    save_path: str | Path,
    display_names: list | None = None,
):
    """Vẽ biểu đồ F1-score từng lớp bệnh."""
    labels = display_names if display_names and len(display_names) == len(class_names) else (
        DISPLAY_NAMES if len(DISPLAY_NAMES) == len(class_names) else class_names
    )
    f1_scores = [report[class_name]["f1-score"] for class_name in class_names]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(labels, f1_scores, color="skyblue")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Class")
    ax.set_title("Per-Class F1-score")

    for index, value in enumerate(f1_scores):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: dict,
    save_path: str | Path,
    prefix: str = "",
):
    """
    Vẽ và lưu biểu đồ trực quan hóa hàm mất mát (Loss) và độ chính xác (Accuracy) qua từng Epoch.
    
    Tự động chuẩn hóa đường dẫn lưu trữ, tránh hiện tượng nhân đôi tiền tố (prefix).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Đồ thị Training vs Validation Loss
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation Loss", color="#ff7f0e")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Đồ thị Training vs Validation Accuracy
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy", color="#2ca02c")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy", color="#d62728")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo tiền tố nhất quán và không bị trùng lặp
    filename = save_path.name
    if prefix:
        if not prefix.endswith("_"):
            prefix = f"{prefix}_"
        if not filename.startswith(prefix):
            filename = f"{prefix}{filename}"

    final_path = save_path.parent / filename
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list,
    device: torch.device,
    results_dir: str | Path,
    figures_dir: str | Path,
    label: str,
    display_names: list | None = None,
    prefix: str = "",
) -> dict:
    """
    Chạy đánh giá toàn diện mô hình trên tập Test set:
    - Thu thập toàn bộ dự đoán thực tế vs mô hình.
    - Tính toán metrics: Accuracy, Macro F1, Weighted F1, Per-class Precision/Recall.
    - Lưu báo cáo phân loại dưới 2 định dạng: JSON và CSV.
    - Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix).
    - Vẽ và lưu Biểu đồ Per-Class F1-score.
    """
    y_true, y_pred = collect_predictions(model, test_loader, device)

    report = compute_classification_metrics(
        y_true,
        y_pred,
        class_names,
    )

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Chuẩn hóa tiền tố file
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    # Lưu Classification Report (JSON + CSV)
    save_classification_report(
        report,
        json_path=results_dir / f"{prefix}classification_report_{label}.json",
        csv_path=results_dir / f"{prefix}classification_report_{label}.csv",
    )

    # Vẽ và lưu Normalized Confusion Matrix
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path=figures_dir / f"{prefix}confusion_matrix_{label}.png",
        normalize=True,
        display_names=display_names,
    )

    # Vẽ và lưu Per-Class F1-Score
    plot_per_class_f1(
        report,
        class_names,
        save_path=figures_dir / f"{prefix}per_class_f1_{label}.png",
        display_names=display_names,
    )

    print(f"\n--- Kết quả đánh giá ({prefix}{label}) ---")
    print(f"Accuracy   : {report['overall_accuracy']:.4f}")
    print(f"Macro F1   : {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    return report
