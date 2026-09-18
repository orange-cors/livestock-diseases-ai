from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(n_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Xây dựng mô hình ResNet-18 cho bài toán phân loại bệnh gia súc, gia cầm."""
    # Khởi tạo mô hình ResNet-18 với trọng số pre-trained ImageNet
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Đóng đóng (Freeze) các lớp backbone nếu được yêu cầu (Dùng cho Phase 1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Thay thế lớp Fully Connected (FC) cuối cùng
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes),
    )

    # Khởi tạo lại trọng số cho lớp Linear mới
    nn.init.kaiming_normal_(model.fc[1].weight, nonlinearity="relu")
    if model.fc[1].bias is not None:
        nn.init.constant_(model.fc[1].bias, 0.0)

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Mở khóa (Unfreeze) toàn bộ bộ bộ tham số để huấn luyện Fine-tuning (Dùng cho Phase 2)."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Thống kê số lượng tham số trong mô hình."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict | None = None) -> None:
    """Lưu trọn vẹn state_dict mô hình cùng với thông tin metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device | str = "cpu") -> dict:
    """Nạp trọng số checkpoint đã lưu vào mô hình và trả về thông tin metadata."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("meta", {})