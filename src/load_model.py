from pathlib import Path
import torch
import torch.nn as nn

from src.model import build_model, load_checkpoint


def load_best_model(
    checkpoint_path: str | Path = None,
    n_classes: int = 10,
    device: torch.device | None = None,
) -> nn.Module:
    """Load mô hình ResNet-18 đã huấn luyện xong để chạy dự đoán (Inference)."""
    # 1. Tự động xác định thiết bị tính toán (CUDA / CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Xử lý đường dẫn checkpoint mặc định nếu không truyền vào
    project_root = Path(__file__).resolve().parent.parent
    if checkpoint_path is None:
        checkpoint_path = project_root / "models" / "resnet18_phase2_best.pth"
    else:
        checkpoint_path = Path(checkpoint_path)

    # 3. Kiểm tra file checkpoint có tồn tại không
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số mô hình tại:\n{checkpoint_path}"
        )

    print("=" * 60)
    print("LOADING AI MODEL FOR INFERENCE")
    print("=" * 60)
    print(f"Model path : {checkpoint_path.name}")
    print(f"Classes    : {n_classes}")
    print(f"Device     : {device}")

    # 4. Xây dựng kiến trúc mô hình ResNet-18 (unfreeze toàn bộ để load weights)
    model = build_model(n_classes=n_classes, freeze_backbone=False)

    # 5. Chuyển mô hình sang thiết bị trước khi nạp checkpoint
    model = model.to(device)

    # 6. Nạp checkpoint kèm tham số map_location=device
    meta = load_checkpoint(model, checkpoint_path, device=device)

    # 7. Chuyển mô hình sang chế độ đánh giá (Evaluation mode)
    model.eval()

    val_acc = meta.get("val_accuracy", "N/A")
    print(f"Model loaded successfully! (Saved Val Acc: {val_acc})")
    print("=" * 60)

    return model