from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def get_transforms(split: str) -> transforms.Compose:
    """Trả về các phép biến đổi ảnh (Augmentation / Normalization) theo từng split."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_dataset(root: str | Path):
    """Nạp dataset từ một thư mục chứa đủ 3 thư mục con: train, val, test."""
    root = Path(root)

    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    missing_dirs = [path for path in (train_dir, val_dir, test_dir) if not path.is_dir()]
    if missing_dirs:
        missing = ", ".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Missing dataset split directories: {missing}")

    train_dataset = ImageFolder(train_dir, transform=get_transforms("train"))
    val_dataset = ImageFolder(val_dir, transform=get_transforms("val"))
    test_dataset = ImageFolder(test_dir, transform=get_transforms("test"))

    if not (train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx):
        raise ValueError(
            "Class folders must match across train, val, and test splits. "
            f"Got train={train_dataset.classes}, val={val_dataset.classes}, "
            f"test={test_dataset.classes}."
        )

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    # Đếm số lượng ảnh từng lớp trong tập train
    class_counts = {class_name: 0 for class_name in class_names}
    for _, label in train_dataset.samples:
        class_name = class_names[label]
        class_counts[class_name] += 1

    counts = np.array([class_counts[c] for c in class_names], dtype=float)
    counts = np.maximum(counts, 1.0)  # Tránh lỗi chia cho 0
    
    # Tính trọng số nghịch đảo tần suất (Inverse Frequency Class Weights)
    class_weights = counts.sum() / (len(counts) * counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    info = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "class_counts": class_counts,
        "class_weights": class_weights_tensor,
        "n_classes": len(class_names),
        "split_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
            "total": len(train_dataset) + len(val_dataset) + len(test_dataset),
        },
    }

    return train_dataset, val_dataset, test_dataset, info


def make_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    """Khởi tạo DataLoader cho các tập train, val, test."""
    is_multiprocess = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if is_multiprocess else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if is_multiprocess else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if is_multiprocess else False,
    )

    return train_loader, val_loader, test_loader
