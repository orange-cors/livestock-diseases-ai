from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)), 
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
    root = Path(root)

    # Đọc trực tiếp từ các folder đã được chia sẵn bằng ImageFolder
    train_dataset = ImageFolder(root / "train", transform=get_transforms("train"))
    val_dataset = ImageFolder(root / "val", transform=get_transforms("val"))
    test_dataset = ImageFolder(root / "test", transform=get_transforms("test"))

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    # Đếm số lượng ảnh mỗi class trong tập train để tính class_weights (Xử lý mất cân bằng)
    class_counts = {class_name: 0 for class_name in class_names}
    for _, label in train_dataset.samples:
        class_name = class_names[label]
        class_counts[class_name] += 1

    counts = np.array([class_counts[c] for c in class_names], dtype=float)
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
        },
    }

    return train_dataset, val_dataset, test_dataset, info

def make_loaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=32, num_workers=0, pin_memory=False, persistent_workers=False
):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader