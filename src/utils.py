import json
from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Danh sách tên hiển thị chuẩn hóa cho các lớp bệnh Gia súc & Gia cầm
DISPLAY_NAMES = [
    # --- Gia súc (Cow / Bò) ---
    "Cow: Ectoparasite Infection",
    "Cow: Foot and Mouth Disease",
    "Cow: Healthy",
    "Cow: Lumpy Skin Disease",
    "Cow: Mange",
    "Cow: Mastitis",
    "Cow: Pink Eye",
    "Cow: Ringworm",
    "Cow: Wound Infection",
    # --- Gia súc (Pig / Lợn) ---
    "Pig: Bacterial Erysipelas",
    "Pig: Bacterial Greasy Skin",
    "Pig: Environmental Dermatitis 1",
    "Pig: Environmental Dermatitis 2",
    "Pig: Fungal Pityriasis Rosea",
    "Pig: Fungal Ringworm",
    "Pig: Healthy",
    "Pig: Parasitic Mange",
    "Pig: Viral Foot and Mouth",
    "Pig: Viral Swinepox",
    # --- Gia cầm (Poultry / Gà) ---
    "Poultry: Coccidiosis",
    "Poultry: Healthy",
    "Poultry: Newcastle Disease",
    "Poultry: Salmonella",
]