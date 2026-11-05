import json
from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Cố định random seed cho tất cả các thư viện để đảm bảo khả năng tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(obj: dict, path: str | Path) -> None:
    """Lưu đối tượng dict ra file định dạng JSON với mã hóa UTF-8."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    """Đọc dữ liệu từ file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Danh mục các lớp bệnh chuẩn theo tên thư mục dataset của từng loài
ANIMAL_CLASSES = {
    "cow": [
        "Cow__Foot-and-mouth",
        "Cow__Healthy",
        "Cow__Lumpy",
        "Cow__Mastitis",
    ],
    "chicken": [
        "Chicken__Cocci",
        "Chicken__Healthy",
        "Chicken__Ncd",
        "Chicken__Salmo",
    ],
    "pig": [
        "Pig__Bacterial_Erysipelas",
        "Pig__Bacterial_Greasy",
        "Pig__Environmental_Dermatitis",
        "Pig__Environmental_Sunburn",
        "Pig__Fungal_Pityriasis_Rosea",
        "Pig__Fungal_Ringworm",
        "Pig__Healthy",
        "Pig__Parasitic_Mange",
        "Pig__Viral_Foot_and_Mouth",
        "Pig__Viral_Swinepox",
    ],
}

# Tên hiển thị thân thiện (display name) cho từng loài
ANIMAL_DISPLAY_NAMES = {
    "cow": [
        "Cow: Foot and Mouth Disease (Lở mồm long móng)",
        "Cow: Healthy (Khỏe mạnh)",
        "Cow: Lumpy Skin Disease (Viêm da nổi cục)",
        "Cow: Mastitis (Viêm vú)",
    ],
    "chicken": [
        "Chicken: Coccidiosis (Bệnh cầu trùng)",
        "Chicken: Healthy (Khỏe mạnh)",
        "Chicken: Newcastle Disease (Dịch tả gà / NCD)",
        "Chicken: Salmonella (Thương hàn)",
    ],
    "pig": [
        "Pig: Bacterial Erysipelas (Đóng dấu lợn)",
        "Pig: Bacterial Greasy Pig (Viêm da nhờn rỉ dịch)",
        "Pig: Environmental Dermatitis (Viêm da môi trường)",
        "Pig: Environmental Sunburn (Cháy nắng)",
        "Pig: Fungal Pityriasis Rosea (Vẩy phấn hồng)",
        "Pig: Fungal Ringworm (Nấm da)",
        "Pig: Healthy (Khỏe mạnh)",
        "Pig: Parasitic Mange (Ghẻ ngứa do ký sinh trùng)",
        "Pig: Viral Foot and Mouth (Lở mồm long móng)",
        "Pig: Viral Swinepox (Đậu lợn)",
    ],
}

# Danh sách tên hiển thị gộp chung (để tương thích ngược)
DISPLAY_NAMES = [
    # --- Gia súc (Cow / Bò) ---
    "Cow: Foot and Mouth Disease",
    "Cow: Healthy",
    "Cow: Lumpy Skin Disease",
    "Cow: Mastitis",
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
    "Chicken: Coccidiosis",
    "Chicken: Healthy",
    "Chicken: Newcastle Disease",
    "Chicken: Salmonella",
]


def get_pipeline_output_paths(
    project_root: str | Path,
    animal: str,
    phase: str,
) -> dict[str, Path]:
    """
    Tạo bộ đường dẫn chuẩn hóa cho các file đầu ra của pipeline huấn luyện và đánh giá.
    
    Quy ước chuẩn:
    - Model Checkpoint:  models/resnet18_<animal>_<phase>_best.pth
    - History JSON:      results/<animal>_history_<phase>.json (kèm fallback history_<animal>_<phase>.json)
    - Report CSV/JSON:   results/<animal>_classification_report_<phase>.csv / .json
    - Training curves:   figures/<animal>_training_curves_<phase>.png
    - Confusion Matrix:  figures/<animal>_confusion_matrix_<phase>.png
    - Per-class F1:      figures/<animal>_per_class_f1_<phase>.png
    """
    root = Path(project_root)
    animal = animal.lower().strip()
    phase = phase.lower().strip()

    models_dir = root / "models"
    results_dir = root / "results"
    figures_dir = root / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "model_checkpoint": models_dir / f"resnet18_{animal}_{phase}_best.pth",
        "history_json": results_dir / f"{animal}_history_{phase}.json",
        "history_legacy_json": results_dir / f"history_{animal}_{phase}.json",
        "report_csv": results_dir / f"{animal}_classification_report_{phase}.csv",
        "report_json": results_dir / f"{animal}_classification_report_{phase}.json",
        "training_curves_png": figures_dir / f"{animal}_training_curves_{phase}.png",
        "confusion_matrix_png": figures_dir / f"{animal}_confusion_matrix_{phase}.png",
        "per_class_f1_png": figures_dir / f"{animal}_per_class_f1_{phase}.png",
    }
