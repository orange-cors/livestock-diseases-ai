import io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.utils import DISPLAY_NAMES

# Chuẩn hóa ảnh đầu vào theo ImageNet cho quá trình suy luận (Inference)
INFERENCE_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def prepare_image(image: Image.Image) -> torch.Tensor:
    """Chuyển đổi ảnh PIL thành Tensor (1, 3, 224, 224) đã được chuẩn hóa."""
    image_rgb = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(image_rgb)
    return tensor.unsqueeze(0)


def get_display_image(image: Image.Image) -> np.ndarray:
    """Tái tạo ảnh RGB gốc (dải giá trị [0, 1]) sau bước crop và normalize để làm nền hiển thị Grad-CAM."""
    image_rgb = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(image_rgb)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # De-normalize: nhân lại std và cộng mean
    rgb = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = rgb * std + mean
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def numpy_to_png_bytes(image: np.ndarray) -> bytes:
    """Chuyển đổi mảng numpy float [0, 1] thành chuỗi byte định dạng PNG."""
    image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def predict_with_gradcam(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """
    Thực hiện dự đoán phân loại bệnh và sinh bản đồ nhiệt Grad-CAM giải thích quyết định.
    
    Tham số:
    - model: Mô hình ResNet-18 đã được nạp trọng số tốt nhất.
    - image: Ảnh đầu vào dạng PIL Image.
    - device: Thiết bị tính toán (torch.device('cuda') hoặc 'cpu').
    - class_names: Danh sách tên các nhãn lớp của loài tương ứng.
    
    Trả về:
    - predicted_class: Tên nhãn có xác suất cao nhất.
    - confidence_percentage: Độ tin cậy tính theo %.
    - top5: Danh sách Top 5 chẩn đoán kèm xác suất.
    - original_image: Byte ảnh đã crop.
    - gradcam_image: Byte ảnh đã phủ heatmap Grad-CAM.
    - visualization_np: Mảng numpy hiển thị trực tiếp bằng matplotlib.
    """
    class_names = class_names or DISPLAY_NAMES
    input_tensor = prepare_image(image).to(device)

    model.eval()

    # 1. Dự đoán phân phối xác suất
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top_k = min(5, probabilities.numel())
        top_prob, top_indices = torch.topk(probabilities, k=top_k)

    top_prob = top_prob.detach().cpu().numpy()
    top_indices = top_indices.detach().cpu().numpy()

    predicted_idx = int(top_indices[0])
    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
    confidence = float(top_prob[0])

    top5 = []
    for idx, prob in zip(top_indices, top_prob):
        c_idx = int(idx)
        c_name = class_names[c_idx] if c_idx < len(class_names) else f"Class_{c_idx}"
        top5.append({
            "index": c_idx,
            "class_name": c_name,
            "probability": float(prob),
            "percentage": float(prob * 100),
        })

    # 2. Sinh bản đồ nhiệt Grad-CAM từ tầng tích chập cuối cùng (layer4) của ResNet-18
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # targets=None sẽ tự động lấy lớp có điểm số cao nhất
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    rgb_img = get_display_image(image)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    original_png = numpy_to_png_bytes(rgb_img)
    heatmap_png = numpy_to_png_bytes(visualization / 255.0)

    return {
        "predicted_index": predicted_idx,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_percentage": confidence * 100,
        "top5": top5,
        "original_image": original_png,
        "gradcam_image": heatmap_png,
        "rgb_image_np": rgb_img,
        "gradcam_image_np": visualization,
    }
