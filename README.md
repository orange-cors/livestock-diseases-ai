# Livestock Diseases AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Model](https://img.shields.io/badge/Model-ResNet18-lightgrey)

Dự án ứng dụng Trí tuệ Nhân tạo (Computer Vision) để nhận diện và phân loại các bệnh trên da ở vật nuôi (Gà, Bò, Lợn) thông qua hình ảnh. Hệ thống sử dụng PyTorch, mô hình ResNet18 với Transfer Learning (học chuyển giao) và được tối ưu hóa qua hai giai đoạn huấn luyện (Phase 1 & Phase 2) để đạt được độ chính xác cao nhất.

##  Luồng chạy của dự án (Execution Flow)

Dự án được cấu trúc theo một pipeline hoàn chỉnh từ xử lý dữ liệu thô đến khi ra được mô hình tốt nhất.

### 1. Chuẩn bị & Tiền xử lý dữ liệu (Data Preparation & Preprocessing)
- **Chuẩn hóa tên tệp:** Dữ liệu thô (từ `Dataset_Capstone/Livestock_Skin`) được xử lý thông qua script `rename_roboflow.py` nhằm rút gọn tên file (sử dụng chuỗi UUID ngẫu nhiên). Bước này rất quan trọng để tránh lỗi giới hạn độ dài đường dẫn của hệ điều hành.
- **Data Loaders:** Quá trình tải dữ liệu (trong `src/dataset.py`) tổ chức tập dữ liệu thành `train`, `val`, và `test` riêng biệt cho từng loại vật nuôi (Gà, Bò, Lợn).
- **Data Augmentation & Class Weights:** 
  - Tập `train` được áp dụng các kĩ thuật tăng cường dữ liệu (Random Resized Crop, Random Horizontal Flip, Random Rotation, Color Jitter) giúp mô hình tránh bị overfitting và có khả năng tổng quát hóa tốt hơn.
  - Tự động tính toán trọng số lớp (Inverse Frequency Class Weights) để giúp hàm Loss khắc phục hiện tượng mất cân bằng dữ liệu (Imbalanced Data) giữa các class bệnh.

### 2. Xây dựng mô hình (Model Architecture)
- Kiến trúc cốt lõi sử dụng là **ResNet18** đã được pre-train các đặc trưng hình ảnh cơ bản trên tập ImageNet.
- Lớp phân loại cuối cùng (Classification Head) được tinh chỉnh lại kích thước đầu ra để tương ứng chính xác với số lượng lớp bệnh của từng loại vật nuôi.

### 3. Quá trình Huấn luyện (Training Strategy)
Quá trình huấn luyện (`src/train.py`) được thiết kế với 2 giai đoạn học chuyển giao chuyên sâu nhằm bảo vệ đặc trưng đã học và làm quen với tập dữ liệu mới:

- **Phase 1 (Frozen Backbone - Khởi động):**
  - Đóng băng (Freeze) toàn bộ các lớp tích chập (Convolutional Layers) của mô hình.
  - Chỉ huấn luyện lớp Classification Head mới với Learning Rate lớn (`0.001`).
  - *Mục tiêu:* Giúp lớp phân loại mới điều chỉnh nhanh với các đặc trưng của dataset bệnh vật nuôi mà không làm hỏng trọng số ImageNet đã nạp.
- **Phase 2 (Full Fine-tuning - Tinh chỉnh):**
  - Mở khóa (Unfreeze) toàn bộ mạng mô hình.
  - Huấn luyện lại toàn bộ mạng lưới với Learning Rate nhỏ hơn nhiều (`0.0001`), kết hợp cùng Cosine Annealing Scheduler.
  - *Mục tiêu:* Tinh chỉnh các đặc trưng hình ảnh ở các tầng sâu để nhận diện chính xác các vết bệnh đặc thù trên da lợn, bò, gà.

### 4. Đánh giá kết quả (Evaluation & Checkpoints)
- Sau mỗi epoch, mô hình được đánh giá trên tập Validation. Trọng số của mô hình có điểm `val_accuracy` tốt nhất sẽ tự động được lưu lại (thư mục `models/`).
- Các chỉ số Loss, Accuracy qua từng vòng lặp được xuất vào các tệp `.json` trong `results/`. Riêng tập Test được đánh giá độ chính xác thực tế, Precision, Recall, và F1-Score (xuất báo cáo `.csv`).

---

##  Kết quả Huấn luyện (Phase 1 & Phase 2 Results)

Số liệu dưới đây được trích xuất chính xác từ các file lịch sử huấn luyện (trong thư mục `results/`) tương ứng với từng loài vật. Sự chênh lệch rõ rệt từ Phase 1 sang Phase 2 chứng minh hiệu quả vượt trội của kỹ thuật **Full Fine-Tuning**.

### 1. Mô hình Bệnh trên Bò (Cow)
*(Dữ liệu từ: `history_cow_phase1.json`, `history_cow_phase2.json`, `classification_report_phase1.csv`, `classification_report_phase2.csv`)*
Nhận diện 4 nhóm lớp: Foot-and-mouth, Healthy, Lumpy, Mastitis.

- **Phase 1 (Frozen Backbone - 10 Epochs):**
  - **Validation Accuracy (Tốt nhất):** 79.04%

- **Phase 2 (Full Fine-Tuning - 15 Epochs):**
  - **Validation Accuracy (Tốt nhất):** 94.76% 

### 2. Mô hình Bệnh trên Gà (Chicken)
*(Dữ liệu từ: `history_chicken_phase1.json`, `history_chicken_phase2.json`)*

- **Phase 1 (Frozen Backbone - 10 Epochs):**
  - **Validation Accuracy (Tốt nhất):** 66.34%
- **Phase 2 (Full Fine-Tuning - 15 Epochs):**
  - **Validation Accuracy (Tốt nhất):** 97.51% 

### 3. Mô hình Bệnh trên Lợn (Pig)
*(Dữ liệu từ: `history_pig_phase1.json`, `history_pig_phase2.json`)*

- **Phase 1 (Frozen Backbone - 10 Epochs):**
  - **Validation Accuracy (Tốt nhất):** 55.81%
- **Phase 2 (Full Fine-Tuning - 15 Epochs):**
  - **Validation Accuracy (Tốt nhất):** 76.10% 

---

##  Cấu trúc Thư mục

```text
livestock-diseases-ai/
├── data/                       # Dữ liệu ảnh chia theo train/val/test
├── models/                     # Chứa các file trọng số (.pth) tốt nhất
│   ├── resnet18_chicken_phase1_best.pth
│   ├── resnet18_chicken_phase2_best.pth
│   ├── ...
├── results/                    # Các file báo cáo kết quả (.json, .csv)
├── src/                        # Chứa logic code chính
│   ├── dataset.py              # Xử lý data loaders, Augmentation, tính class weights
│   ├── model.py                # Định nghĩa kiến trúc và checkpoint
│   ├── train.py                # Logic huấn luyện Phase 1 & 2
│   └── utils.py                
├── rename_roboflow.py          # Script làm sạch, rút gọn tên file đầu vào
└── README.md                   
```