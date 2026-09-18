# VetVision AI: Trợ lý AI Chẩn đoán Bệnh Vật Nuôi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Model](https://img.shields.io/badge/Model-ResNet18-lightgrey)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)

Một dự án ứng dụng Trí tuệ Nhân tạo (Computer Vision) và Web API thực tế nhằm chẩn đoán bệnh ở gia súc (Bò, Lợn) thông qua các vết loét ngoài da và gia cầm (Gà) thông qua hình ảnh phân. Dự án được xây dựng bằng PyTorch, áp dụng kỹ thuật Transfer Learning với mạng ResNet18 và triển khai qua FastAPI. Dự án bao quát toàn bộ quy trình học máy (Machine Learning): từ khâu chuẩn bị dữ liệu, cân bằng trọng số phân lớp, huấn luyện, tinh chỉnh mô hình cho đến khi triển khai hệ thống.

## Tổng quan Dự án

| Tiêu chí | Chi tiết |
|---|---|
| **Bài toán** | Phân loại đa lớp (Multi-class) bệnh gia súc và gia cầm |
| **Công nghệ** | PyTorch & FastAPI |
| **Mô hình** | ResNet18 (Sử dụng trọng số tiền huấn luyện ImageNet) |
| **Cấu trúc luồng** | 2 luồng độc lập (Bệnh ngoài da Gia súc & Phân Gia cầm) |

## Tại sao tôi xây dựng dự án này?

Các dịch bệnh ở động vật có thể lây lan rất nhanh và gây thiệt hại kinh tế nghiêm trọng cho các trang trại. Việc chẩn đoán bằng mắt thường đôi khi rất khó khăn và đòi hỏi phải có bác sĩ thú y chuyên môn cao. Công nghệ thị giác máy tính (Computer Vision) có thể hỗ trợ giải quyết vấn đề này bằng cách cung cấp một bước chẩn đoán nhanh chóng, tự động thông qua những bức ảnh chụp vết thương hoặc phân bằng điện thoại di động.

Trong đồ án này, tôi tập trung vào việc xây dựng một luồng (pipeline) hoàn chỉnh từ đầu đến cuối (End-to-end). Bằng cách tách biệt các mô hình dựa trên phương pháp chẩn đoán lâm sàng (Biểu hiện da đối với Gia súc, Phân đối với Gia cầm), hệ thống đảm bảo được tính logic và độ chính xác cao khi áp dụng vào các Ứng dụng Quản lý Trang trại thực tế.

## Kết quả & Đánh giá trực quan

Các mô hình được huấn luyện qua 2 giai đoạn (phase) để tối đa hóa độ chính xác và tránh hiện tượng học vẹt (overfitting). Phase 1 sẽ "đóng băng" (freeze) kiến trúc gốc của ResNet18 để chỉ huấn luyện lớp phân loại cuối, và Phase 2 sẽ "mở khóa" (unfreeze) toàn bộ mạng để tinh chỉnh (fine-tuning) với tốc độ học cực kỳ nhỏ.

*(Lưu ý: Thay thế các con số dưới đây bằng kết quả thực tế của bạn sau khi train xong)*

| Luồng Mô hình | Test Accuracy (Phase 2) | Macro F1-score | Weighted F1-score |
|---|---:|---:|---:|
| **Gia cầm (Phân)** | ~76.19% | ~75.77% | ~76.00% |
| **Gia súc (Ngoài da)** | *[Cập nhật sau]* | *[Cập nhật sau]* | *[Cập nhật sau]* |

Các báo cáo trực quan bao gồm **Đường cong huấn luyện (Training Curves)** và **Ma trận nhầm lẫn (Confusion Matrices)** sẽ được tự động tạo và lưu vào thư mục `figures/` sau khi bạn chạy xong các file Notebook.

## Bộ dữ liệu (Dataset)

Dự án sử dụng 2 bộ dữ liệu chính, được tách biệt rõ ràng để huấn luyện 2 mô hình độc lập.

1. **Livestock_Skin**: Hình ảnh các vết lở loét/bệnh ngoài da của Bò và Lợn (Ví dụ: Bệnh Viêm da nổi cục, Lở mồm long móng).
2. **Poultry_Feces**: Hình ảnh phân gà bệnh (Ví dụ: Bệnh cầu trùng, Salmonella).

Hệ thống sẽ tự động chia ngẫu nhiên dữ liệu thành các tập Train, Validation và Test trong quá trình chạy code. Trọng số phân lớp (Class weights) cũng được tự động tính toán để xử lý vấn đề mất cân bằng dữ liệu (Imbalanced Data).

## Phương pháp thực hiện (Method)

Quy trình thực thi tuân theo các bước sau:
1. Tổ chức dữ liệu ảnh vào 2 thư mục `Livestock_Skin` và `Poultry_Feces`.
2. Khởi chạy `01_livestock_pipeline.ipynb` và `02_poultry_pipeline.ipynb` trên Jupyter/Colab.
3. Hệ thống tự động tính toán class weights.
4. Huấn luyện các mô hình phân loại ResNet18 (Phase 1: Frozen Backbone -> Phase 2: Full Finetuning).
5. Trích xuất các chỉ số đánh giá và biểu đồ.
6. Triển khai hệ thống 2 mô hình thông qua Backend FastAPI và Frontend HTML/JS.

## Cấu trúc thư mục

```text
livestock-diseases-ai/
├── data/                       
│   ├── Livestock_Skin/         # Dữ liệu Bò và Lợn
│   └── Poultry_Feces/          # Dữ liệu Gà
├── docs/                       
├── figures/                    # Biểu đồ tự động tạo
├── frontend/                   
│   ├── index.html
│   └── ...
├── models/                     
│   ├── livestock_model.pth     # Trọng số tốt nhất của mô hình Gia súc
│   └── poultry_model.pth       # Trọng số tốt nhất của mô hình Gia cầm
├── notebooks/                  
│   ├── 01_livestock_pipeline.ipynb
│   └── 02_poultry_pipeline.ipynb
├── results/                    # Báo cáo đánh giá dạng JSON
├── src/                        
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── app.py                      # FastAPI server (Sẽ phát triển)
├── requirements.txt            
└── README.md
```

## Hướng phát triển tương lai (Next steps)

- Hoàn thiện Backend `app.py` (FastAPI) để có thể phục vụ cả 2 mô hình Gia súc và Gia cầm cùng lúc.
- Mở rộng hệ thống để tích hợp tính năng giám sát trang trại theo thời gian thực (Real-time monitoring) bằng mô hình nhận diện vật thể YOLO.
- Phát triển giao diện Mobile App giúp nông dân dễ dàng chụp ảnh ngay tại chuồng trại và nhận kết quả tức thì.