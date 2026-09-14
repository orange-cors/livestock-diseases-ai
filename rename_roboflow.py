import uuid
from pathlib import Path

# Trỏ vào thư mục chứa dữ liệu lợn của bạn (Livestock_Skin)
dataset_dir = Path(r"\\?\C:\Users\MangOS\livestock-diseases-ai\data\Dataset_Capstone\Livestock_Skin")
print("Đang quét và rút gọn tên file...")

count = 0
# Quét toàn bộ thư mục (train, val, test và các thư mục bệnh con)
for filepath in dataset_dir.rglob("*.*"):
    if filepath.is_file() and filepath.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        # Chỉ can thiệp đổi tên những file có tên dài hơn 50 ký tự
        if len(filepath.name) > 50:
            # Tạo một mã ngẫu nhiên 8 ký tự để không bị trùng lặp
            short_id = str(uuid.uuid4())[:8]
            new_name = f"img_{short_id}{filepath.suffix}"
            new_path = filepath.parent / new_name
            
            try:
                filepath.rename(new_path)
                count += 1
            except Exception as e:
                print(f"Không thể đổi tên tệp: {filepath.name} - Lỗi: {e}")

print(f"Đã rút gọn thành công {count} ảnh!")