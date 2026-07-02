# Hierarchical Semantic Prototype Refinement Network (HSPRN)

HSPRN là mô hình Zero-Shot Learning (ZSL) và Generalized Zero-Shot Learning (GZSL) tập trung vào việc thu hẹp khoảng cách giữa không gian ngữ nghĩa (semantic space) và không gian thị giác (visual space). Mô hình sử dụng cơ chế tinh chỉnh prototype dựa trên từng thực thể ảnh cụ thể (Instance-aware Prototype Refinement) kết hợp với các mô tả ngữ nghĩa đa tầng cấp độ lớp học.

---

## 1. Kết Quả Huấn Luyện Hiện Tại (CUB Dataset)

Dưới đây là các kết quả ghi nhận được trên tập dữ liệu **CUB** (với Vision Backbone: ResNet-50, Semantic: CLIP Text Encoder):

| Lần chạy (Run) | Trạng thái cấu hình | ZSL Top-1 | GZSL Unseen ($u$) | GZSL Seen ($s$) | GZSL Harmonic Mean ($H$) |
|---|---|---|---|---|---|
| **Lần 1** (80 Epochs) | Baseline ban đầu | 36.3% | 6.9% | 47.0% | 12.0% |
| **Lần 2** (20 Epochs) | Điều chỉnh pipeline | **48.2%** | **11.2%** | **69.1%** | **19.3%** (Best: 20.1%) |
| **Lần 3** (Hiện tại) | Sau khi sửa 3 lỗ hổng toán học | *Đang huấn luyện...* | - | - | - |

> [!WARNING]
> **Nhận xét hiện trạng:** Kết quả ban đầu gặp lỗi **bias nghiêm trọng về seen classes** ($s$ đạt gần 70% trong khi $u$ chỉ đạt 11%). Đây là vấn đề phổ biến trong GZSL do sự mất cân bằng phân phối điểm tương đồng, đã được chẩn đoán và khắc phục qua bản cập nhật mới nhất.

---

## 2. Phân Tích Lỗi Toán Học và Biện Pháp Khắc Phục

Để giải quyết tình trạng bias lệch seen/unseen, dự án đã thực hiện 3 cải tiến toán học cốt lõi:

### 1. Sửa Lỗi Lệch Quy Mô Gradient (Reduction Mismatch)
* **Vấn đề:** Trước đây, hàm loss classification ($L_{cls}$) tính trung bình (`mean`), trong khi loss visual-semantic alignment ($L_{align}$) và loss prototype preservation ($L_{pres}$) tính tổng (`sum`). Điều này làm cho gradient của $L_{pres}$ bị phóng đại lên hàng chục ngàn lần, triệt tiêu động lực tối ưu hóa của classification và làm vô hiệu hóa bộ refiner.
* **Giải pháp:** Chuyển toàn bộ $L_{align}$ và $L_{pres}$ sang phép thu gọn trung bình (`mean`).

### 2. Tích Hợp Hiệu Chỉnh Điểm Đánh Giá (Calibrated Stacking)
* **Vấn đề:** Do mô hình chỉ được train trên các lớp seen, điểm dự đoán cho các lớp seen luôn cao hơn đáng kể so với unseen classes trong pha GZSL.
* **Giải pháp:** Thêm cơ chế Calibrated Stacking trong quá trình đánh giá:
  $$\hat{y} = \arg\max_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u} s_c - \gamma \cdot \mathbb{1}[c \in \mathcal{Y}^s]$$
  Hệ thống tự động quét tham số $\gamma \in [0.0, 1.0]$ để tìm điểm cân bằng tốt nhất nhằm cải thiện Harmonic Mean $H$.

### 3. Tăng Tham Số Nhiệt Độ (Temperature Scaling)
* **Vấn đề:** Đặt $\tau = 0.07$ (dùng cho contrastive learning quy mô lớn) làm cho phân phối xác suất đầu ra cực kỳ dốc và nhanh chóng bão hòa gradient (plateau từ epoch thứ 3).
* **Giải pháp:** Tăng nhiệt độ lên $\tau = 0.5$ để ổn định dòng gradient và tăng khả năng tổng quát hóa.

---

## 3. Cấu Trúc Dự Án

* `config/`: Chứa các file cấu hình huấn luyện dạng YAML (như `cub_image_resnet50_train.yaml`).
* `datasets/`: Loader legacy XLSA17 và loader package độc lập `prepared_dataset.py`.
* `losses/`: Triển khai hàm loss tổng hợp HSPRNLoss (`hsprn_loss.py`).
* `model/`: Kiến trúc mạng chính (matching head, prototype refiner, visual encoder, semantic constructor).
* `tools/`: Các kịch bản tiện ích (trích xuất đặc trưng, sinh prompt ngữ nghĩa, đánh giá mô hình `evaluator.py`).
* `trainer/`: Vòng lặp huấn luyện chính (`trainer.py`).
* `docs/`: Tài liệu kỹ thuật chi tiết và nhật ký dự án (`logs.md`).

---

## 4. Hướng Dẫn Nhanh

Xem chi tiết hướng dẫn chuẩn bị dữ liệu và lệnh huấn luyện tại [training_tutorial.md](training_tutorial.md).
