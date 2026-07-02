# Hướng Dẫn Huấn Luyện (Training Tutorial) — HSPRN

Tài liệu này hướng dẫn chi tiết quy trình chuẩn bị môi trường, tiền xử lý dữ liệu, và thực hiện huấn luyện mô hình HSPRN (Hierarchical Semantic Prototype Refinement Network) trên các bộ dữ liệu Zero-Shot Learning (ZSL/GZSL) chuẩn như CUB, AWA2, SUN, và APY.

---

## 1. Chuẩn Bị Môi Trường

Dự án sử dụng `micromamba` (hoặc `conda`) để quản lý các thư viện Python. Môi trường đề xuất là `py310` sử dụng Python 3.10 và PyTorch có hỗ trợ CUDA.

### Kích hoạt môi trường:
```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate py310
```
*(Nếu sử dụng Anaconda/Miniconda, thay thế `micromamba` bằng `conda`).*

---

## 2. Cấu Trúc Dữ Liệu Training

Pipeline image hiện dùng `PreparedZSLDataset` trong `datasets/prepared_dataset.py`. Mỗi package là nguồn dữ liệu độc lập, không cần truy cập lại `data/`, `datasets/` hoặc `data_semantics/`.

### Tổ chức thư mục chuẩn:
```text
HSPRN/
└── prepared_datasets/
    ├── manifest.yaml
    ├── CUB/
    │   ├── manifest.yaml
    │   ├── metadata/
    │   │   ├── attributes.npy
    │   │   ├── classes.json
    │   │   ├── class_semantics.json
    │   │   ├── images.jsonl
    │   │   └── splits.json
    │   └── raw/CUB_200_2011/images/...
    ├── AWA2/...
    ├── SUN/...
    └── APY/...
```

Config image đặt `data.source: prepared` và `data.prepared_root: ./prepared_datasets`. Loader kiểm tra manifest, split order, class count, đủ bốn trường semantic và đường dẫn ảnh tương đối trong package.

---

## 3. Sinh Dữ Liệu Ngữ Nghĩa (Class Descriptions)

Mô hình HSPRN yêu cầu 4 trường ngữ nghĩa cho mỗi lớp: `name_c` (tên lớp), `attr_c` (thuộc tính), `hier_c` (quan hệ phân cấp), và `desc_c` (mô tả ngữ cảnh). Chúng ta sinh offline thông qua kịch bản gọi LLM API:

```bash
python tools/generate_descriptions.py \
  --dataset CUB \
  --classes_file data/CUB/classes.txt \
  --output data_semantics/CUB/class_descriptions_cub.json
```

*Lưu ý:* Công cụ có cơ chế resume-safe, tự động bỏ qua các lớp đã được sinh hoàn chỉnh nếu tiến trình bị gián đoạn.

---

## 4. Trích Xuất Semantic Embeddings Ngoại Tuyến (Tùy Chọn)

Pipeline mặc định mã hóa raw class descriptions bằng text encoder trong model, nên không cần bước này. Chỉ dùng lệnh dưới đây cho thí nghiệm cached embedding riêng:

```bash
python tools/extract_semantic_embeddings.py \
  --metadata data_semantics/CUB/class_descriptions_cub.json \
  --encoder clip \
  --output data_semantics/CUB/clip_text_embeddings.pt
```

---

## 5. Đóng Gói và Kiểm Tra Dữ Liệu

Công cụ `prepare_zsl_dataset_package.py` tạo package tự chứa dùng trực tiếp cho training:

```bash
python tools/prepare_zsl_dataset_package.py \
  --datasets CUB \
  --data-root ./data \
  --image-root ./datasets \
  --action copy \
  --require-complete-images \
  --require-full-semantics
```

Sau khi package được tạo và kiểm tra, có thể lưu trữ/công bố riêng `prepared_datasets`. Các thư mục nguồn chỉ cần khi muốn dựng lại package.

---

## 6. Huấn Luyện Mô Kinh

Cấu hình huấn luyện được khai báo trong file cấu hình YAML. Chạy lệnh để bắt đầu huấn luyện:

### Chế độ 1: Huấn luyện ảnh gốc (Image Mode - Mặc định & Đề xuất)
Mô hình sẽ nhận trực tiếp đường dẫn ảnh gốc, nạp qua visual backbone (như ResNet-50 hoặc ViT) để học các tầng đặc trưng trực tiếp.

```bash
PYTHONNOUSERSITE=1 python train.py \
  --config config/cub_image_resnet50_train.yaml \
  --device cuda
```

Có thể đổi vị trí package mà không sửa config:

```bash
PYTHONNOUSERSITE=1 python train.py \
  --config config/cub_image_resnet50_train.yaml \
  --prepared_root /path/to/prepared_datasets \
  --device cuda
```

### Chế độ 2: Huấn luyện đặc trưng có sẵn (Legacy Feature Mode)
Nếu muốn huấn luyện nhanh bằng các đặc trưng 2048-dim trích xuất sẵn từ `res101.mat` (giảm tải tài nguyên GPU và thời gian chạy):
1. Đặt `data.source: "legacy"` và `data.mode: "feature"`.
2. Chạy lệnh:
```bash
python train.py --config config/cub_feature_train.yaml --device cuda
```

---

## 7. Đánh Giá Mô Hình với Calibrated Stacking

Khi huấn luyện trên Seen classes, mô hình bị bias lệch điểm cosine về phía Seen classes trong pha generalized ZSL (GZSL). Kỹ thuật **Calibrated Stacking** (Chao et al., TPAMI 2016) được áp dụng tại inference để trừ đi một lượng bias $\gamma$ từ điểm cosine của Seen classes:

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u} s_c - \gamma \cdot \mathbb{1}[c \in \mathcal{Y}^s]$$

Hệ thống huấn luyện tự động quét giá trị $\gamma \in [0.0, 1.0]$ để tìm chỉ số Harmonic Mean ($H$) tối ưu nhất sau mỗi epoch đánh giá:
- Chỉ số log hiển thị cả chuẩn thông thường và calibrated: `Cal u: XX% s: XX% H: XX% γ: X.XX`
- Model checkpoint tốt nhất (`best.pth`) sẽ lưu lại dựa trên chỉ số **Calibrated H** cao nhất.

---

## 8. Tìm Bộ Lambda Trên Subset 200 Ảnh

Loss huấn luyện hiện dùng ba thành phần:

$$L = \lambda_1 L_{cls} + \lambda_2 L_{align} + \lambda_3 L_{pres}$$

Trong đó `lambda1`, `lambda2`, `lambda3` lần lượt ứng với `L_cls`, `L_align`, `L_pres`. Tool `tools/build_lambda_search_subset.py` đọc trực tiếp bốn package AWA2, CUB, SUN và APY trong `prepared_datasets`; AWA1 chưa được dùng vì chưa có package ảnh hoàn chỉnh.

### Kiểm tra subset trước khi search

```bash
PYTHONNOUSERSITE=1 /opt/conda/micromamba/envs/py310/bin/python \
  tools/build_lambda_search_subset.py \
  --prepared-root ./prepared_datasets \
  --dry-run \
  --subset-file lambda_search_subset.yaml \
  --output lambda_search_result.yaml
```

Mặc định tool lấy 200 ảnh, chia đều 50 ảnh cho mỗi dataset và chia tiếp thành `30 trainval + 10 test_seen + 10 test_unseen`. Ảnh được mở thật để phát hiện file hỏng; semantic bắt buộc đủ `name_c/attr_c/hier_c/desc_c` và không fallback.

### Chạy random search đề xuất

```bash
PYTHONNOUSERSITE=1 /opt/conda/micromamba/envs/py310/bin/python \
  tools/build_lambda_search_subset.py \
  --prepared-root ./prepared_datasets \
  --semantic-backbone tinyclip \
  --semantic-pretrained-weights ./pre-trained/text/tinyclip-text-3m \
  --search random \
  --trials 12 \
  --epochs 1 \
  --device cuda \
  --subset-file lambda_search_subset.yaml \
  --output lambda_search_result.yaml
```

Grid mặc định gồm:

- `lambda1`: `0.5, 0.7, 1.0`;
- `lambda2`: `0.05, 0.1, 0.2`;
- `lambda3`: `0.05, 0.1, 0.2`.

Random search lấy 12 trong 27 tổ hợp, không lặp. Dùng `--search grid` để chạy đủ 27 tổ hợp hoặc truyền grid riêng qua `--lambda1-values`, `--lambda2-values`, `--lambda3-values`. Mọi candidate dùng cùng ảnh, seed và thứ tự DataLoader để so sánh công bằng.

Metric mặc định là trung bình của `ZSL T1` và calibrated `GZSL H`. Có thể chọn riêng bằng `--metric zsl_T1`, `--metric gzsl_H` hoặc `--metric cal_H`.

### File đầu ra

- `lambda_search_subset.yaml`: seed và index ảnh cố định của từng split/dataset.
- `lambda_search_result.yaml`: score từng candidate trên từng dataset, candidate tốt nhất, block `model.loss` và danh sách config image train có thể nhận bộ lambda.
- `lambda_search_runs/`: log và checkpoint tạm của từng candidate.

Tool không tự sửa config. Chỉ áp block `model.loss` vào các config sau khi search chính thức hoàn tất và kết quả đã được review.

> [!WARNING]
> Search hiện dùng mẫu từ official `test_seen/test_unseen`, vì vậy score chỉ có ý nghĩa tuning diagnostic và không phải final test metric độc lập để công bố. Quy trình báo cáo cần một validation protocol riêng, chốt lambda trên validation, sau đó chỉ đánh giá official test một lần.

Subset 200 ảnh giảm chi phí visual nhưng semantic constructor vẫn phải xử lý toàn bộ class; SUN với 645 seen và 72 unseen class thường là bộ chạy chậm nhất. Bộ lambda tìm bằng template ResNet-50 cũng cần được kiểm tra lại trước khi áp kết luận cho các họ backbone ViT hoặc CLIP-ViT.
