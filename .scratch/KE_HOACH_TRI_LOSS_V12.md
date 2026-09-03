# KẾ HOẠCH TÁI CẤU TRÚC HÀM LOSS TINH GỌN (TRI-LOSS SYSTEM) — HSPRN-v12

*Ngày lập: 2026-09-01. Mục tiêu: Rút gọn 8 hàm loss phức tạp thành 3 hàm loss cốt lõi không xung đột, triệt tiêu u-collapse, đưa u ≥ 50% và bảo tồn Calibrated H ≥ 65%.*

---

## 1. BỐI CẢNH VÀ ĐỘNG LỰC KHOA HỌC

Qua chuỗi thực nghiệm v10, E1, E2, E3, E4:
1. **Thành công lớn nhất:** Ràng buộc góc thị giác ($L_{\text{fm}} = 2.0$) và chưng cất quan hệ ($L_{\text{rel}}, L_{\text{zs}}$) đã giúp nâng **Calibrated H lên 65.1% (kỷ lục dự án)** và hồi phục $u$ từ 29.9% lên 39.6% (refined) / 47.0% (semantic).
2. **Khuyết tật cấu trúc còn lại:** Hệ thống loss đang có **8 thành phần chồng chéo**, trong đó $\mathcal{L}_{\text{cls}}$ (Hard Cross-Entropy trên 150 lớp seen) chiếm >80% tổng giá trị loss, liên tục tạo gradient kéo lệch mô hình về phía seen, đối kháng trực tiếp với các hàm neo giữ zero-shot.
3. **Phát hiện về $L_{\text{attn}}$ hiện tại:** Hàm $\text{MSE}(\text{attr\_pred}, \text{norm}(a_y))$ trên vector 312 chiều cho giá trị cực nhỏ ($\approx 0.0004$), khiến gradient thuộc tính bị chìm nghỉm trước Hard CE.

---

## 2. KIẾN TRÚC HÀM LOSS TINH GỌN (TRI-LOSS FORMULATION)

Hệ thống loss mới loại bỏ hoàn toàn $\mathcal{L}_{\text{cls}}$ (Seen-only Hard CE), $\mathcal{L}_{\text{align}}$ (NCE), $\mathcal{L}_{\text{pres}}$, $\mathcal{L}_{\text{sem}}$, $\mathcal{L}_{\text{tan}}$, chỉ giữ lại đúng **3 thành phần nguyên lý**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{visual}} \cdot \mathcal{L}_{\text{visual}} + \lambda_{\text{attr}} \cdot \mathcal{L}_{\text{attr}}$$

### 2.1. Thành phần 1: Phân loại toàn tập nhãn 200 lớp ($\mathcal{L}_{\text{task}} = \mathcal{L}_{\text{zs}}$)
* **Ý nghĩa:** Thay thế hoàn toàn Hard CE trên 150 lớp seen bằng phân phối mềm zero-shot trên toàn bộ 200 lớp (150 seen + 50 unseen).
* **Công thức chuẩn hóa gradient:**
  $$q_c = \text{softmax}\left( \frac{\text{cls}_{\text{anchor}} \cdot \phi_c}{\tau} \right), \quad p_c = \text{softmax}\left( \frac{\text{score}(x, \hat{p}_c)}{\tau} \right) \quad \forall c \in \{1, \dots, 200\}$$
  $$\mathcal{L}_{\text{task}} = D_{\text{KL}}(q \,\|\, p) = \sum_{c=1}^{200} q_c \log \frac{q_c}{p_c}$$
  *(Loại bỏ nhân tử triệt tiêu $\tau^2 = 0.0049$ của KD phụ trợ cũ, đưa độ lớn gradient của $\mathcal{L}_{\text{task}}$ về thang đo chuẩn tương đương Cross-Entropy $\approx 1.0 - 2.0$).*
* **Đặc tính ZSL:** Tuân thủ 100% chuẩn Inductive ZSL — chỉ đưa ảnh seen vào, nhưng phân phối xác suất tính trên toàn bộ 200 lớp văn bản.

### 2.2. Thành phần 2: Neo giữ đặc trưng thị giác MFA ($\mathcal{L}_{\text{visual}} = \mathcal{L}_{\text{fm}}$)
* **Ý nghĩa:** Bảo tồn góc của vector thị giác tổng hợp $\tilde{v}$ so với CLS CLIP đóng băng (WiSE-FT feature space).
* **Công thức:**
  $$\mathcal{L}_{\text{visual}} = 1 - \cos\left( \tilde{v}, \, \text{cls}_{\text{anchor}} \right), \quad \lambda_{\text{visual}} = 2.0$$
  *(Thực nghiệm E4 chứng minh mức $\lambda = 2.0$ giữ $\cos(\tilde{v}, \text{cls}) \approx 0.90$, đạt Calibrated H 65.1%).*

### 2.3. Thành phần 3: Định hướng IPR bằng Cosine Thuộc tính ($\mathcal{L}_{\text{attr}}$)
* **Ý nghĩa:** Sửa lỗi gradient biến mất của MSE cũ ($10^{-4}$). Sử dụng Cosine Distance scale-invariant để bắt ma trận Cross-Attention $M_{\text{attr}}$ soi đúng 312 đặc điểm hình thái học.
* **Công thức:**
  $$\text{attr\_pred}_j = \max_{\text{patches}} M_{\text{attr}}(j) \quad [B, 312]$$
  $$\mathcal{L}_{\text{attr}} = 1 - \cos\left( \text{attr\_pred}, \, a_y \right) = 1 - \frac{\text{attr\_pred} \cdot a_y}{\|\text{attr\_pred}\|_2 \|a_y\|_2}, \quad \lambda_{\text{attr}} = 0.5$$
  *(Giá trị loss nằm trong khoảng $0.2 - 0.5$, tạo gradient đủ mạnh để huấn luyện $W_{\text{qa}}, W_k$).*

---

## 3. CÁC THỰC NGHIỆM ĐỐI CHỨNG DỰ KIẾN (ABLATION MATRIX)

Toàn bộ thí nghiệm chạy 80 Epochs trên CUB (CLIP ViT-B/16, GPU L40S trên IBM):

| Thử nghiệm | Cấu hình hàm Loss | Mục tiêu kiểm chứng |
|---|---|---|
| **Trial 1 (T1 - Pure Tri-Loss)** | $\mathcal{L}_{\text{task}} (\text{KD 200}) + 2.0 \mathcal{L}_{\text{visual}} + 0.5 \mathcal{L}_{\text{attr}} (\text{Cosine})$ | Đánh giá hiệu năng khi loại bỏ 100% Hard CE seen-only. |
| **Trial 2 (T2 - Tri-Loss + Small CE)** | $\mathcal{L}_{\text{task}} + 0.1 \mathcal{L}_{\text{cls}} + 2.0 \mathcal{L}_{\text{visual}} + 0.5 \mathcal{L}_{\text{attr}}$ | Kiểm tra xem thêm một lượng nhỏ Hard CE ($\lambda=0.1$) có giúp tăng $s$ mà không làm hại $u$ hay không. |
| **Trial 3 (T3 - Tri-Loss + Decoupled IPR)** | T1 nhưng IPR đóng băng $\lambda_r = 0$ lúc phân loại (Prototype = $p^{\text{sem}}$) | Đo lường hiệu năng của IPR khi đóng vai trò bộ trích xuất đặc trưng độc lập. |

---

## 4. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG (SUCCESS CRITERIA)

* **Calibrated H:** $\ge 65.0\%$ (Duy trì hoặc vượt kỷ lục E4).
* **GZSL u (Raw):** $\ge 48.0\%$ (Mục tiêu tối thượng: tiệm cận mốc 50% trên toàn bộ đường suy luận).
* **ZSL T1:** $\ge 73.5\%$.
* **Độ lệch thiên vị $\gamma$:** $\le 0.04$.
* **Độ tinh gọn:** Mã nguồn loss giảm từ >400 dòng với 8 tham số $\lambda$ xuống còn 3 thành phần trực quan.

---

## 5. KẾ HOẠCH TRIỂN KHAI MÃ NGUỒN VÀ ARTEFACTS

1. **Cập nhật `losses/hsprn_loss.py`:** Thêm chế độ `loss_mode: "tri_loss"`, hỗ trợ Cosine Attribute Loss và scaled 200-class KL.
2. **Cấu hình YAML:** Tạo `config/cub_clip_v12_t1_triloss.yaml`, `config/cub_clip_v12_t2_hybrid.yaml`.
3. **Đồng bộ và chạy trên IBM:** Sync code, chạy tuần tự các trial, theo dõi và eval.
4. **Báo cáo kết quả:** Cập nhật tài liệu kỹ thuật và báo cáo tổng kết `.scratch/BAO_CAO_V12_TRILOSS.md`.
