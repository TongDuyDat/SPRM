# KẾ HOẠCH HUẤN LUYỆN PHÂN KỲ 3 PHASE (PHASED TRAINING) — HSPRN-v13

*Ngày lập: 2026-09-02. Tổng số epoch: 50. Mục tiêu: Triệt tiêu sự cộng hưởng phá hoại gradient giữa các module, nâng cao điểm Seen $s$ trong khi bảo tồn $u \ge 50.0\%$ và Calibrated $H \ge 65.0\%$.*

---

## 1. NGUYÊN LÝ VÀ ĐỘNG LỰC KHOA HỌC

Trong các phiên bản trước (v10 $\rightarrow$ v12), mô hình được huấn luyện đồng thời (Joint Training) cả 3 module: $W_{\text{multi}}$ (MFA), IPR (Cross-Attention) và $W_s$ ($I+BA$). Dưới tác động của hàm phân loại, cả 3 module cùng lúc nhận gradient và cộng hưởng bẻ cong không gian biểu diễn về phía tập Seen, gây sụp đổ $u$.

Giải pháp **Huấn luyện phân kỳ (Phased Training)** áp dụng nguyên lý LP-FT (ICLR 2022) và Decoupled Representation Learning: tách biệt rành mạch giữa việc **"Học trích xuất thuộc tính khách quan"**, **"Học thích nghi thị giác đa tầng"**, và **"Hiệu chỉnh tương quan ngữ nghĩa"**. Tại mỗi phase, chỉ có duy nhất một module được mở tham số khả học, các module còn lại bị khóa cứng làm điểm tựa (Anchor).

---

## 2. PHÂN BỔ 3 PHASE TRONG 50 EPOCH

```
 Epoch 1 ─────────────────── Epoch 15 ────────────────────── Epoch 40 ────────────── Epoch 50
    │                          │                               │                         │
    ▼                          ▼                               ▼                         ▼
┌─────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────┐
│ PHASE 1 (Ep 1 -> 15)    │ │ PHASE 2 (Ep 16 -> 40)       │ │ PHASE 3 (Ep 41 -> 50)   │
│ Học Attribute Grounding │ │ Thích nghi Thị giác MFA     │ │ Hiệu chỉnh Ngữ nghĩa Ws │
│ • Mở: IPR (W_qa, W_k)   │ │ • Mở: W_multi               │ │ • Mở: Ws (BA low-rank)  │
│ • Khóa: W_multi=0, Ws=I │ │ • Khóa: IPR (Frozen), Ws=I  │ │ • Khóa: W_multi, IPR    │
│ • Loss: L_attr (Cosine) │ │ • Loss: L_task + 2.0 L_fm   │ │ • Loss: L_rel (RKD 200) │
└─────────────────────────┘ └─────────────────────────────┘ └─────────────────────────┘
```

### Chi tiết từng Phase:

### Phase 1: Học trích xuất thuộc tính khách quan (Epoch 1 $\rightarrow$ 15 — 15 Epochs)
* **Trạng thái tham số:**
  * Khóa cứng: $W_{\text{multi}} \equiv 0$, $W_s \equiv I$ (`requires_grad = False`).
  * Khóa cứng: $\lambda_r = 0.0$ (Prototype hoàn toàn là $p^{\text{sem}}$ đóng băng).
  * Mở huấn luyện: $W_{\text{qa}}, W_k, W_{\text{vca}}$ của IPR.
* **Hàm Loss:** Chỉ sử dụng $\mathcal{L}_{\text{attr}}$ (Cosine Distance 312 thuộc tính, $\lambda_{\text{attr}} = 1.0$). Tắt hoàn toàn mọi hàm phân loại.
* **Mục tiêu:** Ma trận Cross-Attention $M_{\text{attr}}$ học cách định vị chính xác 312 vùng bộ phận cơ thể (mỏ, cánh, chân, màu lông) mà không hề bị nhiễm bất kỳ định kiến phân loại seen nào.

### Phase 2: Thích nghi thị giác đa tầng MFA (Epoch 16 $\rightarrow$ 40 — 25 Epochs)
* **Trạng thái tham số:**
  * Khóa cứng: IPR (đóng băng toàn bộ $W_{\text{qa}}, W_k, W_{\text{vca}}$ vừa học ở Phase 1, $\lambda_r = 0.0$), $W_s \equiv I$.
  * Mở huấn luyện: Duy nhất ma trận $W_{\text{multi}}$ của MFA.
* **Hàm Loss:** $\mathcal{L}_{\text{task}}$ (Soft KD 200 lớp, $\lambda_{\text{task}} = 1.0$) $+ 2.0 \cdot \mathcal{L}_{\text{fm}}$ (Neo góc thị giác) $+ 0.1 \cdot \mathcal{L}_{\text{cls}}$ (Hard CE seen liều nhỏ).
* **Mục tiêu:** $W_{\text{multi}}$ học cách kết hợp đặc trưng từ Block 4, 8, 12 để nâng cao điểm phân biệt $s$, nhưng góc lệch $\cos(\tilde{v}, \text{cls})$ được kiểm soát chặt ở mức $\ge 0.90$.

### Phase 3: Hiệu chỉnh tương quan ngữ nghĩa $W_s$ (Epoch 41 $\rightarrow$ 50 — 10 Epochs)
* **Trạng thái tham số:**
  * Khóa cứng: Toàn bộ Visual Stream (Backbone, MFA $W_{\text{multi}}$) và IPR.
  * Mở huấn luyện: Duy nhất ma trận Low-rank $BA$ của $W_s$.
* **Hàm Loss:** $\mathcal{L}_{\text{rel}}$ (Relational KD 200 lớp, $\lambda_{\text{rel}} = 0.2$) $+ 0.05 \cdot \mathcal{L}_{\text{cls}}$.
* **Mục tiêu:** Tinh chỉnh nhẹ ma trận chiếu ngữ nghĩa để đồng bộ với không gian thị giác đã học mà không làm méo hình học tương đối giữa các lớp.

---

## 3. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG (SUCCESS CRITERIA)

* **GZSL u (Raw):** $\ge 50.0\%$ (Khắc phục hoàn toàn $u$-collapse).
* **GZSL s (Raw):** $\ge 65.0 - 70.0\%$ (Tăng trưởng mạnh mẽ nhờ Phase 2 và Phase 3).
* **Calibrated H:** $\ge 65.0\%$.
* **ZSL T1:** $\ge 73.5\%$.
* **Độ lệch $\gamma$:** $\le 0.04$.

---

## 4. KẾ HOẠCH TRIỂN KHAI MÃ NGUỒN

1. **Cập nhật `trainer/trainer.py`:** Thêm logic điều khiển chuyển Phase tự động theo epoch (`self._apply_phase_schedule(epoch)`):
   * Tự động bật/tắt `requires_grad` cho từng nhóm tham số ($W_{\text{multi}}$, IPR, $W_s$).
   * Tự động điều chỉnh trọng số hàm loss tương ứng với từng Phase.
2. **Tạo config YAML:** `config/cub_clip_v13_phased_50ep.yaml` với tổng số epoch = 50.
3. **Đồng bộ và chạy trên IBM:** Sync code, chạy train 50 epoch, theo dõi qua log và đánh giá các checkpoint.
4. **Báo cáo kết quả:** Tạo báo cáo chi tiết `.scratch/BAO_CAO_V13_PHASED.md`.
