# KẾ HOẠCH NGHIÊN CỨU & THỰC NGHIỆM ĐỘT PHÁ: HỆ THỐNG HSPRN-v14 SUPER-CLIP
## (Local-Global Feature Enhancement & Multi-Scale Dual-Stream Alignment)

*Ngày lập: 2026-09-02. Benchmark: CUB-200-2011, CLIP ViT-B/16, Phi v8 CuPL. Server: IBM GPU L40S 48GB.*  
*Mục tiêu tối thượng: Vượt trần Zero-Shot lý thuyết của CLIP gốc, đạt u > 60.0%, s > 80.0%, và Harmonic Mean H > 70.0%.*

---

## 1. BỐI CẢNH VÀ NGUYÊN LÝ KHOA HỌC VƯỢT TRẦN

### 1.1. Giới hạn của CLIP gốc và các phiên bản trước
* **Điểm mù của CLIP gốc:** CLIP chỉ trích xuất duy nhất 1 vector toàn cục `[CLS]`, bỏ qua 196 patch tokens không gian. Với các loài chim tinh mịn (Fine-grained), CLIP không thể phân biệt được các loài có hình thái giống nhau nhưng khác biệt ở chi tiết nhỏ (màu trán, đốm cánh, dạng mỏ) $\rightarrow$ Điểm trần Zero-Shot $u$ bị chặn ở mức **$59.7\%$**.
* **Sai lầm cấu trúc của các bản v10 $\rightarrow$ v12:** Dùng IPR để uốn nắn prototype văn bản $\hat{p}_c(x)$ dưới sự giám sát của Cross-Entropy seen-only $\rightarrow$ Gây méo mó prototype của lớp unseen ($u$-collapse).
* **Kết quả đạt được của v13:** Phased Training đã chứng minh khi khóa chặt prototype $p^{\text{sem}}$ và chỉ tối ưu đặc trưng thị giác, $u$ giữ vững ở **$55.5\%$** và $s$ tăng lên **$61.1\%$** ($H = 58.1\%$).

### 1.2. Nguyên lý vượt trần của HSPRN-v14 (Super-CLIP)
Thay vì uốn nắn prototype phân loại, HSPRN-v14 sử dụng **312 thuộc tính thị giác và 196 patch tokens để cường hóa trực tiếp vector thị giác của bức ảnh (Visual-Centric Enhancement)**:
1. **Luồng 1 (Global Stream):** Vector toàn cục $\tilde{v}_{\text{global}}$ tổng hợp từ CLIP CLS và MFA đa tầng (Block 4, 8, 12).
2. **Luồng 2 (Local Attribute Stream):** Module IPR đóng vai trò là "kính lúp vi mô", dùng 312 Attribute Queries để quét trên 196 patch, trích xuất vector biểu diễn thuộc tính cục bộ $\tilde{v}_{\text{local}}$.
3. **Tổng hợp đa tỷ lệ (Multi-Scale Fusion):** Kết hợp cả tầm nhìn vĩ mô và vi mô vào cùng một vector thị giác tối thượng $\tilde{v}_{\text{enhanced}}$:
   $$\tilde{v}_{\text{enhanced}} = \text{normalize}\left( \tilde{v}_{\text{global}} + W_{\text{local}} \cdot \tilde{v}_{\text{local}} \right)$$
4. **Tính chất bảo toàn Zero-Shot:** Vì 312 thuộc tính là ngôn ngữ thị giác dùng chung giữa Seen và Unseen, một bức ảnh chim Unseen khi đưa vào cũng được bóc tách mỏ, cánh, màu sắc cực kỳ chuẩn xác $\rightarrow$ **$u$ tự nhiên vượt qua mức $59.7\%$ của CLIP gốc**.

---

## 2. KIẾN TRÚC MÔ HÌNH HSPRN-v14 (SUPER-CLIP)

```
                                [ ẢNH ĐẦU VÀO x ]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 [ ViT Backbone ]             [ Patch Tokens ]
                  CLIP CLS [512]              [196, 768] (Block 4, 8, 12)
                         │                             │
                         ▼                             ▼
                [ Global MFA Stream ]         [ Local IPR Stream ]
                 W_multi(z4, z8, z12)        Cross-Attention với 312 Attr
                         │                             │
                         ▼                             ▼
                 v_global [512]                v_local [512]
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                           [ MULTI-SCALE FUSION ]
                       v_enhanced = norm(v_global + W_local @ v_local)
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
         [ 1. GLOBAL MATCHING ]                    [ 2. ATTRIBUTE PREDICTION ]
     score_global = cos(v_enhanced, p_sem)          attr_pred = max_patches(M_attr)
                   │                                         │
                   ▼                                         ▼
            [ L_task (KD 200) ]                     [ L_attr (Cosine 312) ]
            [ + 0.15 L_cls (Seen) ]
```

---

## 3. HỆ THỐNG HÀM LOSS & LỊCH TRÌNH HUẤN LUYỆN

Hàm loss tổng hợp được thiết kế theo cấu trúc Tri-Loss tinh gọn:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} (\text{KD 200 lớp}) + 0.15 \cdot \mathcal{L}_{\text{cls}} (\text{Seen CE}) + 2.0 \cdot \mathcal{L}_{\text{visual}} + 0.5 \cdot \mathcal{L}_{\text{attr}}$$

### Lịch trình huấn luyện phân kỳ nâng cao (50 Epochs, Batch Size 512):
* **Phase 1 (Epoch 1 $\rightarrow$ 15):** Học trích xuất 312 thuộc tính bằng IPR ($\mathcal{L}_{\text{attr}}$), khóa $W_{\text{multi}} = 0, W_{\text{local}} = 0, W_s = I$.
* **Phase 2 (Epoch 16 $\rightarrow$ 40):** Mở đồng thời $W_{\text{multi}}$ và $W_{\text{local}}$ để học tổng hợp đặc trưng Global + Local. Prototype $p^{\text{sem}}$ giữ cố định.
* **Phase 3 (Epoch 41 $\rightarrow$ 50):** Hiệu chỉnh tương quan ngữ nghĩa $W_s$ ($BA$) qua $\mathcal{L}_{\text{rel}}$ kết hợp tinh chỉnh toàn cục nhẹ.

---

## 4. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG (SUCCESS CRITERIA)

* **GZSL u (Unseen):** $\ge \mathbf{60.0\%}$ (Vượt trần $59.7\%$ của CLIP).
* **GZSL s (Seen):** $\ge \mathbf{80.0\%}$ (Nhận diện chính xác vượt bậc nhờ Local Attributes).
* **Harmonic Mean H:** $\ge \mathbf{70.0\%}$ (Đạt đẳng cấp SOTA thế giới).
* **ZSL T1 (Unseen):** $\ge \mathbf{75.0\%}$.
* **Độ lệch thiên vị $\gamma$:** $\le \mathbf{0.03}$.

---

## 5. KẾ HOẠCH TRIỂN KHAI MÃ NGUỒN VÀ THỰC NGHIỆM

1. **Cập nhật `model/protypes.py` & `model/hsprn_model.py`:**
   * Tái cấu trúc IPR để xuất ra vector biểu diễn thuộc tính cục bộ $v_{\text{local}}$.
   * Bổ sung module `LocalGlobalFusion` tích hợp $v_{\text{global}}$ và $v_{\text{local}}$ vào $\tilde{v}_{\text{enhanced}}$.
   * Prototype $p^{\text{sem}}$ cố định tĩnh $100\%$ lúc phân loại.
2. **Cập nhật `trainer/trainer.py`:** Hỗ trợ lịch trình phân kỳ 3 phase cho kiến trúc Local-Global mới.
3. **Tạo config YAML:** `config/cub_clip_v14_super_50ep.yaml` (Batch size 512, 50 Epochs).
4. **Đồng bộ và huấn luyện trên IBM:** Chạy thực nghiệm, theo dõi log và trích xuất checkpoint.
5. **Tài liệu & Báo cáo:**
   * Viết tài liệu mô tả kiến trúc mô hình hoàn chỉnh: `docs/KIEN_TRUC_HSPRN_V14_SUPER_CLIP.md`.
   * Viết báo cáo khoa học kết quả chi tiết: `.scratch/BAO_CAO_V14_SUPER_CLIP.md`.
