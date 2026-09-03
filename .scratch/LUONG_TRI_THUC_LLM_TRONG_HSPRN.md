# BẢN ĐẶC TẢ KIẾN TRÚC TOÀN DIỆN: HSPRN-v14 SUPER-CLIP
## (Luồng Tích hợp Tri thức Đa tầng LLM & Thuộc tính Sinh học vào Mô hình)

*Tài liệu kỹ thuật giải trình chi tiết về nguồn gốc, quy trình trích xuất và luồng tương tác của tri thức văn bản làm giàu từ LLM (CuPL) trong mô hình HSPRN-v14.*

---

## 1. SƠ ĐỒ LUỒNG TRI THỨC NGỮ NGHĨA TOÀN CẢNH (END-TO-END SEMANTIC PIPELINE)

Sơ đồ dưới đây mô tả chính xác cách **1.813 câu mô tả sâu từ LLM (CuPL) và Cây phân loại sinh học (Taxonomy)** được đưa vào mô hình, tương tác với các module tính toán và tham gia trực tiếp vào việc ra quyết định phân loại:

```
========================================================================================================================================
                                 [ NGUỒN DỮ LIỆU VĂN BẢN ĐƯỢC LLM LÀM GIÀU ]
========================================================================================================================================
  File: prepared_datasets/CUB/metadata/class_descriptions_cub_after_llm.json (200 loài chim)
  Mỗi lớp c gồm 4 trường tri thức:
    1. name_c : Tên định danh loài chim (vd: "Laysan Albatross")
    2. desc_c : ~1.813 câu mô tả hình thái học / tập tính do LLM (GPT-3/CuPL) sinh ra (vd: "A large seabird with white body, dark upperwings...")
    3. hier_c : Phân loại học sinh học Bộ / Họ / Chi (vd: ["Order: Procellariiformes", "Family: Diomedeidae"])
    4. attr_c : 2.000 cụm từ thuộc tính đặc thù được trích lọc
                                                 │
                                                 ▼
========================================================================================================================================
                         [ GIAI ĐOẠN 1: MÃ HÓA NGỮ NGHĨA PHÂN CẤP (PSC ENCODING) ]
========================================================================================================================================
  Áp dụng Prompt Templates chuẩn hóa cho từng câu:
    • Câu LLM desc_c  ──> "a photo of a {name_c}, which {desc_c}."
    • Thuộc tính attr_c ──> "a photo of a {name_c} with {attr_c}."
    • Phân cấp hier_c ──> "a bird of {hier_c}."
                                                 │
                                                 ▼
                                     [ CLIP Text BPE Tokenizer ]
                                      Context Length = 77 Tokens
                                                 │
                                                 ▼
                                  [ FROZEN CLIP ViT-B/16 TEXT ENCODER ]
                                     Transformer Text Tower (Dim = 512)
                                                 │
                                                 ▼
                           [ Itemwise Hierarchical Pooling (PSC Module) ]
                Lấy trung bình vector ngữ nghĩa của tất cả các câu trong lớp c:
                  phi_c = 1/K * sum_{k=1}^K normalize(CLIP_Text_Encoder(prompt_k))
                                                 │
                                                 ▼
                       Ma trận Tri thức Ngữ nghĩa Đóng băng (Frozen Phi Cache)
                                      Phi ∈ R^{200 x 512}
                           (Lưu sẵn tại pre-trained/cub_v8_phi_b16.pt)
                                                 │
                                                 ▼
========================================================================================================================================
                [ GIAI ĐOẠN 2: LUỒNG THAM GIA CỦA TRI THỨC LLM TRONG MÔ HÌNH HSPRN ]
========================================================================================================================================
                                      Phi ∈ R^{200 x 512}
                                                 │
            ┌────────────────────────────────────┼────────────────────────────────────┐
            │                                    │                                    │
            ▼                                    ▼                                    ▼
 [ NHÁNH 1: TẠO PROTOTYPE TĨNH ]     [ NHÁNH 2: TẠO CLASS QUERY IPR ]     [ NHÁNH 3: DÂY NEO ZERO-SHOT ]
      (Semantic Bank p^sem)             (Class Attention Query Q_c)              (Loss Distillation L_zs)
            │                                    │                                    │
     Chiếu qua Ws = I + BA               Chiếu qua W_qc:                      So khớp với CLS ảnh:
     p_c^sem = norm(Ws @ phi_c)          Q_c = W_qc(p_c^sem) [200 x 512]      q_c = softmax(cls @ Phi^T / tau)
            │                                    │                                    │
            ▼                                    ▼                                    ▼
  Prototype tĩnh cố định 200 lớp       Dùng để rà quét 196 patch ảnh:       Teacher phân phối 200 lớp ép
  chứa trọn vẹn tri thức CuPL          M_class = softmax(Q_c @ K^T / sqrt)  mô hình học theo tri thức LLM
            │                                    │                          ngay cả khi chưa thấy ảnh unseen
            │                                    ▼                                    │
            │                         IPR nhận diện vùng cơ thể                       │
            │                                    │                                    │
            └────────────────────────┬───────────┴────────────────────────────────────┘
                                     │
                                     ▼
========================================================================================================================================
                         [ GIAI ĐOẠN 3: SO KHỚP VÀ RA QUYẾT ĐỊNH PHÂN LOẠI ]
========================================================================================================================================
         Vector Thị giác Cường hóa v_enhanced (Chứa CLS + MFA + 312 Attr Pool)
                                     │
                                     ▼
                      [ SEMANTIC MATCHING HEAD (MSH) ]
                  score(x, c) = v_enhanced(x) @ p_c^sem
                                     │
                                     ▼
                     DỰ ĐOÁN LOÀI CHIM CHÍNH XÁC: y_pred
========================================================================================================================================
```

---

## 2. CHI TIẾT 3 VỊ TRÍ TRI THỨC LLM THAM GIA VÀO MÔ HÌNH

### Vị trí 1: Tạo dựng Prototype Ngữ nghĩa chuẩn ($p_c^{\text{sem}}$)
* Vector $\phi_c$ từ ~1.813 câu mô tả LLM đóng vai trò là **tọa độ neo chuẩn** của loài chim $c$ trong không gian 512 chiều.
* Module Primitive Semantic Constructor (PSC) sử dụng ma trận chiếu Low-rank $W_s = I + B \cdot A$ để bảo toàn hình học tương đối giữa các lớp:
  $$p_c^{\text{sem}} = \text{normalize}(W_s \cdot \phi_c) \in \mathbb{R}^{512}$$
* Prototype này đại diện cho toàn bộ hiểu biết về hình thái học của loài chim mà không bị phụ thuộc vào một bức ảnh cụ thể nào.

### Vị trí 2: Điều hướng cơ chế Attention trong IPR ($Q_c = W_{qc}(p_c^{\text{sem}})$)
* Module IPR không đoán mò vị trí đặc trưng của chim mà dùng chính tri thức LLM làm câu truy vấn:
  $$Q_c = W_{qc}(p_c^{\text{sem}}) \in \mathbb{R}^{200 \times 512}$$
* Query $Q_c$ mang câu hỏi: *"Dựa trên mô tả chi tiết của loài $c$, vùng nào trong 196 patch của bức ảnh này chứa đặc điểm nhận dạng của nó?"* $\rightarrow$ Giúp ma trận Attention $M_{\text{class}}$ tập trung đúng vào vùng đầu, thân hay cánh của chim.

### Vị trí 3: Dẫn dắt hàm mất mát Chưng cất Zero-Shot ($\mathcal{L}_{\text{task}} = \mathcal{L}_{\text{zs}}$)
* Trong suốt quá trình huấn luyện, Teacher Zero-Shot liên tục so khớp vector ảnh với ma trận tri thức LLM $\Phi \in \mathbb{R}^{200 \times 512}$:
  $$q_c = \text{softmax}\left( \frac{\text{cls}_{\text{anchor}} \cdot \phi_c}{\tau} \right) \quad \forall c \in \{1, \dots, 200\}$$
* Đây chính là kênh truyền dẫn tri thức giúp các loài chim **Unseen** được bảo vệ trong không gian biểu diễn, giải thích tại sao mô hình đạt điểm Zero-Shot $u = 55.3\%$ (vượt trội so với mức $50.1\%$ của CLIP thông thường).
