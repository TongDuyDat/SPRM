# Plan 1 — Selection Adapter (CLIP query → top-k descriptions → HSPRN)

**Trạng thái:** kế hoạch triển khai để review. **Không** implement code trong bước này.  
**Nhánh git đề xuất (sau khi review):** `feat/plan-1-selection-adapter`  
**Tham chiếu thiết kế:** `docs/research.md` §1–§9 (đặc biệt §3.2–§3.4 Biến thể II/III, §4.1 + §4.5, §5).

---

## 1. Mục tiêu / vấn đề giải quyết

### 1.1. Vấn đề

Pipeline HSPRN hiện train với semantic giàu (LLM lists `name_c` / `attr_c` / `hier_c` / `desc_c`, thường bị **join** thành một prompt). Khi eval “lean” (chỉ attribute matrix / name) hoặc khi **toàn bộ list** được encode bất kể ảnh, xuất hiện:

1. **Train–eval channel mismatch** trên nhánh semantic (research §1.2).
2. **Instance-blind noise:** mọi ảnh cùng class dùng cùng dump text, kể cả item không khớp ảnh.

### 1.2. Mục tiêu plan này

Triển khai **Selection Adapter** $A_\theta$:

- Query = embedding ảnh CLIP $\hat{v}_x = \phi_{\mathrm{img}}(x)$ (freeze dual encoder khớp text bank).
- Bank = các câu mô tả LLM **item-level** đã cache embedding offline.
- Chọn top-$k$ (per-field hoặc global trong class) → pool / dựng semantic → đưa vào HSPRN **cùng** tín hiệu gốc (dual-channel, Biến thể III).
- **Cùng cơ chế** ở train và eval (research Eq. 15).

### 1.3. Phạm vi MVP (phase 1)

| Trong scope | Ngoài scope (phase sau) |
| --- | --- |
| Offline item bank + CLIP text cache | Gumbel-Top-k, cross-attn (§4.4, §4.6) |
| Pure CLIP cosine top-$k$, $\theta=\emptyset$ (§4.1) | Learnable $a_\theta,b_\theta$ (§4.2) trừ khi cosine yếu |
| Per-field structured select (§4.5) | FAISS / ANN scale lớn |
| Path A: lọc list → constructor; hoặc Path B: pool embedding → $W_s$ | Đổi loss HSPRN cốt lõi |
| Biến thể II (instance-aware) + optional Biến thể III fusion | Online LLM |

---

## 2. Thiết kế kỹ thuật chi tiết

### 2.1. Notation (rút từ research)

- Bank: $\mathcal{B} = \{p_j\}$, $e_j = \phi_{\mathrm{txt}}(p_j)$, metadata $\mathrm{cls}(j)$, $\mathrm{type}(j)\in\{\mathrm{name},\mathrm{attr},\mathrm{hier},\mathrm{desc}\}$.
- Score: $s_j(x) = \langle \hat{v}_x, \hat{e}_j \rangle / \tau$ (Eq. 1).
- Class-conditional: $S_k(x,c) = \mathrm{TopK}_{j:\mathrm{cls}(j)=c}\, s_j(x)$ (Eq. 9).
- Per-field: top-$k_f$ trong từng field; luôn giữ `name_c` (Eq. 16).
- Pool: $\phi_{x,c} = \frac{1}{k}\sum_{j\in S} e_j$ (mean) rồi $\tilde{p}^{\mathrm{sem}}_{x,c} = \mathrm{Normalize}(W_s \phi_{x,c})$ (Eq. 10).
- Dual-channel (khuyến nghị):  
  $p^{\mathrm{fuse}}_{x,c} = \mathrm{Normalize}(\beta\, p^{\mathrm{orig}}_c + (1-\beta)\, p^{\mathrm{sel}}_{x,c})$ (Eq. 12).

### 2.2. Kiến trúc module

```text
x --> [HSPRN VisionModel + MFA] --> v_tilde(x) --> IPR --> Matching
                |
                +--> [CLIP image encoder freeze] --> v_x
                            |
                            v
                 SelectionAdapter(v_x, bank, k, fields)
                            |
              +-------------+-------------+
              |                           |
         selected items              p_orig (optional:
         --> pool / W_s_sel           full class prompt or attribute)
              |                           |
              +-----> fuse beta ----------+
                            |
                            v
                      p_sem for IPR  (shape [B, C, D] if instance-aware)
```

### 2.3. Lựa chọn kỹ thuật cố định cho MVP

1. **Query CLIP phụ:** luôn load CLIP image encoder khớp `model.semantic.clip_model` / TinyCLIP, freeze — kể cả khi vision HSPRN là ResNet (giả định A1).
2. **Selection scope:** class-restricted trong candidate set $\mathcal{C}$ (ZSL: unseen; GZSL: all) khi matching; train trên seen classes.
3. **Per-field $k$:** config `selection.k_attr`, `k_hier`, `k_desc` (default 3); `name` always-on.
4. **Aggregate:** mean-pool **per-item embeddings** (không join text rồi CLIP lại — tránh 77-token join, research §7).
5. **Train = eval path:** bật adapter cả hai; ablate `selection.enabled=false` = baseline join hiện tại.

### 2.4. Config YAML (đề xuất schema)

```yaml
model:
  selection:
    enabled: true
    mode: cosine_topk          # cosine_topk | soft_attn (phase 2)
    k_attr: 3
    k_hier: 3
    k_desc: 3
    temperature: 0.07
    fusion_beta: 0.5           # 1.0 = only selected; 0.0 = only orig
    clip_image_for_query: true
    bank_dir: null             # path to item bank artifacts
data:
  item_bank_path: null         # e.g. prepared/.../metadata/item_bank.pt
```

### 2.5. Artifact offline

Mỗi package / dataset:

- `item_texts.jsonl` hoặc index trong `item_index.json`: `{j, class_id, field, text}`
- `item_embeddings.pt`: float tensor `[M, d]`, L2-normalized optional
- `item_meta.pt`: `class_ids [M]`, `field_ids [M]`

Sinh từ `class_semantics.json` / LLM JSON (list per field), encode bằng cùng CLIP text như train.

---

## 3. Các bước triển khai (thứ tự)

### Bước 0 — Chuẩn bị nhánh và baseline

1. Tạo nhánh `feat/plan-1-selection-adapter` từ `main` (hoặc commit review).
2. Ghi baseline metric hiện tại (CUB/AWA2 ZSL + GZSL H) với `semantic_source=class_desc` join — để so sánh sau.
3. Xác nhận `tools/generate_descriptions.py` đã prompt item ≤ ~50 English words (77 tokens/item).

### Bước 1 — Tool offline: build item bank

1. Thêm `tools/build_item_bank.py` (tên có thể điều chỉnh):
   - Input: class semantics JSON (list `attr_c`/`hier_c`/`desc_c` + `name_c`).
   - Flatten từng string item; bỏ item rỗng.
   - Encode CLIP text từng item; **cấm** concatenate.
   - Output: `item_embeddings.pt` + `item_index.json`.
2. Unit smoke: encode 2 class, shape và cosine image–text sanity (optional notebook).
3. Document lệnh trong `training_tutorial.md` (chỉ doc, không bắt buộc trong cùng PR code sau).

### Bước 2 — Module `SelectionAdapter`

1. Tạo `model/selection_adapter.py`:
   - `SelectionAdapter(nn.Module)`: buffers/register bank embeddings; forward `(v_x, class_indices) -> selected_phi [B,C,d] or [C,d]`.
   - API helpers: `topk_per_field`, `mean_pool`, `cosine_scores`.
2. Unit test file (pytest hoặc `__main__` smoke): fake bank, deterministic top-k indices.

### Bước 3 — CLIP image query encoder

1. Tái sử dụng CLIP image path trong `model/sematics/clip/` nếu có; nếu chỉ có text path, thêm thin wrapper load cùng checkpoint, `encode_image`, freeze.
2. Preprocess ảnh query: resize/crop chuẩn CLIP (không dùng train augment mạnh nếu làm lệch bank alignment — quyết định: eval = CLIP val preprocess; train = CLIP preprocess trên ảnh đã augment nhẹ hoặc raw crop).

### Bước 4 — Wire vào `HSPRN.forward`

1. `model/hsprn_model.py`:
   - Nếu `selection.enabled`:
     - Compute `v_x` từ CLIP image (hoặc nhận precomputed).
     - `p_sel = adapter(...)`.
     - `p_orig` từ path hiện tại (constructor full / attribute) — optional.
     - Fuse theo `fusion_beta`.
     - Truyền `p_sem` shape compatible với `InstancePrototypeRefiner` (`[C,D]` hoặc `[B,C,D]` — refiner **đã hỗ trợ** `[B,C,d]`).
2. Giữ `HSPRNOutput` fields; có thể thêm debug `selection_indices` khi `return_full`.

### Bước 5 — Data / trainer / eval hooks

1. `trainer/trainer.py`: load item bank path từ cfg; pass bank vào model (hoặc load trong model `__init__`).
2. `train.py` / `eval.py`: CLI args `--item_bank_path`, flag enable selection.
3. Đảm bảo candidate class indices ZSL/GZSL khớp bank filter (A4, A5).
4. `eval.py` calibrated stacking giữ nguyên sau scores.

### Bước 6 — Config experiments

1. Thêm YAML ví dụ: `config/cub_image_clip_vit_b16_selection.yaml` (copy train cfg + selection block).
2. Defaults an toàn: `enabled: false` trên config cũ để không phá baseline.

### Bước 7 — Ablation checklist (sau khi code xong, manual runs)

1. Baseline join full lists.
2. Selection $k=1,3,5$ per field.
3. `fusion_beta ∈ {0, 0.5, 1}`.
4. ZSL Top-1; GZSL $u,s,H$.
5. Diagnostics research §8.2 nếu có thời gian: retrieval R@k item cùng class.

### Bước 8 — Definition of Done (DoD)

- [ ] Offline bank build reproducibly từ semantics JSON.
- [ ] Train + eval chạy end-to-end với selection bật, không LLM online.
- [ ] Baseline path (`enabled=false`) bit-compatible hoặc metric ~không regress không chủ ý.
- [ ] README/tutorial note ngắn cách bật selection.
- [ ] Không merge cho đến khi review plan + code trên nhánh riêng.

---

## 4. File / module dự kiến bị ảnh hưởng

| File / path | Mức | Việc |
| --- | --- | --- |
| `tools/build_item_bank.py` | **Mới** | Flatten + CLIP encode items |
| `model/selection_adapter.py` | **Mới** | Top-k + pool |
| `model/hsprn_model.py` | Sửa | Wire adapter + fuse |
| `model/sematics/constructor.py` | Sửa nhẹ / optional | Path A filter lists; hoặc bypass join |
| `model/sematics/clip/*` | Sửa / mở rộng | Image encode cho query |
| `trainer/trainer.py` | Sửa | Load bank, forward args |
| `train.py`, `eval.py` | Sửa | CLI + cfg |
| `config/*.yaml` | Thêm | Experiment configs |
| `docs/research.md` | Tham chiếu only | Không bắt buộc sửa |
| `training_tutorial.md` | Doc sau | Hướng dẫn bank + train |

**Không đụng (MVP):** `losses/hsprn_loss.py` logic chính; calibrated stacking; prepared image package layout (chỉ thêm metadata artifacts).

---

## 5. Rủi ro / điểm cần lưu ý

1. **CLIP vs ResNet vision mismatch:** query selection ≠ $\tilde{v}$ — đúng theo A1; đừng nhầm dùng ResNet feature làm cosine với CLIP text.
2. **Chi phí:** instance-aware $\times$ $C$ classes: **bắt buộc cache** text embeddings; chỉ matmul `v_x @ E_c.T`.
3. **Leak nhãn:** adapter không nhận $y$; chỉ $\mathcal{C}$ theo protocol.
4. **77 tokens:** chỉ encode per-item; không join selected texts trước CLIP.
5. **Unseen bank:** phải có LLM items cho unseen classes offline (chuẩn ZSL semantic side).
6. **Determinism:** top-k ties; cố định sort stable.
7. **Regression:** config default off; test smoke cả hai mode.
8. **Memory:** bank CUB ~200 classes × ~31 items × 512 × 4 bytes ≈ nhỏ; SUN lớn hơn — vẫn OK trên GPU mid-range.

---

## 6. Phụ lục — Mapping research → implementation

| Research | Code |
| --- | --- |
| Eq. 1, 3, 9–11 | `selection_adapter.py` + `hsprn_model.py` |
| Eq. 12–13 dual-channel | fuse trong `HSPRN.forward` |
| §4.1 cosine MVP | phase 1 |
| §4.5 per-field | `k_attr/k_hier/k_desc` |
| §5.3 Path A/B | Path B ưu tiên tốc độ (pool embedding) |
| §7 CLIP 77 | `build_item_bank.py` per-item only |
'''