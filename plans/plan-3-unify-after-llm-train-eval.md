# Plan 3 — Unify After-LLM Semantics for Both Train and Eval

**Trạng thái:** đang triển khai trên `labs/after-llm`.  
**Constraint bổ sung:** LLM after-LLM ghi `data_semantics/*/class_descriptions_*_after_llm.json` — **không** ghi đè `class_descriptions_*.json` legacy.  
**Nguồn:** hướng **mới**, **chưa** formal hóa trong `docs/research.md`.

---

## 0. Hình thức hóa vấn đề và giải pháp (mới)

### 0.1. Định nghĩa kênh semantic

Gọi với mỗi class $c$:

| Ký hiệu | Nghĩa | Nguồn điển hình trong repo |
| --- | --- | --- |
| $t^{\mathrm{before}}_c$ | **Before-LLM** / input gốc dataset | Tên lớp raw, attribute vector/string XLSA17, predicate list — **chưa** qua LLM paraphrase |
| $t^{\mathrm{after}}_c$ | **After-LLM** | Output `tools/generate_descriptions.py`: `name_c`, lists `attr_c`/`hier_c`/`desc_c` |

Encoder semantic trong model (CLIP/BERT text hoặc path class_desc):

$$
\phi^{\mathrm{sem}}(t) = W_s\,\phi_{\mathrm{txt}}(\mathrm{prompt}(t))
$$

(hoặc pool item-level sau này; Plan 3 không bắt buộc four-bank).

### 0.2. Vấn đề hiện tại (train–eval mismatch)

Quan sát thiết kế / thực tế pipeline khi so sánh các `semantic_source` và thói quen thí nghiệm:

$$
s_{\mathrm{train}}(x) = t^{\mathrm{after}}_{y}\quad\text{(class_desc LLM already generated)}
$$

$$
s_{\mathrm{eval}}(x) \in \{ t^{\mathrm{before}}_{c},\ \text{attribute matrix only},\ \text{partial fields} \}
$$

Hệ quả:

$$
P_{\mathrm{train}}(s\mid x) \neq P_{\mathrm{eval}}(s\mid x)
$$

Model học matching visual $\leftrightarrow$ **mô tả LLM phong phú**, nhưng lúc đo có thể bị ép visual $\leftrightarrow$ **attribute/name gầy**. Đây là **channel shift** trên semantic side (cùng họ vấn đề research §1.2), không phải label shift ZSL.

Ngay cả khi train/eval cùng đọc `class_semantics.json`, mismatch vẫn xuất hiện nếu:

- eval fallback `semantic_source=attribute` / embedding attribute;
- hoặc constructor join khác nhau;
- hoặc một nhánh thí nghiệm cố ý “lean eval”.

### 0.3. Thay đổi đề xuất (ý tưởng cốt lõi)

**Dùng after-LLM cho cả training và evaluation** — đồng nhất hoàn toàn kênh semantic:

$$
s_{\mathrm{train}}(c) = s_{\mathrm{eval}}(c) = t^{\mathrm{after}}_c
\quad \forall c \in \mathcal{Y}^{s}\cup\mathcal{Y}^{u}
$$

$$
\hat{y} = f_{\mathrm{HSPRN}}\big(x,\ \{t^{\mathrm{after}}_c\}_{c\in\mathcal{C}}\big)
$$

- **Không** yêu cầu selection adapter (Plan 1) hay four-bank retrieval (Plan 2) để “vá” lean eval — mặc dù vẫn tương thích nếu bật sau.
- **Yêu cầu cứng:** mọi class ứng viên (gồm **unseen**) đã có $t^{\mathrm{after}}_c$ offline trước eval.
- **Không** gọi LLM trong train/eval loop.

### 0.4. Ràng buộc kỹ thuật: 77 tokens CLIP **per description item**

CLIP text encoder (OpenAI CLIP / tương thích):

$$
\mathrm{len}_{\mathrm{BPE}}(\mathrm{tokenize}(u)) \le 77
$$

(bao gồm `<|startoftext|>` và `<|endoftext|>`).

Áp dụng cho **từng câu / item** $u$ đưa vào encoder:

- Nếu encode **từng dòng** list `attr_c[i]`: mỗi dòng ≤ 77 tokens.
- Nếu vẫn **join** fields thành một prompt: **toàn bộ chuỗi join** phải ≤ 77 — thường **không khả thi** với 10+10+10 items → Plan 3 **khuyến nghị bắt buộc** chuyển sang:
  - **(P3-A)** encode item-level + mean-pool per field / per class, **hoặc**
  - **(P3-B)** rút gọn số item + join cực ngắn (kém ưu tiên).

Prompt generation (`generate_descriptions.py`) đã hướng dẫn ~50 English words/item; Plan 3 **siết enforcement**: validate tokenizer CLIP sau sinh, regenerate/truncate-policy rõ ràng.

Ký hiệu:

$$
\forall c,r,i:\quad \mathrm{tokens}(u_{c,r,i}) \le 77
$$

**Không** áp dụng “tổng số mô tả × 77”; chỉ **per item**.

### 0.5. So sánh nhanh 3 plans

| | Plan 1 | Plan 2 | Plan 3 (này) |
| --- | --- | --- | --- |
| Mục tiêu chính | Select item theo ảnh | 4 bank + retrieve bằng ảnh+attr | **Đồng nhất after-LLM train/eval** |
| Eval semantic | Selected after-LLM (+ optional orig) | Retrieved after-LLM bank + attr fuse | **Full after-LLM class semantics** |
| Eval query đặc biệt | Ảnh CLIP | Ảnh + attribute | Không cần retrieval query |
| Độ phức tạp | Trung bình–cao | Cao | **Thấp–trung bình** (policy + data) |
| 77-token | Per item bank | Per item bank | **Per item (enforce)** |

---

## 1. Mục tiêu / vấn đề giải quyết (tóm tắt triển khai)

1. Loại bỏ train–eval semantic channel mismatch bằng cách **bắt buộc** `class_desc` after-LLM ở **mọi** entrypoint train/val/test.
2. Đảm bảo artifact $t^{\mathrm{after}}$ đủ cho **unseen** classes trong package.
3. Enforce độ dài CLIP 77 tokens **từng mô tả**.
4. (Khuyến nghị) bỏ join dài: pool item-level để semantic encode hợp lệ và ổn định.

**Không** nhằm thay ZSL protocol (vẫn không train ảnh unseen); chỉ thống nhất **side information text** after-LLM.

---

## 2. Thiết kế kỹ thuật chi tiết

### 2.1. Policy semantic source

```text
semantic_source allowed in train/eval:
  - class_desc  (REQUIRED for Plan 3 experiments)
forbidden for Plan 3 main line:
  - attribute-only eval paired with class_desc train
  - auto → attribute fallback
```

Implementation policy:

1. `resolve_semantic_source`: nếu `cfg.experiment.unify_after_llm: true` → force `class_desc`, error nếu thiếu metadata.
2. `configure_semantic_input`: giữ `input_dim: null`, backbone text CLIP/BERT.
3. Train/eval cùng hàm `semantic_input(...)` đọc **cùng** `get_class_semantics()`.

### 2.2. Data contract after-LLM

Mỗi class record tối thiểu:

```json
{
  "name_c": "zebra",
  "attr_c": ["...", "..."],
  "hier_c": ["...", "..."],
  "desc_c": ["...", "..."]
}
```

- Có mặt cho **mọi** class id trong trainval + test_seen + test_unseen.
- Prepared package: `metadata/class_semantics.json` (đã có pattern trong tutorial).
- Manifest flag: `semantics_mode: after_llm`, `clip_token_validated: true`.

### 2.3. Encode path (đồng nhất)

**MVP encode (P3-A, khuyến nghị):**

1. Encode từng item list bằng CLIP text.
2. Mean-pool trong field → 4 vector; mix/concat → $W_s$ → `p_sem`.
3. Class name: 1 embedding.

**Tương thích ngắn hạn (P3-B):** giữ `_build_prompt` nhưng **chỉ** join `name` + top-1 item/field đã rút gọn — chỉ khi P3-A chưa sẵn; document là interim.

Train và eval gọi **cùng** hàm `encode_class_semantics(record) -> p_sem`.

### 2.4. Generation constraints (prompt + validation)

Trong `tools/generate_descriptions.py` (hoặc post-process):

1. Giữ prompt “self-contained, ≤ ~50 English words”.
2. Thêm bước **CLIP tokenizer length check** sau parse lines:
   - if `len(tokens) > 77`: drop item / regenerate / hard-reject class batch (config policy).
3. Log stats: mean/max tokens per field; % rejected.
4. Optional: max items per field giảm (10 → 5) nếu pool đủ.

### 2.5. Eval flow thống nhất

```text
[Offline once]
  class list --> LLM --> after-LLM JSON --> (validate 77) --> package metadata

[Train]
  image x, label y (seen)
  candidate semantics: { t_after_c for c in Y_seen }  (or full bank as now)
  --> same encode_class_semantics --> HSPRN

[Eval ZSL]
  image x
  candidates C = Y_unseen
  semantics: { t_after_c for c in C }   # NOT attribute-only
  --> same encode_class_semantics --> HSPRN --> scores --> metrics
```

GZSL: $C=Y^s\cup Y^u$, cùng after-LLM.

### 2.6. Metrics / acceptance

- Primary: ZSL Top-1, GZSL $u,s,H$ với **unify_after_llm**.
- Control: cố ý chạy **mismatched** eval (attribute-only) trên cùng checkpoint để đo **channel gap** $\Delta$ (research §8.2) — chứng minh Plan 3 giảm gap về ~0 khi so train/eval cùng after-LLM.
- Token audit report: 0 items > 77 nếu policy reject.

---

## 3. Các bước triển khai (thứ tự)

### Bước 0 — Nhánh và inventory mismatch

1. Tạo nhánh `feat/plan-3-unify-after-llm`.
2. Inventory code paths:
   - `train.py` / `eval.py` `resolve_semantic_source`
   - `trainer/trainer.py` class_desc vs attribute
   - configs `semantic_source`, `semantic_embedding_path`
3. Viết checklist “các lệnh eval hiện có đang lean-semantic?” trong PR description (khi code).

### Bước 1 — Policy flag + hard fail

1. Thêm `experiment.unify_after_llm` (hoặc `data.require_after_llm: true`) trong default/config experiment.
2. Khi bật:
   - force `semantic_source=class_desc`
   - require `class_semantics` full 4 fields cho mọi class index dùng trong split
   - `raise` rõ ràng nếu thiếu unseen descriptions
3. Unit test: mock missing unseen → fail.

### Bước 2 — Data completeness pipeline

1. Script audit: `tools/audit_class_semantics.py`
   - Input: semantics JSON + split class lists
   - Output: missing classes, empty fields, token length histogram
2. Chạy trên CUB/AWA2 packages; generate thiếu bằng `generate_descriptions.py` nếu cần.
3. `prepare_zsl_dataset_package.py`: option `--require-full-semantics` (có thể đã có) + fail nếu audit fail.

### Bước 3 — Enforce 77 tokens

1. Utility `tools/clip_token_budget.py`: `count_tokens(text, clip_model) -> int`.
2. Hook sau `generate_descriptions.py` write:
   - validate mỗi line item
   - config: `on_overflow: reject | regenerate | skip_item`
3. CI/local smoke: sample 100 items, assert max ≤ 77.
4. Document: ràng buộc **per item**, không per-class total.

### Bước 4 — Đồng nhất encode path (P3-A)

1. Implement `encode_class_record_itemwise` (module semantic):
   - pool lists; không join dài.
2. `PriSematicConstructor`: branch `itemwise_pool: true` dùng chung train/eval.
3. So sánh sanity embedding: cosine giữa join-truncated vs itemwise (log only).

### Bước 5 — Eval / train entrypoints

1. `eval.py`: default prepared path luôn class_desc khi unify flag; xóa/chặn silent fallback attribute.
2. `train.py`: same.
3. Tutorials: một lệnh train + một lệnh eval **cùng** metadata path; cấm example “train class_desc, eval attribute” trong section Plan 3.

### Bước 6 — Regression & gap experiment

1. Train 1 short run CUB (hoặc reuse checkpoint) với after-LLM.
2. Eval A: after-LLM (Plan 3).
3. Eval B: attribute-only (mismatch control).
4. Report $\Delta = \mathrm{Acc}_A - \mathrm{Acc}_B$ và khẳng định A là protocol chính Plan 3.

### Bước 7 — DoD

- [ ] Flag unify bật → không thể eval attribute-only nhầm.
- [ ] 100% class splits có after-LLM 4 fields.
- [ ] Token audit: không item > 77 (theo policy).
- [ ] Train/eval cùng encode function (itemwise pool khuyến nghị).
- [ ] Doc ngắn trong tutorial + (optional) đoạn thêm `docs/research.md` sau khi plan được chốt.
- [ ] Code chỉ trên nhánh Plan 3.

---

## 4. File / module dự kiến bị ảnh hưởng

| File / path | Mức | Việc |
| --- | --- | --- |
| `train.py` | Sửa | Force class_desc; CLI flag |
| `eval.py` | Sửa | Force class_desc; no lean fallback |
| `trainer/trainer.py` | Sửa | Assert semantics full; same encode |
| `model/sematics/constructor.py` | Sửa | Itemwise pool path; tránh join dài |
| `tools/generate_descriptions.py` | Sửa | Token validate / policy overflow |
| `tools/clip_token_budget.py` | **Mới** | Count CLIP tokens |
| `tools/audit_class_semantics.py` | **Mới** | Completeness + length audit |
| `tools/prepare_zsl_dataset_package.py` | Sửa | Gate full after-LLM |
| `config/*.yaml` | Thêm | `unify_after_llm` experiments |
| `training_tutorial.md` | Doc | Unified train/eval instructions |
| `docs/research.md` | Optional sau review | Thêm § formalization Plan 3 |
| Prepared `metadata/class_semantics.json` | Data | Đủ seen+unseen |

**Ít đụng:** IPR, matching head, losses (trừ khi đổi dim do concat 4d).

---

## 5. Rủi ro / điểm cần lưu ý

1. **ZSL community protocol:** một số paper eval chỉ attribute vectors. Plan 3 là **experimental protocol** “LLM side info available at test for all classes” — cần nêu rõ trong báo cáo (không claim so sánh unfair với method attribute-only nếu họ không có LLM text unseen).
2. **Unseen leakage qua LLM knowledge:** mô tả unseen do LLM sinh từ **tên class** là phổ biến trong generative ZSL; không được sinh từ **ảnh test**.
3. **Join vs 77 tokens:** nếu ai đó bật lại join full lists, CLIP truncation im lặng → metric ảo. Mitigation: itemwise pool + audit.
4. **Chi phí lưu trữ / sinh lại:** regenerate descriptions khi siết token policy.
5. **Tương thích Plan 1/2:** Plan 3 là baseline đồng nhất; Plan 1/2 có thể build **trên** after-LLM thống nhất. Không trộn flag trong một PR.
6. **BERT backbone:** 77 là ràng buộc CLIP; nếu BERT, dùng max_length model đó — config `token_budget` theo backbone.
7. **Prepared packages cũ:** audit sẽ fail → cần rebuild semantics trước train.

---

## 6. Tiêu chí review plan (trước khi code)

Reviewer kiểm tra:

- [ ] Có đồng ý protocol “after-LLM available for unseen at test”?
- [ ] Chọn P3-A (itemwise pool) vs P3-B (join ngắn) làm mặc định?
- [ ] Policy overflow token: reject vs regenerate?
- [ ] Có cần cập nhật `docs/research.md` § mới trước khi merge code không?

---

## 7. Phụ lục — Công thức tóm tắt Plan 3

**Mismatch hiện tại (cần loại):**

$$
s_{\mathrm{train}} = t^{\mathrm{after}},\quad s_{\mathrm{eval}} = t^{\mathrm{before}} \implies P_{\mathrm{train}}\neq P_{\mathrm{eval}}
$$

**Mục tiêu:**

$$
s_{\mathrm{train}}(c)=s_{\mathrm{eval}}(c)=t^{\mathrm{after}}_c
$$

**Ràng buộc CLIP:**

$$
\forall\ \text{item } u \text{ fed to } \phi_{\mathrm{txt}}:\quad \mathrm{tokens}(u)\le 77
$$

**Encode khuyến nghị:**

$$
e_{c,r,i}=\phi_{\mathrm{txt}}(u_{c,r,i}),\quad
\bar{e}_{c,r}=\mathrm{mean}_i e_{c,r,i},\quad
p^{\mathrm{sem}}_c=\mathrm{Normalize}(W_s\cdot\mathrm{mix}_r(\bar{e}_{c,r}))
$$
'''