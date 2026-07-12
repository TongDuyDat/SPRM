# Plan 2 — Four-Bank Independent Encoding & Eval-time Top-k Retrieval

**Trạng thái:** kế hoạch triển khai để review. **Không** implement code trong bước này.  
**Nhánh git đề xuất (sau khi review):** `feat/plan-2-four-bank-topk`  
**Tham chiếu thiết kế:** `docs/research.md` **§10** (Four-Bank; query eval = **ảnh + attribute**, không LLM text query).  

> **Lưu ý đồng bộ tài liệu:** mô tả miệng “before-LLM text query” là phiên bản cũ. Bản formalization hiện tại trong `docs/research.md` §10.5 quy định query chính là **ảnh $x$ + attribute dataset $a_c$** (protocol ZSL). Plan này triển khai **đúng §10 hiện tại**.

---

## 1. Mục tiêu / vấn đề giải quyết

### 1.1. Vấn đề

1. Constructor hiện **join** nhiều câu LLM thành một prompt → trộn field, dễ vượt 77 tokens CLIP, mất tín hiệu item-level.
2. Train phụ thuộc dump LLM; eval ZSL chuẩn chỉ có **ảnh + attribute matrix** → cần cơ chế dùng bank LLM offline mà **không** đưa LLM text làm query.
3. Cần tách rõ 4 bank field-wise để retrieval / pooling có cấu trúc.

### 1.2. Mục tiêu plan này

1. **Offline:** encode độc lập từng item → 4 bank $B_1$–$B_4$ (`attr_c`, `hier_c`, `desc_c`, `class`/`name_c`) — research Eq. 22–24 / §10.3.
2. **Train:** model dùng embeddings trong từng bank (pool/mix), **không** join text — §10.4.
3. **Eval ZSL/GZSL:** query $q(x)$ hoặc $q(x,c)$ từ **$\hat{v}_x$ + $W_a a_c$** → cosine top-$k$ mỗi bank → nhánh $g_\psi$ (fuse + IPR + matching) — §10.5, Phụ lục C.
4. LLM text **chỉ** là nội dung bank (memory), không phải input query eval.

### 1.3. Phạm vi MVP

| Trong scope | Ngoài / phase 2 |
| --- | --- |
| 4 bank artifacts + loader | Soft attention full bank không top-k |
| Train: mean-pool per field + weighted mix $\phi^{\mathrm{mix}}_c$ | Multi-prototype max-over-items full |
| Eval Q2: $q(x,c)=\mathrm{Normalize}(\lambda_q \hat{v}_x+(1-\lambda_q)\hat{e}^a_c)$ | Learnable bilinear sim |
| Class-restricted top-$k$ per bank | Global vote-only classifier (A) như ablate |
| Fuse $\beta$ selected + $W_a a_c$ (Eq. 41a) | Gumbel / cross-attn |

---

## 2. Thiết kế kỹ thuật chi tiết

### 2.1. Bốn bank

| Bank | Field nguồn | Nội dung |
| --- | --- | --- |
| $B_1$ | `attr_c` | list attribute phrases (LLM) |
| $B_2$ | `hier_c` | list hierarchy / subtype (LLM) |
| $B_3$ | `desc_c` | list context / setting (LLM) |
| $B_4$ | `name_c` / class | class name (thường 1 item; có thể prompt ngắn) |

Mỗi entry: embedding $e_{c,r,i}=\phi_{\mathrm{txt}}(u_{c,r,i})$, class id, item id.  
$N_r=\mathrm{card}(B_r)$.

### 2.2. Train path

Với class $c$:

$$
\hat{\bar{e}}_{c,r} = \mathrm{Normalize}\Big(\frac{1}{M_{c,r}}\sum_i \hat{e}_{c,r,i}\Big)
$$

$$
\phi^{\mathrm{mix}}_c = \sum_{r=1}^{4} w_r \hat{\bar{e}}_{c,r}
\quad\text{hoặc}\quad
\phi^{\mathrm{four}}_c = [\hat{\bar{e}}_{c,1};\ldots;\hat{\bar{e}}_{c,4}]
$$

$$
\tilde{p}^{\mathrm{sem}}_c = \mathrm{Normalize}(W_s \phi^{\mathrm{mix}}_c)
$$

**Không** gọi `_build_prompt` join.

Tùy chọn consistency (khuyến nghị ablate): lúc train cũng subsample top-$k$ theo $q(x,c)$ như eval để giảm train–eval select mismatch (§10.8).

### 2.3. Eval path (protocol ZSL)

**Input được phép:** ảnh $x$, attribute matrix $\{a_c\}_{c\in\mathcal{C}}$, candidate set $\mathcal{C}$.  
**Cấm làm query:** text LLM, free-text before-LLM.

$$
\hat{v}_x = \mathrm{Normalize}(\phi_{\mathrm{img}}(x)),\quad
\hat{e}^{a}_c = \mathrm{Normalize}(W_a a_c)
$$

**Q1:** $q(x)=\hat{v}_x$  
**Q2 (MVP khuyến nghị):** $q(x,c)=\mathrm{Normalize}(\lambda_q \hat{v}_x+(1-\lambda_q)\hat{e}^{a}_c)$

Top-k class-restricted:

$$
S_r(x,c;k_r)=\mathrm{TopK}_{j:\mathrm{cls}_r(j)=c}\ \langle q(x,c),\hat{e}^{r}_j\rangle
$$

Pool + fuse:

$$
h(x,c)=[\tilde{e}_1;\tilde{e}_2;\tilde{e}_3;\tilde{e}_4],\quad
p^{\mathrm{fuse}}_{x,c}=\mathrm{Normalize}(\beta W_s h+(1-\beta)W_a a_c)
$$

Rồi IPR + matching như HSPRN; $\hat{y}=\arg\max_{c\in\mathcal{C}}\mathrm{score}(x,c)$.

### 2.4. Module layout đề xuất

```text
model/
  banks/
    four_bank.py          # load B1-B4, indexing by class
  selection/
    bank_retriever.py     # top-k given q(x,c)
  fusion/
    field_mixer.py        # w_r mix / concat + W_s, W_a
hsprn_model.py            # wire train pool vs eval retrieve
tools/
  build_four_banks.py     # offline encode
```

### 2.5. Config schema đề xuất

```yaml
model:
  four_bank:
    enabled: true
    bank_dir: null
    k: 3                    # hoặc k_attr/k_hier/k_desc/k_class
    lambda_q: 0.7           # weight ảnh trong q(x,c)
    fusion_beta: 0.5        # weight retrieved vs W_a a_c
    field_weights: [0.25, 0.25, 0.25, 0.25]
    train_mode: mean_pool   # mean_pool | train_topk_subsample
    eval_query: q2          # q1 | q2
    clip_image_query: true
data:
  semantic_source: class_desc   # lists vẫn cần để build banks offline
  attribute_always_available: true
```

### 2.6. Quan hệ với Plan 1

- Plan 1: selection adapter tổng quát, bank phẳng + dual-channel.  
- Plan 2: **cấu trúc hóa 4 bank**, query eval **bắt buộc** ảnh+attribute, train path pool theo field.  
- Có thể tái sử dụng code cosine top-k từ Plan 1 nếu Plan 1 merge trước; nếu không, implement retriever độc lập trên nhánh Plan 2.

---

## 3. Các bước triển khai (thứ tự)

### Bước 0 — Nhánh, baseline, kiểm tra attribute

1. Nhánh `feat/plan-2-four-bank-topk`.
2. Xác nhận prepared packages / XLSA17 vẫn cung cấp **attribute matrix** cho mọi class (seen+unseen) — bắt buộc cho $W_a a_c$.
3. Baseline metrics: (a) join class_desc; (b) pure attribute embedding path nếu có.

### Bước 1 — Offline `build_four_banks.py`

1. Đọc semantics JSON list-per-field.
2. Với mỗi field $r$, encode item-level CLIP text → 4 files hoặc 1 dict:
   - `bank_attr.pt`, `bank_hier.pt`, `bank_desc.pt`, `bank_class.pt`
   - mỗi file: `embeddings [N_r,d]`, `class_ids [N_r]`, optional `texts`.
3. Validate: không item nào tokenize > 77 (warn + clip policy: reject item, không silent truncate nếu có thể).
4. Script smoke trên CUB subset.

### Bước 2 — `FourBankStore` loader

1. Module load 4 tensors, build per-class index lists `indices[c][r]`.
2. API: `get_class_items(c, r) -> Tensor [M_c,r, d]`, `all_for_field(r)`.
3. Device move / dtype float32.

### Bước 3 — Train semantic path (thay join)

1. Thêm `FieldBankConstructor` (hoặc mở rộng `PriSematicConstructor`):
   - Input: class ids hoặc pre-sliced bank rows.
   - Output: `p_sem_norm` như constructor hiện tại.
2. Tắt / bypass `_build_prompt` join khi `four_bank.enabled`.
3. `W_s` `in_features`: $d$ nếu mix; $4d$ nếu concat — config rõ.
4. Train loop: semantic bank **class-level** mean-pool (MVP); optional later: top-k subsample theo batch images.

### Bước 4 — Attribute projection $W_a$

1. Linear `W_a: d_a -> d` (hoặc `-> shared_dim` rồi align).
2. Init: Xavier; train với HSPRN losses (gradient qua fuse path).
3. Freeze CLIP text/image encoders.

### Bước 5 — Eval retriever + $g_\psi$

1. `BankRetriever.forward(v_x, a_c_batch_or_bank, class_list)`:
   - build $q(x,c)$; scores; top-k; weighted pool $\tilde{e}_r$.
2. Build $h(x,c)$, fuse với $W_a a_c$, IPR, matching.
3. Wire `eval.py` + `trainer` validation metrics (ZSL/GZSL) dùng **cùng** path.
4. Calibrated stacking giữ sau `score(x,c)`.

### Bước 6 — CLIP image query

1. Freeze CLIP image encoder đồng checkpoint với bank text encode.
2. Preprocess CLIP chuẩn; batch encode images trong eval loop (có thể cache val embeddings).

### Bước 7 — Config + integration tests

1. YAML experiment CUB/AWA2.
2. Smoke test:
   - shapes `[B,C,D]` qua IPR;
   - `lambda_q=1` vs `0.5`;
   - `enabled=false` fallback baseline.
3. Manual ablation matrix (§10.9): $k\in\{1,3,5\}$, $\lambda_q$, $\beta$, Q1 vs Q2.

### Bước 8 — DoD

- [ ] 4 bank artifacts tái tạo được từ semantics JSON.
- [ ] Train không join text; eval không LLM query.
- [ ] Eval chỉ cần image loader + attribute matrix + banks.
- [ ] Metrics ZSL/GZSL report được; so baseline join.
- [ ] Plan/code trên nhánh riêng; không đụng Plan 1/3 trừ khi cherry-pick có chủ đích.

---

## 4. File / module dự kiến bị ảnh hưởng

| File / path | Mức | Việc |
| --- | --- | --- |
| `tools/build_four_banks.py` | **Mới** | Offline 4 banks |
| `model/banks/four_bank.py` | **Mới** | Store + index |
| `model/selection/bank_retriever.py` | **Mới** | Top-k Q1/Q2 |
| `model/sematics/constructor.py` | Sửa lớn | Bypass join; field pool path |
| `model/hsprn_model.py` | Sửa lớn | Train pool vs eval retrieve + fuse |
| `model/protypes.py` | Kiểm tra | IPR `[B,C,d]` |
| `model/matching_head.py` | Kiểm tra | scores |
| `trainer/trainer.py` | Sửa | attribute + bank + v_x |
| `train.py`, `eval.py` | Sửa | cfg/CLI |
| `config/*_four_bank*.yaml` | **Mới** | experiments |
| `tools/generate_descriptions.py` | Có thể | đảm bảo list format ổn định |
| `tools/prepare_zsl_dataset_package.py` | Optional | copy bank artifacts vào package |

**Phụ thuộc dữ liệu:** attribute matrix (`res101.mat` / package) + class_semantics lists + CLIP weights.

---

## 5. Rủi ro / điểm cần lưu ý

1. **$W_a$ alignment:** attribute $d_a$ (85/312/…) khác CLIP $d$; $W_a$ kém → Q2 nhiễu. Mitigation: pretrain $W_a$ align seen (cosine $\hat{v}$ vs $W_a a_y$) trước full HSPRN.
2. **Protocol ZSL:** không đưa class name LLM dài làm query; $B_4$ chỉ memory. Name trong bank OK; query name text không phải Q2.
3. **Train/eval select gap:** mean-pool train vs top-k eval — ablate `train_topk_subsample`.
4. **Cost:** per image × $C$ × 4 banks top-k: cache bank on GPU; pre-normalize rows.
5. **Unseen attributes:** phải có $a_c$ cho unseen (chuẩn ZSL). Thiếu attribute → plan 2 không chạy đúng protocol.
6. **Trùng Plan 1:** tránh double-merge conflict; coordinate `hsprn_model.forward` signatures.
7. **Token 77:** validate lúc build bank; fail loud.
8. **Seed / tie-break** top-k reproducibility.

---

## 6. Phụ lục — Mapping research §10 → code

| §10 | Implementation |
| --- | --- |
| Eq. 22–24 banks | `build_four_banks.py`, `FourBankStore` |
| Eq. 26–28 train pool/mix | `FieldBankConstructor` |
| Eq. 29–30a query | CLIP img + `W_a` |
| Eq. 31–34 top-k | `BankRetriever` |
| Eq. 39–41c $g_\psi$ | fuse + IPR + matching trong `HSPRN` |
| Phụ lục C | integration test asserts shapes |
'''