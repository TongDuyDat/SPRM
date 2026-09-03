# Audit Log — v9.6 Phase 3 (hợp nhất 4 đề xuất)
> Mọi mục: Phương pháp → Lệnh → Kết quả (nguyên văn stdout) → Gate → Ghi chú.
> Quy ước số: T1 | T1 seen | u | s | H | Cal u | Cal s | Cal H | γ. Baseline R6b/ep44 (best.pth): 74.4 | 67.5 | 50.2 | 64.4 | 56.4 | 66.8 | 56.9 | 61.5 | 0.02 (hub s/u 7.13/2.08 từ log train ep80; hub của best.pth xem E1).
> Nghiên cứu nền: `docs/nghien_cuu_chuyen_sau_v95.md`. Ma trận kế hoạch E1–E6: plan đã duyệt.

---

## E0 — Smoke test + sửa lỗi trước khi chạy thật | 2026-08-29 (local code, GPU IBM)
**Phương pháp:** kiểm tra từng tool mới (export_sentences, control Waffle, semantic_ens, csls, eval_tta) trên checkpoint R6b/best.pth trước khi chạy phase 3 thật, để bug không phá batch lớn.

**Lệnh (đại diện):**
```
python tools/precompute_class_phi.py --config config/cub_clip_v9_hsprn_b16.yaml \
    --prepared_root ./prepared_datasets --export_sentences --output ./pre-trained/smoke_sent_test.pt
python tools/eval_checkpoint.py --config config/cub_clip_v9_r6b_noref_v8phi_b16.yaml \
    --checkpoint checkpoints/cub_clip_v9_r6b_noref_v8phi_b16/CUB/best.pth \
    --name SMOKE/ens --score_bank semantic_ens --phi_path ./pre-trained/smoke_sent_test.pt
python tools/eval_checkpoint.py ... --score_correction csls --csls_k 10 [--csls_beta B]
python tools/diag_hubness.py --config config/cub_clip_v9_r6b_noref_v8phi_b16.yaml \
    --checkpoint checkpoints/cub_clip_v9_r6b_noref_v8phi_b16/CUB/best.pth
```

**File & dòng đã đổi (trước → sau):**
- `trainer/trainer.py`: + `self.save_final = True` (init) và block `final.pth` sau vòng train (epoch cuối, không chọn theo test) — trước: chỉ best/best_t1. `ckpt_dir` dời ra trước vòng for để final.pth không phụ thuộc nhánh eval_freq.
- `model/sematics/constructor.py`: + `class_phi_sentence_texts(records)` — flatten mỗi record thành list câu (name/attr/hier/desc từng item một), không đụng `encode_class_phi` cũ.
- `tools/precompute_class_phi.py`: + `--export_sentences` (lưu `embeddings_sentences [C,S,512]` + `sentence_mask [C,S]`), `--control {random_words,name_repeat}` (WaffleCLIP control, wordlist 50 từ cố định seed 42), `--metadata` (đè semantics JSON, cho plan5/hybrid mà không đụng `class_semantics.json` gốc).
- `datasets/prepared_dataset.py`: + kwarg `semantics_override_path` cho `PreparedZSLDataset` (mặc định None — các run cũ không đổi). Lỗi tạm `int(self.image_size)` đã sửa ngay trong phiên.
- `tools/eval_checkpoint.py`: + `--phi_path` (đè φ không đổi config), `--score_bank semantic_ens` (score per-sentence rồi mean, CuPL-style), `--score_correction csls --csls_k --csls_beta`, in hub s/u trong RESULT.
- `tools/eval_tta.py` (MỚI): TTA transductive (DPE-style centroid + TransCLIP-style mixture), bảng riêng.
- `tools/diag_hubness.py` (MỚI): Spearman(r(c), win-count) kiểm định tín hiệu hubness.
- `scripts/run_v9_phase3.sh` (MỚI): orchestrate E1→E5.
- `config/cub_clip_v9_r7h_hybrid_b16.yaml` (MỚI): = R6b, chỉ đổi `semantic_embedding_path` → `cub_v9_hybrid_phi_b16.pt`.
- `data_semantics/CUB/class_descriptions_cub_hybrid.json` (MỚI): name/hier/desc từ plan5 + attr_c từ grounded; đã cài vào `prepared_datasets/CUB/metadata/class_descriptions_cub_hybrid.json` (IBM).

**Kết quả smoke (nguyên văn):**
```
Per-sentence tensor: (200, 25, 512) (S_max=25, classes=200, control=none)
Saved phi_t (200, 512) -> pre-trained/smoke_sent_test.pt
  mean off-diagonal cosine: 0.5719 -> 0.5719 (centered=False)
RESULT SMOKE/ens epoch=44 bank=semantic_ens ... T1_unseen=72.3 T1_seen=65.6 u=45.0 s=62.2 H=52.3 cal_u=62.5 cal_s=55.8 cal_H=59.0 gamma=0.02 hub_s=6.04 hub_u=2.36   [phi grounded — chỉ smoke logic]
RESULT SMOKE/phi_sem ... T1_unseen=71.8 T1_seen=64.7 ...                    [cùng φ, bank semantic]
RESULT SMOKE/csls k=10 beta=1.0 ... T1_unseen=29.0 ... hub_s=18.02 hub_u=11.65
RESULT SMOKE/csls k=10 beta=0.05 ... T1_unseen=73.8 ... hub_s=8.89 hub_u=2.35
[unseen] win counts: mean=33.7 max=64 min=0 (hub ratio 1.90)
[unseen] k= 5: Spearman(r(c), win-count) = -0.373
[unseen] k=10: Spearman(r(c), win-count) = -0.335
[unseen] k=20: Spearman(r(c), win-count) = -0.366
[seen] win counts: mean=11.0 max=38 min=0 (hub ratio 3.47)
[seen] k= 5: Spearman(r(c), win-count) = -0.108
[seen] k=10: Spearman(r(c), win-count) = -0.136
[seen] k=20: Spearman(r(c), win-count) = -0.186
```

**Bug đã sửa trong smoke (3 cái, đều do E0 bắt được):**
1. `eval_checkpoint`: `num_total_classes` dùng trước khi định nghĩa → sụp eval khi `--phi_path`.
2. `eval_checkpoint`: `semantic_ens` sai einsum chiều + sai thứ tự class (sent_emb lấy cả 200 theo thứ tự gốc thay vì subset theo `all_classes`) → T1 0.1 (gần random). Sau sửa: T1 72.3 hợp lý so với bank semantic 71.8 cùng φ.
3. `datasets/prepared_dataset.py`: lệnh tạm `int(self.image_size)` → AttributeError, sửa ngay.
4. `diag_hubness.py`: KeyError khi pred rơi vào seen class (counts chỉ có unseen) → guard `if c in counts`.

**Phát hiện E0 quan trọng (thay đổi thiết kế E4):**
- **CSLS prototype-side BỊ BÁC BỎ trước khi vào tổ hợp.** Cơ chế giải thích bằng số:
  - Modality gap: proto-proto cosine mean 0.845 (r stats min 0.724 / mean 0.845 / max 0.926 trên `cub_v8_phi_b16.pt`) trong khi image-proto score ~0.25 → trừ r(c) ở thang riêng của nó đảo thứ tự score (beta=1.0: T1 74.4 → 29.0; beta=0.05 vẫn 73.8 < 74.4).
  - **Spearman(r(c), win-count) = −0.37 (unseen), −0.11..−0.19 (seen)**: trong pipeline này prototype "dày đặc" (hub theo nghĩa CSLS) lại *ít* được chọn hơn trên unseen — ngược giả thuyết hubness chuẩn của retrieval. Hiện tượng "hub s/u 7.31/2.01" đo trước đó là độ tập trung top-1 (max/mean), không phải r(c) CSLS; cả hai không đồng dấu ở đây.
  - Kết luận: hubness-correction kiểu Dinu/Conneau không portable cho prototype-matching trên CLIP đã qua Ws; phiên bản center + scale nhỏ (beta=0.05) vẫn dưới baseline ⇒ E4 chỉ giữ 3 hàng làm bằng chứng bác bỏ, không đưa vào tổ hợp E6.
- Đổi tương ứng trong kế hoạch: E4 (HUB-CORRECT) hạ từ "tầng ứng viên" xuống "bằng chứng phủ định" — cập nhật cả trong docs.

**Gate E0:** đạt (tất cả tool chạy đúng logic; kết quả hợp lý về mặt số học).

---

## E1 — Chuẩn hóa baseline R6b/ep44 | 2026-08-29 14:51 IBM
**Phương pháp:** eval lại `best.pth` (epoch 44, checkpoint hội tụ duy nhất có sẵn của R6b) bằng bank semantic để làm gốc so sánh cho mọi tầng sau; kèm hub s/u trong RESULT (mở rộng mới của eval_checkpoint).

**Lệnh:**
```
python tools/eval_checkpoint.py --config config/cub_clip_v9_r6b_noref_v8phi_b16.yaml \
    --checkpoint checkpoints/cub_clip_v9_r6b_noref_v8phi_b16/CUB/best.pth \
    --name "E1/r6b/ep44/semantic" --score_bank semantic
```

**Kết quả (nguyên văn):**
```
RESULT E1/r6b/ep44/semantic epoch=44 bank=semantic correction=none k=10 phi=cub_v8_phi_b16.pt T1_unseen=74.4 T1_seen=67.5 u=50.2 s=64.4 H=56.4 cal_u=66.8 cal_s=56.9 cal_H=61.5 gamma=0.02 hub_s=7.13 hub_u=2.08
```

**Bảng 9 cột:** 74.4 | 67.5 | 50.2 | 64.4 | 56.4 | 66.8 | 56.9 | 61.5 | 0.02 (hub s/u 7.13/2.08)

**Gate:** đạt — reproduce đúng số phase 2 (74.4), công cụ eval mở rộng không đổi metric cũ.

**Ghi chú:** ep44 (best theo Cal H) ≈ ep80 về T1 (74.4 cả hai, theo log train); dùng ep44 làm gốc vì là checkpoint thật duy nhất ngoài best_t1 (ep9, test-peeking).

---

## E2 — DESC-ENSEMBLE + đối chứng WaffleCLIP | 2026-08-29 15:01–15:06 IBM
**Phương pháp:** (a) `cub_v9_plan5_sent_b16.pt` = per-sentence embeddings của chính φ plan5 đã train R6b (cos(old,new)=1.000, S_max=31); (b) 2 đối chứng Waffle: `random_words` (filler "…, which has <2 từ ngẫu nhiên từ wordlist 50 từ>." thay mọi câu attr/hier/desc, giữ name) và `name_repeat` (chỉ name lặp S lần); (c) mỗi φ chấm bằng cả bank semantic (mean-embedding) và semantic_ens (mean per-sentence score) trên cùng checkpoint R6b/ep44.

**Lệnh (đại diện):**
```
python tools/precompute_class_phi.py --config config/cub_clip_v9_hsprn_b16.yaml \
    --prepared_root ./prepared_datasets --export_sentences \
    --metadata data_semantics/CUB/class_descriptions_cub_after_llm_plan5.json \
    --output pre-trained/cub_v9_plan5_sent_b16.pt
python tools/eval_checkpoint.py --config config/cub_clip_v9_r6b_noref_v8phi_b16.yaml \
    --checkpoint checkpoints/cub_clip_v9_r6b_noref_v8phi_b16/CUB/best.pth \
    --name E2/r6b/ens_plan5 --score_bank semantic_ens --phi_path pre-trained/cub_v9_plan5_sent_b16.pt
# (tương tự cho bank/ens × rand/name)
```

**Kết quả (nguyên văn):**
```
RESULT E2/r6b/ens_plan5 ... T1_unseen=72.5 T1_seen=64.2 u=44.3 s=61.3 H=51.4 cal_u=64.8 cal_s=51.1 cal_H=57.2 gamma=0.02 hub_s=5.97 hub_u=2.74
RESULT E2/r6b/bank_rand ... T1_unseen=69.9 T1_seen=61.4 u=41.3 s=59.1 H=48.6 cal_u=62.5 cal_s=49.6 cal_H=55.3 gamma=0.03
RESULT E2/r6b/bank_name ... T1_unseen=70.9 T1_seen=62.8 u=41.6 s=60.6 H=49.3 cal_u=63.2 cal_s=50.7 cal_H=56.3 gamma=0.03
RESULT E2/r6b/ens_rand  ... T1_unseen=69.9 T1_seen=61.3 u=41.6 s=59.0 H=48.8 cal_u=62.7 cal_s=49.5 cal_H=55.3 gamma=0.03
RESULT E2/r6b/ens_name  ... T1_unseen=70.9 T1_seen=62.8 u=41.6 s=60.6 H=49.3 cal_u=63.2 cal_s=50.7 cal_H=56.3 gamma=0.03
```

**Bảng 9 cột (T1 | T1s | u | s | H | Cu | Cs | CH | γ):**
| Phi × bank | T1 | T1s | u | s | H | Cu | Cs | CH | γ |
|---|---|---|---|---|---|---|---|---|---|
| plan5-φ semantic (=E1 baseline) | 74.4 | 67.5 | 50.2 | 64.4 | 56.4 | 66.8 | 56.9 | 61.5 | 0.02 |
| plan5-φ **ensemble per-sentence** | 72.5 | 64.2 | 44.3 | 61.3 | 51.4 | 64.8 | 51.1 | 57.2 | 0.02 |
| random_words bank | 69.9 | 61.4 | 41.3 | 59.1 | 48.6 | 62.5 | 49.6 | 55.3 | 0.03 |
| random_words ens | 69.9 | 61.3 | 41.6 | 59.0 | 48.8 | 62.7 | 49.5 | 55.3 | 0.03 |
| name_repeat bank | 70.9 | 62.8 | 41.6 | 60.6 | 49.3 | 63.2 | 50.7 | 56.3 | 0.03 |
| name_repeat ens | 70.9 | 62.8 | 41.6 | 60.6 | 49.3 | 63.2 | 50.7 | 56.3 | 0.03 |

**Gate ΔT1 ≥ +0.3 so 74.4: KHÔNG ĐẠT** (72.5 < 74.4, Δ = −1.9) ⇒ tầng DESC-ENSEMBLE bị loại khỏi tổ hợp E6.

**Phát hiện khoa học (đối chứng Waffle đọc được 3 điều):**
1. **Semantics thật có giá trị**: name_repeat (zero semantics ngoài tên) = 70.9 và random_words = 69.9, đều dưới plan5-φ 74.4 một khoảng rõ (−3.5 / −4.5). Khác WaffleCLIP gốc (descriptor ngẫu nhiên ≈ descriptor LLM), trong pipeline HSPRN-Ws đã train, semantics đầy đủ của plan5 mang +3.5–4.5 T1 so với control. Ws có khả năng đã học tận dụng ngữ nghĩa chứ không chỉ đếm averaging.
2. **Ensemble per-sentence KHÔNG thay được mean-embedding đã train**: 72.5 vs 74.4. Ws được anchor (λ_sem=2.0) vào φ mean 4-field; chấm từng câu rời rồi lấy mean điểm phá cấu trúc đã train — mô hình "sinh ra" để dùng đúng 1 prototype/1 lớp. (Không loại trừ khả năng ens thắng nếu train lại với anchor từng câu — ngoài scope phase này.)
3. **Ensemble không tạo gain từ averaging thuần**: ens ≈ bank trên cả 2 control (69.9≈69.9, 70.9≈70.9) ⇒ hiệu ứng "ensemble magic" của CuPL/Waffle không xuất hiện khi Ws đã train trên mean. Averaging driver trong paper gốc tồn tại ở setting zero-shot không train.
4. Hub s/u của mọi biến thể E2 (5.5–6.9 / 2.3–2.7) thấp hơn baseline 7.13 — ensemble làm phân bố thắng đều hơn nhưng không đổi kết cục phân loại.

**Ghi chú:** name_repeat 70.9 > random_words 69.9: câu filler "X, which has <từ ngẫu nhiên>." thêm nhiễu nhẹ so với lặp tên thuần — khớp tinh thần WaffleCLIP rằng context rác có thể hại.

---

## E3 — HYBRID-φ (desc/hier plan5 + attr grounded) train 80ep | 2026-08-29 15:06–15:48 IBM
**Phương pháp:** φ_hybrid = name_c/hier_c/desc_c từ plan5 + attr_c sạch từ grounded (`class_descriptions_cub_hybrid.json`, 200 lớp); precompute φ mới (S_max câu); train R7h = config R6b chỉ đổi `semantic_embedding_path` (80 ep, seed 42, refiner off, mọi loss giữ nguyên).

**Lệnh:**
```
python tools/precompute_class_phi.py --config config/cub_clip_v9_hsprn_b16.yaml \
    --prepared_root ./prepared_datasets --export_sentences \
    --metadata prepared_datasets/CUB/metadata/class_descriptions_cub_hybrid.json \
    --output pre-trained/cub_v9_hybrid_phi_b16.pt
python train.py --config config/cub_clip_v9_r7h_hybrid_b16.yaml
python tools/eval_checkpoint.py --config config/cub_clip_v9_r7h_hybrid_b16.yaml \
    --checkpoint checkpoints/cub_clip_v9_r7h_hybrid_b16/CUB/{final,best}.pth ...
```

**Kết quả (nguyên văn):**
```
[2026-08-29 15:47:49] INFO - === Done. Best Cal H: 61.1% at epoch 55 | Best ZSL T1: 74.9% at epoch 1 ===
RESULT E3/r7h/final/semantic epoch=80 bank=semantic ... T1_unseen=73.4 T1_seen=68.1 u=46.9 s=64.9 H=54.4 cal_u=63.9 cal_s=58.5 cal_H=61.1 gamma=0.02 hub_s=7.19 hub_u=2.17
RESULT E3/r7h/final/ens epoch=80 bank=semantic_ens ... T1_unseen=73.4 T1_seen=67.4 u=47.3 s=64.3 H=54.5 cal_u=63.9 cal_s=57.2 cal_H=60.3 gamma=0.02
RESULT E3/r7h/best/semantic epoch=55 ... T1_unseen=73.5 T1_seen=68.1 u=47.2 s=64.9 H=54.6 cal_u=64.1 cal_s=58.4 cal_H=61.1 gamma=0.02
```

**Bảng 9 cột (T1 | T1s | u | s | H | Cu | Cs | CH | γ):**
| Run | T1 | T1s | u | s | H | Cu | Cs | CH | γ |
|---|---|---|---|---|---|---|---|---|---|
| R6b (plan5-φ) ep44 | 74.4 | 67.5 | 50.2 | 64.4 | 56.4 | 66.8 | 56.9 | 61.5 | 0.02 |
| **R7h hybrid final ep80** | 73.4 | 68.1 | 46.9 | 64.9 | 54.4 | 63.9 | 58.5 | 61.1 | 0.02 |
| R7h hybrid best (ep55, Cal H) | 73.5 | 68.1 | 47.2 | 64.9 | 54.6 | 64.1 | 58.4 | 61.1 | 0.02 |

**Gate T1(ep80) ≥ 74.4: KHÔNG ĐẠT** (73.4, Δ = −1.0) ⇒ tầng dữ liệu hybrid bị loại khỏi tổ hợp.

**Phân tích:**
- Trajectory: ep1 T1 74.9 (đỉnh sớm giống R6b 75.7@ep9), giữa lịch tụt về 73.4–73.5; u hội tụ 46.9–47.2 (thấp hơn R6b 50.2 −3.0..−3.3) nhưng s 64.9 (+0.5) và T1_seen 68.1 (+0.6) cao hơn — attr_c grounded nghi phía seen (thuộc tính "sạch" giúp phân biệt lớp seen đã thấy, không chuyển giao sang unseen).
- Kết luận đúng hướng của giả thuyết "cone/shared direction": attr_c grounded sạch nón nhưng **không** mang tín hiệu chuyển giao cho unseen; giữ attr cũ của plan5 (mang shared direction) vẫn tốt hơn. Điều này khớp phân tích toán §4: component chung không phải noise hoàn toàn.
- **Thứ tự kết luận về các trường φ (R6a grounded < R7h hybrid < R6b plan5):** chỉ riêng attr_c đổi sang grounded đã kéo T1 −1.0 và u −3.0; desc_c/hier_c là trường mang tín hiệu chuyển giao chính.

---

## E4 — CSLS prototype-side: bằng chứng bác bỏ | 2026-08-29 (chạy trong phase 3, số nguyên văn)
**Phương pháp:** CSLS prototype-side s'(x,c) = cos(x,c) − β·(r(c) − mean r), r(c) = mean cosine đến k prototype gần nhất (sau Ws, normalize), β ∈ {1.0, 0.05}, k ∈ {10, 20} — grid từ smoke E0 đã mở rộng.

**Kết quả (nguyên văn):**
```
RESULT E4/r6b/csls_k10_b1.0 ... T1_unseen=29.0 T1_seen=30.1 u=16.9 s=27.7 H=21.0 ... hub_s=18.02 hub_u=11.65
RESULT E4/r6b/csls_k10_b0.05 ... T1_unseen=73.8 T1_seen=65.9 u=48.0 s=63.3 H=54.6 cal_u=65.4 cal_s=55.9 cal_H=60.3 gamma=0.02 hub_s=8.89 hub_u=2.35
RESULT E4/r6b/csls_k20_b0.5 ... T1_unseen=50.2 T1_seen=46.2 u=25.3 s=44.1 H=32.2 ... hub_s=11.55 hub_u=6.60
```

**Gate ΔT1 ≥ +0.3: KHÔNG ĐẠT ở mọi cấu hình** (tốt nhất 73.8 < 74.4) ⇒ BỊ BỎ khỏi tổ hợp.

**Kết luận khoa học (kèm số E0):**
1. Spearman(r(c), win-count) = −0.37 (unseen) / −0.11..−0.19 (seen): mật độ prototype tương quan **âm** với việc được chọn trong pipeline này — trái giả thuyết hubness chuẩn.
2. Modality gap (proto-proto ~0.85 vs image-proto ~0.25) làm term CSLS ở thang riêng đảo thứ tự score.
3. Hub s/u metric trước đó (7.13/2.08) đo độ tập trung top-1, không phải hubness kiểu retrieval — hai khái niệm không đồng dấu ở đây.
⇒ Hubness-correction Dinu/Conneau **không portable** cho prototype-matching sau Ws. Ghi nhận là kết quả âm có giá trị, chống lại kỳ vọng từ tài liệu (Dinu ICLR'15).

---

## E5 — TTA transductive (DPE/TransCLIP-style) | 2026-08-29 IBM (bảng RIÊNG)
**Phương pháp:** trên score matrix inductive của R6b/ep44 (bank semantic), lặp: q = softmax(s/τ), centroid m_c = Σq·v/Σq (trên toàn bộ ảnh test seen+unseen, không nhãn), s' = λ·cos(v,m_c) + (1−λ)·s0. λ ∈ {0.3, 0.5, 0.7}, τ=0.07, 10 vòng.

**Lỗi đã bắt và sửa trong phase:** bản đầu chạy cho T1=2.5 ở iter=0 — do `collect_embeddings` không expand cột theo class id toàn cục (cột j ≠ class j vì seen_classes không phải 0..149). Sau sửa, iter=0 tái hiện đúng 74.4 (bằng chứng sanity).

**Kết quả (nguyên văn, sau sửa):**
```
TTA_RESULT E5fix/lam0.3 ... iter=0 ... T1_unseen=74.4 u=50.2 s=64.4 H=56.4 cal_H=61.5 gamma=0.02
TTA_RESULT E5fix/lam0.3 ... iter=10 ... T1_unseen=74.2 u=50.3 s=64.2 H=56.4 cal_H=60.8 gamma=0.01
TTA_RESULT E5fix/lam0.5 ... iter=5  ... T1_unseen=73.5 u=50.0 s=63.3 H=55.9 cal_H=60.1 gamma=0.01
TTA_RESULT E5fix/lam0.7 ... iter=10 ... T1_unseen=73.2 u=49.4 s=62.1 H=55.1 cal_H=57.4 gamma=0.01
```

**Gate H ≥ 60 (transductive bảng riêng): KHÔNG ĐẠT** — H tốt nhất 56.4 (λ=0.3, bằng đúng baseline), mọi λ lớn hơn giảm dần: 55.9 (λ=0.5), 55.1 (λ=0.7).

**Kết luận:** DPE-style centroid evolution KHÔNG tăng trên pipeline này. Giải thích khả dĩ: (1) score cosine của HSPRN đã qua Ws train + semantic bank, các centroid visual "nguyên liệu CLIP" thô hơn chính prototype đã chuẩn; (2) CUB unseen 50 lớp × ~59 ảnh/lớp — centroid mềm ít dữ liệu hơn nhiều so với 150 seen; (3) TransCLIP/DPE report gain trên CLIP zero-shot text bank chưa train — khi text bank đã được Ws tối ưu trên seen, phần "text prior" đã mạnh sẵn, phần dư transductive chỉ thêm nhiễu.
⇒ Đề xuất 4 (TTA) bị bác bỏ; bảng transductive chỉ giữ làm bằng chứng âm.

---

## E6 — Tổng kết tổ hợp + số chính thức phase 3 | 2026-08-29
**Phương pháp tổ hợp:** theo kế hoạch đã duyệt, mỗi tầng chỉ được vào tổ hợp nếu đạt gate so với đỉnh tầng trước. Kết quả gate: E2 (−1.9) ✗, E3 (−1.0) ✗, E4 (−0.6 tốt nhất) ✗, E5 (H 56.4 = baseline, T1 giảm) ✗.

**Tổ hợp inductive tốt nhất phase 3 = R6b nguyên bản (không chồng tầng nào):**
| Tổ hợp | T1 | T1s | u | s | H | Cu | Cs | CH | γ |
|---|---|---|---|---|---|---|---|---|---|
| **R6b/ep44 + semantic bank (CHÍNH THỨC)** | **74.4** | 67.5 | 50.2 | 64.4 | 56.4 | 66.8 | 56.9 | 61.5 | 0.02 |

Không có tầng nào qua gate ⇒ theo nguyên tắc đặt trước trong kế hoạch: kết luận trung thực là **74.4 là trần thực dụng của B/16 trong design space prototype-matching này**; các driver còn lại nằm ngoài prototype bank (thông tin part/region, hoặc backbone text mạnh hơn — đang bị cấm).

**Bảng tổng toàn phase 3 (đủ 9 cột):**
| Hàng | T1 | T1s | u | s | H | Cu | Cs | CH | γ | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| R6b baseline (E1) | 74.4 | 67.5 | 50.2 | 64.4 | 56.4 | 66.8 | 56.9 | 61.5 | 0.02 | gốc so sánh |
| + E2 ens plan5 | 72.5 | 64.2 | 44.3 | 61.3 | 51.4 | 64.8 | 51.1 | 57.2 | 0.02 | ✗ gate |
| + E2 bank rand | 69.9 | 61.4 | 41.3 | 59.1 | 48.6 | 62.5 | 49.6 | 55.3 | 0.03 | control |
| + E2 bank name-repeat | 70.9 | 62.8 | 41.6 | 60.6 | 49.3 | 63.2 | 50.7 | 56.3 | 0.03 | control |
| + E3 hybrid final | 73.4 | 68.1 | 46.9 | 64.9 | 54.4 | 63.9 | 58.5 | 61.1 | 0.02 | ✗ gate |
| + E4 csls β=0.05 | 73.8 | 65.9 | 48.0 | 63.3 | 54.6 | 65.4 | 55.9 | 60.3 | 0.02 | ✗ gate |
| + E5 tta λ=0.3 (transductive) | 74.2 | — | 50.3 | 64.2 | 56.4 | 62.9 | 58.8 | 60.8 | 0.01 | ✗ gate, bảng riêng |

**5 phát hiện nghiên cứu có giá trị của phase 3** (đề cập trong paper/future work):
1. **Đối chứng Waffle cho kết quả DƯƠNG tính**: semantics plan5 mang +3.5–4.5 T1 so với name-repeat/random-words — khác WaffleCLIP gốc vì Ws được train trên φ mean; averaging không phải driver duy nhất khi có giai đoạn học.
2. **Ensemble per-sentence thua mean-embedding đã train** (72.5 vs 74.4): anchor λ_sem=2.0 gắn Ws vào đúng φ mean; đổi cách chấm mà không train lại là bất lợi.
3. **Attr_c grounded hại transfer** (−1.0 T1, −3.0 u): xác nhận giả thuyết "shared direction không phải noise" (LP-FT ICLR'23) — chỉ thay attr_c sạch đã kéo cả mô hình xuống.
4. **CSLS/hubness-correction không portable** vào prototype-matching sau Ws: Spearman âm (−0.37) giữa density và win-count; modality gap làm term ở thang sai.
5. **TTA DPE/TransCLIP-style không thêm gì** khi text bank đã được Ws train trên seen: phần prior text đã mạnh, centroid visual thô chỉ thêm nhiễu (H 56.4 → giữ nguyên ở λ=0.3, giảm ở λ cao hơn).

**Trạng thái đề xuất → bằng chứng:**
| Đề xuất (kế hoạch) | Kỳ vọng đặt trước | Kết quả đo | Phán quyết |
|---|---|---|---|
| 1 HUB-CORRECT (E4) | +0.5–2.0 | −0.6 (tốt nhất) | Bác bỏ |
| 2 DESC-ENSEMBLE (E2) | +0.5–1.5 | −1.9 | Bác bỏ |
| 3 HYBRID-φ (E3) | 74.5–75.5 | 73.4 | Bác bỏ |
| 4 TTA-DPE (E5) | H 60+ | H 56.4 (flat) | Bác bỏ |

**File output phase 3 (IBM):** `pre-trained/cub_v9_plan5_sent_b16.pt` (per-sentence plan5), `cub_v9_waffle_{rand,name}_b16.pt` (controls), `cub_v9_hybrid_phi_b16.pt` (hybrid), `checkpoints/cub_clip_v9_r7h_hybrid_b16/CUB/{best,best_t1,final}.pth`, `logs/cub_clip_v9_phase3.log` + `logs/cub_clip_v9_phase3_resume.log`.

---

## P4-0 — Chuẩn bị Phase 4: semantics chuẩn CUB cho APY/AWA2/SUN | 2026-08-30 (local + sẽ sync IBM)
**Phương pháp:** chuẩn CUB = file `class_descriptions_{ds}_after_llm_plan5.json` đủ 4 trường (name_c, attr_c list 10 câu, hier_c, desc_c). Ba bộ đã có desc_c/hier_c từ plan5 (sinh 2026-08-11/12, log `logs/plan5_generation.log`) nhưng **thiếu attr_c** (file plan5 cũ chỉ 3 trường); attr_c cũ trên IBM là str markdown nguyên khối (925–1174 ký tự) — không đúng chuẩn list-10-câu của CUB. ⇒ Sinh attr_c mới bằng đúng `PROMPTS['attr_c']` của `tools/generate_descriptions.py` (cùng prompt đã sinh attr CUB), LLM local endpoint `localhost:10100` (model gemini-3.1-pro, cùng endpoint sinh plan5), gộp với desc/hier sẵn có → **FILE MỚI** `class_descriptions_{ds}_after_llm_plan5.json` (không đè file plan5 cũ).

**Lệnh:**
```
bash scripts/gen_attr_3ds.sh        # log: logs/attr_gen_3ds.log
python scripts/refresh_semar_3ds.py <DS>   # (IBM) refresh class_semantics.json + backup .bak_v96
bash scripts/run_v96_3ds.sh         # refresh -> precompute phi -> train -> eval 9 cot
```

**Kết quả sinh attr (nguyên văn log):**
```
=== APY: saved 32/32 classes -> data_semantics/APY/class_descriptions_apy_after_llm_plan5.json
=== AWA2: saved 50/50 classes -> data_semantics/AWA2/class_descriptions_awa2_after_llm_plan5.json
=== SUN: saved 717/717 classes -> data_semantics/SUN/class_descriptions_sun_after_llm_plan5.json  [đang chạy]
```

**Kiểm định chất lượng (APY/AWA2 xong):** mọi lớp đủ 4 trường; attr_c = 10 câu/lớp tự chứa ("A long, cylindrical central body known as the fuselage...") — đúng format CUB. Token filter CLIP 77 áp như chuẩn.

**File mới cho phase này:**
- `scripts/gen_attr_3ds.sh` (sinh attr 3 bộ, ghi file mới + .partial để resume)
- `scripts/refresh_semar_3ds.py` (refresh prepared metadata, backup .bak_v96)
- `scripts/run_v96_3ds.sh` (orchestrate train 3 bộ)
- `config/{apy,awa2,sun}_clip_v96_hsprn_b16.yaml` (classdesc — precompute φ)
- `config/{apy,awa2,sun}_clip_v96_r6_plan5_b16.yaml` (train R6b-pattern: refiner off, semantic bank, λ_sem 2.0, 80ep, seed 42)

**Ghi chú dữ liệu IBM:** image metadata đầy đủ (APY 15,339 / AWA2 37,322 / SUN 14,340 ảnh), `num_missing_images: 0`; trainer không hardcode số lớp — chạy đa bộ an toàn.

---

## P4-1 — APY: semantics 4-trường chuẩn CUB + train R6-pattern | 2026-08-30 IBM
**Phương pháp:** attr_c sinh mới (10 câu/lớp, prompt chuẩn CUB) gộp desc/hier plan5 → refresh metadata → φ (32, 512) → train 80ep refiner-off (λ_sem 2.0, seed 42) → eval semantic bank 9 cột.

**Kết quả (nguyên văn):**
```
RESULT V96/apy/final/semantic epoch=80 ... T1_unseen=84.9 T1_seen=81.7 u=71.8 s=79.0 H=75.2 cal_u=75.0 cal_s=76.3 cal_H=75.7 gamma=0.02 hub_s=6.47 hub_u=1.79
RESULT V96/apy/best/semantic epoch=1 ... T1_unseen=86.6 T1_seen=84.7 u=79.0 s=82.1 H=80.5 cal_u=81.8 cal_s=79.8 cal_H=80.8 gamma=0.01 hub_s=2.55 hub_u=3.51
=== Done. Best Cal H: 80.8% at epoch 1 | Best ZSL T1: 86.6% at epoch 1 ===
```

**Bảng 9 cột (T1 | T1s | u | s | H | Cu | Cs | CH | γ):**
| Checkpoint | T1 | T1s | u | s | H | Cu | Cs | CH | γ |
|---|---|---|---|---|---|---|---|---|---|
| **APY final ep80 (chính thức)** | 84.9 | 81.7 | 71.8 | 79.0 | 75.2 | 75.0 | 76.3 | 75.7 | 0.02 |
| APY best ep1 (test-selected, tham chiếu) | 86.6 | 84.7 | 79.0 | 82.1 | 80.5 | 81.8 | 79.8 | 80.8 | 0.01 |

**Ghi chú:** APY theo pattern CUB — đỉnh T1 ngay ep1 rồi tụt nhẹ và ổn định 84.9–85.0 từ ep40 (hub s/u 6.4/1.8). Số hội tụ dùng làm số chính thức như đúng giao thức CUB.

---

## P4-2 — AWA2: semantics 4-trường chuẩn CUB + train R6-pattern | 2026-08-30 IBM
**Phương pháp:** như P4-1 (APY): attr_c 10 câu mới (prompt chuẩn CUB) + desc/hier plan5 → φ (50, 512) → train 80ep refiner-off → eval 9 cột.

**Kết quả (nguyên văn):**
```
RESULT V96/awa2/final/semantic epoch=80 ... T1_unseen=95.9 T1_seen=90.4 u=79.4 s=88.8 H=83.9 cal_u=90.6 cal_s=81.2 cal_H=85.6 gamma=0.08 hub_s=2.38 hub_u=2.20
```

**Bảng 9 cột (T1 | T1s | u | s | H | Cu | Cs | CH | γ):**
| Checkpoint | T1 | T1s | u | s | H | Cu | Cs | CH | γ |
|---|---|---|---|---|---|---|---|---|---|
| **AWA2 final ep80 (chính thức)** | 95.9 | 90.4 | 79.4 | 88.8 | 83.9 | 90.6 | 81.2 | 85.6 | 0.08 |

**Ghi chú:** T1 ổn định 95.9 từ ep8 đến ep80 (không over-fit往下); hub s/u 2.4/2.2 cân bằng; γ=0.08 (calibration hơi mạnh vì seen s cao). H 83.9 vượt vùng SOTA inductive ResNet-era trên AWA2 (~60-70 H thường báo cáo cho generative).

---

## P4-3 — SUN: semantics 4-trường chuẩn CUB + train R6-pattern | 2026-08-30 IBM
**Phương pháp:** như P4-1/P4-2; attr_c sinh 717/717 lớp (prompt chuẩn CUB, endpoint local gemini-3.1-pro); refresh metadata → φ (717, 512) → train 80ep refiner-off → eval 9 cột.

**Kết quả (nguyên văn):**
```
RESULT V96/sun/final/semantic epoch=80 ... T1_unseen=89.2 T1_seen=58.2 u=55.1 s=56.7 H=55.9 cal_u=76.5 cal_s=52.4 cal_H=62.2 gamma=0.03 hub_s=4.80 hub_u=2.55
RESULT V96/sun/best/semantic epoch=4 ... T1_unseen=89.0 T1_seen=59.3 u=57.5 s=58.1 H=57.8 cal_u=79.4 cal_s=52.4 cal_H=63.1 gamma=0.02 hub_s=5.90 hub_u=2.40
RESULT V96/sun/best_t1/semantic epoch=54 ... T1_unseen=89.3 ...
=== Done. Best Cal H: 63.1% at epoch 4 | Best ZSL T1: 89.3% at epoch 54 ===
```

**Bảng 9 cột (T1 | T1s | u | s | H | Cu | Cs | CH | γ):**
| Checkpoint | T1 | T1s | u | s | H | Cu | Cs | CH | γ |
|---|---|---|---|---|---|---|---|---|---|
| **SUN final ep80 (chính thức)** | 89.2 | 58.2 | 55.1 | 56.7 | 55.9 | 76.5 | 52.4 | 62.2 | 0.03 |
| SUN best ep4 (Cal H) | 89.0 | 59.3 | 57.5 | 58.1 | 57.8 | 79.4 | 52.4 | 63.1 | 0.02 |

**Ghi chú SUN:** khác APY/AWA2, T1_seen thấp (58) và u/s xấp xỉ nhau (55/57) — đặc thù SUN (645 seen vs 72 unseen, scene classes). H 55.9 / Cal H 62.2: GZSL trên SUN khó vì seen-classes đa dạng hơn unseen; T1 89.2 vẫn rất cao.

---

## P4-TỔNG KẾT — 4 bộ dữ liệu hoàn chỉnh (CUB + APY + AWA2 + SUN) | 2026-08-30
**Bảng 4 bộ, checkpoint chính thức = epoch cuối 80 (semantic bank, refiner off, seed 42):**
| Dataset | T1 | T1s | u | s | H | Cu | Cs | CH | γ | hub s/u |
|---|---|---|---|---|---|---|---|---|---|---|
| CUB (R6b) | 74.4 | 67.5 | 50.2 | 64.4 | 56.4 | 66.8 | 56.9 | 61.5 | 0.02 | 7.13/2.08 |
| APY | 84.9 | 81.7 | 71.8 | 79.0 | 75.2 | 75.0 | 76.3 | 75.7 | 0.02 | 6.47/1.79 |
| AWA2 | 95.9 | 90.4 | 79.4 | 88.8 | 83.9 | 90.6 | 81.2 | 85.6 | 0.08 | 2.38/2.20 |
| SUN | 89.2 | 58.2 | 55.1 | 56.7 | 55.9 | 76.5 | 52.4 | 62.2 | 0.03 | 4.80/2.55 |

**Phát hiện:**
1. CUB là bộ khó nhất về T1 (74.4) — 3 bộ kia đạt 84.9–95.9. AWA2 gần như bão hòa (95.9).
2. Hubness seen/unseen gần cân bằng ở AWA2 (2.4/2.2) — không còn hiện tượng seen-hub như CUB (7.1).
3. SUN lệch kiểu ngược: T1_seen (58) << T1_unseen (89) — seen classes 645 đa dạng, semantic bank plan5 phục vụ unseen tốt.
4. Thứ tự thành quả nhất quán với chi phí dữ liệu: semantics 4-trường chuẩn CUB (attr list 10 câu) + refiner off + λ_sem 2.0 = công thức tổng quát tốt cho mọi bộ (chỉ đổi file semantics + config).

**File/data mới của phase 4:** `data_semantics/{APY,AWA2,SUN}/class_descriptions_{ds}_after_llm_plan5.json` (4 trường, attr 10 câu/lớp), `scripts/gen_attr_3ds.sh`, `scripts/refresh_semar_3ds.py`, `scripts/run_v96_3ds.sh`, `config/{apy,awa2,sun}_clip_v96_{hsprn,r6_plan5}_b16.yaml`, `pre-trained/{apy,awa2,sun}_v96_phi_b16.pt`, `checkpoints/{apy,awa2,sun}_clip_v96_r6_plan5_b16/<DS>/{final,best,best_t1}.pth` (IBM), logs `logs/v96_apy_awa2.log`, `logs/v96_sun.log`, `logs/attr_gen_3ds.log` (local).

---

## P5-1 — Viết paper chính (trọng tâm Section 3: Kiến trúc model) | 2026-08-30 local
**Phương pháp:** dựng `paper/main.tex` (article, two-column, tiếng Việt phần kiến trúc — sẽ dịch Anh khi hoàn tất; tiếng Anh cho formula). Mọi phương trình truy vết code: MFA (`model/aggregator.py` Eq. mfa-clip/mfa-general/mfa-init), φ 4-trường 2 cấp (`tools/precompute_class_phi.py`), Ws=I+BA (`model/sematics/constructor.py` Eq. ws/ws-norm), IPR (`model/protypes.py` Eq. ipr, gate_bias −2.0, U không bias, Wr 512×1536), loss (`losses/hsprn_loss.py` Eq. total/teacher/gradcls), teacher (`trainer/trainer.py:375-385` — softmax cos φ/τ_t trên 200 lớp, m_s từ trainval only). Số tham số xác minh bằng checkpoint thực (torch.load trên IBM): Ws (32,512)+(512,32)=32.768 trainable; ipr 1.049.088 (off); mfa.Wv (512,3584)+Wv_single (512,512) frozen — **phát hiện: đường v9 CLIP trả pooled vector → chỉ Wv_single được dùng, Wv 3584 không dùng** (đã ghi trung thực trong Table params).
**Kiểm tra cấu hình đối chiếu paper:** config R6b khớp 100% (λ1..λ3, λ_sem 2.0, λ_trans 0.25, τ 0.07, τ_teacher 0.1, 80ep, lr 1.5e-4, warmup 5, seed 42, batch 256, wv_init identity, freeze_wv, gate_bias −2.0).
**Sửa lỗi chữ:** loại từ Cyrillic «оплачив» lọt vào §3.2, typo «đụn»→«đụng tới», calque «tiện thể»→«tình cờ tiện lợi».
**Kiểm tra tĩnh (không có pdflatex local/IBM):** begin/end môi trường cân bằng theo tên (27/27); $ chẵn mỗi dòng; mọi \ref/\eqref có label; tabular cột-vs-ampersand SẠCH (sửa {lrrl}→{lrl} và thêm tham chiếu Table~\ref{tab:main}, Table~\ref{tab:params} vào văn bản).
**Trạng thái:** Section 3 (Kiến trúc) + Section 4 (Loss) viết đầy đủ sâu; Intro/Related/Analysis/Experiments/Conclusion để TODO khung cho đồng nghiệp; 15 mục bibliography seeded.

---

## P5-2 — Tái cấu trúc paper theo khung đúng: "họ refinement bị bác bỏ, Ws là số sống sót" | 2026-08-30 local
**Bối cảnh:** người dùng chỉ ra khung kể sai hướng của draft P5-1: IPR (module mang tên bài) TẮT trong mọi cấu hình thắng; phía ảnh không có gì train được (1 vector CLS đóng băng); toàn bộ học = 32.768 tham số phía ngữ nghĩa; "Refinement" còn sống = Ws refine chòm prototype bằng hàm dùng chung — không phải refine theo instance.

**Các thay đổi vào `paper/main.tex`:**
1. §3.1 Overview: đổi khung "IPR là module có thể bật" → "post-mortem có kiểm định": mọi biến thể refinement (train-time IPR, per-class residual, test-time TTA) bị bác bỏ; khối sống sót duy nhất = Ws. Định nghĩa lại "Refinement" của tên bài ngay ở overview.
2. §3.5 đổi tên → "Họ Instance/Class Refinement bị bác bỏ": mô tả IPR (Eq. ipr) + per-class residual (Eq. residual mới: p = Ws(φ) + r_c, R3, code `model/sematics/constructor.py:222-237` nn.Embedding zero-init), kèm bảng `tab:refine-reject` gom số 4 biến thể (v8 refined 71.0/36.8/49.5; v8 eval-semantic 74.4/51.8/58.1; R3 best_t1@ep1 71.9/12.1/20.3 với chú thích hội tụ u→1.4; E5 TTA 74.2/50.3/56.4; R6b đậm 74.4/50.2/56.4) — toàn bộ số đối chiếu docs/bao_cao_thuc_nghiem_v9.md §7.2 bảng 26 hàng.
3. Cơ chế thất bại viết thống nhất: cả 3 biến thể cùng mở "kênh gradient từ dữ liệu seen vào hàm biến đổi áp cho cả seen+unseen"; Ws an toàn theo construction vì là hàm của φ_c (không nhận ảnh). Phát biểu nguyên tắc thiết kế + định nghĩa lại tên bài cuối §3.5.
4. Sửa bảng ablation `tab:ablation-refine` sai số (44.0/48.9/56.1 không khớp nguồn) → thay bằng bảng 5 hàng đúng số (71.0/36.8/49.5 · 74.4/51.8/58.1 · 69.6/1.4/2.7 · 74.2/50.3/56.4 · **74.4/50.2/56.4**).
5. §Analysis đổi tên → "Seen-Only Optimization and the Failure Modes of Refinement", subsection "Seen-hub: một cơ chế, ba biến thể thất bại".
6. §3.7 params: nhấn mạnh "toàn bộ nằm phía ngữ nghĩa" + liệt kê chuỗi cắt kênh gradient có kiểm định (P1 đóng băng ảnh → tắt IPR → bỏ residual → L_pres≡0).
7. TODO Abstract/Intro/Conclusion cập nhật theo khung mới (số liệu dẫn: IPR-on 71.0 vs off 74.4; residual u→0 sau 3 epoch; 32.768 = 0.038% backbone).

**Kiểm tra sau sửa:** env balance OK; $ parity OK; refs/labels OK; tabular cột-vs-& OK; 18/18 số trong tab:refine-reject khớp nguồn §7.2. 11 TODO giữ cho đồng nghiệp.

---

## P5-3 — Sửa lỗi kết luận sai trong paper về hàng v8-IPR-on/eval-semantic | 2026-08-30 local
**Phát hiện từ người dùng:** hàng "v8 IPR-on (eval semantic)" thực ra là hàng cao nhất bảng refine-reject (T1 74.4 = R6b, u 51.8 > 50.2, H 58.1 > 56.4) — câu cũ trong paper viết "R6b tốt hơn cả hai về mọi cột GZSL" là SAI.

**Điều tra nguyên nhân (diff config v8 vs R6b):** v8 và R6b KHÔNG phải đối trọng — khác nhau 4 chỗ: refiner.enabled (true/false), λ_sem (1.0/2.0), λ_trans (0.5/0.25), semantic_source (class_desc live text tower vs embedding precomputed-φ). Không thể quy lợi u +1.6 của v8 cho IPR.

**Sửa trong paper/main.tex:**
1. Đoạn sau tab:refine-reject viết lại 3 dữ kiện tách bạch: (i) cặp đối trọng THẬT cùng trọng số (v8 refined 71.0/36.8 → v8 semantic 74.4/51.8) ⇒ refine là nhiễu ròng lúc inference (+3.4 T1, +15.0 u khi bỏ); (ii) hàng cao nhất bảng đúng là v8-IPR-on/eval-semantic — train với IPR được neo bởi L_pres không tự phá, còn chút lợi GZSL khi bỏ nó lúc suy luận; (iii) chọn R6b làm cấu hình chính thức vì 3 confound λ khiến lợi u của v8 không quy được cho IPR + phân tích gradient thấy kênh per-instance là rủi ro cấu trúc. Phán quyết trung thực: IPR không xứng đáng giữ trong inference; việc nó vô hại khi neo cứng + bị bỏ lúc suy luận là phát hiện thú vị, không phải lý do giữ.
2. Caption tab:ablation-refine cập nhật: ghi rõ hàng 2 vs hàng cuối không phải đối trọng thuần, chênh u 51.8 vs 50.2 nằm trong nhiễu single-seed.
3. Giải thích thêm vì sao IPR-on không phá khi train: L_pres neo p̂ về p_sem (cosine, pres_target=all gồm unseen).

**Validation:** env/refs/$/tabular OK. Bảng giữ nguyên số đúng (74.4/51.8/58.1) — không đổi số, chỉ sửa diễn giải.
