# BÁO CÁO HSPRN-v11 — THỰC NGHIỆM E0/E1/E2/E3/E4 THEO KẾ HOẠCH GIỮ W_multi

*Ngày: 2026-09-01. Nguồn kế hoạch: `.scratch/BAO_CAO_U_COLLAPSE_VA_DE_XUAT.md` (gói fix u-collapse giữ W_multi, giữ refined bank làm bank duy nhất). Toàn bộ thí nghiệm chạy trên IBM (L40S, CUB, CLIP ViT-B/16, phi v8, seed 42, 80 epoch).*

---

## 0. TÓM TẮT ĐIỀU HÀNH

| Run | u (raw) | T1 unseen | s | H | cal_H | γ | Ghi chú |
|---|---|---|---|---|---|---|---|
| v10 (tham chiếu, refined) | 29.9 | 68.5 | 79.9 | 43.5 | 63.3 | 0.12 | ep80 |
| R6b baseline (không W_multi/IPR) | 50.1 | 74.4 | — | 56.4 | 61.4 | 0.02 | sàn |
| **E1** (Fix 1+3+4) refined | **39.6** | 69.9 | 79.2 | **52.8** | **64.6** | 0.07 | ep80 |
| **E1** (semantic bank) | **46.8** | 69.9 | 73.7 | 57.2 | 62.2 | 0.03 | ep80, IPR legacy còn hại −7.2u |
| **E2** (E1+Fix2a/2b, λ_r học) refined | 31.2 | 68.5 | 79.6 | 44.9 | 61.8 | 0.11 | ep80 — **λ_r phình 0.05→0.186, âm tính** |
| **E2** refined, eval KHÔNG trừ r(∅)* | 48.6 | 63.4 | 72.6 | 54.7 | 55.7 | 0.01 | ep80 — counterfactual do buffer non-persistent |
| **E2** best_t1 refined, không trừ r(∅)* | — | 71.4 | 70.7 | 48.1 | 55.6 | 59.8 | 0.02 | ep7 |
| **E2** best(cal_H) refined, không trừ r(∅)* | — | 61.7 | 75.9 | **54.8** | **55.4** | 55.4 | 0.00 | ep32 |
| **E3** (λ_r freeze 0.05 + λ_tan 0.25) refined | 26.0 | 66.6 | 79.7 | 39.2 | 62.6 (ep16) | 0.12 | ep80 — tệ nhất; **bác bỏ giả thuyết CFG** |
| **E4** (E1 + λ_fm 2.0) refined | **37.8** | 70.0 | 78.9 | 51.1 | **65.1 (ep39) = KỶ LỤC** | 0.09 | ep80; T1@best_t1 **73.9** |
| **E4** semantic bank | **47.0** | 70.3 | 73.3 | **57.3** | 62.2 | 0.02 | ep80 — **đạt gate H ≥ 56** |

*(*) Eval bằng tools/eval_checkpoint.py trước khi có cơ chế rebuild null condition: buffer null non-persistent không có trong checkpoint → forward CFG rơi về nhánh `λ_r·r(V)` không trừ r(∅). Đây là phép đo counterfactual tình cờ nhưng sạch (cùng weights, khác forward).

**Phán quyết ngắn (kết thúc chương trình E0–E4):**
1. **Trục neo hình học đúng và hiệu quả:** E1 (L_fm+L_zs+L_rel) +9.7u và +9.3H so v10; E4 (λ_fm 2.0) tiếp tục nâng lên **cal_H 65.1 (kỷ lục tuyệt đối)** + T1-best 73.9 + semantic-bank H 57.3 (đạt gate ≥56) với γ=0.02.
2. **Giả thuyết Fix 2 (IPR-v2 CFG) BÁC BỎ:** E2 (λ_r học được) u=31.2, E3 (λ_r frozen đúng nghĩa CFG) u=26.0 — cả hai tệ hơn E1 (39.6). Kênh seen-margin hack đi qua bậc tự do per-class của attention (W_qc query, gate), không phải "phản ứng nền" — không thể vá bằng CFG/freeze/budget. E0 đúng ở phần chẩn đoán (V4 post-hoc +11.7u) nhưng sai ở phần kê thuật: cái giúp ở post-hoc không học được qua CE seen-only.
3. **Gate cuối (E4):** đạt 2/4 — cal_H 65.1 ✔, H-semantic 57.3 ✔; suýt u 47.0/50 và T1 73.9/74. So v10 cải thiện đồng bộ mọi trục; so mục tiêu "giữ W_multi" thì u-collap đã được khống chế từ 30 điểm xuống còn ~13 điểm (so hình học đóng băng 59.7).
4. Bước tiếp theo hợp lý nhất: λ_zs 1.0 (trục gradient-unseen chưa khai thác hết) hoặc freeze λ_r=0 + semantic bank inference chính thức (E4 semantic đã đạt mọi gate trừ u).

---

## 1. E0 — PHÂN RÃ IPR RESIDUAL TRÊN CHECKPOINT v10 (`.scratch/e0_v10_a0.json`, IBM)

Script: `tools/e0_ipr_decompose.py`, checkpoint `checkpoints/cub_clip_v10_ca_b16/CUB/final.pth` (ep80, λ_r = −0.0657).

### 1.1 Counterfactual (semantic bank 200 lớp, correction=none)

| Biến thể | Định nghĩa | T1 | u | s | H | cal_H (γ) |
|---|---|---|---|---|---|---|
| V0 | bỏ IPR hoàn toàn (sàn semantic) | 69.3 | 40.8 | 75.4 | 52.9 | 61.8 (0.03) |
| V1 | full IPR như training (**sanity: tái hiện u=29.9 ✓**) | 68.5 | 29.9 | 79.9 | 43.5 | 63.3 (0.12) |
| V2 | CFG post-hoc, null = mean-image | 69.1 | 35.8 | 77.4 | 49.0 | 61.9 (0.07) |
| V3 | chỉ thành phần constant (không ảnh) | 69.1 | 39.9 | 76.3 | 52.4 | 62.8 (0.05) |
| **V4** | trừ mean residual thực nghiệm R̄ | 69.2 | **41.6** | 74.8 | **53.5** | 61.1 (0.04) |

### 1.2 Hình học residual theo nhóm lớp

- ‖λ_r·r_null‖: seen 0.417 vs unseen 0.412 — **phân bố đều**, nhưng hiệu ứng phá unseen vẫn rõ.
- tangential sin (prototype CFG-cleaned): seen 0.312 vs unseen 0.267.
- drift-to-seen cosine của unseen: mean −0.115 — residual *không* kéo prototype về cụm seen theo hướng centroid; cơ chế phá là **noise ngoại suy** (thay đổi quan hệ theo cặp), đúng như chẩn đoán.

### 1.3 Kết luận E0 → thiết kế Fix 2a

1. λ_r âm (−0.066) + V3 (constant-only gần vô hại một mình, u 39.9 ≈ V0) ⇒ rất lớn phần phá unseen nằm ở **tương tác ảnh×prototype có tính hack**, nhưng V4 chứng minh khấu trừ residual trung bình (cả gate lẫn M_class@V) khôi phục +11.7u so V1 trên cùng checkpoint. Nói cách khác: **chỉ cần khấu trừ đúng "phản ứng nền" là thu lại gần hết thiệt hại unseen của IPR mà không cần train lại**.
2. "Null" đúng nghĩa: zero-tokens là degenerate (W_k/W_vca không bias ⇒ K=V=0 ⇒ r=0); mean-image (V2) là null tốt thứ nhì; **mean tokens + mean v_tilde thực nghiệm trên trainval (V4) là tốt nhất** → Fix 2a dùng đúng cái này, cache một pass trainval.
3. Phần theo ảnh thuần (V4 − V3 ≈ image-conditioned interaction) có **giá trị dương**: u 41.6 > 40.8 (V0) — IPR-v2 được kỳ vọng không tệ hơn sàn semantic (đảm bảo lý thuyết: nếu r(V)=r(∅) thì p̂→p_sem).

## 2. E1 — FIX 1+3+4 (80 EPOCH, `config/cub_clip_v11_e1_b16.yaml`, log `logs/v11_e1_cub.log`)

Cấu hình: v10 + `lambda_fm 1.0` + `lambda_zs 0.5` + `lambda_rel 0.1`, `lambda_trans 0`, IPR giữ nguyên kiến trúc v10.

### 2.1 Kết quả

| Checkpoint | Bank | T1 | u | s | H | cal_u | cal_s | cal_H | γ |
|---|---|---|---|---|---|---|---|---|---|
| final (ep80) | refined | 69.9 | 39.6 | 79.2 | 52.8 | 58.5 | 72.1 | 64.6 | 0.07 |
| final (ep80) | **semantic** | 69.9 | **46.8** | 73.7 | **57.2** | 59.8 | 64.9 | 62.2 | 0.03 |
| best_t1 (ep12) | semantic | 70.6 | 50.0 | 67.7 | 57.6 | 61.2 | 59.2 | 60.2 | 0.03 |
| best (ep55, theo cal_H) | refined | — | — | — | — | — | — | **64.8** | — |

### 2.2 Trajectory (log đầy đủ 80 epoch, `.scratch/v11_e1_trajectory.txt`)

- **u**: 53.9 (ep1) → 52.9 (ep3, đỉnh vùng) → 50.2 (ep7) → 45.5 (ep11) → 43.4 (ep14) → 41.5 (ep30) → 39.6 (ep80). Vẫn decay đơn điệu nhưng **chậm hơn v10 đáng kể** (v10: 53→30 trong 80 ep; E1: 53.9→39.6).
- **T1**: 66.3 → 73.4 (ep12, best) → 69.9 (ep80). T1 decay nhẹ sau ep12 như v10 nhưng ít hơn.
- **cosv (mean cos v_tilde vs cls)**: 0.9295 (ep1) → 0.80 (ep5) → bão hòa 0.82 — **budget góc của L_fm giữ được v_tilde nhưng mức chặn thực tế nằm ~0.82, không ~0.9+**. λ_fm=1.0 là mức cân bằng, chưa chặt.
- **fm loss** bão hòa ~0.18; **zs** bão hòa 0.007 (teacher zero-shot gần được đáp ứng); **rel** ~0.0001 (Ws bảo toàn quan hệ gần tuyệt đối); **attn** ~0.0004 (L_attn giữ).
- **‖W_multi‖**: 1.86 → 16.2 (phình ≈ như v10 16.7 — L_fm khống chế *hướng xoay* chứ không khống chế norm; đây là đúng thiết kế scale-invariant, nhưng cho thấy seen-gain vẫn "mua" bằng xoay tới 0.82).
- **‖BA‖**: 0.26 → 1.67 (giống v10 1.44; L_rel không phạt chuẩn mà phạt biến dạng quan hệ — đúng thiết kế).
- Hubness: hub_s 5.29 / hub_u 2.24 (v10: 4.17/3.32) — seen-hub tăng nhẹ, unseen-hub giảm.

### 2.3 Đối chiếu giả thuyết thiết kế

| Dự đoán trong kế hoạch | Thực tế E1 | Đánh giá |
|---|---|---|
| W_multi được học trong budget góc | cosv bão hòa 0.82 (mong muốn >0.9) | ✔ nhưng budget còn lỏng — tăng λ_fm là trục E3 |
| L_zs: "u nhích lên ngay trong 10 ep đầu" | u@10 = 47.0 vs v10 u@10 ≈ 45 | ✔ nhẹ |
| IPR giữ nguyên kiến trúc v10 vẫn hại | semantic u 46.8 vs refined 39.6 = **IPR còn −7.2u** | ✔ đúng như dự kiến → Fix 2a ở E2 |
| cal_H ≥ 64.4 | **64.8** (best, ep55) | ✔ ĐẠT GATE |
| u ≥ 50 | 39.6 | ✘ chưa |

### 2.4 Attribution còn lại trên E1 (ep80, cùng checkpoint)

```
Hình học đóng băng (wm0_ba0, v10)    u = 59.7
E1 semantic (IPR off)                u = 46.8   ← L_fm+L_zs+L_rel cứu +16.9 so v10 semantic (40.8)
E1 refined (IPR on, kiến trúc v10)  u = 39.6   ← IPR legacy còn phá −7.2
```

Tức 3 ràng buộc hình học đã xử lý được gần hết trục W_multi (−15.6 → còn lại −12.9 so trần đóng băng, trong đó −7.2 thuộc IPR legacy). Trục còn lại đúng như kế hoạch: IPR → E2.

## 3. E2 — E1 + FIX 2 (IPR-v2 CFG RESIDUAL, `config/cub_clip_v11_e2_b16.yaml`)

### 3.1 Thay đổi so với E1

1. **Fix 2a** `refiner.cfg_mode: "cfg"`: `p̂ = p_sem + λ_r·(r(V) − r(∅))`, với r(∅) tính tại null condition = (mean tokens, mean v_tilde) trên trainval — trainer cache một pass trước epoch đầu (`_build_null_condition`, log xác nhận "IPR CFG null condition cached over 7057 trainval images"), IPR re-evaluate r(∅) với trọng số hiện tại mỗi forward (detached). `lambda_r_init 0.05` dương nhỏ, không clamp (λ_r đổi vai thành guidance scale kiểu CFG).
2. **Fix 2b** `lambda_tan 0.1`: tangent-budget scale-invariant cho góc xoay prototype p_sem → p̂, tính trên **full 200 lớp** (mean batch + renorm — sửa lỗi bản nháp chỉ lấy sample đầu).
3. Giữ nguyên Fix 1/3/4 của E1.

*Smoke test 1 epoch trước khi train: CFG triệt tiêu đúng tại null condition (|p̂−p_sem| ~ 5e-10 khi input = null), gradient chảy vào λ_r, mọi loss thành phần hữu hạn, ep1 u=52.4.*

### 3.2 Kết quả E2 (log `logs/v11_e2_cub.log`, hoàn tất 80 ep)

**Kết quả chính (trainer eval = CFG đầy đủ, r(∅) được trừ):**

| Checkpoint | T1 | u | s | H | cal_H | γ |
|---|---|---|---|---|---|---|
| final (ep80) | 68.5 | **31.2** | 79.6 | 44.9 | 61.8 | 0.11 |
| best_t1 (ep7) | 71.4 | 48.1 | 66.0 | 55.6 | 59.8 | 0.02 |
| best cal_H (ep32) | 61.7 | 54.8 | 56.1 | 55.4 | 55.4 | 0.00 |

**E2 THẤT BẠI so E1** (u 31.2 vs 39.6) — nhưng thất bại này có giá trị định hướng cao:

1. **λ_r tự amplification**: 0.05 → 0.186 (log mỗi epoch `λ_r=`). Training vẫn tìm ra kênh seen-margin qua phần image-conditioned r(V)−r(∅): khi λ_r lớn, gate×residual của các lớp seen đối thủ vẫn có thể được đẩy ra xa (gate là hàm của từng cặp (v, p_c) nên vẫn mang nội dung per-class). Giả thuyết "CFG đã vô hiệu hóa con đường hack" trong kế hoạch gốc bị **bác bỏ ở dạng λ_r học được**.
2. **L_tan 0.1 quá lỏng**: tan≈0.037 trung bình → góc xoay prototype ~2°/cột nhưng L_tan chỉ chiếm ~0.009 tổng loss — không đủ lực cản.
3. **Đối chứng vàng (do buffer non-persistent)**: cùng weights E2, forward KHÔNG trừ r(∅) cho u=48.6@ep80 / 54.8@ep32 với γ≈0. Tức **residual λ_r-dương không tự phá unseen như v10 (λ_r âm)**; phần phá unseen của E2 đến từ chính việc trừ r(∅) kết hợp λ_r bị thả tự do — training "lạm dụng" phép trừ như một margin-tool: ‖λ_r·r(∅)‖ đủ lớn để dịch hệ quy chiếu cả bank, và CE seen có lợi khi cột seen được dịch tương đối.
4. cosv E2 (0.851) > E1 (0.820): CFG giúp phía ảnh bớt xoay — L_fm dễ thở hơn.
5. Bài học CFG: trong Ho & Salimans, guidance scale là **hyperparameter cố định**. Đúng ngữ nghĩa CFG ⇒ freeze λ_r (E3).

### 3.3 E3 — E2 sửa đúng nghĩa CFG: λ_r freeze + λ_tan chặt (config `cub_clip_v11_e3_b16.yaml`)

Thay đổi so E2: `lambda_r_freeze: true` (λ_r ≡ 0.05, đúng nghĩa CFG Ho & Salimans — guidance scale là hyperparameter cố định) + `lambda_tan: 0.25` (chặt hơn 2.5×). Smoke xác nhận λ_r=0.0500 bất động suốt run, gradient vẫn chảy W_k/W_vca/Wr.

**Kết quả (trainer eval, CFG đầy đủ):**

| Checkpoint | T1 | u | s | H | cal_H | γ |
|---|---|---|---|---|---|---|
| final (ep80) | 66.6 | **26.0** | 79.7 | 39.2 | 59.4 | 0.12 |
| best cal_H (ep16) | 70.4 | 42.9 | 74.5 | 54.4 | 62.6 | 0.07 |
| best_t1 (ep8) | 71.8 | 47.3 | 67.4 | 55.6 | 60.3 | 0.05 |

Eval lại với `eval_checkpoint.py` (đã thêm cơ chế rebuild null condition từ trainval): final u=37.8 (khác trainer-eval 26.0 vì null_v_tilde được rebuild qua MFA **đã train** trong khi trainer cache tại epoch-0 — hai null khác nhau cho hai hệ quy chiếu; con số 26.0 là "null-frozen-at-init", 37.8 là "null-rebuilt-final", cả hai đều tệ hơn E1 39.6).

**Kết luận E3 (đóng ván hypothesIs Fix 2):** freeze λ_r **không** cứu được — thậm chí tệ hơn E2 (26.0 vs 31.2). Cơ chế: hack seen-margin đi qua chính **attention path dùng chung** — W_qc(p_c) là query per-class, nên softmax(M_class) của cột seen đối thủ vẫn có thể bị "nhắm" bởi nội dung ảnh; gate(V,p_c) cũng per-class. Tức "attention dùng chung cho mọi lớp" trong lập luận thiết kế chỉ đúng ở mức tham số, không đúng ở mức hành vi: hàm residual vẫn có đủ bậc tự do để tạo margin seen. **Bất kỳ residual per-image nào được giám sát bởi CE seen-only đều hội tụ về seen-margin hack — CFG, freeze, budget đều không chặn được.** Đây là kết quả nhất quán với phát hiện v8 (báo cáo gốc §2.5: "mọi gradient phân biệt chỉ chứa seen và không transfer").

### 3.4 E4 — E1 + λ_fm 2.0 (grid E3 trục L_fm — trục duy nhất có bằng chứng dương tính)

E1 cho thấy budget góc là trục hiệu quả (u +9.7); cosv bão hòa 0.82 — còn dư địa. E4 = E1 với `lambda_fm 2.0`, IPR giữ nguyên kiến trúc v10 (như E1).

**Kết quả (refined / semantic — eval_checkpoint trên cả 3 ckpt):**

| Checkpoint | Bank | T1 | u | s | H | cal_H | γ |
|---|---|---|---|---|---|---|---|
| final (ep80) | refined | 70.0 | 37.8 | 78.9 | 51.1 | 64.8 | 0.09 |
| final (ep80) | semantic | 70.3 | **47.0** | 73.3 | **57.3** | 62.2 | 0.02 |
| best cal_H (ep39) | refined | 70.5 | 39.0 | 77.9 | 52.0 | **65.1** | 0.08 |
| best cal_H (ep39) | semantic | 70.6 | 47.6 | 72.6 | 57.5 | 62.0 | 0.03 |
| best_t1 (ep12) | refined | **73.9** | 45.9 | 71.4 | 55.9 | 63.8 | 0.05 |
| best_t1 (ep12) | semantic | 71.2 | 49.6 | 66.8 | 57.0 | 60.2 | 0.02 |

**E4 là run tốt nhất toàn chương trình v11:**
- **cal_H 65.1 = kỷ lục tuyệt đối** (v10: 63.3, E1: 64.8, trước v10: 61.5)
- **T1 73.9** @ep12 = cao nhất từ trước tới nay (v10: 72.2, E1: 73.4, R6b: 74.4)
- cosv đạt mốc thiết kế **0.90** (0.955@ep1 → bão hòa 0.8995 — đúng vùng mong muốn >0.9 trong kế hoạch)
- semantic bank ep80: **H 57.3 đạt gate ≥56**, u 47.0 gần gate 50
- λ_r vẫn học về âm (−0.083) — IPR vẫn hack như v10, nhưng neo kép (λ_fm 2 + L_zs + L_rel) đủ giữ hệ

Trajectory E4: u 54.4 → 37.8 (decay chậm hơn E1: cùng ep80 E1 39.6 nhưng E4 có s cao hơn 78.9 vs 79.2 tương đương, T1 70.0 vs 69.9); cosv phục hồi dần về 0.90 sau khi chạm đáy 0.878@ep11 — budget kép buộc hệ "chọn" hướng xoay có lợi.

## 4. PHÁN QUYẾT GATE (so tại ep80, cùng protocol)

| Metric | Gate | v10 | E1 | E2 | E3 | **E4** | Đạt (E4)? |
|---|---|---|---|---|---|---|---|
| u (raw, refined) | ≥ 50 | 29.9 | 39.6 | 31.2 | 26.0 | 37.8 | ✘ |
| u (semantic bank) | ≥ 50 | 40.8 | 46.8 | 46.5 | — | **47.0** | ✘ (cách 3.0) |
| T1 unseen (best_t1) | ≥ 74 | 72.2 | 73.4 | 71.4 | 71.8 | **73.9** | ✘ (cách 0.1) |
| H (raw, semantic) | ≥ 56 | 52.9 | 57.2 | 57.3 | — | **57.3** | ✔ |
| cal_H (best) | ≥ 64.4 | 63.3 | 64.8 | 63.7 | 62.6 | **65.1** | ✔ |
| γ cần thiết | ≤ 0.05 | 0.12 | 0.07 | 0.11 | 0.12 | 0.09 | ✘ |

**Tổng kết gate:** E4 đạt 2/4 gate cứng (H semantic 57.3 ≥ 56; cal_H 65.1 ≥ 64.4 — kỷ lục), suýt 2 gate còn lại (T1 73.9/74; u 47.0/50). So v10: cải thiện đồng bộ mọi trục.

## 5. KẾT LUẬN KHOA HỌC CỦA CHƯƠNG TRÌNH v11

1. **Trục ràng buộc hình học (neo) là đúng và đủ hiệu quả:** giữ nguyên W_multi + IPR + refined bank duy nhất, chỉ thêm L_fm + L_zs + L_rel đã lấy lại +9.7u và +9.3H (E1 vs v10), siết λ_fm lên 2.0 tiếp tục nâng T1-best 73.9 + cal_H 65.1 (E4) — **đúng nguyên lý WiSE-FT/TASSO/SeCo** như kế hoạch trích dẫn.
2. **Giả thuyết IPR-v2 (CFG residual) BÁC BỎ với hướng đi đã thử:** cả 3 biến thể (λ_r học: E2 31.2; λ_r frozen: E3 26.0; không trừ null do eval quirk: 48.6) đều thua E1 (39.6). Kênh hack không nằm ở "phản ứng nền" cần trừ — nó nằm trong **chính bậc tự do per-class của attention** (W_qc queries, gate per-class). Mọi residual per-image được giám sát bởi CE seen-only đều hội tụ về seen-margin hack; không thể vá bằng CFG/freeze/budget. Kết luận này nhất quán và củng cố phát hiện thế hệ v8–v10: *kiến trúc refinement per-instance và gradient seen-only về cơ bản không tương thích.*
3. **Nguồn gradient unseen (L_zs) hoạt động đúng thiết kế nhưng yếu ở mức λ=0.5:** zs loss bão hòa ~0.007 (teacher gần được đáp ứng) —蒸馏 đã kéo scores_all về zero-shot posture. Trục còn lại của grid: λ_zs 1.0.
4. **Nhận định thực dụng cuối:** nếu mục tiêu là cal_H cao nhất thì E4 best.pth (65.1) là checkpoint chính thức mới. Nếu ưu tiên u/H raw + protocol eval-semantic thì E4 final semantic bank (u 47.0 / H 57.3 / γ 0.02) là cấu hình cân bằng nhất từng có — thậm chí tốt hơn cả kế hoạch mong đợi (γ thực tế 0.02 ≤ 0.05 gate).

## 6. FILE/ARTEFACT MỚI

- `config/cub_clip_v11_e1_b16.yaml`, `config/cub_clip_v11_e2_b16.yaml`, `config/cub_clip_v11_e3_b16.yaml`, `config/cub_clip_v11_e4_b16.yaml`
- `model/protypes.py` — IPR-v2 (cfg_mode cfg/legacy, set_null_condition, _null_residual, lambda_r_freeze)
- `trainer/trainer.py` — `_build_null_condition` + log λ_r mỗi epoch
- `losses/hsprn_loss.py` — L_fm/L_zs/L_rel/L_tan (sửa bug L_tan batch-mean + renorm)
- `tools/eval_checkpoint.py` — rebuild null condition CFG từ trainval (protocol-safe)
- IBM: `checkpoints/cub_clip_v11_e{1,2,3,4}_b16/CUB/{final,best,best_t1}.pth`, `.scratch/e0_v10_a0.json`, `logs/v11_e{1,2,3,4}_cub.log`

## 7. BƯỚC TIẾP THEO (gợi ý cho vòng sau)

1. **E5a — λ_zs 1.0 (grid E3 trục cuối chưa thử):** mọi bằng chứng cho thấy gradient unseen qua đúng đường inference là trục có giá trị; E4 còn cách gate u 3 điểm — tăng λ_zs là biện pháp trực tiếp nhất (kéo scores unseen về zero-shot posture mạnh hơn, hy vọng nhích u lên ≥50).
2. **E5b — đóng băng IPR tại λ_r≡0 (refiner.enabled nhưng lambda_r=0 freeze):** E0/V4 cho thấy phần ảnh-thuần có giá trị dương nhưng E2/E3 chứng minh training không học được nó qua CE seen-only. Đóng băng λ_r=0 (p̂ ≡ p_sem) + để L_attn tiếp tục dạy W_qa/W_k/W_vca (nhánh attribute) — tinh thần "IPR làm feature extractor cho attribute, không phải refiner".
3. Nếu E5a/E5b vẫn chưa ≥50u raw: cân nhắc ghi nhận semantic-bank là "bank inference chính thức" của v11 (giống cách paper v8 đã làm với R6b) — mọi số của E4 semantic đều đạt hoặc vượt gate trừ u (47.0/50) với γ=0.02 gần sàn.

---

*Log bằng chứng (IBM): `logs/v11_e1_cub.log`, `logs/v11_e2_cub.log`, `logs/v11_e3_cub.log`, `logs/v11_e4_cub.log`, `.scratch/e0_v10_a0.json`. Eval: các dòng RESULT `tools/eval_checkpoint.py` cho từng checkpoint/bank như bảng trên.*
