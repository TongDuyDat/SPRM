# BÁO CÁO — HSPRN v10: Sync IBM + E0 + E1 + E2-4a (APY/AWA2/SUN theo yêu cầu user)

Ngày: 31-08-2026 · WSL → IBM (ubuntu@128.168.135.246, 1× L40S) · Seed 42 · Log train: `logs/v10_cub_a0.log`

> **Lưu ý:** E1 CUB fail cổng T1 (72.6 < 74.7). User chỉ đạo trực tiếp "train APY/AWA2/SUN" (bước 4a) — chạy tiếp phần 4a với phương pháp mới v10 như yêu cầu, bỏ qua cổng chặn E2.

## 1. E0 — **PASSED** (all checks green, exit 0)

Output đầy đủ các dòng `[PASS]/[FAIL]` (verify lại lần 2 sau khi code fix, cùng lệnh `python tools/e0_sanity_v10.py --baseline config/cub_clip_v9_r6b_noref_v8phi_b16.yaml --v10 config/cub_clip_v10_ca_b16.yaml --device cuda`):

```
=== E0.1: attribute metadata (./prepared_datasets) ===
[PASS] metadata CUB — attributes (200, 312), names 312, expected n_attr 312
[PASS] metadata APY — attributes (32, 64), names 64, expected n_attr 64
[PASS] metadata AWA2 — attributes (50, 85), names 85, expected n_attr 85
[PASS] metadata SUN — attributes (717, 102), names 102, expected n_attr 102
=== E0.2/E0.3: build fresh v10 + R6b models ===
=== E0.2: extract_levels_anchored vs backbone(x) ===
[PASS] extract_levels_anchored shapes — tokens [(32, 197, 768), (32, 197, 768), (32, 197, 768)], cls (32, 512)
[PASS] cls == backbone(x) — max|d| = 0.00e+00
=== E0.3: bit-exact epoch-0 anchor (v10 vs R6b) ===
[PASS] epoch-0 anchor: scores — max|d| = 0.00e+00
[PASS] epoch-0 anchor: scores_sem — max|d| = 0.00e+00
[PASS] epoch-0 anchor: v_embed — max|d| = 0.00e+00
=== E0.4: L_attn + 10 optimizer steps ===
[PASS] GT attribute matrix rows == classes — gt (200, 312) vs phi (200, 512)
[PASS] labels <-> GT attribute row — label 196: 167 active attributes (row 167 vs other 171)
  trainable tensors: 11, total numel 3,310,593
  loss_attn trajectory (10 steps, same batch): 0.00088 0.00087 0.00084 0.00081 0.00076 0.00072 0.00074 0.00071 0.00067 0.00066
[PASS] L_attn present and > 0 — step0 = 0.0008837858913466334
[PASS] L_attn finite
[PASS] L_attn decreases over 10 steps — 0.00088 -> 0.00066
E0 PASSED — all checks green.
```

- Sync md5: **24/24 file khớp** bảng §2 (kể cả 2 file fix mới: `model/vision/__init__.py` = `26dc1e51…`, `tools/e0_sanity_v10.py` = `fb191ef8…`).
- 4 attrq bank có sẵn từ lần sync trước: `pre-trained/{cub,apy,awa2,sun}_attrq_b16.pt`.

## 2. Bảng kết quả

E1 (A0 CUB) + E2-4a (A0 APY/AWA2/SUN — theo chỉ đạo trực tiếp của user sau khi E1 fail cổng).

| dataset | arm | config | ckpt dùng | T1 (ZSL) | u | s | H (GZSL) | seed | ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| CUB | A0 | cub_clip_v10_ca_b16.yaml | best_t1.pth (epoch 9) | **72.6** | 49.1 | 69.9 | 57.7 | 42 | Best T1 epoch 9; best Cal H 63.2 @ep14; @ep80: T1 68.4 / u 39.9 / s 78.5 / H 52.9 |
| CUB | A0 | cub_clip_v10_ca_b16.yaml | best.pth (epoch 14) | 71.5 | 43.7 | 73.5 | 54.8 | 42 | Best Cal H = 63.2 @ep14 (Cal u 63.0 s 63.5) |
| APY | A0 | apy_clip_v10_ca_b16.yaml | best_t1.pth (epoch 2) | **83.6** | 64.7 | 79.9 | 71.5 | 42 | Best T1 = best Cal H cùng ep2 (Cal H 74.8); @ep80: T1 76.3 / u 37.6 / s 84.6 / H 52.1 |
| AWA2 | A0 | awa2_clip_v10_ca_b16.yaml | best_t1.pth (epoch 8) | **97.0** | 58.5 | 95.8 | 72.6 | 42 | Best T1 @ep8 vượt cổng 96.2; best Cal H 91.7 @ep11 (Cal u 94.1 s 89.3); @ep80: T1 94.4 / u 10.5 / s 96.2 / H 19.0 |
| SUN | A0 | sun_clip_v10_ca_b16.yaml | best_t1.pth (epoch 20) | **88.3** | 47.8 | 65.2 | 55.1 | 42 | Best T1 @ep20 (Cal H 67.6); best Cal H 68.7 @ep58 (T1 87.6, Cal u 77.5 s 61.7); @ep80: T1 87.6 / u 45.1 / s 67.5 / H 54.1 |

Trajectory T1/u theo epoch (trích log):

**CUB:**
```
ep   1: T1 64.1  u 53.2   ep   9: T1 72.6  u 49.1  ← BEST_T1
ep   3: T1 68.3  u 53.3   ep  14: T1 71.5  u 43.7  ← BEST (Cal H)
ep   5: T1 70.7  u 52.2   ep  20: T1 69.9  u 43.9
ep   7: T1 72.4  u 51.0   ep  29: T1 69.9  u 41.7
ep  40: T1 68.6  u 40.3   ep  80: T1 68.4  u 39.9 (s 78.5, H 52.9)
```

**APY:**
```
ep 1: T1 83.1  u 62.8 (s 73.7)   ep 2: T1 83.6  u 64.7 (s 79.9)  ← BEST_T1 + BEST Cal H
ep 5: T1 79.4  u 50.4            ep 10: T1 76.4  u 46.4
ep 20: (xem log)                  ep 80: T1 76.3  u 37.6 (s 84.6, H 52.1)
```

**AWA2:**
```
ep 1: T1 91.0  u 59.6   ep 3: T1 96.7  u 62.4 (s 93.8, H 74.9)
ep 5: T1 96.5  u 60.1   ep 8: T1 97.0  u 58.5 (s 95.8, H 72.6)  ← BEST_T1 (PASS cổng 96.2)
ep 11: T1 96.9  u 52.2  (Cal H 91.7 ← BEST Cal)   ep 20: T1 95.3  u 26.7
ep 80: T1 94.4  u 10.5  (s 96.2, H 19.0 — unseen sụp về cuối)
```

**SUN:**
```
ep 1: T1 86.7  u 52.8 (s 53.1)   ep 7: T1 87.1  u 49.4 (s 60.3)   ep 8: T1 87.6  u 49.9
ep 20: T1 88.3  u 47.8 (s 65.2, H 55.1, Cal H 67.6)  ← BEST_T1
ep 58: T1 87.6  u 45.2 (s 67.9, H 54.3, Cal H 68.7)  ← BEST Cal H
ep 80: T1 87.6  u 45.1 (s 67.5, H 54.1)
```

## 3. Phán quyết cổng

| Dataset | Baseline T1 (R6b) | Cổng T1 | Best T1 (epoch) | Δ so baseline | Phán quyết T1 |
|---|---|---|---|---|---|
| CUB | 74.4 | ≥ 74.7 | **72.6** (ep 9) | **−1.8** | **FAIL** (thiếu 2.1 so cổng) |
| APY | 84.9 | ≥ 85.2 | **83.6** (ep 2) | **−1.3** | **FAIL** (thiếu 1.6 so cổng) |
| AWA2 | 95.9 | ≥ 96.2 | **97.0** (ep 8) | **+1.1** | **PASS** ✅ |
| SUN | 89.2 | ≥ 89.5 | **88.3** (ep 20) | **−0.9** | **FAIL** (thiếu 1.2 so cổng) |

- Cổng u (u ≥ baseline_u − 1.0): baseline_u của APY/AWA2/SUN không có log R6b trên IBM (chỉ CUB có). @best-T1: CUB u=49.1 (đạt 49.1), APY u=64.7, AWA2 u=58.5, SUN u=47.8 — không chấm chính thức được phần u cho 3 dataset vì thiếu baseline_u.
- **Pattern chung cả 4 dataset**: T1 đạt đỉnh rất sớm (CUB ep9, APY ep2, AWA2 ep8, SUN ep20) rồi giảm/bão hòa; u giảm liên tục trong khi s tăng — model lệch dần về seen classes theo epoch. AWA2 cực đoan nhất: u 62.4 → 10.5 ở ep80.
- Lưu ý: best T1 thường rơi vào/khoảng warmup (5 epochs) — LR schedule/warmup tương tác với các loss term mới (L_attn, L_sem) là nghi vấn chính.

## 4. Log loss_attn

- **Trong log train: `loss_attn` KHÔNG được in ra** — trainer chỉ log `sem`, `wsanchor`, `wvanchor`, `||BA||` vào dòng epoch, và postfix `cls/align/pres/lr` trên progress bar. Đây là thiếu sót logging, không phải lỗi train (loss_attn có trong loss_dict, chỉ không được ghi).
- Giá trị thay thế từ E0.4 (10 optimizer steps, cùng batch, λ_attn=0.3): **đầu 0.00088 → cuối 0.00066** (giảm đều, finite, trajectory đầy đủ ở mục 1).
- Tham chiếu từ log train CUB A0: `sem` 0.0014 → 0.0341, `wsanchor` 0.0002 → 0.0251, `||BA||` 0.2580 → 1.5849 (epoch 1 → 80).

## 5. Vấn đề gặp phải

1. **[CHÍNH] v10 A0 thua baseline ở 3/4 dataset** (CUB −1.8, APY −1.3, SUN −0.9; chỉ AWA2 +1.1). T1 đạt đỉnh sớm rồi giảm; u giảm liên tục trong khi s tăng → unseen bị đánh đổi cho seen trong suốt training. KHÔNG sửa config/code theo §5 — người phụ trách chính quyết định (nghi vấn: λ_attn/λ_sem, LR/warmup schedule, tương tác MFA–CLS-anchor).
2. `[minor]` Loss_attn không được ghi vào log train (mô tả mục 4) — nếu cần trajectory thật của run thì phải thêm logging (do người phụ trách chính quyết).
3. `[minor]` Chuỗi train lần đầu tưởng fail vì ssh command timeout — thực ra nohup vẫn chạy tốt trên IBM; các lần sau viết thành file script cho chắc.
4. Không NaN/crash/OOM (SUN 645 lớp chạy bình thường ~27s/epoch). Sync md5 24/24.

## 6. Checkpoint sinh ra (đường dẫn đầy đủ trên IBM)

- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v10_ca_b16/CUB/best_t1.pth` (384,574,903 B — epoch 9, best T1)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v10_ca_b16/CUB/best.pth` (384,563,225 B — epoch 14, best Cal H)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v10_ca_b16/CUB/final.pth` (358,070,311 B — epoch 80)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/apy_clip_v10_ca_b16/APY/best_t1.pth` (384,574,903 B — epoch 2, best T1)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/apy_clip_v10_ca_b16/APY/best.pth` (384,563,225 B — epoch 2, best Cal H)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/apy_clip_v10_ca_b16/APY/final.pth` (358,070,311 B — epoch 80)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/awa2_clip_v10_ca_b16/AWA2/best_t1.pth` (384,574,903 B — epoch 8, best T1)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/awa2_clip_v10_ca_b16/AWA2/best.pth` (384,563,225 B — epoch 11, best Cal H)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/awa2_clip_v10_ca_b16/AWA2/final.pth` (358,070,311 B — epoch 80)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/sun_clip_v10_ca_b16/SUN/best_t1.pth` (384,574,903 B — epoch 20, best T1)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/sun_clip_v10_ca_b16/SUN/best.pth` (384,563,225 B — epoch 58, best Cal H)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/sun_clip_v10_ca_b16/SUN/final.pth` (358,070,311 B — epoch 80)
- Log train: `logs/v10_cub_a0.log`, `logs/v10_apy_a0.log`, `logs/v10_awa2_a0.log`, `logs/v10_sun_a0.log`, `logs/v10_chain_3ds.log`

## Kết luận

- E0 **PASSED** (14/14 checks, anchor bit-exact 0.00e+00, loss_attn giảm 0.00088→0.00066).
- Train A0 v10 hoàn tất cả 4 dataset (80 epochs/run, không NaN/crash/OOM):
  - **CUB 72.6** (FAIL cổng 74.7, −1.8 vs baseline 74.4)
  - **APY 83.6** (FAIL cổng 85.2, −1.3 vs baseline 84.9)
  - **AWA2 97.0** (PASS cổng 96.2 ✅, +1.1 vs baseline 95.9)
  - **SUN 88.3** (FAIL cổng 89.5, −0.9 vs baseline 89.2)
- **1/4 dataset qua cổng.** Pattern nhất quán: T1 đỉnh sớm (ep 2–20) rồi giảm, u suy giảm liên tục — nghi vấn về lịch LR/warmup hoặc λ_attn/λ_sem.
- Chưa chạy: ablation A2/A3 (4b) và A1 eval-only (4c).
- Trạng thái IBM khi kết thúc: sạch (không process, GPU 0 MiB).
