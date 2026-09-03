# BÁO CÁO — HSPRN v10 (L_trans fix): E0 rerun + E1 CUB A0 — E1 FAIL CỔNG → DỪNG

Ngày: 31-08-2026 · IBM (ubuntu@128.168.135.246, 1× L40S) · Seed 42 · Log: `logs/v10_cub_a0_lt.log`
Phiên bản hướng dẫn: `.scratch/HUONG_DAN_AI_KHAC.md` (31-08, "E0 rerun + E1 train — bật lại L_trans")

## 1. E0 rerun — **PASSED** (14/14 checks green)

Lệnh: `python tools/e0_sanity_v10.py --baseline config/cub_clip_v9_r6b_noref_v8phi_b16.yaml --v10 config/cub_clip_v10_ca_b16.yaml --device cuda`

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

- md5 verify 14 file mới sync trên IBM: **khớp 24/24** bảng §5 (trainer.py=89fd0f5d…, e0_sanity_v10.py=3c5cf7bf…, 12 config mới).

## 2. E1 verify L_trans — **ĐẠI**

Dòng bắt buộc xuất hiện trong `logs/v10_cub_a0_lt.log` (08:33:55):

```
[2026-08-31 08:33:55] INFO - Transfer-alignment ON (lambda_trans=0.25, tau_teacher=0.1, levels=3, teacher=(150, 200)).
```

30 dòng đầu log (lọc warning timm):

```
[2026-08-31 08:33:40] INFO - Precomputed semantic embeddings -> frozen phi cache shape=(200, 512) (text tower NOT loaded).
[2026-08-31 08:33:40] INFO - Ws identity anchor ON (weight=0.01g, init=identity 512x512).
Loading dataset: CUB from ./prepared_datasets (prepared)
  Data mode: image (raw images -> visual backbone)
  Train: 7057 samples
  Test seen: 1764 samples
  Test unseen: 2967 samples
  Seen classes: 150, Unseen: 50
  Semantic dim: 512 (./pre-trained/cub_v8_phi_b16.pt)
Model parameters: 89,503,233 total, 3,310,593 trainable
[2026-08-31 08:33:55] INFO - Transfer-alignment ON (lambda_trans=0.25, tau_teacher=0.1, levels=3, teacher=(150, 200)).
[2026-08-31 08:33:55] INFO - === Training HSPRN on CUB ===
[2026-08-31 08:33:55] INFO - Seen classes: 150, Unseen classes: 50
[2026-08-31 08:33:55] INFO - Train samples: 7057, Test seen: 1764, Test unseen: 2967
Train 1/80:  0%|  0/28 … loss=4.0576 … (không crash ở _build_trans_caches)
```

Không Traceback, không crash init. `attn=` và `trans=` xuất hiện trong log epoch từ ep1.

## 3. Bảng kết quả

| dataset | arm | config | ckpt dùng | T1 (ZSL) | u | s | H (GZSL) | seed | ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| CUB | A0 (L_trans=0.25) | cub_clip_v10_ca_b16.yaml | best_t1.pth (ep 11) | **72.2** | 41.9 | 72.8 | 53.2 | 42 | Best T1 @ep11; best Cal H 64.4 @ep40 (T1 70.4, Cal u 60.9 s 68.3); @ep80: T1 68.5 / u 29.9 / s 79.9 / H 43.5 |
| CUB | A0 (L_trans=0.25) | cub_clip_v10_ca_b16.yaml | best.pth (ep 40) | 70.4 | 35.2 | 79.3 | 48.7 | 42 | Best Cal H = 64.4 |

E2 (4a/4b/4c) **KHÔNG chạy** — E1 fail cổng (§3: DỪNG).

Trajectory (mỗi 5 epoch, run mới `_lt`):

```
ep  5: T1 71.7  u 51.1   ep 11: T1 72.2  u 41.9  ← BEST_T1
ep 10: T1 71.5  u 44.2   ep 20: T1 70.6  u 41.2
ep 30: T1 70.5  u 37.9   ep 40: T1 70.4  u 35.2  ← BEST Cal H (64.4)
ep 50: T1 69.4  u 31.1   ep 80: T1 68.5  u 29.9 (s 79.9, H 43.5)
```

## 4. Phán quyết cổng E1 — **FAIL**

| Tiêu chí | Cổng | Thực tế | Phán quyết |
|---|---|---|---|
| Best T1 | ≥ 74.7 | **72.2** (ep 11) | ❌ thiếu 2.5 điểm |
| u @best T1 | ≥ 49.1 | **41.9** (ep 11) | ❌ thiếu 7.2 điểm |
| Δ so baseline R6b (74.4) | — | **−2.2** | ❌ |

So với run trước (L_trans tắt, T1 72.6 / u@80 39.9): run mới **không tốt hơn** — T1 −0.4, u@80 −10.0.

**Dấu hiệu fix có hiệu lực (§3) — kiểm tra bắt buộc:**

| Dấu hiệu | Kỳ vọng | Thực tế | Đạt? |
|---|---|---|---|
| u không giảm đơn điệu; u@ep80 ≥ ~45 | ngừng sụp | u 51.1 @ep5 → 29.9 @ep80, giảm đơn điệu gần như toàn trình | ❌ |
| sem, ||BA|| không tăng liên tục | ổn định | sem 0.0014→0.0089 (nhỏ hơn run trước, có giảm về cuối); ||BA|| 0.2537→1.4396 vẫn tăng liên tục (nhẹ hơn run trước 1.58) | ⚠️ một nửa |
| trans > 0 mỗi epoch | có | trans=0.0368 @ep1 → 0.0994 @ep80, luôn > 0, có trong mọi dòng epoch | ✅ |

Kết luận dấu hiệu: L_trans đã **bật và hoạt động** (trans>0, sem/||BA|| nhỏ hơn run trước) nhưng **không cứu được u** — u vẫn sụp đơn điệu, thậm chí nhanh hơn run trước ở nửa sau (29.9 vs 39.9 @ep80). Theo §3: DỪNG, không chạy E2, không đổi config.

## 5. Loss trajectory (đầu → cuối, run CUB A0 `_lt`)

```
attn=     0.0008 (ep1)  →  0.0004 (ep80)   — nhỏ, giảm nhẹ, ổn định
trans=    0.0368 (ep1)  →  0.0994 (ep80)   — tăng từ 0.0368 → đỉnh 0.1811 (ep5) rồi giảm dần → 0.0994
sem=      0.0014 (ep1)  →  0.0089 (ep80)   — đỉnh 0.0706 @ep8, giảm về cuối (run trước tăng lên 0.0341)
wsanchor= 0.0002 (ep1)  →  0.0207 (ep80)
||BA||=   0.2537 (ep1)  →  1.4396 (ep80)   — vẫn tăng liên tục (run trước: 0.2580 → 1.5849)
Loss:     3.5479 (ep1)  →  0.8972 (ep80)
```

## 6. Vấn đề gặp phải

1. **[CHÍNH] L_trans bật nhưng kết quả không cải thiện — ngược lại xấu hơn run trước về u**: T1 72.2 (vs 72.6), u@bestT1 41.9 (vs 49.1), u@ep80 29.9 (vs 39.9). Trans loss hoạt động (0.037→0.18→0.099), sem/||BA|| thấp hơn run trước, nhưng u vẫn giảm đơn điệu cả 80 epochs. Giả thuyết "L_trans tắt là nguyên nhân u sụp" **không được xác nhận** — có yếu tố khác (nghi vấn: MFA cross-attn path, λ_attn, λ_sem=2.0 quá mạnh, hoặc teacher/cache cls means chưa đúng semantics). KHÔNG sửa gì theo §6 — người phụ trách chính quyết định.
2. `[minor]` Lệnh ssh wrapper WSL timeout (SIGTERM phía local) khi launch nohup — train trên IBM vẫn chạy bình thường (đã xác minh PID + log). Không ảnh hưởng kết quả.
3. Không NaN/crash/OOM. Sync md5 khớp.

## 7. Checkpoint + backup

**Mới (run `_lt`, ghi đè vị trí cũ):**
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v10_ca_b16/CUB/best_t1.pth` (384,574,903 B — epoch 11)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v10_ca_b16/CUB/best.pth` (384,563,225 B — epoch 40)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v10_ca_b16/CUB/final.pth` (358,070,311 B — epoch 80)

**Backup run trước (pre-fix, L_trans tắt, T1 72.6):**
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/pre_lt_fix_backup/cub_a0_pre_fix/CUB/best_t1.pth` (ep 9)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/pre_lt_fix_backup/cub_a0_pre_fix/CUB/best.pth` (ep 14)
- `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/pre_lt_fix_backup/cub_a0_pre_fix/CUB/final.pth` (ep 80)

Log: `/home/ubuntu/woker/Zeroshot/HSPRM/logs/v10_cub_a0_lt.log`

## Kết luận

- E0 rerun **PASSED** 14/14; L_trans verify **ĐẠI** (dòng Transfer-alignment ON đúng spec).
- E1 (L_trans=0.25) train đủ 80 epochs, không lỗi kỹ thuật, nhưng **FAIL cổng nặng hơn**: T1 72.2 < 74.7, u 41.9 < 49.1; u vẫn sụp đơn điệu (29.9 @ep80) dù trans>0 và sem/||BA|| ổn định hơn.
- **E2 đã dừng đúng theo §3.** IBM sạch (không process, GPU rảnh).
- So sánh 3 run CUB A0: R6b baseline 74.4 → v10 pre-fix 72.6 → v10 L_trans fix **72.2**. Fix L_trans không phải lời giải cho hiện tượng T1-đỉnh-sớm + u-sụp.
