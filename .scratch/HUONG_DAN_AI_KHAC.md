# HƯỚNG DẪN CHO AI PHỤ TRÁCH — HSPRN v10: E0 rerun + E1 train (bật lại L_trans)

> **Vai trò của bạn:** Thực thi các bước dưới đây trên server IBM.
> KHÔNG sửa code — code đã hoàn tất, sync và verify md5. Chỉ chạy lệnh, đọc log, ghi kết quả.
> Sau khi hoàn thành (hoặc gặp lỗi), báo cáo lại theo mục **BÁO CÁO** ở cuối.

## 0. Bối cảnh — đã thay đổi gì so với run trước

- **Nguyên nhân run trước fail cổng (3/4 dataset):** L_trans bị tắt nhầm (`lambda_trans: 0.0` trong cả 12 config v10) vì trainer cũ chưa hỗ trợ "dict vision path" của v10. Không có L_trans, model bị kéo về cụm seen → T1 đỉnh sớm rồi giảm, **u sụp liên tục** (AWA2: u 62.4 → 10.5 @ep80).
- **Fix đã xong (KHÔNG làm lại):**
  1. `trainer/trainer.py` — `_build_trans_caches` đã hỗ trợ dict path (GAP token levels + cache cls means → `_level_means = {"tokens": [3 means], "cls": cls_mean}`; mfa tiêu thụ trực tiếp cả mode anchored lẫn cls_only). Epoch log giờ in thêm `attn=` và `trans=`.
  2. `tools/e0_sanity_v10.py` — force λ_trans=0 trong 10-step check (E0 không chạm L_trans; đường L_trans thật được verify bằng dòng log khi train).
  3. **12 config v10** — `lambda_trans: 0.25` + `tau_teacher: 0.1` (đúng giá trị baseline R6b).
- **14/14 file đã sync lên IBM, md5 đã verify khớp bảng §5.** KHÔNG sync lại, KHÔNG sửa bất kỳ file .py/.yaml nào.
- Kỳ vọng: loss v10 ≡ loss R6b (+ L_attn + các đường zero-init) → theo LP-FT, nếu module mới không đóng góp thì hội tụ ≈ baseline (CUB ~74.4), nếu đóng góp thì vượt cổng 74.7. **u phải ngừng sụp.**

## TRẠNG THÁI HIỆN TẠI (31-08)

- **Tất cả bước chuẩn bị (fix code + 12 config + sync + md5) ĐÃ XONG.** Việc của bạn bắt đầu từ **Bước 1 (E0 rerun)**.
- Kết quả run trước (L_trans tắt, 1/4 dataset pass) đã lưu trong `.scratch/BAO_CAO_V10.md` — chỉ để tham chiếu, không phải kết quả mới.
- Server IBM: user `ubuntu@128.168.135.246`, code tại `/home/ubuntu/woker/Zeroshot/HSPRM` (luôn `cd` vào trước khi chạy).
- Môi trường: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310` (torch, GPU L40S **duy nhất** — mọi run xếp hàng tuần tự).
- SSH từ WSL: `bash ~/ssh_ibm.sh "lệnh"`.
- **Checkpoints của run trước** (pre-fix, T1 CUB 72.6) đang nằm tại `checkpoints/cub_clip_v10_ca_b16/CUB/`. Run mới SẼ GHI ĐÈ — backup trước khi train (Bước 2).

## 1. E0 rerun (~3 phút, chạy trực tiếp)

```bash
cd /home/ubuntu/woker/Zeroshot/HSPRM && source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310 && python tools/e0_sanity_v10.py --baseline config/cub_clip_v9_r6b_noref_v8phi_b16.yaml --v10 config/cub_clip_v10_ca_b16.yaml --device cuda
```

- Kỳ vọng **14/14 PASS** như lần trước (E0 force λ_trans=0 nên không chạm đường L_trans; bit-exact anchor không đổi).
- `E0 PASSED — all checks green.` → sang Bước 2.
- Bất kỳ FAIL nào → DỪNG, báo lại tên check FAIL + 30 dòng log quanh chỗ fail.

## 2. Backup checkpoints cũ + Train E1 CUB A0

Backup run trước (giữ làm bằng chứng pre-fix):

```bash
cd /home/ubuntu/woker/Zeroshot/HSPRM
mkdir -p checkpoints/pre_lt_fix_backup && mv checkpoints/cub_clip_v10_ca_b16 checkpoints/pre_lt_fix_backup/cub_a0_pre_fix
```

Rồi khởi động train (nohup, ~40–60 phút, 80 epochs):

```bash
nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310 && CUDA_VISIBLE_DEVICES=0 python train.py --config config/cub_clip_v10_ca_b16.yaml' > logs/v10_cub_a0_lt.log 2>&1 &
```

**Verify trong 3 phút đầu** (`head -50 logs/v10_cub_a0_lt.log`):
- PHẢI thấy dòng: `Transfer-alignment ON (lambda_trans=0.25, tau_teacher=0.1, levels=3, teacher=(150, 200))` — bằng chứng L_trans đã bật. `teacher=(150, 200)` = 150 seen × 200 classes của CUB.
- KHÔNG có dòng này → DỪNG, báo lại 30 dòng đầu log.
- Traceback/crash ngay đầu (khoảng `_build_trans_caches`) → giữ log, báo lại 30 dòng cuối. Đó là bug dict path — không tự sửa.

## 3. Theo dõi + Cổng E1

- `tail -5 logs/v10_cub_a0_lt.log` mỗi ~10 phút. Loss giảm đều, không NaN.
- Log epoch giờ in thêm `attn=` (L_attn) và `trans=` (L_trans) — ghi lại giá trị đầu/cuối để báo cáo.

**Cổng E1 (chấm bằng best T1 / best.pth trong log):** `T1 ≥ 74.7` VÀ `u ≥ 49.1`.

**Dấu hiệu fix có hiệu lực (bắt buộc kiểm tra, so với run trước):**
- u KHÔNG còn giảm đơn điệu theo epoch (run trước: u 51.0 @ep7 → 39.9 @ep80; u@ep80 ≥ ~45 là đạt).
- `sem` và `||BA||` không còn tăng liên tục (run trước sem 0.0014→0.0341, ||BA|| 0.26→1.58).
- `trans` > 0 và có trong log mỗi epoch.

PASS → chạy tiếp Bước 4. FAIL → DỪNG, thu thập T1/u/s/H mỗi epoch (hoặc 10 epoch cuối) + trajectory attn/trans/sem/||BA||, rồi báo cáo. KHÔNG tự ý đổi config.

## 4. E2 — Mở rộng 3 dataset + ablation (chỉ khi E1 PASS)

1 GPU duy nhất — tuần tự, chỉ khởi động run tiếp khi run trước xong (`pgrep -f train.py` trả rỗng, hoặc `tail` log thấy kết quả cuối).

**Bước 4a — A0 3 dataset còn lại** (log riêng mỗi run):

```bash
nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310 && CUDA_VISIBLE_DEVICES=0 python train.py --config config/apy_clip_v10_ca_b16.yaml' > logs/v10_apy_a0_lt.log 2>&1 &
# apy XONG → awa2 → sun (tuần tự):
nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310 && CUDA_VISIBLE_DEVICES=0 python train.py --config config/awa2_clip_v10_ca_b16.yaml' > logs/v10_awa2_a0_lt.log 2>&1 &
nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310 && CUDA_VISIBLE_DEVICES=0 python train.py --config config/sun_clip_v10_ca_b16.yaml' > logs/v10_sun_a0_lt.log 2>&1 &
```

**Bước 4b — Ablation A2 (MFA cls_only) và A3 (không L_attn) × 4 dataset** — 8 run, cùng lệnh, đổi config `<ds>_clip_v10_a2_mfacls.yaml` / `<ds>_clip_v10_a3_noattn.yaml`, log `logs/v10_<ds>_a2_lt.log` / `logs/v10_<ds>_a3_lt.log`.

**Bước 4c — A1 (eval-only, KHÔNG train)** dùng checkpoint A0 mới:

```bash
python tools/eval_checkpoint.py --config config/cub_clip_v10_ca_b16.yaml \
    --checkpoint checkpoints/cub_clip_v10_ca_b16/CUB/best.pth --score_bank semantic
# lặp cho apy/awa2/sun; chấm cả best_t1.pth nếu tồn tại
```

**Cổng từng dataset (ΔT1 ≥ +0.3 so baseline, u không sập):**

| Dataset | Baseline T1 | Cổng T1 | u tối thiểu |
|---|---|---|---|
| CUB | 74.4 | ≥ 74.7 | ≥ 49.1 |
| APY | 84.9 | ≥ 85.2 | baseline_u − 1.0 |
| AWA2 | 95.9 | ≥ 96.2 | baseline_u − 1.0 |
| SUN | 89.2 | ≥ 89.5 | baseline_u − 1.0 |

## 5. Bảng md5 — 24 file (14 file mới sync đậm, 10 file cũ không đổi)

| File | md5 (LF) |
|---|---|
| model/protypes.py | f4b82a284a7864bc69b1b07f4117fc63 |
| model/hsprn_model.py | 41890001f3144669ad81bae1f260fd53 |
| model/vision_model.py | 56df7eb359bac9f9ed4557b24a7f0dfc |
| model/vision/__init__.py | 26dc1e51c282583f650ad2056616785b |
| model/sematics/constructor.py | d7598c038ede119656f6ad6ca8c6de36 |
| tools/semantic_policy.py | e3c8caf12441445b55777de209660da3 |
| model/hsprn_outputs.py | 82ea01fed28ae160f8035130d8adfdc3 |
| losses/hsprn_loss.py | c4f925ecd276208f42606c16091aca44 |
| model/aggregator.py | cf0266b00d865c4a5c8d97bb302abc86 |
| tools/precompute_attribute_queries.py | 6d698941b7da239fb2aaca6b3f1bbfdf |
| **trainer/trainer.py** | **89fd0f5d1fd92a3161758273c953be8c** |
| **tools/e0_sanity_v10.py** | **3c5cf7bfee4025d3e301aa210bb3a24e** |
| **config/cub_clip_v10_ca_b16.yaml** | **493073a0e0754d236eb5d521fc68c79e** |
| **config/cub_clip_v10_a2_mfacls.yaml** | **c50658245fbd375a0bbf3e485e6d4640** |
| **config/cub_clip_v10_a3_noattn.yaml** | **31049b9d43df1156d145d9c998301596** |
| **config/apy_clip_v10_ca_b16.yaml** | **de48a70377078006927529f8503ea3ec** |
| **config/apy_clip_v10_a2_mfacls.yaml** | **a3d3194b2b50fa4bc17cc194e7f34f3d** |
| **config/apy_clip_v10_a3_noattn.yaml** | **6ed2879fdfbd6045c0351fcf6f28b458** |
| **config/awa2_clip_v10_ca_b16.yaml** | **f46b442a83824e1cf61cacc8525ed652** |
| **config/awa2_clip_v10_a2_mfacls.yaml** | **1d24bf5da5a9234b9265de40ca6326a6** |
| **config/awa2_clip_v10_a3_noattn.yaml** | **a4a16dbed8a8c1d3b6206972a3dfd693** |
| **config/sun_clip_v10_ca_b16.yaml** | **78fdeef245b97e73d4456692307d01bb** |
| **config/sun_clip_v10_a2_mfacls.yaml** | **856f254c40480d23e37eaefc4c824407** |
| **config/sun_clip_v10_a3_noattn.yaml** | **4fd2109c31c884f9408c12d63b512dc6** |

(Nếu nghi ngờ, verify bằng `md5sum` trên IBM — KHÔNG re-upload.)

## 6. Xử lý sự cố

- **E0 fail "bit-exact anchor"**: báo lại, không sửa.
- **Train crash ở init / traceback trong 2 phút đầu**: giữ log, báo 30 dòng cuối. Không sửa file nào.
- **OOM trên SUN** (645 lớp): báo lại, không sửa.
- **Train crash / NaN**: giữ log, báo 30 dòng cuối.
- **`pgrep -f train.py` trả về PID nhưng GPU 0 MiB**: process tháp ma — check `ps -o pid,etime,cmd -p <PID>`, nếu trống thì bỏ qua, GPU rỗi.
- KHÔNG sửa bất kỳ file .py/.yaml nào. Nếu nghi code sai: mô tả, đừng sửa.

## 7. BÁO CÁO (bắt buộc khi xong hoặc khi gặp blocker)

1. **E0**: PASS/FAIL + toàn bộ output các dòng `[PASS]/[FAIL]`.
2. **E1 verify**: có dòng `Transfer-alignment ON (lambda_trans=0.25…)` hay không + 30 dòng đầu log.
3. **Bảng kết quả** (mỗi run đã hoàn thành — A0/A1/A2/A3 × dataset):

```
| dataset | arm | config | ckpt dùng | T1 (ZSL) | u | s | H (GZSL) | seed | ghi chú |
```

4. **Phán quyết cổng** từng dataset theo bảng Bước 4 + dấu hiệu u (giảm đơn điệu hay không, u@ep80).
5. **Loss trajectory**: đầu/cuối của `attn=`, `trans=`, `sem`, `||BA||` từ log epoch.
6. **Vấn đề gặp phải** + 30 dòng log liên quan (nếu có).
7. **Danh sách checkpoint** sinh ra (đường dẫn đầy đủ) + vị trí backup pre-fix.
