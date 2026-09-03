# BÁO CÁO — HSPRN v10: Sync + E0 (E0 FAILED → DỪNG, không train E1/E2)

Ngày chạy: 31-08 (WSL → IBM ubuntu@128.168.135.246, GPU L40S duy nhất, 46068 MiB)
Log đầy đủ: `.scratch/v10_sync_e0_output.log` (WSL)

## 1. Kết quả Sync + Precompute + E0

### Sync (bước 1) — OK
- 24 file v10 đã scp lên IBM, CRLF fix OK (`CRLF_OK`).
- **Bước 2 — md5 check: 24/24 KHỚP** bảng trong hướng dẫn (so từng cặp, diff = rỗng).
- 4 attribute-query bank đã tạo OK trên IBM:
  - `pre-trained/cub_attrq_b16.pt` — CUB 312 attr → (312, 512), norm mean=1.0000
  - `pre-trained/apy_attrq_b16.pt` — APY 64 attr → (64, 512)
  - `pre-trained/awa2_attrq_b16.pt` — AWA2 85 attr → (85, 512)
  - `pre-trained/sun_attrq_b16.pt` — SUN 102 attr → (102, 512)

### E0 sanity — **E0 FAILED** (4 check FAIL + 1 crash)

Toàn bộ output các dòng `[PASS]/[FAIL]`:

```
=== E0.1: attribute metadata (./prepared_datasets) ===
[PASS] metadata CUB — attributes (200, 312), names 312, expected n_attr 312
[PASS] metadata APY — attributes (32, 64), names 64, expected n_attr 64
[PASS] metadata AWA2 — attributes (50, 85), names 85, expected n_attr 85
[PASS] metadata SUN — attributes (717, 102), names 102, expected n_attr 102
=== E0.2/E0.3: build fresh v10 + R6b models ===
=== E0.2: extract_levels_anchored vs backbone(x) ===
[PASS] extract_levels_anchored shapes — tokens [(32, 197, 768), (32, 197, 768), (32, 197, 768)], cls (32, 512)
[FAIL] cls == backbone(x) — max|d| = 2.70e+00
=== E0.3: bit-exact epoch-0 anchor (v10 vs R6b) ===
[FAIL] epoch-0 anchor: scores — max|d| = 2.24e-01
[FAIL] epoch-0 anchor: scores_sem — max|d| = 2.24e-01
[FAIL] epoch-0 anchor: v_embed — max|d| = 3.54e-01
=== E0.4: L_attn + 10 optimizer steps ===
[PASS] GT attribute matrix rows == classes — gt (200, 312) vs phi (200, 512)
[PASS] labels <-> GT attribute row — label 196: 167 active attributes (row 167 vs other 171)
  trainable tensors: 11, total numel 3,310,593
Traceback (most recent call last):
  File "/home/ubuntu/woker/Zeroshot/HSPRM/tools/e0_sanity_v10.py", line 261, in <module>
    main()
  File "/home/ubuntu/woker/Zeroshot/HSPRM/tools/e0_sanity_v10.py", line 253, in main
    check_train_steps(model_v10, cfg_v10, splits, device, args.batch_size)
  File "/home/ubuntu/woker/Zeroshot/HSPRM/tools/e0_sanity_v10.py", line 201, in check_train_steps
    loss, loss_dict = criterion(
  File ".../torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File ".../torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/woker/Zeroshot/HSPRM/losses/hsprn_loss.py", line 202, in forward
    raise ValueError(
ValueError: lambda_sem>0 requires phi_t and p_sem_unnorm tensors.
```

Script thoát trước khi in kết luận `E0 FAILED: ...` (crash trong E0.4), exit code 1.

## 2. Bảng kết quả train

**Không có run nào.** E0 fail → theo hướng dẫn §1: DỪNG, không chạy E1/E2.

| dataset | arm | config | ckpt dùng | T1 (ZSL) | u | s | H (GZSL) | seed | ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | E0 FAILED → không train |

## 3. Phán quyết cổng

Không đánh giá được — chưa tới E1/E2 (điều kiện vào E1 là E0 PASSED).

## 4. Log loss_attn

**Không có.** E0.4 crash ở lần gọi criterion đầu tiên (trước khi chạy 10 optimizer steps), nên không có giá trị `loss_attn` đầu/cuối nào được in ra.

## 5. Vấn đề gặp phải (KHÔNG sửa gì theo §5 — mô tả để người phụ trách chính xử lý)

### Vấn đề A — "bit-exact anchor" FAIL (E0.2 + E0.3), lỗi init/code
`extract_levels_anchored` (model/vision/__init__.py, dòng 89–141) tính CLS từ **residual stream**:
`cls = ln_post(x[:, 0, :]) @ proj` với `x` chạy qua vòng `for blk in resblocks: x = blk(x)` bình thường.

Nhưng CLS "chuẩn" mà check so sánh — `backbone(x)` = `CLIPVisualBackbone.forward` → `self.visual(x)` → `VisionTransformer.forward` (model/sematics/clip/model.py, dòng 426) — **không phải CLIP vanilla**. Forward đó với default `model_type='ClearCLIP'`, `ignore_residual=True`, `last_n_layers=1`:
- các block trước cuối: chạy bình thường (LND format);
- block cuối: **không dùng block normally** — thay vào đó `output = custom_attn(blk.attn, blk.ln_1(x), 'ClearCLIP')` = attention **q@qᵀ (bỏ key)**, **bỏ residual + bỏ MLP**; rồi `x_ori = ln_post(output[:, 0, :]) @ proj`.

Tức hai đường tính CLS khác nhau căn bản (một bên residual-stream CLS, một bên ClearCLIP no-residual CLS) → `max|d| = 2.70e+00`. E0.3 fail (scores 2.24e-01, scores_sem 2.24e-01, v_embed 3.54e-01) là hệ quả kéo dài của cùng lỗi upstream ở CLS: mọi embedding/scores epoch-0 của v10 lệch khỏi R6b. Lưu ý: VPT prompt không phải nguyên nhân (vision_depth=0 → VPT_shallow=False); nguyên nhân là semantics ClearCLIP của forward chuẩn so với loop thủ công trong `extract_levels_anchored`.

### Vấn đề B — crash E0.4: `ValueError: lambda_sem>0 requires phi_t and p_sem_unnorm tensors`
- Config v10 A0 CUB có `loss.lambda_sem: 2.0` (> 0).
- `tools/e0_sanity_v10.py` dòng 201–205 gọi criterion chỉ với `v_tilde, p_sem, p_ref, attr_pred, gt_attr` — **không truyền `phi_t` và `p_sem_unnorm`**, trong khi `losses/hsprn_loss.py:200-204` raise khi `lambda_sem>0` mà thiếu 2 tensor này.
- Tham khảo: `trainer/trainer.py` dòng 653–654 (training thật) có truyền đủ `phi_t=phi_bank, p_sem_unnorm=out.p_sem_unnorm` → bản thân trainer/loss có vẻ consistent; chỗ thiếu argument là **lời gọi trong script E0** (`check_train_steps`).
- Không sửa gì — người phụ trách chính quyết định (sửa script E0, hoặc E0 gọi criterion theo cách khác).

### 30 dòng log quanh chỗ fail (E0.2 → E0.4, đã lọc bớt warning timm)

```
=== E0.2/E0.3: build fresh v10 + R6b models ===
=== E0.2: extract_levels_anchored vs backbone(x) ===
[PASS] extract_levels_anchored shapes — tokens [(32, 197, 768), (32, 197, 768), (32, 197, 768)], cls (32, 512)
[FAIL] cls == backbone(x) — max|d| = 2.70e+00
=== E0.3: bit-exact epoch-0 anchor (v10 vs R6b) ===
[FAIL] epoch-0 anchor: scores — max|d| = 2.24e-01
[FAIL] epoch-0 anchor: scores_sem — max|d| = 2.24e-01
[FAIL] epoch-0 anchor: v_embed — max|d| = 3.54e-01
=== E0.4: L_attn + 10 optimizer steps ===
[PASS] GT attribute matrix rows == classes — gt (200, 312) vs phi (200, 512)
[PASS] labels <-> GT attribute row — label 196: 167 active attributes (row 167 vs other 171)
  trainable tensors: 11, total numel 3,310,593
Traceback (most recent call last):
  File "/home/ubuntu/woker/Zeroshot/HSPRM/tools/e0_sanity_v10.py", line 261, in <module>
    main()
  File "/home/ubuntu/woker/Zeroshot/HSPRM/tools/e0_sanity_v10.py", line 253, in main
    check_train_steps(model_v10, cfg_v10, splits, device, args.batch_size)
  File "/home/ubuntu/woker/Zeroshot/HSPRM/tools/e0_sanity_v10.py", line 201, in check_train_steps
    loss, loss_dict = criterion(
  File ".../torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File ".../torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/woker/Zeroshot/HSPRM/losses/hsprn_loss.py", line 202, in forward
    raise ValueError(
ValueError: lambda_sem>0 requires phi_t and p_sem_unnorm tensors.
```

## 6. Danh sách checkpoint / artefact sinh ra

**Không có checkpoint train nào.** Trên IBM:
- `checkpoints/` — không có thư mục `*v10*` nào (đã kiểm tra).
- Không có tiến trình `train.py` đang chạy; GPU idle (0 MiB used).
- Chỉ có 4 attrq bank mới trong `pre-trained/`:
  - `/home/ubuntu/woker/Zeroshot/HSPRM/pre-trained/cub_attrq_b16.pt`
  - `/home/ubuntu/woker/Zeroshot/HSPRM/pre-trained/apy_attrq_b16.pt`
  - `/home/ubuntu/woker/Zeroshot/HSPRM/pre-trained/awa2_attrq_b16.pt`
  - `/home/ubuntu/woker/Zeroshot/HSPRM/pre-trained/sun_attrq_b16.pt`

## Kết luận

- Sync + md5 + precompute: **HOÀN HẢO** (24/24 file khớp, 4 bank OK).
- **E0 FAILED** do (A) bit-exact anchor không đạt — `extract_levels_anchored` tính CLS theo residual-stream trong khi `backbone(x)` theo semantics ClearCLIP; (B) script E0 thiếu `phi_t`/`p_sem_unnorm` khi gọi criterion với `lambda_sem=2.0`.
- Theo hướng dẫn: **DỪNG — chưa chạy E1/E2**, không sửa code. Cần người phụ trách chính xử lý 2 vấn đề A/B rồi chạy lại E0.
