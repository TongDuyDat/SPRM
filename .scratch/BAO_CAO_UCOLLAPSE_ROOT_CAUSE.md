# BÁO CÁO: NGUYÊN NHÂN GỐC u-COLLAPSE VÀ GIẢI PHÁP (v10)

**Ngày:** 2026-08-31 · **Phạm vi:** CUB, CLIP ViT-B/16, chuẩn ZSL/GZSL Xian
**Phương pháp:**Attribution bằng phẫu thuật checkpoint (zero-权重 at inference) + ma trận eval 2×2×2 — toàn bộ chạy trên IBM, không train lại.

---

## 1. Kết luận một dòng

> u-collapse KHÔNG do Ws drift (đã loại từ R6b) mà do **2 khối train-mới của v10 tự bẻ không gian thị giác / prototype theo hướng lợi-cho-seen**: W_multi (−15.7 u) và IPR refinement (−11→12 u). L_trans làm tệ thêm vì truyền gradient THÊM vào W_multi. Điểm bối rối nhất: **CLIP zero-shot thuần (đóng băng hoàn toàn) đạt raw H = 56.7 — cao hơn mọi biến thể đã train, kể cả R6b (56.4)**.

---

## 2. Bằng chứng định lượng — ma trận attribution tại ep80

Phẫu thuật checkpoint (zero `W_multi` và/hoặc `Ws=BA·A=0→I`) trên `final.pth`, eval bằng `tools/eval_checkpoint.py`:

| Biến thể | bank | T1 | u | s | H | cal_H | γ (offset seen) |
|---|---|---|---|---|---|---|---|
| final (như train) | refined | 68.5 | 29.9 | 79.9 | 43.5 | 63.3 | **0.12** |
| final | semantic | 69.3 | 40.8 | 75.4 | 52.9 | 61.8 | 0.03 |
| final_wm0 (W_multi=0) | refined | 57.7 | 44.9 | 45.7 | 45.3 | 45.3 | 0.00 |
| **final_wm0** | **semantic** | **73.0** | **56.4** | 60.8 | **58.5** | 58.5 | 0.00 |
| final_ba0 (Ws→I) | refined | 70.0 | 32.2 | 76.8 | 45.4 | 63.0 | 0.11 |
| final_ba0 | semantic | 71.3 | 44.0 | 72.2 | 54.7 | 62.0 | 0.03 |
| final_wm0_ba0 | refined | 56.8 | 47.7 | 38.9 | 42.8 | 42.8 | 0.00 |
| **final_wm0_ba0** | **semantic** | **73.1** | **59.7** | 53.9 | **56.7** | 56.7 | 0.00 |
| best_t1 (ep11) | refined | 72.2 | 41.9 | 72.8 | 53.2 | 63.3 | 0.07 |
| best_t1 (ep11) | semantic | 72.1 | 45.7 | 71.0 | 55.6 | 62.4 | 0.03 |

### Phân rã tổng sụp u: 59.7 → 29.9 (bank semantic, tính từ hình học thuần)

| Trục | Mất u | Ghi chú |
|---|---|---|
| **W_multi** (anchored MFA) | **−15.7** | \|W_multi\|_F phình 7.9 (ep11) → 16.7 (ep80) |
| **IPR refinement** (refined vs semantic) | **−10.9 → −12.0** | tăng dần theo epoch: ep11 chỉ −3.8, ep80 −10.9 |
| Ws drift (BA) | −3.3 | nhỏ — xác nhận lại kết luận từ R6b |
| **Tổng** | ≈ −29.8 ✓ | gần cộng tính |

Kèm theo, trong checkpoint: `λ_r = −0.066`, gate/W_qc/W_vca đã học xong — IPR không "yếu", nó **hoạt động nhưng theo hướng thấy-bias**.

---

## 3. Cơ chế (tại sao từng trục hại u)

### 3.1 W_multi — kẻ hại lớn nhất (−15.7)
`v~ = normalize(cls + W_multi·[GAP(z4); GAP(z8); GAP(z12)])`. W_multi chỉ được huấn luyện qua **l_cls/l_align trên ảnh seen** — nó kéo v~ ra khỏi manifold text-aligned của CLIP CLS (nơi chứa toàn bộ năng lực zero-shot) để siết cụm seen cho gọn. Ảnh unseen test bị kéo theo, rơi gần prototype seen hơn → γ tăng. Nó mua +14.6 s với giá −15.6 u, và **H khi bỏ nó đi LÊN** (52.9→58.5). Không có lý do gì giữ nó ở inference.

### 3.2 L_trans — phản tác dụng, cơ chế đã tìm ra
`trainer/trainer.py:626`: `m_s = self.model.mfa(self._level_means)` — `_level_means` detached nhưng **mfa chứa W_multi trainable** → L_trans truyền gradient THÊM vào W_multi, tức tăng thêm áp lực fit-seen lên đúng không gian đang gây hại. Đó là lý do run có L_trans sụp nhanh hơn (u@80: 39.9 → 29.9, γ 0.07→0.12). Ý định của L_trans (ràng buộc hình học) đúng, nhưng đường truyền gradient sai đích.

### 3.3 IPR — tín hiệu thật, offset thấy-bias
- Refinement chỉ được giám sát bởi l_cls **trên các hàng seen**. Hàm shared học được: "thêm residual giúp prototype seen hút ảnh" — áp cho MỌI ảnh test, kể cả ảnh unseen → offset hệ thống γ 0.03→0.12 (toàn bộ offset do IPR tạo ra; semantic path chỉ γ=0.03, ngang R6b).
- **Nhưng**: cal_H refined 63.3 > semantic 61.8, cal_u 60.4 > 57.3 → IPR có tín hiệu phân biệt thật +3, chỉ bị chôn dưới offset. Không được vứt — phải khử offset.

### 3.4 Chấm hết ảo tưởng: baseline đóng băng
`wm0_ba0 + semantic` = v~ = normalize(CLIP cls), prototype = normalize(φ) — đúng CLIP zero-shot: **u 59.7 / s 53.9 / H 56.7 / T1 73.1**. Mọi thứ đã train (R6b gồm cả) chỉ đổi u lấy s, không thắng nổi baseline này trên raw H. Đây là chuẩn mới phải vượt.

---

## 4. Giải pháp

### 4.0 Không cần train lại — có ngay hôm nay (config/eval-only)
Áp `aggregator.mode: cls_only` + `--score_bank semantic` cho checkpoint hiện tại ⇒ chính là số dòng **final_wm0/semantic: T1 73.0, u 56.4, s 60.8, H 58.5** — vượt mọi gate cũ (R6b: T1 74.4/u 50.1/H 56.4) về u và H. Nếu cần "mô hình v10 triển khai được ngay", đây là nó.

**CSLS đã test và LOẠI** (logs/attr_v10_csls.log): final/semantic u 40.8→24.8, T1 69.3→52.5, hub_u nảy lên 7.09 — CSLS sửa quá mức trên phân phối score này, tự tạo hub unseen. Không dùng cho pipeline này.

### 4.1 Công thức v11 (đề xuất chính — 1 run 80 epoch CUB)
1. **`aggregator.mode: cls_only`** — triệt trục −15.7 bằng cấu hình, không code mới.
2. **`lambda_trans: 0.0`** — đã chứng minh phản tác dụng; bỏ luôn.
3. **Giữ IPR + L_attn**, thêm term mới **L_zsl (zero-shot distillation)** khử offset của IPR:
   - teacher (frozen, detach): `T = softmax(v~.detach() @ φ_all^T / τ_z)` — hậu nghiệm zero-shot thuần của từng ảnh (vì cls_only ⇒ v~ chính là cls đóng băng).
   - student: `S = softmax(v~ @ p_ref_all^T / τ)`.
   - `L_zsl = KL(T ‖ S)` — kéo phân phối điểm refine (kể cả hàng seen) về mức zero-shot; trực tiếp triệt offset γ đã đo được, trong khi l_cls vẫn được tự do tăng chất lượng hàng seen.
   - Cost triển khai: ~15 dòng trong `losses/hsprn_loss.py` + plumbing `p_ref_all`, `phi_all_norm` (đa phần đã có sẵn trong train loop vì pres_target=all đã truyền `p_ref_all`).
4. Giữ nguyên λ_sem=2.0, ws_anchor=0.01, L_attn, pres_target=all.

**Tiêu chí thành công v11 @ep80:** u ≥ 56 (sàn wm0-semantic), T1 ≥ 73, raw H ≥ 58.5 (sàn cls_only), cal_H ≥ 64 (ngang best v10). Đạt đồng thời ⇒ IPR lần đầu **cộng** vào unseen thay vì trừ.

### 4.2 Phương án dự phòng nếu IPR vẫn hại ở v11
- Eval với `--score_bank semantic` (đã có tool) và đóng khung IPR như training-time regularizer; hoặc
- Khóa/cap `λ_r` (gate_residual cap), hoặc chỉ áp residual cho hàng seen của student trong L_zsl bằng mask.

### 4.3 Thí nghiệm phụ (rẻ, làm khi tiện)
- `best_t1.pth` + wm0 surgery: kiểm tra u@ep11 sau khi bỏ W_multi (dự đoán ~52-56, củng cố khuyến nghị chọn checkpoint theo Cal H).
- AWA2/APY: lặp lại attribution cube để khái quát kết luận trước khi viết paper.

---

## 5. Nhật ký thí nghiệm đã chạy

| # | Thí nghiệm | Kết quả | File log IBM |
|---|---|---|---|
| 1 | 2×2: best_t1/final × refined/semantic | bảng §2 | `logs/attr_v10_2x2.log` |
| 2 | Surgery wm0/ba0/wm0_ba0 × refined/semantic | bảng §2 | `logs/attr_v10_abl.log` |
| 3 | Kiểm tra checkpoint: λ_r, \|W_multi\|, \|BA\| | §2 | (inline) |
| 4 | CSLS trên final/best_t1 (semantic) | **LOẠI** — u 40.8→24.8, T1 69.3→52.5 | `logs/attr_v10_csls.log` |

Checkpoint phẫu thuật lưu tại `checkpoints/cub_clip_v10_ca_b16/CUB/final_{wm0,ba0,wm0_ba0}.pth`.
