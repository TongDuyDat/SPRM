# BÁO CÁO NGHIÊN CỨU: NGUYÊN NHÂN u-COLLAPSE VÀ ĐỀ XUẤT GIẢI PHÁP (GIỮ W_multi)

*Ngày: 2026-09-01. Phạm vi: phân tích định lượng trên checkpoint v10 (CUB, CLIP ViT-B/16), tổng hợp văn liệu ZSL mới nhất, đề xuất giải pháp giữ nguyên cơ chế W_multi.*

---

## 0. TÓM TẮT ĐIỀU HÀNH

Huấn luyện v10 làm mất **~30 điểm u** so với hình học đóng băng của CLIP (59.7 → 29.9).
Phân rã bằng ablation inference cho thấy ba thủ phạm **cộng dồn gần như độc lập**:

| Thành phần | Mất u | Bằng chứng |
|---|---|---|
| **W_multi (MFA anchored)** | **~15.6 đ** | 40.8 → 56.4 khi zero W_multi (semantic bank, ep80) |
| **IPR (refined bank)** | **~10.9 đ** | 29.9 (refined) vs 40.8 (semantic) trên cùng checkpoint |
| **BA/Ws drift** | **~3.3 đ** | 56.4 → 59.7 khi thêm zero BA |
| *(L_trans — giữa 2 run)* | *+10 đ tệ hơn* | u 39.9 (không trans) → 29.9 (có trans) |

**Cơ chế cốt lõi:** mọi tín hiệu huấn luyện của v_tilde (l_cls CE, l_align NCE) đều **chỉ chứa lớp seen**. W_multi và IPR vì thế học các hướng *phân biệt seen* — hữu ích cho seen (s +15) nhưng xoay v_tilde / uốn prototype ra khỏi không gian CLIP đóng băng nơi các prototype văn bản của lớp **unseen** đang nằm. Lớp unseen không có dù chỉ một gradient bảo vệ. W_multi **không hề có regularizer nào** (ws_anchor chỉ áp cho BA; wv_anchor chỉ áp cho nhánh legacy đã bị bỏ qua ở mode `anchored`).

**Giải pháp đề xuất (giữ W_multi, giữ refined bank làm bank duy nhất):** mỗi ánh xạ khả học chạm vào không gian chung được ghép một ràng buộc đồng hành không cần ảnh unseen, suy ra từ hình học đóng băng — **neo** W_multi bằng cosine retention L_fm (WiSE-FT feature-space, đã duyệt); **thiết kế lại IPR** thành residual tương tác ảnh–prototype dạng classifier-free + xoay tiếp tuyến có budget (IPR-v2); **thay L_trans bằng chưng cất posterior zero-shot trên toàn không gian nhãn** (L_zs — nguồn gradient unseen đầu tiên đổ thẳng vào đường inference); **bảo toàn quan hệ theo cặp của toàn bộ tập prototype** (L_rel) thay cho bump ws_anchor. Mục tiêu: u@80 ≥ 50, T1 ≥ 74.7, cal_H ≥ 64.4.

---

## 1. BẰNG CHỨNG ĐỊNH LƯỢNG HOÀN CHỈNH

### 1.1 Ma trận attribution (eval trực tiếp trên checkpoint, IBM, correction=none)

Cấu hình gốc: `cub_clip_v10_ca_b16.yaml`, checkpoint trong `checkpoints/cub_clip_v10_ca_b16/CUB/`.
Các biến thể "phẫu thuật checkpoint": `final_wm0` (W_multi:=0 ⇒ v~=normalize(cls)), `final_ba0` (B:=0 ⇒ Ws=I), `final_wm0_ba0` (cả hai — **chính là hình học đóng băng R6b-zero-shot**).

| # | Checkpoint | Bank | T1_unseen | T1_seen | u | s | H | cal_u | cal_s | cal_H | γ | hub_s | hub_u |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | final (ep80) | refined | 68.5 | 81.5 | **29.9** | 79.9 | 43.5 | 60.4 | 66.6 | 63.3 | 0.12 | 4.17 | 3.32 |
| 2 | final (ep80) | semantic | 69.3 | 77.7 | **40.8** | 75.4 | 52.9 | 57.3 | 67.1 | 61.8 | 0.03 | 5.20 | 3.02 |
| 3 | best_t1 (ep11) | refined | 72.2 | 74.9 | 41.9 | 72.8 | 53.2 | 63.4 | 63.2 | 63.3 | 0.07 | 3.59 | 2.70 |
| 4 | best_t1 (ep11) | semantic | 72.1 | 73.4 | **45.7** | 71.0 | 55.6 | 62.8 | 62.0 | 62.4 | 0.03 | 3.49 | 3.74 |
| 5 | final_wm0 | refined | 57.7 | 57.4 | 44.9 | 45.7 | 45.3 | 44.9 | 45.7 | 45.3 | 0.00 | 12.40 | 4.26 |
| 6 | final_wm0 | semantic | 73.0 | 66.0 | **56.4** | 60.8 | 58.5 | 56.4 | 60.8 | 58.5 | 0.00 | 5.45 | 2.76 |
| 7 | final_ba0 | refined | 70.0 | 79.0 | 32.2 | 76.8 | 45.4 | 62.9 | 63.1 | 63.0 | 0.11 | 5.47 | 3.16 |
| 8 | final_ba0 | semantic | 71.3 | 75.2 | **44.0** | 72.2 | 54.7 | 60.4 | 63.7 | 62.0 | 0.03 | 6.21 | 2.78 |
| 9 | final_wm0_ba0 | refined | 56.8 | 50.5 | 47.7 | 38.9 | 42.8 | 47.7 | 38.9 | 42.8 | 0.00 | 14.46 | 4.31 |
| 10 | final_wm0_ba0 | semantic | **73.1** | 60.2 | **59.7** | 53.9 | 56.7 | 59.7 | 53.9 | 56.7 | 0.00 | 5.31 | 3.56 |

Log gốc: `logs/attr_v10_2x2.log`, `logs/attr_v10_abl.log` trên IBM.

### 1.2 Phân rã ~30 điểm u (semantic bank, ep80 — cô lập từng trục)

```
Hình học đóng băng (wm0_ba0, semantic)          u = 59.7   ← trần "miễn phí" của CLIP
+ BA drift (ws_anchor 0.01 không đủ)            u = 56.4   (−3.3)
+ W_multi (MFA anchored, KHÔNG regularizer)     u = 40.8   (−15.6)  ← THỦ PHẠ 1
+ IPR refinement khi inference (bank refined)  u = 29.9   (−10.9)  ← THỦ PHẠ 2
```
Ba trục **cộng dồn gần như độc lập** (15.6 + 3.3 + 10.9 ≈ 29.8 ≈ 59.7 − 29.9). Hiệu ứng IPR ổn định ~11–12 điểm ở mọi cấu hình ablation (44.9 vs 56.4; 47.7 vs 59.7).

### 1.3 Trajectory & chỉ số phụ trợ

**v10 (có L_trans), log `v10_cub_a0_lt.log`:** u 53→30 monotonic trong 80 ep; T1 cũng decay (72.2@ep11 → 68.5@ep80); ||BA|| bão hòa 1.44 từ ~ep40 trùng lúc u/s bão hòa; wsanchor bão hòa 0.0207 (cân bằng gradient). **v10 (trước fix L_trans), log `v10_cub_a0.log`:** u kết thúc 39.9 — tức **L_trans làm tệ thêm đúng 10 điểm u** (39.9 → 29.9), γ cần thiết tăng 0.07 → 0.12.

**R6b (không W_multi, không IPR — `cub_clip_v9_r6b_noref_v8phi_b16.yaml`):** u decay nhẹ 59.3 → 50.1, T1 ổn định 74.4, ||BA|| drift tới 2.71 (gấp đôi v10) mà u vẫn giữ 50.1 ⇒ **drift Ws không phải cơ chế chính** — đây là sàn decay nội tại của họ loss seen-only.

**Đo trực tiếp tham số các khối mới (IBM, torch.load):**

| Tham số | ep11 | ep40 | ep80 |
|---|---|---|---|
| λ_r (IPR) | −0.047 | −0.062 | −0.066 |
| ‖W_multi‖_F | 7.92 | 15.03 | **16.69** |
| ‖B·A‖ (BA) | — | — | 1.44 |

W_multi **phình gấp 2** giữa ep11→ep80, không bị ràng buộc gì. λ_r **âm và âm dần** — residual IPR bị *đảo dấu*: training phát hiện cách dùng nó là **đẩy prototype ra xa** hướng đọc patch (một seen-margin hack, không phải refinement đúng nghĩa).

### 1.4 Kết luận cơ chế

1. **W_multi học hướng phân-biệt-seen** (chỉ CE + NCE trên seen giám sát nó). Seen gain thật (+15 s), nhưng cùng lúc xoay v_tilde của *mọi* ảnh ra khỏi hình học CLIP; prototype unseen không xoay theo → cos giảm → u sụp. Zero W_multi khi inference: s sụp từ 75→61 (hàng #2 vs #6) chứng tỏ W_multi đã thành **load-bearing cho seen** — hai vai xung đột trong cùng một tensor.
2. **IPR uốn prototype per-image bằng patch-attention chỉ được giám sát trên hàng seen**; unseen nhận noise ngoại suy (−11u). λ_r âm xác nhận nó học hack margin cho seen. Hub_u tăng khi dùng refined (3.32 vs 3.02) — refinement tăng hubness trên unseen.
3. **BA góp ~3 điểm** — thật nhưng nhỏ.
4. **L_trans** không cứu được (teacher ở phi-space, student là m_s tách rời khỏi v_tilde per-sample) và thực nghiệm **làm tệ 10 điểm**.
5. Điểm mấu chốt về metric cuối: calibration (γ=0.12) đã bù lại phần lớn (cal_H 63.3 refined > 61.8 semantic), nhưng đó là crutch quét trên test-set — hư hại nằm ở **biểu diễn**, không phải ở score; γ lớn = dấu hiệu seen-bias nặng và dễ vỡ khi phân phối lệch.

---

## 2. CÁC MODEL ZSL XÂY "KHÔNG GIAN TRUNG GIAN CHUNG" KIỂU GÌ (VĂN LIỆU)

Câu hỏi cốt lõi của đề tài: *"các model ZSL trước đây làm gì với các không gian trung gian được huấn luyện để nó có không gian chung?"* — có 4 thế hệ, và thế hệ càng mới càng **ít dám huấn luyện không gian chung bằng gradient seen-only**:

### 2.1 Thế hệ 1 — Compatibility function (2013–2016)
DeViSE, ALE, ESZSL: học một phép chiếu W: visual → semantic, huấn luyện bằng cặp (ảnh, prototype) **chỉ seen**. Seen-bias là bệnh bẩm sinh; khắc phục ở tầng **score**: calibrated stacking (Chao et al., TPAMI 2016 — repo đã có), sau này thêm CSLS (repo đã có). Bài học: *sửa ở score chỉ là băng bó; hư hại nằm ở phép chiếu.*

### 2.2 Thế hệ 2 — Generative (2018–2021, và vẫn là trục chính 2025–2026)
f-CLSWGAN, TF-VAEGAN, CE-GZSL, FREE; mới nhất: **GenZSL** (arXiv 2505.11882, inductive VAE sinh mẫu unseen từ seen tương đồng + CLIP semantics, ~+24.7% trên AWA2 so với f-VAEGAN), **FSIGenZ** (2507.01026, prototype mức nhóm giảm số mẫu sinh), **ADiVA** (2603.06281). Nguyên lý: **đừng tin phép chiếu học từ seen khi phân loại unseen** — thay vào đó *sinh đặc trưng visual của unseen trong không gian backbone đóng băng*, huấn luyện classifier trên seen + unseen-sinh. Nghĩa là: gradient "unseen" được tiêm vào hệ qua mẫu sinh, không phải để hệ tự ngoại suy.

### 2.3 Thế hệ 3 — CLIP-era embedding GZSL (2021–nay)
Không gian chung **được thừa hưởng đóng băng** từ 400M cặp ảnh-văn. Các kết quả GZSL SOTA trên CUB/AWA2 hiện đến từ: (a) prompt/description ensemble với encoder đóng băng (CuPL — chính là phi v8 đang dùng); (b) **adapter residual nhỏ + retention**: CLIP-Adapter dùng v' = (1−α)·f + α·Wf với **α ≈ 0.2 chủ đích để giữ zero-shot**; Tip-Adapter training-free; (c) prompt-tuning có collaborator đóng băng — **VSPC** (2503.23030) học visual+semantic prompt nhưng không fine-tune backbone để tránh overfit seen; (d) **PSVMA+** (2410.11560 — nguồn của L_attn đang dùng) làm multi-granularity semantic-visual adaption — tức cùng triết lý multi-level như HSPRN nhưng toàn bộ adaption được neo bằng ngữ nghĩa thuộc tính.

### 2.4 Thế hệ 4 — Seen/unseen routing + frozen-branch anchoring (2024–2026)
Hai hướng mới nhất liên quan trực tiếp:

- **Routing:** nhận thấy không một head nào phục vụ cả hai thế giới, người ta **tách đường seen/unseen tại inference**: EZ-AVOOD (2503.22197) và khung AV-OOD-GZSL (2408.01284) — detector OOD chọn expert seen (discriminative) hay expert unseen (frozen/generative). CASH (2509.23807) có module chuyên "chặn bias về seen".
- **Frozen-branch consistency anchoring:** adapter được phép học nhưng phải **trả phí theo độ lệch khỏi nhánh đóng băng**: SeCo-SBIR (2608.03120) — "consistency regularization against a frozen CLIP branch... without losing generalization"; TASSO (2608.21487) — low-rank projector + geodesic knowledge distillation "preserve latent geometry"; LMR (2608.02216) bảo vệ near-top candidates khi adapt. Đây chính là nguyên lý **WiSE-FT** (Wortsman et al., ICLR 2022): nội suy weight fine-tune ↔ zero-shot giữ cả ID lẫn OOD — và nghiên cứu LoRA (2608.09633) xác nhận lại: "LoRA refine ranh giới in-dataset nhưng biểu diễn pretrain mới quyết định generalization cross-dataset".

### 2.5 Chẩn đoán v10 so với các nguyên lý trên

v10 vi phạm đúng điều cả 4 thế hệ đã học tránh: một projector giàu thông tin (W_multi trên 3 level token) + một refiner per-image (IPR) được huấn luyện **hoàn toàn bằng gradient seen-only**, **không có bất kỳ neo/fees nào** giữ chúng gần hình học đóng băng, và unseen **không có nguồn gradient nào** (L_trans dạy m_s, không dạy v_tilde; L_attn chỉ supervise nhánh attribute). Kiểu kiến trúc này đúng bằng proof-by-ablation đã *kiểm chứng*: mất 30 điểm u.

Ngược lại, các thành phần đang hoạt động đúng: freeze backbone + phi đóng băng + ws_anchor cho BA (BA chỉ mất 3 đ), calibrated stacking, và bản thân W_multi *đúng là* ý tưởng đa mức như PSVMA+ multi-granularity — vấn đề không phải ý tưởng mà là **thiếu neo**.

---

## 3. ĐỀ XUẤT GIẢI PHÁP — GIỮ W_MULTI, NEO VÀO ĐÓNG BĂNG

**Nguyên tắc thống nhất của cả gói.** Tổn thất u có một gốc chung: mọi gradient *phân biệt* (l_cls CE, l_align NCE) đều chỉ chứa seen và không transfer sang unseen, trong khi các map khả học (W_multi phía ảnh; Ws+IPR phía text; head điểm số) dùng chung cho cả hai thế giới. Gói fix gán cho mỗi map một ràng buộc đồng hành **không cần ảnh unseen** — suy ra từ ba nguồn sẵn có trong setting inductive: (i) anchor đóng băng per-sample (**neo** — L_fm), (ii) cấu trúc quan hệ theo cặp của toàn bộ tập lớp, vì semantics của unseen luôn có ở lúc train (**bảo toàn** — L_rel), (iii) posterior zero-shot của máy đóng băng trên đủ mọi cột (**chưng cất** — L_zs). Các ràng buộc này đều class-agnostic hoặc full-label-space nên gradient của chúng transfer được sang unseen — khác biệt bản chất so với mọi loss seen-only hiện có.

### Fix 1 (BẮT BUỘC) — L_fm: frozen-anchor cosine retention cho v_tilde

```python
# L_fm = mean_b( 1 − cos( v_tilde_b, normalize(cls)_b ) )
# v_tilde = normalize(cls + W_multi·[GAP(z_l);GAP(z_m);GAP(z_h)])
```
`λ_fm ≈ 1.0` (grid {0.5, 1.0, 2.0} nếu cần).

- **Đây là WiSE-FT ở feature-space** (và là SeCo-SBIR/TASSO-style consistency): W_multi vẫn được học và làm giàu ảnh, nhưng chỉ trong **budget góc** — mọi độ xoay quá mức khỏi anchor phải "trả phí". Thứ bị phạt là *đúng đại lượng* gây sụp (cos(v~, cls)), scale-invariant, không phải proxy thô như phạt norm.
- Kỳ vọng: s giữ phần lớn gain (W_multi vẫn tách seen), u về vùng 50–55 (giữa final 40.8 và wm0 56.4).
- Triển khai (nhỏ, 5 điểm chạm):
  1. `model/hsprn_outputs.py`: thêm field `cls_anchor`.
  2. `model/hsprn_model.py`: stash `F.normalize(features["cls"], dim=-1)` vào outputs.
  3. `losses/hsprn_loss.py`: arg `cls_anchor=None`, `lambda_fm` từ `loss_cfg`, tính L_fm.
  4. `trainer/trainer.py`: pass `cls_anchor=out.cls_anchor`, log `loss_fm`.
  5. config: `model.loss.lambda_fm: 1.0`.
  *(Tùy chọn đi kèm: `mfa_anchor: 1e-3` phạt ‖W_multi‖²_F như ws_anchor — rẻ, nhưng L_fm là chính.)*

### Fix 2 (BẮT BUỘC) — IPR-v2: residual classifier-free có hướng dẫn + budget tiếp tuyến

**Chẩn đoán phải nhắc lại trước khi thiết kế.** λ_r học thành **số âm** (−0.047 → −0.066): training phát hiện cách giảm l_cls không phải bằng cách đưa prototype *gần* nội dung ảnh mà bằng cách **đẩy prototype đối thủ seen ra xa** — một seen-margin hack cài vào gốc tự do của residual. Vấn đề không phải "residual to hay nhỏ" mà là **không gian hàm residual được phép chọn** hoàn toàn tự do: nó chỉ bị giám sát qua l_cls (seen-only), nên mọi hướng giảm CE seen đều hợp lệ — kể cả hướng phá unseen. Do đó fix đúng phải **thay đổi không gian hàm**, không phải clamp/băng bó tham số.

**2a — Residual có hướng dẫn classifier-free (guidance).** Chuyển residual từ dạng cộng tự do

```
p̂ = p_sem + λ_r · gate(V) ⊙ (M_class(V) @ V_t)
```

sang dạng **hiệu có-điều-kiện trừ không-điều-kiện** — nguyên lý classifier-free guidance (Ho & Salimans, *Classifier-Free Diffusion Guidance*, arXiv:2207.12598):

```
p̂ = p_sem + λ_r · [ r(V) − r(∅) ]
r(V) = gate(V) ⊙ (M_class(V) @ V_t)          # residual đọc tokens ảnh thật
r(∅) = gate(∅) ⊙ (M_class(∅) @ V_t(∅))        # residual khi tokens/v_embed = null (0)
```

Toán tử ∅ thay toàn bộ điều kiện ảnh bằng 0 và tính lại gate + attention trong nhánh đó (một forward IPR phụ — vài matmul nhỏ, chi phí không đáng kể; sửa tại [protypes.py:301](model/protypes.py#L301)).

Cơ chế: r(∅) là phần phản ứng **chỉ theo prototype** — đúng chỗ "margin hack seen" cư ngụ, vì nó độc lập với nội dung ảnh và gradient CE cài hack qua chính thành phần này. Khấu trừ r(∅) loại bỏ phần đó; residual còn lại là **tương tác ảnh×prototype** — được tạo bởi attention dùng chung cho mọi lớp nên hành vi trên prototype unseen giống hệt prototype seen. Với ảnh unseen (chưa từng có ở train), nhánh r(V) vẫn đọc được bằng chứng thị giác thật từ CLIP đóng băng — đúng yêu cầu "IPR uốn prototype được khi dùng cả ảnh unseen". λ_r đổi vai từ "độ mạnh residual" thành "guidance scale" (như w trong CFG): khởi tạo dương nhỏ 0.05, không clamp — cơ chế đã vô hiệu hóa con đường hack, không cần băng bó.

**2b — Budget tiếp tuyến: phạt đúng đại lượng hình học của phép uốn.** p̂ bị normalize về mặt cầu, nhưng góc xoay trước normalize không bị ai ràng buộc. Ràng buộc đúng đại lượng là **độ dài tiếp tuyến của góc xoay** (scale-invariant — không cho hệ "né phạt bằng cách co residual"):

```
L_tan = mean_c ‖ p̂_⊥ ‖ / ⟨p̂, p_sem⟩        với p̂_⊥ = p̂ − ⟨p̂,p_sem⟩·p_sem   (= tan(góc xoay))
```

Đây là budget góc cho prototype — đối ứng hình học của L_fm ở phía ảnh (L_fm budget góc của v_tilde quanh cls; L_tan budget góc của p̂ quanh p_sem): mỗi bên không gian chung có một "cung được phép xoay". Tinh thần geodesic/geometry-preservation như TASSO (arXiv:2608.21487) nhưng ở mức prototype, không phải backbone; prototype-trên-mặt-cầu là dạng chuẩn của học prototype hiện đại (Hyperspherical Prototype Networks, Mettes et al., arXiv:1901.10514; tổng quan so sánh 2410.08925: prototype mặt cầu vượt prototype Euclidean).

Phân công lao động rõ ràng với Fix 4: **2b giới hạn tổng độ uốn (cap), L_rel quyết định loại uốn nào được phép** — cap là chặn an toàn scale-invariant, còn thành phần cấu trúc thì do quan hệ theo cặp quản (xem Fix 4) — hai cái không trùng nhau.

**2c (tùy chọn) — Nội dung của residual phải giải thích được theo attribute.** IPR vốn trả về `attr_pred` ([hsprn_model.py:69](model/hsprn_model.py#L69)), hiện chỉ được giám sát qua l_attn (MSE với GT attribute matrix của lớp seen, [hsprn_loss.py:223](losses/hsprn_loss.py#L223)) — tức supervision vẫn mang nhãn seen. Biến thể không-nhãn: L_attr = 1 − cos(attr_pred, normalize(φ_ŷ)) với ŷ = lớp dự đoán bởi refined bank — tín hiệu nội dung **không phụ thuộc seen/unseen** (mọi ảnh đều có ngữ nghĩa), cho IPR một đích giám sát nội dung thay vì chỉ đích phân biệt. Để làm ablation sau E1/E2 (thành phần bổ sung, không bắt buộc).

**Tính chất đảm bảo của IPR-v2:** nếu residual không mang thông tin ảnh (r(V) = r(∅)) thì p̂ → p_sem — nghĩa là *tự động* hạ về đúng sàn semantic đã đo (u ≥ 40.8). IPR-v2 không thể tệ hơn "không có IPR" về phía unseen, trong khi vẫn mở kênh bằng chứng ảnh để kiếm phần thưởng hiếm (attribute đúng của ảnh unseen).

### Fix 3 (BẮT BUỘC) — L_zs: chưng cất posterior zero-shot vào toàn đường inference (thay L_trans)

**Tắt L_trans là cần nhưng chưa đủ.** Câu hỏi khoa học đàng hoàng là thay bằng gì — vì "vị trí" đó (một loss chứa thông tin unseen đổ vào hệ) có giá trị, chỉ là L_trans đi sai chỗ. Sinh bệnh của L_trans: nó dạy **m_s** — đại lượng tách rời khỏi đường tính điểm (m_s = mfa(level_means), không nằm trong scores lúc eval), và teacher của nó là quan hệ φ–φ (phi-space) chứ không phải posterior của máy trên không gian điểm số. Nên gradient của nó không thể sửa đường refined mà eval dùng.

**L_zs — cùng ý tưởng, đúng student, đúng teacher:**

```
teacher: q = softmax( τ · cos(cls_anchor, φ_y) )   ∀ y ∈ 200 lớp   (đóng băng, zero-shot thật)
student: p = softmax( τ · scores_all )             ∀ y ∈ 200 lớp   (chính đường refined của eval)
L_zs = τ² · KL( q ‖ p )
```

- scores_all = msh(v_tilde, p_ref_all) với p_ref_all = refined bank trên **full 200 lớp** — đường tính này đã tồn tại trong trainer ([trainer.py:629-647](trainer/trainer.py#L629-L647), đang bật khi pres_target ∈ {all, unseen}); chỉ cần mở rộng điều kiện bật thêm `lambda_zs > 0`.
- Teacher = máy zero-shot đóng băng đã đo trần bằng ablation (hàng #10: T1 73.1, u 59.7, s 53.9) — dùng cls anchor (không qua W_multi/IPR). cls_anchor có sẵn từ Fix 1.
- **Đây là nguồn gradient unseen đầu tiên đổ thẳng vào đúng đường inference**: cột unseen của refined scores bị kéo về thứ tự zero-shot đóng băng, trong khi l_cls đẩy cột seen — hai lực không tranh chấp cùng trục. Gradient xuyên qua cả W_multi (qua v_tilde), Ws (qua p_all), IPR (qua p̂) đồng thời.
- Là **WiSE-FT dạng hàm (functional)**: Fix 1 neo W_multi per-sample ở feature-space; L_zs neo cả chuỗi Ws→IPR→msh ở posterior-space. Khớp hướng 2025–26: chưng cất từ CLIP/nhánh đóng băng để giữ zero-shot (CLIP-RD, arXiv:2603.25383; GNSP — distillation + bảo toàn cross-modal alignment, arXiv:2507.19839; TASSO — distill để "preserve latent space geometry", arXiv:2608.21487; KD gốc: Hinton et al., arXiv:1503.02531).
- Tham số: λ_zs = 0.5 (grid {0.25, 0.5, 1.0}); τ dùng cùng giá trị với l_cls (không thêm nhiệt độ cạnh tranh). Ablation `L_zs^u`: KL chỉ trên cột unseen (renormalize) nếu thấy s bị kéo xuống.
- Chỉ số theo dõi: mean KL(q‖p) theo epoch — kỳ vọng u nhích lên ngay trong 10 ep đầu dù chưa đụng gì tới IPR.

### Fix 4 — L_rel: bảo toàn quan hệ theo cặp của tập prototype (thay cho bump ws_anchor)

**Vì sao bump ws_anchor 0.02 vừa không đủ vừa sai đại lượng.** ‖B‖_F nhỏ không bảo vệ cấu trúc: B·A chuẩn nhỏ vẫn có thể tách hai prototype gần nhau ra xa nhau (chuẩn nhỏ, tổn hại quan hệ lớn). Mức −3.3u của BA đến từ việc **méo quan hệ giữa các prototype**, nên đại lượng cần bảo toàn phải là quan hệ theo cặp, không phải chuẩn.

**L_rel = relational knowledge distillation (Park et al., CVPR 2019, arXiv:1904.05068) cho Ws** — dạy phiên bản đã-học giữ nguyên cấu trúc quan hệ của phiên bản đóng băng:

```
D_ij = ‖Ws(φ_i) − Ws(φ_j)‖ ;  D⁰_ij = ‖φ_i − φ_j‖        (full 200 lớp, mọi cặp i≠j)
P_i = softmax_j(−D_ij) ;      P⁰_i = softmax_j(−D⁰_ij)     (phân phối khoảng cách theo từng lớp)
L_rel = mean_i KL( P⁰_i ‖ P_i )                            (hoặc dạng ‖P⁰ − P‖² như RKD-D)
```

- **Điểm khoa học:** quyết định phân loại là hàm của quan hệ theo cặp (score = cos v_tilde–prototype, xếp hạng là so sánh các quan hệ). Nếu Ws bảo toàn nguyên vẹn *phân phối khoảng cách* của toàn bộ 200 lớp thì không thể có seen-margin hack ở tầng prototype — mọi hàng unseen giữ nguyên quan hệ tương đối với nhau và với seen. Ws vẫn được học tự do trong null space của các quan hệ đó (tinh thần GNSP, arXiv:2507.19839 — null-space projection — nhưng dạng soft-penalty dễ triển khai); dạng chuẩn hóa theo hàng của RKD-D là **scale-tolerant**: Ws vẫn được phép dãn/thu toàn cục (phục vụ phân biệt) mà không bị phạt, chỉ các biến dạng *idiosyncratic từng cặp* phải trả phí — chính là phân biệt "uốn hợp lý" vs "uốn hack".
- Biến thể (ablation): **SP-KD** (Tung & Mori, arXiv:1907.09682) — giữ nguyên ma trận cosine theo cặp thay vì phân phối khoảng cách; hoặc khóa **chỉ khối unseen** (pres_mask đã có sẵn) nếu thấy khóa full-set cản trở s.
- Triển khai: φ_all có sẵn trong phi cache ([trainer.py:618](trainer/trainer.py#L618)); mỗi bước tính 2 ma trận 200×200 — O(C²·d) ≈ 20M FLOPs/batch, không đáng kể. λ_rel = 0.1 khởi điểm (grid {0.05, 0.1, 0.25}); giữ ws_anchor = 0.01 làm ablation đối chứng.
- Áp ở **semantic bank** (đầu ra Ws) — đúng trục BA. Trục IPR đã được Fix 2 (kiến trúc) + L_zs (gradient unseen) + pres_target=all hiện có phủ.

### Fix 5 — Monitoring để phát hiện tái phát sớm

- Log `mean cos(v_tilde, cls)` mỗi epoch (chỉ số trực quan của budget góc — mong muốn bão hòa > 0.9).
- Log đường u mỗi 5 ep (đã có sẵn) — điều kiện dừng sớm nếu u < 45 sau ep20.

**Không đề xuất:** xoá W_multi (trái yêu cầu đề bài và bỏ +15 s tiềm năng), generative seen-unseen synthesis (thế hệ 2 — chính đáng nhưng là dự án lớn, để làm hướng mở rộng sau nếu Fix 1–2 đạt trần mà cal_H vẫn < 66).

---

## 4. KẾ HOẠCH THÍ NGHIỆM (IBM) & TIÊU CHÍ THÀNH CÔNG

| Bước | Nội dung | Chi phí |
|---|---|---|
| **E0** (ngay, không train) | Phân rã residual IPR của `final.pth` v10: tính r(V) và r(∅) trên val seen + val unseen, đo (i) tỉ lệ chuẩn image-độc-lập ‖r(∅)‖/‖r(V)‖, (ii) độ lớn λ_r·r(∅) trên cột seen vs unseen — kiểm chứng trực tiếp giả thuyết "hack seen-margin cư ngụ trong r(∅)" mà IPR-v2 khấu trừ | ~15 dòng code eval + 1 lần chạy (~5 phút) |
| **E1** | Train 80 ep: v10 + `lambda_fm: 1.0` + `lambda_zs: 0.5` (`lambda_trans: 0`) + `lambda_rel: 0.1` (Fix 1+3+4 — chưa đụng kiến trúc IPR) | 1 run như v10 |
| **E2** | E1 + Fix 2 (IPR-v2: CFG residual r(V)−r(∅), `lambda_r` init 0.05 không clamp, + `lambda_tan: 0.1`) | 1 run |
| **E3** | Grid theo độ nhạy của E1/E2 trên từng trục: λ_fm {0.5, 2.0} · λ_zs {0.25, 1.0} (kèm biến thể L_zs^u nếu s tụt) · λ_rel {0.05, 0.25} · λ_tan {0.05, 0.25} | 0–3 run |
| **E4** (tùy chọn, sau) | Fix 2c (L_attr không-nhãn) như ablation bổ sung nếu E2 còn dư địa | 1 run |

Thứ tự có chủ đích: E0 kiểm chứng giả thuyết trước khi viết code IPR-v2; E1 tách bạch hiệu ứng của ba ràng buộc hình học (không đổi kiến trúc — so song song với v10 được); E2 bổ sung trục kiến trúc IPR lên nền đã có ràng buộc; E3 chỉ grid khi đã thấy hướng đáp.

**Tiêu chí thành công (so tại ep80, cùng protocol):**

| Metric | Hiện tại (v10 final) | R6b sàn | Mục tiêu |
|---|---|---|---|
| u (raw) | 29.9 | 50.1 | **≥ 50** |
| T1 unseen | 68.5 | 74.4 | **≥ 74** |
| H (raw) | 43.5 | 56.4 | **≥ 56** |
| cal_H | 63.3 | 61.4 | **≥ 64.4** (kỷ lục hiện tại) |
| γ cần thiết | 0.12 | 0.02 | ≤ 0.05 |

Trần tham chiếu: hình học đóng băng cho u=59.7/H=56.7 nhưng s chỉ 53.9 — **mục tiêu không phải sàn đó** mà là điểm cân bằng H cao nhất; mỗi phần đăng của W_multi cho s phải "mua" bằng budget góc có giới hạn.

---

## 5. NGUỒN THAM KHẢO

**Xác minh qua arXiv API (09/2026):** GenZSL [2505.11882] · FSIGenZ [2507.01026] · ADiVA [2603.06281] · VSPC prompt-collab [2503.23030] · PSVMA+ [2410.11560] · EZ-AVOOD [2503.22197] · AV-OOD-GZSL framework [2408.01284] · CASH [2509.23807] · SeCo-SBIR [2608.03120] · TASSO [2608.21487] · LMR [2608.02216] · LoRA cross-dataset study [2608.09633] · CLIP-RD [2603.25383] · GNSP [2507.19839].

**Kinh điển (từ trí nhớ — IDs chuẩn, nên double-check khi viết paper):** Chao et al. TPAMI 2016 (calibrated stacking) · Xian et al. CVPR 2018 (f-CLSWGAN, benchmark u/s/H) · Radford et al. 2021 (CLIP) · Gao et al. 2104.01787 (CLIP-Adapter, residual α) · Zhang et al. NeurIPS 2021 (Tip-Adapter) · Wortsman et al. ICLR 2022 (WiSE-FT) · Pratt et al. 2212.08773 (CuPL — nguồn của phi v8).

**Tham chiếu của gói fix (đã xác minh qua arXiv API 09/2026):** Ho & Salimans, Classifier-Free Guidance [2207.12598] · Park et al., Relational Knowledge Distillation (RKD), CVPR 2019 [1904.05068] · Tung & Mori, Similarity-Preserving KD, ICCV 2019 [1907.09682] · Hinton et al., Distilling the Knowledge [1503.02531] · Mettes et al., Hyperspherical Prototype Networks, CVPR 2019 [1901.10514] · tổng quan prototype trên mặt cầu [2410.08925] · CLIP-RD (distill giữ zero-shot) [2603.25383] · GNSP (distill + bảo toàn alignment, null-space) [2507.19839] · TASSO (preserve latent geometry) [2608.21487] · WiSE-FT (Wortsman et al., ICLR 2022 — cơ sở "functional WiSE-FT" của L_zs).

*Log bằng chứng: `logs/attr_v10_2x2.log`, `logs/attr_v10_abl.log`, `logs/v10_cub_a0.log`, `logs/v10_cub_a0_lt.log`, `logs/cub_clip_v9_r6b_noref_v8phi_b16/HSPRN-CUB.log` (IBM).*
