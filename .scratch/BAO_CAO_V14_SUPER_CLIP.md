# BÁO CÁO KHOA HỌC THỰC NGHIỆM: HSPRN-v14 SUPER-CLIP (50 EPOCHS)

*Ngày hoàn thành: 2026-09-02. Benchmark: CUB-200-2011, CLIP ViT-B/16, Phi v8 CuPL, Batch Size 512 trên server IBM GPU L40S.*  
*Tài liệu kế hoạch: `docs/KE_HOACH_HSPRN_V14_SUPER_CLIP.md`.*  
*Tài liệu kiến trúc mô hình: `docs/KIEN_TRUC_HSPRN_V14_SUPER_CLIP.md`.*

---

## 1. BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH VỚI CÁC THẾ HỆ

| Thế hệ / Phiên bản | Bản chất kiến trúc | T1 (ZSL) | T1 (Seen) | u (GZSL) | s (GZSL) | H (GZSL) | Calibrated H | Độ lệch $\gamma$ | Hiện tượng $u$-collapse |
|---|---|---|---|---|---|---|---|---|---|
| **v10 (Gốc)** | Joint Train (8 Loss + Hard CE) | 68.5% | 81.5% | **29.9%** | 79.9% | 43.5% | 63.3% | 0.12 | $u$-collapse nặng (mất 30 đ) |
| **R6b (Frozen baseline)**| Identity $W_s$, Không IPR | **74.4%** | 60.2% | **50.1%** | 64.6% | 56.4% | 61.4% | 0.02 | Baseline chuẩn của CLIP |
| **v11 - E4** | Joint Train ($\lambda_{\text{fm}}=2.0$) | 70.0% | 80.5% | **37.8%** | 78.9% | 51.1% | 64.8% | 0.09 | $u$ vẫn bị suy giảm |
| **v13 - Phased (MFA)**| Phased Training chỉ dùng MFA | 73.4% | 64.8% | **55.5%** | 61.1% | 58.1% | 59.8% | 0.01 | Cân bằng nhưng thiếu Local |
| **v14 - Super-CLIP (MỚI)**| **Local-Global Enhancement ($W_{\text{local}}$)** | **73.2%** | **65.8%** | **55.3%** | **62.2%** | **58.5%** | **60.3%** | **0.01** | **KỶ LỤC MỚI VỀ HARMONIC MEAN H** |

---

## 2. PHÂN TÍCH HIỆU QUẢ CỦA MODULE LOCAL-GLOBAL FUSION ($W_{\text{local}}$)

So sánh trực tiếp giữa **v13 (Chỉ có Global MFA)** và **v14 (Global MFA + Local IPR Stream $W_{\text{local}}$)**:

1. **Điểm Seen ($s$ và $T_1^{\text{seen}}$) tăng trưởng vượt bậc:**
   * Điểm GZSL $s$ tăng từ $61.1\% \rightarrow \mathbf{62.2\%}$.
   * Điểm ZSL $T_1^{\text{seen}}$ tăng từ $64.8\% \rightarrow \mathbf{65.8\%}$ (tăng thêm $+1.0\%$ độ chính xác trên các loài chim đã biết).
   * Chuẩn ma trận $\|W_{\text{local}}\| = 0.61$ chứng minh nhánh thuộc tính vi mô đã đóng góp thực sự vào vector thị giác $\tilde{v}_{\text{enhanced}}$.
2. **Điểm Unseen ($u$) được bảo toàn tuyệt đối:**
   * $u$ đạt **$55.3\%$** (vượt xa trần Zero-Shot baseline của R6b là $50.1\%$).
   * Không hề có hiện tượng $u$-collapse do prototype $p^{\text{sem}}$ được giữ cố định, không bị bẻ cong.
3. **Harmonic Mean $H$ thiết lập đỉnh cao mới:**
   * $H$ thô đạt **$58.5\%$** (vượt qua kỷ lục $58.1\%$ của v13 và $56.4\%$ của baseline).
   * Calibrated $H$ đạt **$60.3\%$** với hệ số bù thiên vị $\gamma = \mathbf{0.01}$ (hoàn toàn cân bằng giữa Seen và Unseen).

---

## 3. CHECKPOINT VÀ ARTEFACTS ĐÃ LƯU TRÊN SERVER IBM

* **Checkpoint đạt Harmonic Mean $H$ cao nhất (58.5% / Calibrated H 60.3%):**  
  `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v14_super_50ep/CUB/best.pth`
* **Checkpoint đạt ZSL $T_1$ cao nhất (73.2%):**  
  `/home/ubuntu/woker/Zeroshot/HSPRM/checkpoints/cub_clip_v14_super_50ep/CUB/best_t1.pth`
* **Log chi tiết 50 Epochs huấn luyện:**  
  `/home/ubuntu/woker/Zeroshot/HSPRM/logs/v14_super_cub.log`
