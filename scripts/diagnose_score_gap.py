"""
Diagnostic: Phân tích score distribution giữa seen/unseen prototypes.
Trả lời câu hỏi: Tại sao ZSL T1 ~40% nhưng GZSL u chỉ ~3%?

Chạy: python scripts/diagnose_score_gap.py --ckpt <path> --config <config>
"""
import argparse, os, sys, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from tools.config import load_config
from model.hsprn_model import HSPRN
from tools.dataset import build_datasets
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build datasets
    train_ds, test_seen, test_unseen, attribute, seen_classes, unseen_classes = build_datasets(cfg)
    
    model = HSPRN(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get all-class prototypes
    class_semantics = test_seen.get_class_semantics()
    phi_cache = model.psc.encode_class_phi(class_semantics).to(device)
    p_all = F.normalize(model.psc.Ws(phi_cache), dim=-1)  # (200, D)

    seen_set = set(int(c) for c in seen_classes)
    unseen_set = set(int(c) for c in unseen_classes)

    # Score test unseen samples
    loader = DataLoader(test_unseen, batch_size=128, shuffle=False, num_workers=4)
    
    all_cos_gt = []
    all_cos_best_seen = []
    all_cos_best_unseen = []
    all_rank_gt = []
    n_processed = 0

    with torch.no_grad():
        for batch in loader:
            if n_processed >= args.max_samples:
                break
            features, labels = batch[:2]
            features = features.to(device)
            labels = labels.to(device)
            
            # Get visual embeddings
            visual_levels = model.vis_encoder(features)
            v_tilde = model.mfa(visual_levels)
            v_tilde = F.normalize(v_tilde, dim=-1)
            
            # Cosine with all prototypes
            cos_all = v_tilde @ p_all.T  # (B, 200)
            
            for i in range(cos_all.shape[0]):
                gt = int(labels[i])
                cos_gt = cos_all[i, gt].item()
                
                # Best seen score
                seen_scores = cos_all[i, list(seen_set)]
                best_seen = seen_scores.max().item()
                
                # Best unseen score (excluding GT)
                unseen_scores = cos_all[i, list(unseen_set)]
                best_unseen = unseen_scores.max().item()
                
                # Rank of GT among all 200
                rank = (cos_all[i] > cos_gt).sum().item() + 1
                
                all_cos_gt.append(cos_gt)
                all_cos_best_seen.append(best_seen)
                all_cos_best_unseen.append(best_unseen)
                all_rank_gt.append(rank)
            
            n_processed += features.shape[0]

    cos_gt = np.array(all_cos_gt)
    cos_best_seen = np.array(all_cos_best_seen)
    cos_best_unseen = np.array(all_cos_best_unseen)
    ranks = np.array(all_rank_gt)

    print(f"\nAnalyzed {len(cos_gt)} unseen test samples")
    print("="*60)
    print("SCORE DISTRIBUTION (cosine similarity)")
    print("="*60)
    print(f"  GT prototype:     mean={cos_gt.mean():.4f}, std={cos_gt.std():.4f}")
    print(f"  Best seen proto:  mean={cos_best_seen.mean():.4f}, std={cos_best_seen.std():.4f}")
    print(f"  Best unseen proto:mean={cos_best_unseen.mean():.4f}, std={cos_best_unseen.std():.4f}")
    print(f"\n  Gap (best_seen - GT): mean={np.mean(cos_best_seen - cos_gt):.4f}")
    print(f"  Gap (best_seen - best_unseen): mean={np.mean(cos_best_seen - cos_best_unseen):.4f}")
    print(f"\n  GT rank among 200: mean={ranks.mean():.1f}, median={np.median(ranks):.0f}")
    print(f"  GT in top-10: {(ranks <= 10).mean()*100:.1f}%")
    print(f"  GT in top-50: {(ranks <= 50).mean()*100:.1f}%")
    print(f"  GT in top-100: {(ranks <= 100).mean()*100:.1f}%")
    
    print("
Done!")

if __name__ == "__main__":
    main()
