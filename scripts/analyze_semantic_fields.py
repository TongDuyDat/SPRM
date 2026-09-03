"""
Phân tích chuyên sâu: Khả năng phân biệt của từng trường semantic
trong không gian CLIP (name_c, attr_c, hier_c, desc_c).

Mục tiêu: Xác định xem equal-weight mean pooling có làm loãng tín hiệu
phân biệt hay không, và trọng số tối ưu cho từng trường là gì.

Chạy trên IBM (cổng G2 của v9, đo trên đúng backbone ViT-B/16):
  python scripts/analyze_semantic_fields.py \
      --semantics ./data_semantics/CUB/class_descriptions_cub_grounded_after_llm.json \
      --clip_model ./pre-trained/ViT-B-16.pt
"""
import json, os, sys, argparse, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_clip_text(model_path="./pre-trained/ViT-B-32.pt", device="cuda"):
    from model.sematics.clip import load_text, tokenize
    model = load_text(model_path, device=device)
    return model, tokenize

def encode_texts(model, tokenize_fn, texts, device="cuda", batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenize_fn(batch, context_length=77, truncate=True).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
        all_embs.append(embs.float())
    return F.normalize(torch.cat(all_embs, dim=0), dim=-1)

def _as_item_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if lines:
            return lines
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []

def encode_field(model, tokenize_fn, records, field_keys, device="cuda"):
    embs = []
    for record in records:
        items = []
        for key in field_keys:
            val = record.get(key)
            if val is not None:
                items = _as_item_list(val)
                if items:
                    break
        if not items:
            items = ["unknown"]
        field_emb = encode_texts(model, tokenize_fn, items, device)
        embs.append(field_emb.mean(dim=0))
    return F.normalize(torch.stack(embs, dim=0), dim=-1)

def inter_class_stats(emb_matrix, name):
    cos = emb_matrix @ emb_matrix.T
    n = cos.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=cos.device)
    off_diag = cos[mask]
    return {
        "field": name,
        "mean_cos": off_diag.mean().item(),
        "std_cos": off_diag.std().item(),
        "max_cos": off_diag.max().item(),
        "min_cos": off_diag.min().item(),
        "median_cos": off_diag.median().item(),
        "pct_above_0.9": (off_diag > 0.9).float().mean().item() * 100,
        "pct_above_0.8": (off_diag > 0.8).float().mean().item() * 100,
    }

def nn_overlap(emb_a, emb_b):
    cos_a = emb_a @ emb_a.T
    cos_b = emb_b @ emb_b.T
    cos_a.fill_diagonal_(-float('inf'))
    cos_b.fill_diagonal_(-float('inf'))
    nn_a = cos_a.argmax(dim=1)
    nn_b = cos_b.argmax(dim=1)
    return (nn_a == nn_b).float().mean().item()

def main():
    # Cổng G2 của v9: đo từng field trên đúng backbone sẽ dùng để train.
    parser = argparse.ArgumentParser(description="Per-field semantic discriminability.")
    parser.add_argument("--semantics",
                        default="./data_semantics/CUB/class_descriptions_cub_after_llm.json")
    parser.add_argument("--clip_model", default="./pre-trained/ViT-B-32.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_path = args.semantics
    
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    
    records = list(data.values())
    n_classes = len(records)
    print(f"Loaded {n_classes} classes")
    
    print(f"Loading {args.clip_model} (text only)...")
    model, tokenize_fn = load_clip_text(model_path=args.clip_model, device=device)
    
    fields = {
        "name_c": ["name_c", "name"],
        "attr_c": ["attr_c", "attr", "attributes"],
        "hier_c": ["hier_c", "hier", "hierarchy"],
        "desc_c": ["desc_c", "desc", "description"],
    }
    
    field_embs = {}
    for field_name, keys in fields.items():
        print(f"  Encoding {field_name}...")
        field_embs[field_name] = encode_field(model, tokenize_fn, records, keys, device)
    
    combined = F.normalize(
        torch.stack([field_embs[f] for f in fields], dim=0).mean(dim=0), dim=-1
    )
    
    print("\n" + "="*70)
    print("1. INTER-CLASS COSINE SIMILARITY (thấp = phân biệt tốt)")
    print("="*70)
    print(f"{'Field':<12} {'Mean':>8} {'Std':>8} {'Max':>8} {'Med':>8} {'>0.9%':>8} {'>0.8%':>8}")
    print("-"*70)
    for name in list(fields.keys()) + ["combined"]:
        emb = field_embs.get(name, combined)
        stats = inter_class_stats(emb, name)
        print(f"{stats['field']:<12} {stats['mean_cos']:>8.4f} {stats['std_cos']:>8.4f} "
              f"{stats['max_cos']:>8.4f} {stats['median_cos']:>8.4f} "
              f"{stats['pct_above_0.9']:>7.1f}% {stats['pct_above_0.8']:>7.1f}%")
    
    print("\n" + "="*70)
    print("2. NEAREST-NEIGHBOR OVERLAP (cao = 2 trường có cấu trúc tương tự)")
    print("="*70)
    field_names = list(fields.keys())
    for i in range(len(field_names)):
        for j in range(i+1, len(field_names)):
            ov = nn_overlap(field_embs[field_names[i]], field_embs[field_names[j]])
            print(f"  {field_names[i]} vs {field_names[j]}: {ov*100:.1f}%")
    
    print("\n" + "="*70)
    print("3. NN-MARGIN (gap giữa NN gần nhất và thứ 2, cao = phân biệt rõ)")
    print("="*70)
    for name in list(fields.keys()) + ["combined"]:
        emb = field_embs.get(name, combined)
        cos = emb @ emb.T
        cos.fill_diagonal_(-float('inf'))
        top2 = cos.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        print(f"  {name:<12}: mean NN-margin={margin.mean():.4f}, "
              f"min={margin.min():.4f}, pct_margin<0.01={((margin < 0.01).float().mean()*100):.1f}%")
    
    print("\n" + "="*70)
    print("4. WEIGHTED COMBINATION SEARCH")
    print("="*70)
    weights_to_try = [
        ("equal (1,1,1,1)", [1, 1, 1, 1]),
        ("name-heavy (3,1,1,1)", [3, 1, 1, 1]),
        ("attr-heavy (1,3,1,1)", [1, 3, 1, 1]),
        ("name+attr (2,2,1,1)", [2, 2, 1, 1]),
        ("no-desc (1,1,1,0)", [1, 1, 1, 0]),
        ("name+attr only (1,1,0,0)", [1, 1, 0, 0]),
        ("attr-only (0,1,0,0)", [0, 1, 0, 0]),
        ("name-only (1,0,0,0)", [1, 0, 0, 0]),
    ]
    
    for wname, weights in weights_to_try:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        stacked = torch.stack([field_embs[f] for f in fields], dim=0)
        weighted = (w.view(4, 1, 1) * stacked).sum(dim=0)
        weighted = F.normalize(weighted, dim=-1)
        stats = inter_class_stats(weighted, wname)
        print(f"  {wname:<30}: mean_cos={stats['mean_cos']:.4f}, "
              f"max={stats['max_cos']:.4f}, >0.9={stats['pct_above_0.9']:.1f}%")
    
    print("\n" + "="*70)
    print("5. ABLATION: Bỏ từng trường")
    print("="*70)
    base_stats = inter_class_stats(combined, "all")
    print(f"  Baseline (all 4 fields): mean_cos={base_stats['mean_cos']:.4f}")
    for drop_field in fields:
        remaining = [f for f in fields if f != drop_field]
        stacked = torch.stack([field_embs[f] for f in remaining], dim=0)
        ablated = F.normalize(stacked.mean(dim=0), dim=-1)
        stats = inter_class_stats(ablated, f"drop {drop_field}")
        delta = stats['mean_cos'] - base_stats['mean_cos']
        direction = "BETTER" if delta < 0 else "WORSE"
        print(f"  Drop {drop_field:<8}: mean_cos={stats['mean_cos']:.4f} "
              f"(delta={delta:+.4f} -> {direction})")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

