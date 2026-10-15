# SPRN — Latent Prototype Refinement (LPR)

CLIP ViT-B/16 (frozen) + K latent slots that read patch tokens (`ipr`), anchored multi-level
aggregator (`mfa`), low-rank semantic adapter `Ws = I + BA` (`psc`), 3-stage training
(grounding -> local-global -> Ws calibration). Datasets: CUB, AWA2, aPY, SUN | MIT-States,
UT-Zappos, C-GQA | COCO, NUS-WIDE. Config = `methods/lpr` < `families/*` < `datasets/*`.

```
train.py  eval.py  config/{methods,families,datasets}  model/  losses/  trainer/  datasets/  tools/  tests/
```

```bash
python train.py --dataset cub                      # any of: cub awa2 apy sun mit utzap cgqa coco nuswide
python train.py --dataset cub --smoke              # 3-epoch pipeline check
python eval.py  --dataset cub --checkpoint checkpoints/lpr/cub/CUB/final.pth --output /tmp/cub.json
python tools/build_latent_slots.py --dataset cub --n_slots 64 --device cuda   # slots + slot targets
python tools/precompute_phi.py --dataset cub --output pre-trained/cub_phi_b16.pt
python -m pytest -q tests                          # CPU, no weights
```

phi (class text embedding) is the mean of the field-means (name, attr, hier, desc) of raw,
un-normalised CLIP sentence embeddings; it is not the Eq. 1 sentence-mean.
