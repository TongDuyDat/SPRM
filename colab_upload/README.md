# Colab upload pack

## Files
- `cub_prepared.tar.gz` — prepared CUB package (Plan 3 after-LLM metadata + raw images)

## Google Drive layout
Upload to:
```text
MyDrive/HSPRN_colab/
├── code.zip
├── cub_prepared.tar.gz          # this file (or extract to prepared_datasets/CUB/)
├── prepared_datasets/           # optional if you extract archive offline
│   └── CUB/
├── pre-trained/                 # optional
├── checkpoints/
└── logs/
```

## Extract on Colab
Notebook `notebooks/colab_train.ipynb` auto-extracts `cub_prepared.tar.gz` if
`prepared_datasets/` is missing.

Manual extract:
```bash
mkdir -p prepared_datasets
tar -xzf cub_prepared.tar.gz -C prepared_datasets
# result: prepared_datasets/CUB/{manifest.yaml,metadata,raw}
```

## Local re-pack
```bash
tar -cf colab_upload/cub_prepared.tar.gz -C prepared_datasets CUB
```
