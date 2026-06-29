#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_CACHE="${TORCH_HOME:-${HOME}/.cache/torch}/hub/checkpoints"
CLIP_CACHE="${CLIP_CACHE:-${HOME}/.cache/clip}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-${ROOT_DIR}/pre-trained/text}"
if [[ -z "${HF_PYTHON:-}" && -x /opt/conda/micromamba/envs/py310/bin/python ]]; then
    HF_PYTHON="/opt/conda/micromamba/envs/py310/bin/python"
else
    HF_PYTHON="${HF_PYTHON:-python3}"
fi

declare -A URLS=()
declare -A CACHE_KIND=()
declare -A UNAVAILABLE=()
declare -A HF_REPOS=()
declare -A HF_BACKENDS=()
declare -A HF_OUTPUTS=()

URLS[resnet50]="https://download.pytorch.org/models/resnet50-0676ba61.pth"
URLS[resnet101]="https://download.pytorch.org/models/resnet101-63fe2227.pth"

URLS[vit_small_patch16_224]="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth"
URLS[vit_base_patch16_224]="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
URLS[vit_base_patch16_384]="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth"
URLS[vit_base_patch32_384]="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth"
URLS[vit_large_patch16_224]="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth"
URLS[vit_large_patch16_384]="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth"
URLS[vit_large_patch32_384]="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth"

URLS[clip_vit_b32]="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
URLS[clip_vit_b16]="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
URLS[clip_vit_l14]="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
URLS[clip_text_b32]="${URLS[clip_vit_b32]}"

HF_REPOS[tinyclip_text_3m]="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
HF_REPOS[tinyclip_text_19m]="wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
HF_REPOS[sbert_minilm_l3]="sentence-transformers/paraphrase-MiniLM-L3-v2"
HF_REPOS[sbert_minilm_l6]="sentence-transformers/all-MiniLM-L6-v2"
HF_BACKENDS[tinyclip_text_3m]="tinyclip"
HF_BACKENDS[tinyclip_text_19m]="tinyclip"
HF_BACKENDS[sbert_minilm_l3]="sbert"
HF_BACKENDS[sbert_minilm_l6]="sbert"
HF_OUTPUTS[tinyclip_text_3m]="tinyclip-text-3m"
HF_OUTPUTS[tinyclip_text_19m]="tinyclip-text-19m"
HF_OUTPUTS[sbert_minilm_l3]="sbert-minilm-l3"
HF_OUTPUTS[sbert_minilm_l6]="sbert-minilm-l6"

UNAVAILABLE[vit_base_patch32_224]="registry entry has no pretrained URL"
UNAVAILABLE[vit_large_patch32_224]="registry entry has no pretrained URL"

for name in "${!URLS[@]}"; do
    CACHE_KIND["${name}"]="torch"
done
CACHE_KIND[clip_vit_b32]="clip"
CACHE_KIND[clip_vit_b16]="clip"
CACHE_KIND[clip_vit_l14]="clip"
CACHE_KIND[clip_text_b32]="clip"

REGISTRY_MODELS=(
    resnet50
    resnet101
    vit_small_patch16_224
    vit_base_patch16_224
    vit_base_patch32_224
    vit_base_patch16_384
    vit_base_patch32_384
    vit_large_patch16_224
    vit_large_patch32_224
    vit_large_patch16_384
    vit_large_patch32_384
    clip_vit_b32
    clip_vit_b16
    clip_vit_l14
)

STANDARD_MODELS=(
    resnet50
    resnet101
    vit_base_patch16_224
    vit_base_patch32_224
    clip_vit_b16
    clip_vit_b32
)

TEXT_MODELS=(
    clip_text_b32
    tinyclip_text_3m
    tinyclip_text_19m
    sbert_minilm_l3
    sbert_minilm_l6
)

ALL_MODELS=(
    "${REGISTRY_MODELS[@]}"
    tinyclip_text_3m
    tinyclip_text_19m
    sbert_minilm_l3
    sbert_minilm_l6
)

usage() {
    printf '%s\n' \
        "Usage:" \
        "  $0 standard          # models used by current training configs (default)" \
        "  $0 text              # current and candidate semantic text encoders" \
        "  $0 all               # image registry plus text encoder candidates" \
        "  $0 MODEL [MODEL ...] # selected registry models" \
        "  $0 --list            # show availability without downloading"
}

list_models() {
    local name
    printf '%s\n' '[image backbones]'
    for name in "${REGISTRY_MODELS[@]}"; do
        if [[ -n "${URLS[${name}]:-}" ]]; then
            printf 'available   %s\n' "${name}"
        else
            printf 'unavailable %s (%s)\n' "${name}" "${UNAVAILABLE[${name}]}"
        fi
    done
    printf '%s\n' '[text encoders]'
    printf 'supported    %s (OpenAI CLIP text-only backend)\n' clip_text_b32
    for name in tinyclip_text_3m tinyclip_text_19m sbert_minilm_l3 sbert_minilm_l6; do
        printf 'supported    %s (%s; text-only extraction)\n' "${name}" "${HF_REPOS[${name}]}"
    done
}

verify_clip_checksum() {
    local path="$1"
    local url="$2"
    local expected actual
    expected="$(basename "$(dirname "${url}")")"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]]
}

download_hf_model() {
    local name="$1"
    local repo="${HF_REPOS[${name}]}"
    local backend="${HF_BACKENDS[${name}]}"
    local target="${HF_MODEL_ROOT}/${HF_OUTPUTS[${name}]}"

    if [[ -f "${target}/text_encoder_metadata.json" ]]; then
        printf '[cached] %s -> %s\n' "${name}" "${target}"
        return 0
    fi
    if ! PYTHONNOUSERSITE=1 "${HF_PYTHON}" -c "import transformers" >/dev/null 2>&1; then
        printf '[error] %s must provide compatible transformers; set HF_PYTHON if needed\n' "${HF_PYTHON}" >&2
        return 1
    fi

    printf '[extract-text] %s -> %s\n' "${repo}" "${target}"
    PYTHONNOUSERSITE=1 "${HF_PYTHON}" "${ROOT_DIR}/tools/extract_text_encoder.py" \
        --backend "${backend}" --source "${repo}" --output "${target}"
}

download_model() {
    local name="$1"
    local url kind cache_dir target partial

    if [[ -n "${HF_REPOS[${name}]:-}" ]]; then
        download_hf_model "${name}"
        return
    fi
    if [[ -n "${UNAVAILABLE[${name}]:-}" ]]; then
        printf '[skip] %s: %s\n' "${name}" "${UNAVAILABLE[${name}]}"
        return 0
    fi
    if [[ -z "${URLS[${name}]:-}" ]]; then
        printf '[error] unknown backbone: %s\n' "${name}" >&2
        usage >&2
        return 2
    fi

    url="${URLS[${name}]}"
    kind="${CACHE_KIND[${name}]}"
    if [[ "${name}" =~ ^clip_(text|vit)_b32$ && -s "${ROOT_DIR}/pre-trained/ViT-B-32.pt" ]]; then
        printf '[cached] %s -> %s\n' "${name}" "${ROOT_DIR}/pre-trained/ViT-B-32.pt"
        return 0
    fi
    cache_dir="${TORCH_CACHE}"
    [[ "${kind}" == "clip" ]] && cache_dir="${CLIP_CACHE}"
    mkdir -p "${cache_dir}"
    target="${cache_dir}/$(basename "${url}")"

    if [[ -s "${target}" ]]; then
        if [[ "${kind}" != "clip" ]] || verify_clip_checksum "${target}" "${url}"; then
            printf '[cached] %s -> %s\n' "${name}" "${target}"
            if [[ "${name}" =~ ^clip_(vit_)?b32$ && ! -e "${ROOT_DIR}/pre-trained/ViT-B-32.pt" ]]; then
                mkdir -p "${ROOT_DIR}/pre-trained"
                ln -s "${target}" "${ROOT_DIR}/pre-trained/ViT-B-32.pt"
            fi
            return 0
        fi
        mv "${target}" "${target}.invalid.$(date +%s)"
    fi

    partial="${target}.part"
    printf '[download] %s -> %s\n' "${name}" "${target}"
    curl --location --fail --retry 3 --continue-at - \
        --output "${partial}" "${url}"
    mv "${partial}" "${target}"

    if [[ "${kind}" == "clip" ]] && ! verify_clip_checksum "${target}" "${url}"; then
        printf '[error] SHA256 mismatch: %s\n' "${target}" >&2
        return 1
    fi

    if [[ "${name}" =~ ^clip_(vit_)?b32$ ]]; then
        mkdir -p "${ROOT_DIR}/pre-trained"
        if [[ ! -e "${ROOT_DIR}/pre-trained/ViT-B-32.pt" ]]; then
            ln -s "${target}" "${ROOT_DIR}/pre-trained/ViT-B-32.pt"
        fi
    fi
}

command -v curl >/dev/null || {
    printf '[error] curl is required\n' >&2
    exit 1
}

case "${1:-standard}" in
    --help|-h)
        usage
        exit 0
        ;;
    --list)
        list_models
        exit 0
        ;;
    standard)
        models=("${STANDARD_MODELS[@]}")
        ;;
    text)
        models=("${TEXT_MODELS[@]}")
        ;;
    all)
        models=("${ALL_MODELS[@]}")
        ;;
    *)
        models=("$@")
        ;;
esac

for model_name in "${models[@]}"; do
    download_model "${model_name}"
done
