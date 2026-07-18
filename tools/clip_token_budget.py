#!/usr/bin/env python
"""CLIP BPE token budget helpers (77 tokens per text sample).

The OpenAI CLIP text encoder accepts at most ``context_length`` tokens per
sample, including ``<|startoftext|>`` and ``<|endoftext|>``. Plan 3 enforces
this budget **per description item**, not across a whole class.

This module loads the repo BPE vocab directly so audit/generation can run
without importing the full ``model`` package (torch/timm) or optional
``ftfy`` when unavailable.
"""

from __future__ import annotations

import argparse
import gzip
import html
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_CONTEXT_LENGTH = 77
_BPE_PATH = (
    _REPO_ROOT / "model" / "sematics" / "clip" / "bpe_simple_vocab_16e6.txt.gz"
)

try:
    import regex as _re  # type: ignore
except ImportError:  # pragma: no cover
    import re as _re


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def _basic_clean(text: str) -> str:
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except Exception:
        pass
    return html.unescape(html.unescape(text)).strip()


def _whitespace_clean(text: str) -> str:
    return _re.sub(r"\s+", " ", text).strip()


class ClipBpeTokenizer:
    """Minimal OpenAI CLIP BPE tokenizer for length checks only."""

    def __init__(self, bpe_path: Path | str = _BPE_PATH):
        self.byte_encoder = _bytes_to_unicode()
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = _re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[^\s\w]+""",
            _re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in _re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens


@lru_cache(maxsize=1)
def get_clip_tokenizer() -> ClipBpeTokenizer:
    """Return a cached CLIP BPE tokenizer for token counting."""
    if not _BPE_PATH.is_file():
        raise FileNotFoundError(f"CLIP BPE vocab not found: {_BPE_PATH}")
    return ClipBpeTokenizer(_BPE_PATH)


def count_tokens(text: str, tokenizer=None, context_length: int = DEFAULT_CONTEXT_LENGTH) -> int:
    """Return CLIP token count for one string, including SOT/EOT.

    Matches ``model.sematics.clip.clip.tokenize``: encode content, then wrap
    with start/end special tokens.
    """
    if tokenizer is None:
        tokenizer = get_clip_tokenizer()
    text = "" if text is None else str(text)
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot] + tokenizer.encode(text) + [eot]
    return len(tokens)


def filter_items_by_budget(
    items: Sequence[str],
    *,
    max_tokens: int = DEFAULT_CONTEXT_LENGTH,
    tokenizer=None,
    on_overflow: str = "skip_item",
) -> list[str]:
    """Filter list items that exceed the CLIP token budget.

    Args:
        items: Candidate description items.
        max_tokens: Inclusive max token length (default 77).
        tokenizer: Optional preloaded tokenizer.
        on_overflow: ``skip_item`` drops overflow items; ``reject`` raises
            if any item overflows.

    Returns:
        Items that fit within the budget (order preserved).
    """
    on_overflow = str(on_overflow or "skip_item").lower()
    if on_overflow not in {"skip_item", "reject", "reject_class"}:
        raise ValueError(
            f"Unknown on_overflow={on_overflow!r}. Expected skip_item or reject."
        )
    if tokenizer is None:
        tokenizer = get_clip_tokenizer()

    kept: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        n = count_tokens(text, tokenizer=tokenizer)
        if n > max_tokens:
            if on_overflow in {"reject", "reject_class"}:
                raise ValueError(
                    f"CLIP token budget exceeded ({n} > {max_tokens}): {text[:120]!r}"
                )
            continue
        kept.append(text)
    return kept


def token_stats(items: Iterable[str], tokenizer=None) -> dict:
    """Return min/max/mean token counts and overflow count for items."""
    if tokenizer is None:
        tokenizer = get_clip_tokenizer()
    counts = []
    for item in items:
        text = str(item).strip()
        if text:
            counts.append(count_tokens(text, tokenizer=tokenizer))
    if not counts:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "overflow_77": 0,
        }
    overflow = sum(1 for c in counts if c > DEFAULT_CONTEXT_LENGTH)
    return {
        "count": len(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": sum(counts) / len(counts),
        "overflow_77": overflow,
    }


def main():
    parser = argparse.ArgumentParser(description="Count CLIP BPE tokens for text(s).")
    parser.add_argument("text", nargs="*", help="Text sample(s) to count.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional text file; one sample per line.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Budget used for overflow reporting (default 77).",
    )
    args = parser.parse_args()

    samples: list[str] = list(args.text or [])
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            samples.extend(line.strip() for line in f if line.strip())
    if not samples:
        parser.error("Provide text args and/or --file.")

    tokenizer = get_clip_tokenizer()
    for sample in samples:
        n = count_tokens(sample, tokenizer=tokenizer)
        flag = "OK" if n <= args.max_tokens else "OVERFLOW"
        print(f"{n:3d}  {flag}  {sample[:100]}")


if __name__ == "__main__":
    main()
