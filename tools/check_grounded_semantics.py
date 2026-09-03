#!/usr/bin/env python
"""Gate G1: static text audit of grounded class semantics (no model, no GPU).

Checks the contract that the v9 semantic design depends on:
  1. coverage   : every class has name_c + >=8 attr_c + >=8 desc_c + >=1 hier_c
  2. form       : every item is one uniform sentence that names its own class
  3. cleanliness: no markdown, no numbering, <=12 descriptor words, <=77 CLIP tokens
  4. cone check : how often the same content recurs across classes, measured as
                  exact-phrase recurrence, raw word Jaccard and IDF-weighted
                  cosine (the failure mode that made the old semantics pure noise)

Usage:
    python tools/check_grounded_semantics.py --semantics data_semantics/CUB/class_descriptions_cub_grounded_after_llm.json
"""

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.clip_token_budget import count_tokens

FIELDS = ("attr_c", "desc_c", "hier_c")
MIN_ITEMS = {"attr_c": 8, "desc_c": 8, "hier_c": 1}
MAX_DESCRIPTOR_WORDS = 12
MAX_CLIP_TOKENS = 77
MARKDOWN_RE = re.compile(r"[*_`#]|^\s*[-•]|\[\d+\]")
STOP_WORDS = {
    "a", "an", "the", "which", "has", "have", "is", "are", "of", "and", "with",
    "in", "on", "to", "for", "its", "it", "that", "this", "<cls>",
}


def norm(text: str) -> str:
    return " ".join(str(text).lower().replace("+", " ").replace("_", " ").split())


def descriptor_of(sentence: str, class_name: str) -> str:
    """Strip the template shell so recurrence can be measured across classes."""
    text = sentence.strip().rstrip(".")
    text = norm(text).replace(norm(class_name), " <cls> ")
    text = re.sub(r"^a\s+<cls>\s*,?\s*which\s+(has|is|lacks)\s+", "", text.strip())
    return " ".join(text.split())


def audit(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Semantics file must be a dict keyed by class name.")

    problems = []
    phrases = {field: collections.Counter() for field in FIELDS}
    wordsets = {field: {} for field in FIELDS}
    lengths = []
    for class_name, record in data.items():
        if not isinstance(record, dict) or record.get("_error"):
            problems.append(f"{class_name}: lỗi sinh dữ liệu {record.get('_error') if isinstance(record, dict) else record}")
            continue
        if norm(record.get("name_c", "")) != norm(class_name):
            problems.append(f"{class_name}: name_c không khớp key")
        words_by_field = {field: set() for field in FIELDS}
        for field in FIELDS:
            items = record.get(field) or []
            if len(items) < MIN_ITEMS[field]:
                problems.append(f"{class_name}.{field}: {len(items)} items (cần >= {MIN_ITEMS[field]})")
            for item in items:
                lengths.append(count_tokens(item))
                if MARKDOWN_RE.search(item):
                    problems.append(f"{class_name}.{field}: ký tự markdown -> {item}")
                    continue
                if not item.lower().startswith("a "):
                    problems.append(f"{class_name}.{field}: không bắt đầu bằng 'a ' -> {item}")
                elif norm(class_name) not in norm(item):
                    problems.append(f"{class_name}.{field}: thiếu tên lớp -> {item}")
                elif not re.search(r", which (has|is|lacks) ", item):
                    problems.append(f"{class_name}.{field}: sai template -> {item}")
                if not item.endswith("."):
                    problems.append(f"{class_name}.{field}: thiếu dấu chấm -> {item}")
                if count_tokens(item) > MAX_CLIP_TOKENS:
                    problems.append(f"{class_name}.{field}: {count_tokens(item)} CLIP tokens -> {item}")
                descriptor = descriptor_of(item, class_name)
                if len(descriptor.split()) > MAX_DESCRIPTOR_WORDS:
                    problems.append(f"{class_name}.{field}: descriptor dài -> {item}")
                phrases[field][descriptor] += 1
                words_by_field[field].update(w for w in descriptor.split() if w not in STOP_WORDS)
        for field in FIELDS:
            wordsets[field][class_name] = words_by_field[field]
    return data, problems, lengths, phrases, wordsets


def mean_pairwise_jaccard(wordsets, samples=4000, seed=0):
    """Độ chồng từ vựng giữa hai lớp bất kỳ: proxy tại chỗ cho hiện tượng nón."""
    sets = [s for s in wordsets.values() if s]
    if len(sets) < 2:
        return float("nan")
    rng = random.Random(seed)
    total, count = 0.0, 0
    for _ in range(samples):
        first, second = rng.sample(sets, 2)
        union = len(first | second)
        if union:
            total += len(first & second) / union
            count += 1
    return total / max(count, 1)


def idf_cosine(wordsets, samples=4000, seed=0):
    """Cosine trọng số IDF giữa hồ sơ từ của hai lớp bất kỳ.

    Jaccard thô phạt mọi từ chung, kể cả những từ tất yếu ("bill", "wing"). attr_c
    của v9 được sinh tất định từ attribute GT nên hiển nhiên trùng cụm từ ngắn; IDF
    mới trả lời đúng câu hỏi "hai lớp có nói chung một nội dung HIẾM GẶP không",
    tức là đại lượng gần với hiện tượng nón hơn.
    """
    names = [name for name, words in wordsets.items() if words]
    if len(names) < 2:
        return float("nan")
    doc_count = collections.Counter()
    for name in names:
        doc_count.update(wordsets[name])
    n = len(names)
    vectors = []
    for name in names:
        weights = {w: math.log(n / (1 + doc_count[w])) for w in wordsets[name]}
        norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        vectors.append((weights, norm))
    rng = random.Random(seed)
    total = 0.0
    for _ in range(samples):
        first, second = rng.sample(vectors, 2)
        small, big = (first, second) if len(first[0]) <= len(second[0]) else (second, first)
        total += sum(w * big[0].get(word, 0.0) for word, w in small[0].items()) / (
            small[1] * big[1]
        )
    return total / samples


def summarize(tag, field, phrases_counter, wordsets):
    total = sum(phrases_counter.values())
    if not total:
        return
    shared = sum(count for count in phrases_counter.values() if count >= 10)
    words = collections.Counter(word for words in wordsets.values() for word in words)
    print(f"{tag} {field:7s}: {total} item | Jaccard từ giữa 2 lớp = "
          f"{mean_pairwise_jaccard(wordsets):.3f} | IDF-cos giữa 2 lớp = "
          f"{idf_cosine(wordsets):.3f}")
    print(f"       {100.0 * shared / total:4.1f}% item thuộc cụm lặp >=10 lớp | "
          f"{len(phrases_counter)} cụm khác nhau")
    print(f"       từ lặp nhiều nhất: "
          + ", ".join(f"{w}({n})" for w, n in words.most_common(6)))
    print(f"       cụm lặp nhiều nhất: "
          + " ; ".join(f"{n}x {p}" for p, n in phrases_counter.most_common(2)))


def field_stats(data):
    phrases = {field: collections.Counter() for field in FIELDS}
    wordsets = {field: {} for field in FIELDS}
    for class_name, record in data.items():
        if not isinstance(record, dict):
            continue
        for field in FIELDS:
            words = set()
            for item in record.get(field) or []:
                if not isinstance(item, str):
                    continue
                descriptor = descriptor_of(item, class_name)
                phrases[field][descriptor] += 1
                words.update(w for w in descriptor.split() if w not in STOP_WORDS)
            wordsets[field][class_name] = words
    return phrases, wordsets


def report_redundancy(phrases, wordsets, baseline_path: Path):
    print("\n--- B redundancy: cùng một nội dung có bị lặp giữa các lớp không ---")
    for field in FIELDS:
        summarize("MỚI", field, phrases[field], wordsets[field])
    if baseline_path and baseline_path.is_file():
        base_phrases, base_wordsets = field_stats(
            json.loads(baseline_path.read_text(encoding="utf-8"))
        )
        print(f"\n  baseline cũ: {baseline_path.name}")
        for field in FIELDS:
            summarize("CŨ", field, base_phrases[field], base_wordsets[field])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semantics", required=True)
    parser.add_argument("--baseline", default="data_semantics/CUB/class_descriptions_cub_after_llm_plan5.json")
    parser.add_argument("--expect_classes", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.semantics)
    data, problems, lengths, phrases, wordsets = audit(path)
    print(f"File: {path}")
    print(f"Số lớp: {len(data)} (kỳ vọng {args.expect_classes})")
    print(f"CLIP tokens/item: max={max(lengths) if lengths else 0}, "
          f"trung bình={sum(lengths) / len(lengths):.1f}")

    if len(data) != args.expect_classes:
        problems.append(f"Thiếu lớp: {len(data)}/{args.expect_classes}")
    report_redundancy(phrases, wordsets, Path(args.baseline))

    print("\n--- Kết luận ---")
    if problems:
        print(f"KHÔNG ĐẠT: {len(problems)} lỗi")
        for line in problems[:40]:
            print("  -", line)
        sys.exit(1)
    print("ĐẠT G1: toàn bộ lớp đủ 4 kênh, đúng template, sạch markdown, trong ngân sách token.")


if __name__ == "__main__":
    main()
