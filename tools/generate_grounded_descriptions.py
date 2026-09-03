#!/usr/bin/env python
"""Attribute-grounded class semantics for HSPRN v9.

RCNn (ICCV 2025) §3.3 keeps the LLM on atomic descriptors and assembles the
sentences in code, so every text sample shares one uniform shape. This script
keeps that discipline and additionally grounds it in the dataset's own
attributes:

  name_c : raw class name                                  (0 API calls)
  attr_c : top-10 ground-truth attributes, rarity-weighted (0 API calls)
  desc_c : LLM atomic visual descriptors, RCNn one-shot prompt, grounded on the
           class attributes and assembled by code
  hier_c : LLM hyponyms, accepted only when the returned name really contains
           the class name (RCNn Supp. §E reports sibling-species hallucination
           here), plus the most specific superclass

attr_c selection scores every attribute by certainty * log(1/prevalence) over
the 200 classes, so the channel carries what *discriminates* the class instead
of the "solid breast / black eyes" phrases that recur across the whole
taxonomy. Measured mean pairwise Jaccard of the selected sets: 0.065 versus
0.156 for plain top-certainty picking.

Output is a dict keyed by class name, the same contract as the after-LLM files
consumed by tools/prepare_zsl_dataset_package.py.

Usage:
    python tools/generate_grounded_descriptions.py --dataset CUB --workers 4
    python tools/generate_grounded_descriptions.py --dataset CUB --dry_run 5
"""

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import generate_descriptions as gen
from tools.clip_token_budget import filter_items_by_budget

DEFAULT_SEMANTIC_ROOT = "./data_semantics"
MAX_ITEMS_PER_FIELD = 10
MAX_ITEM_WORDS = 12

# ---------------------------------------------------------------------------
# Ground-truth attribute channel (no LLM, cannot hallucinate)
# ---------------------------------------------------------------------------

PART_NOUNS = {
    "wing": "wings",
    "upperparts": "upperparts",
    "underparts": "underparts",
    "back": "back",
    "breast": "breast",
    "throat": "throat",
    "forehead": "forehead",
    "nape": "nape",
    "belly": "belly",
    "tail": "tail",
    "crown": "crown",
    "primary": "primaries",
    "leg": "legs",
    "bill": "bill",
    "eye": "eyes",
    "upper_tail": "upper tail coverts",
    "under_tail": "under tail coverts",
}
PLURAL_PARTS = {"wings", "upperparts", "underparts", "primaries", "legs", "eyes",
                "upper tail coverts", "under tail coverts"}

BILL_SHAPES = {
    "curved_(up_or_down)": "curved bill",
    "dagger": "dagger-shaped bill",
    "hooked": "hooked bill",
    "needle": "needle-like bill",
    "hooked_seabird": "hooked seabird bill",
    "spatulate": "spatulate bill",
    "all-purpose": "all-purpose bill",
    "cone": "conical bill",
    "specialized": "specialized bill",
}
BILL_LENGTHS = {
    "about_the_same_as_head": "bill about as long as the head",
    "longer_than_head": "bill longer than the head",
    "shorter_than_head": "bill shorter than the head",
}
TAIL_SHAPES = {
    "forked_tail": "forked tail",
    "rounded_tail": "rounded tail",
    "notched_tail": "notched tail",
    "fan-shaped_tail": "fan-shaped tail",
    "pointed_tail": "pointed tail",
    "squared_tail": "squared tail",
}
WING_SHAPES = {
    "rounded-wings": "rounded wings",
    "pointed-wings": "pointed wings",
    "broad-wings": "broad wings",
    "tapered-wings": "tapered wings",
    "long-wings": "long wings",
}
SIZES = {
    "very_small_(3_-_5_in)": "very small body (3 to 5 inches)",
    "small_(5_-_9_in)": "small body (5 to 9 inches)",
    "medium_(9_-_16_in)": "medium body (9 to 16 inches)",
    "large_(16_-_32_in)": "large body (16 to 32 inches)",
    "very_large_(32_-_72_in)": "very large body (32 to 72 inches)",
}
HEAD_PATTERNS = {
    "malar": "malar marking",
    "crested": "crested crown",
    "masked": "masked face",
    "unique_pattern": "unique head pattern",
    "eyebrow": "eyebrow stripe",
    "eyering": "eyering round the eye",
    "eyeline": "eyeline behind the eye",
    "capped": "capped crown",
    "plain": "plain unmarked head",
    "spotted": "spotted head pattern",
    "striped": "striped head pattern",
}
SHAPE_PHRASES = {
    "upright-perching_water-like": "upright perching water-like body shape",
    "chicken-like-marsh": "chicken-like marsh bird shape",
    "long-legged-like": "long-legged shape",
    "duck-like": "duck-like body shape",
    "owl-like": "owl-like body shape",
    "gull-like": "gull-like body shape",
    "hummingbird-like": "hummingbird-like body shape",
    "pigeon-like": "pigeon-like body shape",
    "tree-clinging-like": "tree-clinging shape",
    "hawk-like": "hawk-like body shape",
    "sandpiper-like": "sandpiper-like body shape",
    "upland-ground-like": "upland ground-dwelling shape",
    "swallow-like": "swallow-like body shape",
    "perching-like": "perching bird shape",
}
STRUCTURAL_FAMILIES = {
    "bill_shape": BILL_SHAPES,
    "bill_length": BILL_LENGTHS,
    "tail_shape": TAIL_SHAPES,
    "wing_shape": WING_SHAPES,
    "size": SIZES,
    "shape": SHAPE_PHRASES,
}


def a_an(phrase: str) -> str:
    """Indefinite article; 'uni-'/'use-'/'eu-' words start with a /j/ sound."""
    lowered = phrase[:1].lower()
    if lowered in "aeiou" and not phrase.lower().startswith(("uni", "use", "eu")):
        return "an"
    return "a"


def verbalize(family: str, value: str) -> str:
    """Turn one CUB attribute into a self-contained English noun phrase."""
    if family in STRUCTURAL_FAMILIES:
        return STRUCTURAL_FAMILIES[family][value]
    if family == "head_pattern":
        return HEAD_PATTERNS[value]
    if family.endswith("_color"):
        part = PART_NOUNS[family[:-len("_color")]]
        return f"{value} {part}"
    if family.endswith("_pattern"):
        part = PART_NOUNS[family[:-len("_pattern")]]
        return f"{value} {part}"
    raise ValueError(f"No verbalizer for attribute family {family!r}.")


def attr_phrase_with_article(family: str, value: str) -> str:
    """Noun phrase with its own article when the head noun is singular."""
    phrase = verbalize(family, value)
    if family == "wing_shape" or phrase in PLURAL_PARTS or phrase.split()[-1].endswith("s"):
        return phrase
    return f"{a_an(phrase)} {phrase}"


def load_attribute_table(repo_root: Path):
    attr_lines = [
        line.strip()
        for line in (repo_root / "datasets" / "attributes.txt").read_text().splitlines()
        if line.strip()
    ]
    families, values = [], []
    for line in attr_lines:
        group, value = line.split(" ", 1)[1].split("::")
        families.append(group[len("has_"):])
        values.append(value)
    matrix = np.loadtxt(repo_root / "datasets" / "CUB_200_2011" / "attributes"
                        / "class_attribute_labels_continuous.txt")
    return families, values, matrix


def read_indexed_names(path: Path):
    """Same normalisation as tools/prepare_zsl_dataset_package.read_indexed_text_file."""
    names = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) > 1 and parts[0].isdigit():
            value = " ".join(parts[1:])
        elif "." in line and line.split(".", 1)[0].isdigit():
            value = line.split(".", 1)[1]
        else:
            value = line
        names.append(value.replace("+", " ").replace("_", " "))
    return names


def select_attributes(matrix, families, values, k=MAX_ITEMS_PER_FIELD, min_struct=2):
    """Per-class attribute indices, ranked by certainty x rarity, capped per family."""
    classes = matrix.shape[0]
    prevalence = (matrix >= 50).mean(axis=0)
    rarity = np.log(1.0 / np.clip(prevalence, 1e-3, 1.0))
    score = matrix * rarity
    color_family = {f for f in set(families) if f.endswith("_color")}

    selected = []
    for index in range(classes):
        ranked = list(np.argsort(-score[index]))
        confident = [j for j in ranked if matrix[index, j] >= 50]
        relaxed = [j for j in ranked if 20 <= matrix[index, j] < 50]
        structural = [j for j in confident if families[j] in STRUCTURAL_FAMILIES]
        chosen, used_families = [], set()
        color_count = collections.Counter()

        def take(j):
            family = families[j]
            if family in used_families:
                return
            if family in color_family and color_count[values[j]] >= 3:
                return
            chosen.append(j)
            used_families.add(family)
            if family in color_family:
                color_count[values[j]] += 1

        for j in structural[:min_struct]:
            take(j)
        for j in confident + relaxed:
            if len(chosen) >= k:
                break
            take(j)
        selected.append(chosen[:k])
    return selected


# ---------------------------------------------------------------------------
# LLM channels (RCNn Supp. §B prompts, atomic output, code-side templating)
# ---------------------------------------------------------------------------

ONE_SHOT = """EXAMPLE
Q: What are useful features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- a flat rectangular screen
- a thin dark frame around the screen
- a stand at the bottom
- buttons on the side of the set
- a small power indicator light
- a remote control in front of it
- cables running behind the screen
- a glossy reflective surface
- a speaker grille below the screen
- a mounting bracket on the back"""

DESC_PROMPT = """You list visual features that distinguish one bird species in a photo.

Output rules:
- One feature per line, each a short noun phrase of at most 12 words.
- No sentences, no periods, no numbering, no markdown, no preamble, no closing remark.
- Do not repeat or paraphrase the known attributes below; add other visible features.
- Only state features that are visually true of this species.

{one_shot}

Q: What are useful features for distinguishing a {class_name} in a photo?
Known attributes of a {class_name}: {known}
A: There are several useful visual features to tell there is a {class_name} in a photo:
(Please generate it according to the example style above.)
-"""

DESC_MORE_PROMPT = DESC_PROMPT.replace(
    "(Please generate it according to the example style above.)",
    "(Give 10 different ones, not listed above, in the example style.)",
)

SUB_PROMPT = """Output only one list of short names separated by '&', no numbering, no other text.

Generate a list of 10 types of the following category and output the list separated by '&' (without numbers): {class_name}"""

SUPER_PROMPT = """Output only one list of short names separated by '&', no numbering, no other text.

Generate a list of 3 super-categories of the following category and output the list separated by '&' (without numbers): {class_name}"""

GENERIC_SUPER = {
    "bird", "birds", "animal", "animals", "creature", "creatures", "organism",
    "species", "living thing", "living things", "life", "vertebrate", "bird species",
    "type of bird", "kind of bird", "group of birds", "avian", "birdlife",
}

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.\)])\s*")


def clean_item(text: str) -> str:
    """Normalize one LLM line into an atomic descriptor, or '' when unusable."""
    item = _BULLET_RE.sub("", str(text)).strip()
    item = item.replace("**", "").replace("`", "").strip(" \t\"';,.")
    item = " ".join(item.split())
    if not item or len(item.split()) > MAX_ITEM_WORDS:
        return ""
    if "." in item or ":" in item or item.startswith("("):
        return ""
    return item


def split_amp(text: str) -> list[str]:
    return [clean_item(part) for part in re.split(r"[&\n]", text) if clean_item(part)]


def tmpl_has(class_name: str, descriptor: str) -> str:
    return f"a {class_name}, which has {descriptor}."


def tmpl_is(subject: str, superclass: str) -> str:
    return f"a {subject}, which is {a_an(superclass)} {superclass}."


def norm_key(text: str) -> str:
    return " ".join(str(text).replace("+", " ").replace("_", " ").lower().split())


def singular(term: str) -> str:
    """'Albatrosses' -> 'albatross', 'Auks' -> 'auk' (superclasses read as common nouns)."""
    word = term.strip().lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("shes", "ches", "xes", "zes", "ses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def build_entry(args, class_name: str, attr_pairs) -> dict:
    entry = {"name_c": class_name}
    entry["attr_c"] = [tmpl_has(class_name, attr_phrase_with_article(f, v)) for f, v in attr_pairs]
    known = "; ".join(v.replace("_", " ") for _, v in attr_pairs)

    prompt = DESC_PROMPT.format(one_shot=ONE_SHOT, class_name=class_name, known=known)
    items = [clean_item(x) for x in gen.parse_items(gen.call_with_retry(args, prompt))]
    if len(items) < args.min_items:
        more = DESC_MORE_PROMPT.format(one_shot=ONE_SHOT, class_name=class_name, known=known)
        seen = {norm_key(i) for i in items}
        for item in [clean_item(x) for x in gen.parse_items(gen.call_with_retry(args, more))]:
            if norm_key(item) not in seen:
                items.append(item)
                seen.add(norm_key(item))
    if len(items) < args.min_items:
        raise ValueError(f"{class_name}: only {len(items)} usable descriptors.")
    entry["desc_c"] = [tmpl_has(class_name, item) for item in items[:MAX_ITEMS_PER_FIELD]]

    hyponyms = [
        item for item in split_amp(gen.call_with_retry(args, SUB_PROMPT.format(class_name=class_name)))
        if class_name.lower() in item.lower() and norm_key(item) != norm_key(class_name)
    ]
    supers = [
        item for item in split_amp(gen.call_with_retry(args, SUPER_PROMPT.format(class_name=class_name)))
        if norm_key(item) != norm_key(class_name) and len(item.split()) <= 4
    ]
    specific = [item for item in supers if singular(item) not in GENERIC_SUPER]
    hier = [tmpl_is(item, class_name) for item in hyponyms[:3]]
    for super_name in (specific or supers)[:2]:
        hier.append(tmpl_is(class_name, singular(super_name)))
    hier = list(dict.fromkeys(hier))
    if not hier:
        raise ValueError(f"{class_name}: no valid hierarchy statement.")
    entry["hier_c"] = hier[:4]

    for field, field_min in (("attr_c", args.min_items), ("desc_c", args.min_items), ("hier_c", 1)):
        entry[field] = filter_items_by_budget(
            entry[field], max_tokens=args.max_clip_tokens, on_overflow="skip_item"
        )
        if len(entry[field]) < field_min:
            raise ValueError(f"{class_name}::{field}: {len(entry[field])} items left after token filter.")
    return entry


def complete_entry(entry) -> bool:
    return bool(
        isinstance(entry, dict)
        and entry.get("name_c")
        and len(entry.get("attr_c") or []) >= 8
        and len(entry.get("desc_c") or []) >= 8
        and len(entry.get("hier_c") or []) >= 1
        and "_error" not in entry
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="CUB")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--image_root", default="./datasets")
    parser.add_argument("--semantic_root", default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry_run", type=int, default=None,
                        help="Print the code-side attr_c channel for N classes and exit (no API call).")
    parser.add_argument("--api_base", default=gen.DEFAULT_API_BASE)
    parser.add_argument("--api_key", default=gen.DEFAULT_API_KEY)
    parser.add_argument("--model", default=gen.DEFAULT_MODEL)
    parser.add_argument("--max_tokens", type=int, default=gen.DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=gen.DEFAULT_TEMPERATURE)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_new_classes", type=int, default=None)
    parser.add_argument("--min_items", type=int, default=8)
    parser.add_argument("--max_clip_tokens", type=int, default=77)
    parser.add_argument("--force_overwrite_legacy", action="store_true")
    args = parser.parse_args()
    if not args.output:
        ds = args.dataset.upper()
        args.output = str(Path(args.semantic_root) / ds
                          / f"class_descriptions_{ds.lower()}_grounded_after_llm.json")
    return args


def load_classes(args) -> list[str]:
    dataset = args.dataset.upper()
    allclasses = Path(args.data_root) / dataset / "allclasses.txt"
    if not allclasses.exists():
        raise FileNotFoundError(f"Missing {allclasses}; run the package builder first.")
    names = read_indexed_names(allclasses)
    if dataset == "CUB":
        source = Path(args.image_root) / "CUB_200_2011" / "classes.txt"
        by_name = {}
        for line in source.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ident, label = parts[0], " ".join(parts[1:])
            by_name[" ".join(label.split(".", 1)[1].replace("_", " ").split())] = int(ident) - 1
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Classes without an attribute row: {missing[:5]}")
        return names, [by_name[name] for name in names]
    raise NotImplementedError(f"Grounded attributes for {dataset} need their own loader.")


def main():
    args = parse_args()
    names, rows = load_classes(args)
    families, values, matrix = load_attribute_table(_REPO_ROOT)
    picks = select_attributes(matrix[rows], families, values)

    if args.dry_run:
        for i in range(min(args.dry_run, len(names))):
            print(f"### {names[i]}")
            for j in picks[i]:
                print("   ", tmpl_has(names[i], attr_phrase_with_article(families[j], values[j])))
        return

    output_path = Path(args.output)
    gen.assert_safe_output_path(output_path, args.force_overwrite_legacy)
    data = gen.load_output(output_path)
    pending = [i for i, name in enumerate(names) if not complete_entry(data.get(name))]
    if args.max_new_classes is not None:
        pending = pending[:args.max_new_classes]
    print(f"{len(names)} classes, {len(pending)} to generate -> {output_path}")

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(build_entry, args, names[i],
                            [(families[j], values[j]) for j in picks[i]]): names[i]
            for i in pending
        }
        with tqdm(total=len(futures)) as progress:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    data[name] = future.result()
                except Exception as exc:  # keep partial progress, resume later
                    failed.append(name)
                    data[name] = {"name_c": name, "_error": str(exc)}
                gen.write_output(output_path, data)
                progress.update(1)

    done = sum(complete_entry(data.get(name)) for name in names)
    print(f"Complete: {done}/{len(names)}. Saved: {output_path}")
    if failed:
        print("Failed (rerun the same command to resume):")
        for name in failed:
            print(f"- {name}")


if __name__ == "__main__":
    main()
