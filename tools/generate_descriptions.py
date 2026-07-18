#!/usr/bin/env python
"""Offline class text generation for semantic-granularity ZSL.

Run this once per dataset before training. The script writes raw text only:
name_c is copied from the class list, while attr_c, hier_c, and desc_c are
generated with three separate prompts. Encoding these text samples is a
training/semantic-encoding concern; no LLM call belongs in the training loop.

CLIP's text encoder accepts at most 77 tokens per text sample (including the
special tokens <|startoftext|> and <|endoftext|>). Each list item is encoded
independently, so the 77-token budget applies per item — not shared across the
whole list. Prompts ask the LLM for self-contained items of at most ~50 English
words (~75 content tokens at ~1.3 BPE tokens/word after reserving SOT/EOT).

Plan 3: write after-LLM output to a **separate** file
``data_semantics/<DATASET>/class_descriptions_<ds>_after_llm.json``.
Never overwrite legacy ``class_descriptions_<ds>.json`` unless
``--force-overwrite-legacy`` is set explicitly.
"""

import argparse
import contextlib
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PROMPTS = {
    "attr_c": (
        "List 10 distinct visual features useful for recognizing a {class_name} "
        "in a photo. Each item must be one self-contained phrase or short sentence "
        "that makes sense on its own (do not rely on other lines for context), "
        "at most about 50 English words. One item per line, no numbering, "
        "no preamble or extra explanation."
    ),
    "hier_c": (
        "List 10 specific subtypes, breeds, or closely related kinds of "
        "{class_name}. Each item must be one self-contained name or short "
        "description that stands alone (do not refer to other lines), at most "
        "about 50 English words. One item per line, no numbering, no preamble "
        "or extra explanation."
    ),
    "desc_c": (
        "List 10 backgrounds or settings where a {class_name} commonly appears "
        "in a photo. Each item must be one self-contained phrase or short "
        "sentence that makes sense on its own (do not rely on other lines for "
        "context), at most about 50 English words. One item per line, no "
        "numbering, no preamble or extra explanation."
    ),
}
PROMPT_CLASS_ALIASES = {
    "monkey": "a non-human primate animal from the simian group",
}

DEFAULT_API_BASE = "http://localhost:8082/v1/messages"
DEFAULT_API_KEY = "sk-jeFXnJyYFOjO64bsNQ5lvCyQ48rtDAYeKGFJ95UW4vFSU8Df"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


@contextlib.contextmanager
def request_deadline(seconds: float):
    """Enforce a total wall-clock deadline for one streaming request.

    urllib's timeout is a socket inactivity timeout; SSE servers can keep the
    socket alive with heartbeat lines forever. The alarm below caps the whole
    request, including parsing the response body.
    """
    seconds = float(seconds or 0)
    if seconds <= 0:
        yield
        return

    def raise_timeout(signum, frame):
        raise TimeoutError(f"SSE request exceeded {seconds:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            return yaml.safe_load(f) or {}
        return json.load(f) or {}


def default_after_llm_output(dataset: str, semantic_root: str = "./data_semantics") -> Path:
    ds = str(dataset).strip()
    return Path(semantic_root) / ds.upper() / f"class_descriptions_{ds.lower()}_after_llm.json"


def assert_safe_output_path(path: Path, force_overwrite_legacy: bool = False) -> None:
    """Refuse to write legacy class_descriptions_<ds>.json unless forced."""
    try:
        from tools.semantic_policy import is_legacy_class_descriptions_path
    except ModuleNotFoundError:
        from semantic_policy import is_legacy_class_descriptions_path

    if is_legacy_class_descriptions_path(path) and not force_overwrite_legacy:
        raise ValueError(
            f"Refusing to overwrite legacy semantics file: {path}\n"
            "Plan 3 writes after-LLM data to class_descriptions_<ds>_after_llm.json.\n"
            "Pass a different --output, or --force-overwrite-legacy only if intentional."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate raw name_c/attr_c/hier_c/desc_c text for one dataset."
    )
    parser.add_argument("--config", default=None,
                        help="Optional JSON/YAML config; CLI args override config values.")
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g. nus_wide or awa2.")
    parser.add_argument("--classes_file", "--class_file", default=None,
                        help="Text file containing class names, one class per line. "
                             "If omitted, names are read from data/<DATASET>/att_splits.mat.")
    parser.add_argument("--data_root", default=None,
                        help="XLSA17 metadata root used when --classes_file is omitted.")
    parser.add_argument(
        "--semantic_root",
        default=None,
        help="Root for default after-LLM output (default: ./data_semantics).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: "
             "data_semantics/<DATASET>/class_descriptions_<ds>_after_llm.json",
    )
    parser.add_argument(
        "--force-overwrite-legacy",
        action="store_true",
        help="Allow writing to legacy class_descriptions_<ds>.json (discouraged).",
    )
    parser.add_argument(
        "--on-overflow",
        default=None,
        choices=["skip_item", "reject_class"],
        help="Policy when an item exceeds CLIP 77 tokens (default: skip_item).",
    )
    parser.add_argument(
        "--max-clip-tokens",
        type=int,
        default=None,
        help="CLIP token budget per item (default: 77).",
    )
    parser.add_argument(
        "--skip-token-validate",
        action="store_true",
        help="Skip CLIP token validation after parse (not recommended).",
    )
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--backoff", type=float, default=None,
                        help="Initial retry delay in seconds; doubled after each failure.")
    parser.add_argument("--request_delay", type=float, default=None,
                        help="Optional delay between successful field calls.")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of classes generated concurrently.")
    parser.add_argument("--max_new_classes", type=int, default=None,
                        help="Generate at most N incomplete classes, for smoke/resume runs.")
    parser.add_argument("--test_api", action="store_true",
                        help="Call the sample greeting prompt and exit.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, default in (
        ("dataset", None),
        ("classes_file", None),
        ("data_root", "./data"),
        ("semantic_root", "./data_semantics"),
        ("output", None),
        ("on_overflow", "skip_item"),
        ("max_clip_tokens", 77),
        ("api_base", DEFAULT_API_BASE),
        ("api_key", DEFAULT_API_KEY),
        ("model", DEFAULT_MODEL),
        ("max_tokens", DEFAULT_MAX_TOKENS),
        ("retries", 3),
        ("backoff", 2.0),
        ("request_delay", 0.0),
        ("timeout", 120.0),
        ("workers", 1),
    ):
        value = getattr(args, key, None)
        if value is None:
            setattr(args, key, cfg.get(key, default))

    if not args.test_api:
        if not args.dataset:
            parser.error("--dataset is required unless --test_api is used.")
        if args.workers < 1:
            parser.error("--workers must be at least 1.")
        if args.max_new_classes is not None and args.max_new_classes < 1:
            parser.error("--max_new_classes must be at least 1.")
        if not args.output:
            args.output = str(default_after_llm_output(args.dataset, args.semantic_root))
    return args


def clean_class_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    if "." in line and line.split(".", 1)[0].isdigit():
        line = line.split(".", 1)[1].strip()
    else:
        parts = line.split()
        if len(parts) > 1 and parts[0].isdigit():
            line = " ".join(parts[1:])

    return line.replace("+", " ").replace("_", " ").strip()


def load_classes(path: str) -> list[str]:
    classes = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            class_name = clean_class_line(line)
            if class_name:
                classes.append(class_name)

    if not classes:
        raise ValueError(f"No class names found in {path}.")
    return classes


def load_xlsa_classes(data_root: str, dataset: str) -> list[str]:
    import scipy.io as sio

    path = Path(data_root) / str(dataset).upper() / "att_splits.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing XLSA17 split metadata: {path}")
    data = sio.loadmat(path)
    if "allclasses_names" not in data:
        raise KeyError(f"{path} does not contain allclasses_names.")

    classes = []
    for value in data["allclasses_names"].squeeze():
        while hasattr(value, "size") and getattr(value, "size", 0) == 1:
            value = value.item()
        class_name = clean_class_line(str(value))
        if class_name:
            classes.append(class_name)
    if not classes:
        raise ValueError(f"No class names found in {path}.")
    return classes


def load_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Output file must be a JSON object keyed by class name: {path}")
    return obj


def write_output(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.\)])\s+")


def parse_items(text: str) -> list[str]:
    """Split LLM response into clean self-contained list items."""
    if not text or not str(text).strip():
        return []
    items: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Drop common chatty preambles / follow-ups.
        lower = line.lower()
        if lower.startswith("if you want") or lower.startswith("sure —") or lower.startswith("sure -"):
            continue
        if lower.startswith("here") and ("list" in lower or "background" in lower):
            continue
        line = _BULLET_RE.sub("", line).strip()
        line = line.strip("`\"'")
        if not line:
            continue
        items.append(line)
    # Fallback: whole block as one item if no newlines produced anything.
    if not items:
        compact = " ".join(str(text).split())
        if compact:
            items = [compact]
    return items


def _field_nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    return value is not None and bool(str(value).strip())


def complete_entry(entry: Any) -> bool:
    """True when name_c and three fields have usable content (list or non-empty str)."""
    if not isinstance(entry, dict):
        return False
    if not isinstance(entry.get("name_c"), str) or not entry["name_c"].strip():
        return False
    for key in ("attr_c", "hier_c", "desc_c"):
        if not _field_nonempty(entry.get(key)):
            return False
    return True


def validate_and_filter_items(
    items: list[str],
    *,
    max_clip_tokens: int = 77,
    on_overflow: str = "skip_item",
    skip_token_validate: bool = False,
) -> list[str]:
    if skip_token_validate:
        return [item for item in items if item.strip()]
    try:
        from tools.clip_token_budget import filter_items_by_budget
    except ModuleNotFoundError:
        from clip_token_budget import filter_items_by_budget

    return filter_items_by_budget(
        items,
        max_tokens=max_clip_tokens,
        on_overflow=on_overflow,
    )


def request_body(model: str, prompt: str, max_tokens: int) -> bytes:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    return json.dumps(body).encode("utf-8")


def extract_delta(obj: Any) -> str:
    pieces: list[str] = []

    if isinstance(obj, dict):
        delta = obj.get("delta")
        if isinstance(delta, dict):
            for key in ("text", "content"):
                value = delta.get(key)
                if isinstance(value, str):
                    pieces.append(value)
                elif isinstance(value, list):
                    pieces.extend(extract_delta(item) for item in value)
        elif isinstance(delta, str):
            pieces.append(delta)

        if isinstance(obj.get("text"), str):
            pieces.append(obj["text"])

        content = obj.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            pieces.extend(extract_delta(item) for item in content)

        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                pieces.append(extract_delta(choice))

        message = obj.get("message")
        if isinstance(message, dict):
            pieces.append(extract_delta(message))

    return "".join(piece for piece in pieces if piece)


def is_stop_event(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    event_type = obj.get("type")
    if event_type in {"message_stop", "done", "response.completed", "response.done"}:
        return True

    if obj.get("done") is True:
        return True

    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                return True

    return False


def parse_sse_lines(lines) -> str:
    chunks: list[str] = []
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue

        payload = line[len("data:"):].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break

        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict) and obj.get("type") == "error":
            raise ValueError(f"SSE API error: {obj.get('error')}")

        chunks.append(extract_delta(obj))
        if is_stop_event(obj):
            break

    text = "".join(chunks).strip()
    if not text:
        raise ValueError("SSE stream finished without text delta.")
    return text


def call_streaming_api(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> str:
    req = urllib.request.Request(
        api_base,
        data=request_body(model, prompt, max_tokens),
        headers={
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request_deadline(timeout):
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return parse_sse_lines(response)


def call_with_retry(args, prompt: str) -> str:
    delay = args.backoff
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            text = call_streaming_api(
                args.api_base,
                args.api_key,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout,
            )
            if text.strip():
                return text.strip()
            raise ValueError("Empty response text.")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = exc
            if attempt < args.retries:
                print(
                    f"[retry] attempt {attempt}/{args.retries} failed: {exc}; "
                    f"sleep {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                delay *= 2.0

    raise RuntimeError(f"LLM SSE request failed after {args.retries} attempts: {last_error}")


def generate_for_class(args, class_name: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"name_c": class_name}
    prompt_class_name = PROMPT_CLASS_ALIASES.get(class_name.lower(), class_name)
    on_overflow = getattr(args, "on_overflow", "skip_item") or "skip_item"
    max_clip_tokens = int(getattr(args, "max_clip_tokens", 77) or 77)
    skip_token_validate = bool(getattr(args, "skip_token_validate", False))

    for field, template in PROMPTS.items():
        prompt = template.format(class_name=prompt_class_name)
        print(f"[generate] {class_name}::{field}", file=sys.stderr, flush=True)
        raw = call_with_retry(args, prompt)
        items = parse_items(raw)
        items = validate_and_filter_items(
            items,
            max_clip_tokens=max_clip_tokens,
            on_overflow=on_overflow,
            skip_token_validate=skip_token_validate,
        )
        if not items:
            raise ValueError(
                f"{class_name}::{field}: no items remaining after parse/token filter."
            )
        entry[field] = items
        if args.request_delay > 0:
            time.sleep(args.request_delay)
    return entry


def generate_descriptions(args, classes: list[str], output_path: Path) -> list[str]:
    assert_safe_output_path(
        output_path,
        force_overwrite_legacy=bool(getattr(args, "force_overwrite_legacy", False)),
    )
    data = load_output(output_path)
    all_pending = [name for name in classes if not complete_entry(data.get(name))]
    pending = all_pending
    max_new_classes = getattr(args, "max_new_classes", None)
    if max_new_classes is not None:
        pending = pending[:max_new_classes]
    failed: list[str] = []

    progress_total = len(classes) if max_new_classes is None else len(pending)
    progress_initial = len(classes) - len(all_pending) if max_new_classes is None else 0
    with tqdm(total=progress_total, initial=progress_initial) as progress:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending_iter = iter(pending)
            futures = {}
            for class_name in pending_iter:
                futures[executor.submit(generate_for_class, args, class_name)] = class_name
                if len(futures) == args.workers:
                    break

            while futures:
                future = next(as_completed(futures))
                class_name = futures.pop(future)
                existing = data.get(class_name)
                try:
                    data[class_name] = future.result()
                except Exception as exc:
                    failed.append(class_name)
                    partial = existing if isinstance(existing, dict) else {"name_c": class_name}
                    partial["name_c"] = class_name
                    partial["_error"] = str(exc)
                    data[class_name] = partial
                write_output(output_path, data)
                progress.update(1)
                try:
                    next_class = next(pending_iter)
                except StopIteration:
                    continue
                futures[executor.submit(generate_for_class, args, next_class)] = next_class

    ordered = {name: data[name] for name in classes if name in data}
    write_output(output_path, ordered)
    return failed


def run_test_api(args) -> None:
    text = call_with_retry(args, "Greet me in one sentence.")
    print(text)


def main():
    args = parse_args()

    if args.test_api:
        run_test_api(args)
        return

    classes = (
        load_classes(args.classes_file)
        if args.classes_file
        else load_xlsa_classes(args.data_root, args.dataset)
    )
    output_path = Path(
        args.output
        or default_after_llm_output(args.dataset, getattr(args, "semantic_root", "./data_semantics"))
    )
    assert_safe_output_path(output_path, bool(getattr(args, "force_overwrite_legacy", False)))
    failed = generate_descriptions(args, classes, output_path)
    final_data = load_output(output_path)
    completed = sum(
        complete_entry(final_data.get(class_name))
        for class_name in classes
    )

    if failed:
        print("Failed classes:")
        for class_name in failed:
            print(f"- {class_name}")
    elif completed == len(classes):
        print("All classes completed.")
    else:
        print(f"Batch completed. Semantic progress: {completed}/{len(classes)} classes.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
