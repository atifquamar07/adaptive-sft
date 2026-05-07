import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import DatasetDict, load_dataset, load_from_disk

try:
    from .modeling import ensure_dirs, load_config, load_tokenizer
except ImportError:
    from modeling import ensure_dirs, load_config, load_tokenizer


def _messages_from_example(example: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        cleaned = []
        for msg in messages:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", ""))
            if role and content:
                cleaned.append({"role": role, "content": content})
        if cleaned:
            return cleaned
    prompt = str(example.get("prompt", ""))
    return [{"role": "user", "content": prompt}]


def _format_fallback(messages: Sequence[Dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    pos = 0
    role_prefix = {"user": "User: ", "assistant": "Assistant: ", "system": "System: "}
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        prefix = role_prefix.get(role, f"{role.title()}: ")
        parts.append(prefix)
        pos += len(prefix)
        start = pos
        parts.append(content)
        pos += len(content)
        end = pos
        if role == "assistant" and end > start:
            spans.append((start, end))
        parts.append("\n")
        pos += 1
    return "".join(parts), spans


def _format_with_chat_template(tokenizer, messages: Sequence[Dict[str, str]]) -> Optional[Tuple[str, List[Tuple[int, int]]]]:
    if not getattr(tokenizer, "chat_template", None):
        return None
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        spans: List[Tuple[int, int]] = []
        for idx, msg in enumerate(messages):
            if msg.get("role", "").lower() != "assistant":
                continue
            before = "" if idx == 0 else tokenizer.apply_chat_template(messages[:idx], tokenize=False, add_generation_prompt=False)
            upto = tokenizer.apply_chat_template(messages[: idx + 1], tokenize=False, add_generation_prompt=False)
            content = msg.get("content", "")
            start = upto.find(content, max(0, len(before) - 32))
            if start < 0:
                return None
            spans.append((start, start + len(content)))
        if not spans:
            return None
        return full_text, spans
    except Exception:
        return None


def format_conversation(tokenizer, example: Dict[str, Any]) -> Tuple[str, List[Tuple[int, int]], str]:
    messages = _messages_from_example(example)
    templated = _format_with_chat_template(tokenizer, messages)
    if templated is not None:
        return templated[0], templated[1], "chat_template"
    text, spans = _format_fallback(messages)
    return text, spans, "fallback"


def tokenize_with_assistant_mask(tokenizer, text: str, spans: Sequence[Tuple[int, int]], max_seq_len: int) -> Dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
    )
    labels = [-100] * len(encoded["input_ids"])
    for tok_idx, (start, end) in enumerate(encoded["offset_mapping"]):
        if end <= start:
            continue
        active = any(start < span_end and end > span_start for span_start, span_end in spans)
        if active:
            labels[tok_idx] = int(encoded["input_ids"][tok_idx])
    active_count = sum(1 for value in labels if value != -100)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "text": text,
        "assistant_spans_json": json.dumps([[int(a), int(b)] for a, b in spans]),
        "active_label_tokens": int(active_count),
        "ignored_tokens": int(len(labels) - active_count),
    }


def _assistant_spans(example: Dict[str, Any]) -> List[Tuple[int, int]]:
    try:
        return [(int(a), int(b)) for a, b in json.loads(example.get("assistant_spans_json", "[]"))]
    except Exception:
        return []


def _first_assistant_text(example: Dict[str, Any]) -> str:
    spans = _assistant_spans(example)
    text = str(example.get("text", ""))
    if not spans:
        return ""
    start, end = spans[0]
    return text[start:end]


def _replace_assistant_text(text: str, spans: Sequence[Tuple[int, int]], replacement: str) -> Tuple[str, List[Tuple[int, int]]]:
    parts: List[str] = []
    new_spans: List[Tuple[int, int]] = []
    pos = 0
    for start, end in spans:
        parts.append(text[pos:start])
        current = sum(len(part) for part in parts)
        parts.append(replacement)
        new_spans.append((current, current + len(replacement)))
        pos = end
    parts.append(text[pos:])
    return "".join(parts), new_spans


def _truncate_response(tokenizer, response: str, tokens: int = 20) -> str:
    ids = tokenizer(response, add_special_tokens=False)["input_ids"][:tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip() or response[:80]


def _add_noise_metadata(dataset, is_noise: bool = False, noise_type: str = ""):
    return dataset.map(
        lambda ex: {"is_synthetic_noise": bool(is_noise), "noise_type": str(noise_type)},
        desc="Adding noise metadata",
    )


def _apply_synthetic_noise(train_pool, tokenizer, cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    if not bool(data_cfg.get("enable_synthetic_noise", False)):
        return _add_noise_metadata(train_pool, False, "")
    fraction = float(data_cfg.get("noise_fraction", 0.0))
    if fraction <= 0.0:
        return _add_noise_metadata(train_pool, False, "")
    rng = random.Random(int(cfg["seed"]) + 909)
    count = max(1, int(round(len(train_pool) * min(max(fraction, 0.0), 1.0))))
    noisy_indices = set(rng.sample(range(len(train_pool)), min(count, len(train_pool))))
    noise_types = list(data_cfg.get("noise_types") or ["generic_response"])
    donor_responses = [_first_assistant_text(train_pool[i]) for i in range(len(train_pool))]
    donor_responses = [response for response in donor_responses if response]
    generic = "I am not sure. Please provide more information."
    verbose = (
        "I am not sure. Please provide more information. "
        "This answer is generic and may not address the request. "
    ) * 12

    def corrupt(example, idx):
        if idx not in noisy_indices:
            return {"is_synthetic_noise": False, "noise_type": ""}
        spans = _assistant_spans(example)
        original = _first_assistant_text(example)
        noise_type = noise_types[idx % len(noise_types)]
        if noise_type == "shuffled_response" and donor_responses:
            replacement = donor_responses[rng.randrange(len(donor_responses))]
        elif noise_type == "truncated_response":
            replacement = _truncate_response(tokenizer, original)
        elif noise_type == "verbose_distractor":
            replacement = verbose
        else:
            noise_type = "generic_response"
            replacement = generic
        text, new_spans = _replace_assistant_text(str(example["text"]), spans, replacement)
        item = tokenize_with_assistant_mask(tokenizer, text, new_spans, int(cfg["data"]["max_seq_len"]))
        item["format_source"] = example.get("format_source", "")
        item["is_synthetic_noise"] = True
        item["noise_type"] = noise_type
        return item

    return train_pool.map(corrupt, with_indices=True, desc="Applying synthetic train-pool noise")


def _load_raw_dataset(cfg: Dict[str, Any]):
    name = cfg["data"]["dataset_name"]
    split = cfg["data"].get("split", "train_sft")
    try:
        return load_dataset(name, split=split)
    except Exception:
        all_splits = load_dataset(name)
        if split in all_splits:
            return all_splits[split]
        if "train_sft" in all_splits:
            return all_splits["train_sft"]
        return all_splits["train"]


def prepare_data(cfg: Dict[str, Any], force: bool = False) -> DatasetDict:
    ensure_dirs(cfg)
    cache_dir = Path(cfg["paths"]["data_cache_dir"])
    if cache_dir.exists() and not force:
        return load_from_disk(str(cache_dir))

    tokenizer = load_tokenizer(cfg)
    raw = _load_raw_dataset(cfg)
    seed = int(cfg["seed"])
    total_needed = int(cfg["data"]["train_pool_size"]) + int(cfg["data"]["utility_val_size"]) + int(cfg["data"]["test_size"])
    margin = int(cfg["data"].get("preprocessing_margin", 0))
    select_count = min(len(raw), total_needed + margin)
    raw = raw.shuffle(seed=seed).select(range(select_count))

    def process(example):
        text, spans, source = format_conversation(tokenizer, example)
        item = tokenize_with_assistant_mask(tokenizer, text, spans, int(cfg["data"]["max_seq_len"]))
        item["format_source"] = source
        return item

    processed = raw.map(process, remove_columns=raw.column_names, desc="Tokenizing with assistant-only labels")
    processed = processed.filter(lambda ex: ex["active_label_tokens"] > 0, desc="Dropping examples without active assistant labels")
    if len(processed) < total_needed:
        raise RuntimeError(f"Only {len(processed)} usable examples after masking; need {total_needed}. Increase preprocessing_margin.")

    train_end = int(cfg["data"]["train_pool_size"])
    val_end = train_end + int(cfg["data"]["utility_val_size"])
    train_pool = processed.select(range(0, train_end))
    train_pool = _apply_synthetic_noise(train_pool, tokenizer, cfg)
    utility_val = _add_noise_metadata(processed.select(range(train_end, val_end)), False, "")
    test = _add_noise_metadata(processed.select(range(val_end, total_needed)), False, "")
    datasets = DatasetDict({"train_pool": train_pool, "utility_val": utility_val, "test": test})
    datasets.save_to_disk(str(cache_dir))
    return datasets


def load_processed_data(cfg: Dict[str, Any]) -> DatasetDict:
    cache_dir = Path(cfg["paths"]["data_cache_dir"])
    if not cache_dir.exists():
        return prepare_data(cfg, force=False)
    return load_from_disk(str(cache_dir))


def _active_runs(labels: Sequence[int]) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for idx, value in enumerate(labels):
        if value != -100 and start is None:
            start = idx
        elif value == -100 and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(labels)))
    return runs


def masking_sanity(cfg: Dict[str, Any], split: str = "train_pool", index: int = 0) -> str:
    datasets = load_processed_data(cfg)
    tokenizer = load_tokenizer(cfg)
    example = datasets[split][index]
    runs = _active_runs(example["labels"])
    lines = []
    lines.append("FULL TEXT")
    lines.append(example["text"])
    lines.append("")
    lines.append("ACTIVE ASSISTANT LABEL SPANS")
    for start, end in runs:
        decoded = tokenizer.decode(example["input_ids"][start:end], skip_special_tokens=False)
        lines.append(f"tokens[{start}:{end}] {decoded}")
    lines.append("")
    lines.append(f"active_label_tokens: {example['active_label_tokens']}")
    lines.append(f"ignored_tokens: {example['ignored_tokens']}")
    output = "\n".join(lines)
    out_path = Path(cfg["paths"]["output_dir"]) / "masking_sanity.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=["prepare", "sanity"], default="prepare")
    parser.add_argument("--split", default="train_pool")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    if args.mode == "prepare":
        datasets = prepare_data(cfg, force=args.force)
        print({split: len(ds) for split, ds in datasets.items()})
    else:
        masking_sanity(cfg, split=args.split, index=args.index)


if __name__ == "__main__":
    main()
