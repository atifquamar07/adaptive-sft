import contextlib
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_output_paths(cfg: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    out = Path(output_dir or cfg["paths"]["output_dir"])
    cfg["paths"]["output_dir"] = str(out)
    cfg["paths"]["data_cache_dir"] = str(out / "data" / "ultrachat_sft_masked")
    cfg["paths"]["warmup_dir"] = str(out / "checkpoints" / "warmup")
    cfg["paths"]["utility_labels"] = str(out / "utility_labels.pt")
    cfg["paths"]["utility_labels_jsonl"] = str(out / "utility_labels.jsonl")
    cfg["paths"]["evaluator"] = str(out / "evaluator.pt")
    cfg["paths"]["evaluator_metrics"] = str(out / "evaluator_metrics.json")
    cfg["paths"]["evaluator_dev_predictions"] = str(out / "evaluator_dev_predictions.csv")
    cfg["paths"]["curves_dir"] = str(out / "curves")
    cfg["paths"]["logs_dir"] = str(out / "logs")
    cfg["paths"]["plots_dir"] = str(out / "plots")


def load_config(path: str = "configs/default.yaml", smoke: bool = False) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_override = os.environ.get("ADAPTIVE_SFT_SEED")
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    if env_flag("ADAPTIVE_SFT_NOISE", default=False):
        cfg.setdefault("data", {})["enable_synthetic_noise"] = True
        if float(cfg["data"].get("noise_fraction", 0.0)) <= 0.0:
            cfg["data"]["noise_fraction"] = 0.25
    noise_fraction_override = os.environ.get("ADAPTIVE_SFT_NOISE_FRACTION")
    if noise_fraction_override is not None:
        cfg.setdefault("data", {})["noise_fraction"] = float(noise_fraction_override)
    if smoke:
        smoke_cfg = cfg.get("smoke", {})
        deep_update(cfg, smoke_cfg.get("overrides", {}))
        _resolve_output_paths(cfg, smoke_cfg.get("output_dir", "outputs/smoke"))
    else:
        _resolve_output_paths(cfg, cfg["paths"]["output_dir"])
    output_override = os.environ.get("ADAPTIVE_SFT_OUTPUT_DIR")
    if output_override:
        _resolve_output_paths(cfg, output_override)
    hf_home = Path(cfg["paths"]["output_dir"]) / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ["output_dir", "curves_dir", "logs_dir", "plots_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["warmup_dir"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["data_cache_dir"]).parent.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def maybe_data_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if (
        device.type == "cuda"
        and env_flag("ADAPTIVE_SFT_DATA_PARALLEL", default=False)
        and torch.cuda.device_count() > 1
    ):
        print(f"using DataParallel across {torch.cuda.device_count()} visible CUDA devices")
        return torch.nn.DataParallel(model)
    return model


def preferred_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=preferred_dtype(device))


def load_tokenizer(cfg: Dict[str, Any]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _detect_lora_targets(model: torch.nn.Module, preferred: Sequence[str]) -> List[str]:
    linear_suffixes = set()
    attention_suffixes = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            linear_suffixes.add(suffix)
            if "attn" in name.lower() or "attention" in name.lower():
                attention_suffixes.add(suffix)
    targets = [name for name in preferred if name in linear_suffixes]
    if targets:
        return targets
    if attention_suffixes:
        return sorted(attention_suffixes)
    return sorted(linear_suffixes)


def load_lora_model(
    cfg: Dict[str, Any],
    adapter_path: Optional[str] = None,
    train: bool = True,
):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    device = get_device()
    dtype = preferred_dtype(device)
    model_kwargs = {
        "trust_remote_code": cfg["model"].get("trust_remote_code", True),
    }
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = dtype
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], **model_kwargs)
    base.config.use_cache = False

    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=train)
    else:
        targets = _detect_lora_targets(base, cfg["lora"].get("target_modules", []))
        if not targets:
            raise RuntimeError("Could not detect any Linear modules for LoRA targets.")
        lora_cfg = LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["alpha"]),
            lora_dropout=float(cfg["lora"]["dropout"]),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)

    model.to(device)
    model = maybe_data_parallel(model, device)
    model.train(train)
    tokenizer = load_tokenizer(cfg)
    return model, tokenizer, device


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    path: str,
    cfg: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(out)
    tokenizer.save_pretrained(out)
    write_json(out / "run_metadata.json", {"config": cfg, "metadata": metadata or {}})


def collate_sft(examples: Sequence[Dict[str, Any]], tokenizer, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(ex["input_ids"]) for ex in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for row, ex in enumerate(examples):
        n = len(ex["input_ids"])
        input_ids[row, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[row, :n] = torch.tensor(ex["attention_mask"], dtype=torch.long)
        labels[row, :n] = torch.tensor(ex["labels"], dtype=torch.long)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if "is_synthetic_noise" in examples[0]:
        batch["is_synthetic_noise"] = torch.tensor(
            [float(bool(ex.get("is_synthetic_noise", False))) for ex in examples],
            dtype=torch.float32,
        )
    return move_batch_to_device(batch, device) if device is not None else batch


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_from_indices(dataset, indices: Sequence[int], tokenizer, device: torch.device) -> Dict[str, torch.Tensor]:
    return collate_sft([dataset[int(i)] for i in indices], tokenizer, device)


def sample_indices(rng: random.Random, dataset_size: int, batch_size: int) -> List[int]:
    return [rng.randrange(dataset_size) for _ in range(batch_size)]


def sample_batch(dataset, tokenizer, device: torch.device, rng: random.Random, batch_size: int) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    indices = sample_indices(rng, len(dataset), batch_size)
    return batch_from_indices(dataset, indices, tokenizer, device), indices


def static_candidate_batch_count(section: Dict[str, Any], continuation_steps: int) -> Tuple[int, int, bool]:
    configured = int(section["candidate_batches"])
    selected_fraction = float(section["selected_fraction"])
    if not 0.0 < selected_fraction <= 1.0:
        raise ValueError("selected_fraction must be in (0, 1].")
    ensure_steps = bool(section.get("ensure_selected_steps", True))
    needed = math.ceil(int(continuation_steps) / selected_fraction) if ensure_steps else configured
    effective = max(configured, needed)
    return configured, effective, effective > configured


def static_candidate_indices(
    rng: random.Random,
    dataset_size: int,
    batch_size: int,
    candidate_batches: int,
) -> List[List[int]]:
    if dataset_size <= 0:
        raise ValueError("Cannot sample candidates from an empty dataset.")
    order: List[int] = []
    cursor = 0
    batches = []
    for _ in range(candidate_batches):
        indices = []
        while len(indices) < batch_size:
            if cursor >= len(order):
                order = list(range(dataset_size))
                rng.shuffle(order)
                cursor = 0
            take = min(batch_size - len(indices), len(order) - cursor)
            indices.extend(order[cursor : cursor + take])
            cursor += take
        batches.append(indices)
    return batches


def static_training_schedule(
    selected: Sequence[Dict[str, Any]],
    steps: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not selected:
        raise ValueError("Cannot build a static training schedule with no selected batches.")
    selected = list(selected)
    schedule: List[Dict[str, Any]] = []
    while len(schedule) < steps:
        epoch = list(selected)
        rng.shuffle(epoch)
        schedule.extend(epoch)
    return schedule[:steps]


def selected_pool_stats(selected: Sequence[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    indices = [int(idx) for row in selected for idx in row.get("batch_indices", [])]
    selected_batches = len(selected)
    return {
        "selected_batches": selected_batches,
        "selected_examples": len(indices),
        "selected_unique_examples": len(set(indices)),
        "steps_per_selected_batch": float(steps) / max(selected_batches, 1),
        "selected_pool_covers_steps": selected_batches >= int(steps),
    }


def sft_loss_and_stats(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    return_logits: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=None,
        use_cache=False,
    )
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs[0]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = batch["labels"][:, 1:].contiguous()
    active = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~active, 0)
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    token_losses = token_losses * active.float()
    active_per_example = active.sum(dim=1)
    denom = active.sum().clamp_min(1)
    loss = token_losses.sum() / denom
    per_example_loss = token_losses.sum(dim=1) / active_per_example.clamp_min(1)
    stats = {
        "loss": float(loss.detach().cpu()),
        "token_loss_sum": float(token_losses.sum().detach().cpu()),
        "active_tokens": int(active.sum().detach().cpu()),
        "per_example_loss": per_example_loss.detach().cpu(),
        "active_per_example": active_per_example.detach().cpu(),
    }
    return loss, stats, logits if return_logits else None


def make_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(
        params,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )


def make_scaler(device: torch.device):
    enabled = device.type == "cuda" and preferred_dtype(device) == torch.float16
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_on_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device):
        loss, stats, _ = sft_loss_and_stats(model, batch, return_logits=False)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(cfg["training"].get("max_grad_norm", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(cfg["training"].get("max_grad_norm", 1.0)),
        )
        optimizer.step()
    return stats


@torch.no_grad()
def evaluate_dataset_loss(
    model: torch.nn.Module,
    dataset,
    tokenizer,
    cfg: Dict[str, Any],
    split_name: str,
    max_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    batch_size = int(batch_size or cfg["evaluation"].get("batch_size", cfg["training"]["micro_batch_size"]))
    if max_batches is None:
        max_batches = int(cfg["training"].get("eval_batches", 0))
    if max_batches == 0:
        max_batches = math.ceil(len(dataset) / batch_size)
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for start in range(0, len(dataset), batch_size):
        if batches >= max_batches:
            break
        examples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = collate_sft(examples, tokenizer, device)
        with autocast_context(device):
            _, stats, _ = sft_loss_and_stats(model, batch, return_logits=False)
        total_loss += stats["token_loss_sum"]
        total_tokens += stats["active_tokens"]
        batches += 1
    if was_training:
        model.train()
    loss = total_loss / max(total_tokens, 1)
    return {
        "split": split_name,
        "loss": loss,
        "perplexity": float(math.exp(min(loss, 20.0))),
        "active_tokens": int(total_tokens),
        "batches": int(batches),
    }


def maybe_eval_curve(
    model: torch.nn.Module,
    val_dataset,
    tokenizer,
    cfg: Dict[str, Any],
    method: str,
    step: int,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    eval_every = int(cfg["training"].get("eval_every", 0))
    if not force and (eval_every <= 0 or step % eval_every != 0):
        return None
    metrics = evaluate_dataset_loss(
        model,
        val_dataset,
        tokenizer,
        cfg,
        split_name="utility_val",
        max_batches=int(cfg["training"].get("eval_batches", 0)),
        batch_size=int(cfg["evaluation"].get("batch_size", cfg["training"]["micro_batch_size"])),
    )
    record = {"method": method, "step": int(step), "wall_time": time.time(), **metrics}
    curve_path = Path(cfg["paths"]["curves_dir"]) / f"{method}.jsonl"
    if force:
        existing = read_jsonl(curve_path)
        if existing and int(existing[-1].get("step", -1)) == int(step):
            return existing[-1]
    append_jsonl(curve_path, record)
    return record


def write_json(path: os.PathLike, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(path: os.PathLike, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def reset_curve(cfg: Dict[str, Any], method: str) -> None:
    path = Path(cfg["paths"]["curves_dir"]) / f"{method}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def read_jsonl(path: os.PathLike) -> List[Dict[str, Any]]:
    records = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_csv(path: os.PathLike, rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_overhead(method: str) -> Dict[str, Any]:
    return {
        "method": method,
        "optimizer_steps": 0,
        "candidate_batches_scored": 0,
        "gradient_calls": 0,
        "wall_clock_seconds": 0.0,
    }


def save_overhead(cfg: Dict[str, Any], method: str, overhead: Dict[str, Any]) -> None:
    path = Path(cfg["paths"]["logs_dir"]) / f"{method}_overhead.json"
    write_json(path, overhead)


def method_checkpoint_dir(cfg: Dict[str, Any], method: str) -> str:
    return str(Path(cfg["paths"]["output_dir"]) / "checkpoints" / method)
