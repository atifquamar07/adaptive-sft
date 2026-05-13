import argparse
import random
import time
from pathlib import Path

import torch

try:
    from .data import load_processed_data
    from .grad_utils import compute_lora_gradient, cosine_utility
    from .modeling import (
        append_jsonl,
        batch_from_indices,
        ensure_dirs,
        init_overhead,
        load_config,
        load_lora_model,
        make_optimizer,
        make_scaler,
        maybe_eval_curve,
        method_checkpoint_dir,
        reset_curve,
        sample_indices,
        save_checkpoint,
        save_overhead,
        set_seed,
        train_on_batch,
    )
except ImportError:
    from data import load_processed_data
    from grad_utils import compute_lora_gradient, cosine_utility
    from modeling import (
        append_jsonl,
        batch_from_indices,
        ensure_dirs,
        init_overhead,
        load_config,
        load_lora_model,
        make_optimizer,
        make_scaler,
        maybe_eval_curve,
        method_checkpoint_dir,
        reset_curve,
        sample_indices,
        save_checkpoint,
        save_overhead,
        set_seed,
        train_on_batch,
    )


def _select_index(scores: torch.Tensor, rng: random.Random, selection: str) -> int:
    if selection == "top1":
        return int(torch.argmax(scores).item())
    if selection != "softmax":
        raise ValueError("oracle.selection must be 'top1' or 'softmax'.")
    probs = torch.softmax(scores, dim=0).cpu().numpy()
    return int(rng.choices(range(len(probs)), weights=probs, k=1)[0])


def _rank_of(scores: torch.Tensor, idx: int) -> int:
    value = float(scores[idx].item())
    return 1 + int((scores > value).sum().item())


def run(cfg):
    method = "oracle_gradient_sft"
    if not bool(cfg.get("oracle", {}).get("enabled", True)):
        print("oracle_gradient_sft disabled by config.")
        return
    ensure_dirs(cfg)
    reset_curve(cfg, method)
    set_seed(int(cfg["seed"]))
    datasets = load_processed_data(cfg)
    model, tokenizer, device = load_lora_model(cfg, adapter_path=cfg["paths"]["warmup_dir"], train=True)
    optimizer = make_optimizer(cfg, model)
    scaler = make_scaler(device)
    rng = random.Random(int(cfg["seed"]) + 701)
    overhead = init_overhead(method)
    start_time = time.time()
    log_path = Path(cfg["paths"]["logs_dir"]) / f"{method}_selected_batches.jsonl"
    log_path.write_text("", encoding="utf-8")

    steps = int(cfg["training"]["continuation_steps"])
    batch_size = int(cfg["training"]["micro_batch_size"])
    val_batch_size = int(cfg["utility"].get("val_batch_size", batch_size))
    k = int(cfg.get("oracle", {}).get("candidate_count", cfg["adaptive"].get("candidate_count", 8)))
    refresh = max(1, int(cfg.get("oracle", {}).get("val_gradient_refresh_steps", 1)))
    selection = cfg.get("oracle", {}).get("selection", "top1")
    val_grad = None
    val_indices = []

    for step in range(1, steps + 1):
        gradient_calls_this_step = 0
        if val_grad is None or (step - 1) % refresh == 0:
            val_indices = sample_indices(rng, len(datasets["utility_val"]), val_batch_size)
            val_batch = batch_from_indices(datasets["utility_val"], val_indices, tokenizer, device)
            val_grad, _ = compute_lora_gradient(model, val_batch, device, cfg)
            overhead["gradient_calls"] += 1
            gradient_calls_this_step += 1

        candidates = []
        utilities = []
        for _ in range(k):
            indices = sample_indices(rng, len(datasets["train_pool"]), batch_size)
            batch = batch_from_indices(datasets["train_pool"], indices, tokenizer, device)
            train_grad, grad_stats = compute_lora_gradient(model, batch, device, cfg)
            overhead["gradient_calls"] += 1
            gradient_calls_this_step += 1
            overhead["candidate_batches_scored"] += 1
            utility = cosine_utility(train_grad, val_grad)
            utilities.append(float(utility))
            candidates.append({"indices": indices, "batch": batch, "gradient_loss": float(grad_stats["loss"])})

        utility_scores = torch.tensor(utilities, dtype=torch.float32)
        selected_idx = _select_index(utility_scores, rng, selection)
        selected = candidates[selected_idx]
        stats = train_on_batch(model, optimizer, scaler, selected["batch"], cfg, device)
        overhead["optimizer_steps"] += 1
        utility_std = float(utility_scores.std(unbiased=False).item()) if utility_scores.numel() > 1 else 0.0
        record = {
            "method": method,
            "step": int(step),
            "selection": selection,
            "candidate_count": int(k),
            "val_indices": [int(i) for i in val_indices],
            "selected_candidate": int(selected_idx),
            "selected_true_utility": float(utility_scores[selected_idx].item()),
            "candidate_utilities": [float(x) for x in utilities],
            "selected_rank": int(_rank_of(utility_scores, selected_idx)),
            "candidate_utility_std": utility_std,
            "candidate_utility_range": float((utility_scores.max() - utility_scores.min()).item()),
            "gradient_calls_this_step": int(gradient_calls_this_step),
            "gradient_calls_cumulative": int(overhead["gradient_calls"]),
            "train_loss": float(stats["loss"]),
            "selected_gradient_loss": float(selected["gradient_loss"]),
        }
        append_jsonl(log_path, record)
        if step % int(cfg["training"].get("log_every", 20)) == 0 or step == 1:
            print(
                f"{method} step={step} train_loss={stats['loss']:.4f} "
                f"utility={record['selected_true_utility']:.4f}"
            )
        maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, step)

    maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, steps, force=True)
    overhead["wall_clock_seconds"] = time.time() - start_time
    save_checkpoint(model, tokenizer, method_checkpoint_dir(cfg, method), cfg, {"method": method})
    save_overhead(cfg, method, overhead)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
