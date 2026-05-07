import argparse
import random
import time
from pathlib import Path

import torch

try:
    from .data import load_processed_data
    from .features import compute_batch_features, feature_names
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
        sample_batch,
        sample_indices,
        save_overhead,
        set_seed,
        train_on_batch,
    )
except ImportError:
    from data import load_processed_data
    from features import compute_batch_features, feature_names
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
        sample_batch,
        sample_indices,
        save_overhead,
        set_seed,
        train_on_batch,
    )


def _collect_at_step(cfg, model, tokenizer, device, datasets, rng, current_step, total_steps, overhead):
    use_grad_features = bool(cfg["utility"].get("use_gradient_features", False))
    names = feature_names(use_grad_features, bool(cfg.get("data", {}).get("feed_noise_feature_to_evaluator", False)))
    val_batches_per_state = max(1, int(cfg["utility"].get("val_batches_per_state", 1)))
    val_indices_by_batch = []
    val_grad = None
    val_active_tokens = 0
    val_loss_sum = 0.0
    for _ in range(val_batches_per_state):
        val_indices = sample_indices(rng, len(datasets["utility_val"]), int(cfg["utility"]["val_batch_size"]))
        val_batch = batch_from_indices(datasets["utility_val"], val_indices, tokenizer, device)
        grad, stats = compute_lora_gradient(model, val_batch, device, cfg)
        val_grad = grad if val_grad is None else val_grad + grad
        val_indices_by_batch.append([int(i) for i in val_indices])
        val_loss_sum += float(stats["token_loss_sum"])
        val_active_tokens += int(stats["active_tokens"])
        overhead["gradient_calls"] += 1
    val_grad = val_grad / float(val_batches_per_state)
    val_loss = val_loss_sum / max(val_active_tokens, 1)
    records = []
    for cand_idx in range(int(cfg["utility"]["candidates_per_state"])):
        indices = sample_indices(rng, len(datasets["train_pool"]), int(cfg["utility"]["candidate_batch_size"]))
        batch = batch_from_indices(datasets["train_pool"], indices, tokenizer, device)
        feats = compute_batch_features(
            model,
            batch,
            device,
            cfg,
            current_step=current_step,
            total_steps=total_steps,
            use_gradient_features=use_grad_features,
        )
        if use_grad_features:
            overhead["gradient_calls"] += 1
        train_grad, grad_stats = compute_lora_gradient(model, batch, device, cfg)
        overhead["gradient_calls"] += 1
        utility = cosine_utility(train_grad, val_grad)
        overhead["candidate_batches_scored"] += 1
        record = {
            "step": int(current_step),
            "candidate_index": int(cand_idx),
            "batch_indices": [int(i) for i in indices],
            "val_indices": val_indices_by_batch,
            "val_batches_per_state": int(val_batches_per_state),
            "val_loss": float(val_loss),
            "features": feats,
            "feature_names": names,
            "utility": float(utility),
            "gradient_loss": float(grad_stats["loss"]),
        }
        records.append(record)
        append_jsonl(cfg["paths"]["utility_labels_jsonl"], record)
    print(f"collected {len(records)} utility labels at step {current_step}")
    return records


def run(cfg):
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))
    datasets = load_processed_data(cfg)
    model, tokenizer, device = load_lora_model(cfg, adapter_path=cfg["paths"]["warmup_dir"], train=True)
    optimizer = make_optimizer(cfg, model)
    scaler = make_scaler(device)
    rng = random.Random(int(cfg["seed"]) + 101)
    overhead = init_overhead("utility_label_collection")
    start_time = time.time()
    records = []
    label_steps = sorted(int(s) for s in cfg["utility"]["label_steps"])
    current_step = int(cfg["training"]["warmup_steps"])
    total_steps = current_step + int(cfg["training"]["continuation_steps"])
    if label_steps[0] < current_step or label_steps[-1] > total_steps:
        raise ValueError(
            "utility.label_steps must be absolute optimizer steps between "
            f"warmup_steps={current_step} and total_steps={total_steps}."
        )

    Path(cfg["paths"]["utility_labels_jsonl"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["utility_labels_jsonl"]).write_text("", encoding="utf-8")

    for target_step in label_steps:
        while current_step < target_step:
            batch, _ = sample_batch(
                datasets["train_pool"],
                tokenizer,
                device,
                rng,
                int(cfg["training"]["micro_batch_size"]),
            )
            train_on_batch(model, optimizer, scaler, batch, cfg, device)
            current_step += 1
            overhead["optimizer_steps"] += 1
        records.extend(_collect_at_step(cfg, model, tokenizer, device, datasets, rng, current_step, total_steps, overhead))

    overhead["wall_clock_seconds"] = time.time() - start_time
    torch.save(records, cfg["paths"]["utility_labels"])
    save_overhead(cfg, "utility_label_collection", overhead)
    print(f"saved {len(records)} utility-label records to {cfg['paths']['utility_labels']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    run(cfg)


if __name__ == "__main__":
    main()
