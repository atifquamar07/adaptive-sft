import argparse
import random
from pathlib import Path

import numpy as np
import torch

try:
    from .data import load_processed_data
    from .grad_utils import compute_lora_gradient, cosine_utility
    from .modeling import (
        autocast_context,
        batch_from_indices,
        ensure_dirs,
        load_config,
        load_lora_model,
        make_optimizer,
        make_scaler,
        sample_indices,
        sft_loss_and_stats,
        train_on_batch,
        write_csv,
        write_json,
    )
except ImportError:
    from data import load_processed_data
    from grad_utils import compute_lora_gradient, cosine_utility
    from modeling import (
        autocast_context,
        batch_from_indices,
        ensure_dirs,
        load_config,
        load_lora_model,
        make_optimizer,
        make_scaler,
        sample_indices,
        sft_loss_and_stats,
        train_on_batch,
        write_csv,
        write_json,
    )


def _trainable_state(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def _restore_trainable_state(model, state):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in state:
                p.copy_(state[name])


def _batch_loss(model, batch, device):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        with autocast_context(device):
            _, stats, _ = sft_loss_and_stats(model, batch, return_logits=False)
    if was_training:
        model.train()
    return float(stats["loss"])


def _pearson(x, y):
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x, y):
    return _pearson(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def run(cfg):
    ensure_dirs(cfg)
    datasets = load_processed_data(cfg)
    model, tokenizer, device = load_lora_model(cfg, adapter_path=cfg["paths"]["warmup_dir"], train=True)
    rng = random.Random(int(cfg["seed"]) + 901)
    n = int(cfg.get("diagnostics", {}).get("gradient_signal_candidates", 16))
    batch_size = int(cfg["training"]["micro_batch_size"])
    val_batch_size = int(cfg.get("diagnostics", {}).get("val_batch_size", cfg["utility"].get("val_batch_size", batch_size)))
    val_indices = sample_indices(rng, len(datasets["utility_val"]), val_batch_size)
    val_batch = batch_from_indices(datasets["utility_val"], val_indices, tokenizer, device)
    val_grad, _ = compute_lora_gradient(model, val_batch, device, cfg)
    base_state = _trainable_state(model)
    rows = []

    for idx in range(n):
        _restore_trainable_state(model, base_state)
        train_indices = sample_indices(rng, len(datasets["train_pool"]), batch_size)
        train_batch = batch_from_indices(datasets["train_pool"], train_indices, tokenizer, device)
        train_grad, grad_stats = compute_lora_gradient(model, train_batch, device, cfg)
        cosine = cosine_utility(train_grad, val_grad)
        before = _batch_loss(model, val_batch, device)
        optimizer = make_optimizer(cfg, model)
        scaler = make_scaler(device)
        train_stats = train_on_batch(model, optimizer, scaler, train_batch, cfg, device)
        after = _batch_loss(model, val_batch, device)
        delta = before - after
        rows.append(
            {
                "candidate": idx,
                "cosine_utility": float(cosine),
                "delta_val_loss": float(delta),
                "val_loss_before": float(before),
                "val_loss_after": float(after),
                "gradient_loss": float(grad_stats["loss"]),
                "temporary_train_loss": float(train_stats["loss"]),
            }
        )

    _restore_trainable_state(model, base_state)
    cosines = [row["cosine_utility"] for row in rows]
    deltas = [row["delta_val_loss"] for row in rows]
    summary = {
        "candidates": n,
        "val_indices": [int(i) for i in val_indices],
        "pearson_cosine_delta_val": _pearson(cosines, deltas),
        "spearman_cosine_delta_val": _spearman(cosines, deltas),
        "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "std_cosine": float(np.std(cosines)) if cosines else 0.0,
        "mean_delta_val_loss": float(np.mean(deltas)) if deltas else 0.0,
        "std_delta_val_loss": float(np.std(deltas)) if deltas else 0.0,
    }
    write_json(Path(cfg["paths"]["output_dir"]) / "gradient_signal_check.json", summary)
    write_csv(Path(cfg["paths"]["output_dir"]) / "gradient_signal_check.csv", rows)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
