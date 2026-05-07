import argparse
import math
import random
import time
from pathlib import Path

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
        selected_pool_stats,
        set_seed,
        static_candidate_batch_count,
        static_candidate_indices,
        static_training_schedule,
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
        selected_pool_stats,
        set_seed,
        static_candidate_batch_count,
        static_candidate_indices,
        static_training_schedule,
        train_on_batch,
    )


def run(cfg):
    method = "static_gradient_sft"
    ensure_dirs(cfg)
    reset_curve(cfg, method)
    set_seed(int(cfg["seed"]))
    datasets = load_processed_data(cfg)
    model, tokenizer, device = load_lora_model(cfg, adapter_path=cfg["paths"]["warmup_dir"], train=True)
    optimizer = make_optimizer(cfg, model)
    scaler = make_scaler(device)
    rng = random.Random(int(cfg["seed"]) + 401)
    overhead = init_overhead(method)
    start_time = time.time()
    log_path = Path(cfg["paths"]["logs_dir"]) / f"{method}_selection.jsonl"
    log_path.write_text("", encoding="utf-8")

    val_indices = sample_indices(rng, len(datasets["utility_val"]), int(cfg["utility"]["val_batch_size"]))
    val_batch = batch_from_indices(datasets["utility_val"], val_indices, tokenizer, device)
    val_grad, _ = compute_lora_gradient(model, val_batch, device, cfg)
    overhead["gradient_calls"] += 1

    steps = int(cfg["training"]["continuation_steps"])
    configured_candidates, effective_candidates, expanded_candidates = static_candidate_batch_count(
        cfg["static_gradient"], steps
    )
    if expanded_candidates:
        print(
            f"{method} expanded candidate_batches from {configured_candidates} to "
            f"{effective_candidates} so the selected pool covers {steps} steps"
        )

    scored = []
    candidate_indices = static_candidate_indices(
        rng,
        len(datasets["train_pool"]),
        int(cfg["training"]["micro_batch_size"]),
        effective_candidates,
    )
    for idx, indices in enumerate(candidate_indices):
        batch = batch_from_indices(datasets["train_pool"], indices, tokenizer, device)
        grad, stats = compute_lora_gradient(model, batch, device, cfg)
        utility = cosine_utility(grad, val_grad)
        row = {"candidate_index": idx, "batch_indices": indices, "score": utility, "loss": stats["loss"]}
        scored.append(row)
        append_jsonl(log_path, row)
        overhead["candidate_batches_scored"] += 1
        overhead["gradient_calls"] += 1
    count = max(1, int(math.ceil(len(scored) * float(cfg["static_gradient"]["selected_fraction"]))))
    selected = sorted(scored, key=lambda row: row["score"], reverse=True)[:count]
    schedule = static_training_schedule(selected, steps, rng)
    print(f"{method} selected_batches={len(selected)}")

    for step, choice in enumerate(schedule, start=1):
        batch = batch_from_indices(datasets["train_pool"], choice["batch_indices"], tokenizer, device)
        stats = train_on_batch(model, optimizer, scaler, batch, cfg, device)
        overhead["optimizer_steps"] += 1
        if step % int(cfg["training"].get("log_every", 20)) == 0 or step == 1:
            print(f"{method} step={step} loss={stats['loss']:.4f}")
        maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, step)

    maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, steps, force=True)
    overhead["wall_clock_seconds"] = time.time() - start_time
    overhead["configured_candidate_batches"] = configured_candidates
    overhead["effective_candidate_batches"] = effective_candidates
    overhead["expanded_candidate_batches"] = expanded_candidates
    overhead.update(selected_pool_stats(selected, steps))
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
