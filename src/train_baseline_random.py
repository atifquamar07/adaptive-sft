import argparse
import random
import time

try:
    from .data import load_processed_data
    from .modeling import (
        ensure_dirs,
        init_overhead,
        load_config,
        load_lora_model,
        make_optimizer,
        make_scaler,
        maybe_eval_curve,
        method_checkpoint_dir,
        reset_curve,
        sample_batch,
        save_checkpoint,
        save_overhead,
        set_seed,
        train_on_batch,
    )
except ImportError:
    from data import load_processed_data
    from modeling import (
        ensure_dirs,
        init_overhead,
        load_config,
        load_lora_model,
        make_optimizer,
        make_scaler,
        maybe_eval_curve,
        method_checkpoint_dir,
        reset_curve,
        sample_batch,
        save_checkpoint,
        save_overhead,
        set_seed,
        train_on_batch,
    )


def run(cfg):
    method = "random_sft"
    ensure_dirs(cfg)
    reset_curve(cfg, method)
    set_seed(int(cfg["seed"]))
    datasets = load_processed_data(cfg)
    model, tokenizer, device = load_lora_model(cfg, adapter_path=cfg["paths"]["warmup_dir"], train=True)
    optimizer = make_optimizer(cfg, model)
    scaler = make_scaler(device)
    rng = random.Random(int(cfg["seed"]) + 201)
    overhead = init_overhead(method)
    start_time = time.time()
    steps = int(cfg["training"]["continuation_steps"])

    for step in range(1, steps + 1):
        batch, _ = sample_batch(datasets["train_pool"], tokenizer, device, rng, int(cfg["training"]["micro_batch_size"]))
        stats = train_on_batch(model, optimizer, scaler, batch, cfg, device)
        overhead["optimizer_steps"] += 1
        if step % int(cfg["training"].get("log_every", 20)) == 0 or step == 1:
            print(f"{method} step={step} loss={stats['loss']:.4f}")
        maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, step)

    maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, steps, force=True)
    overhead["wall_clock_seconds"] = time.time() - start_time
    save_checkpoint(model, tokenizer, method_checkpoint_dir(cfg, method), cfg, {"method": method, "steps": steps})
    save_overhead(cfg, method, overhead)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
