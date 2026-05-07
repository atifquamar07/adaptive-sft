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
        reset_curve,
        sample_batch,
        save_checkpoint,
        save_overhead,
        set_seed,
        train_on_batch,
    )


def run(cfg):
    method = "warmup"
    ensure_dirs(cfg)
    reset_curve(cfg, method)
    set_seed(int(cfg["seed"]))
    datasets = load_processed_data(cfg)
    model, tokenizer, device = load_lora_model(cfg, train=True)
    optimizer = make_optimizer(cfg, model)
    scaler = make_scaler(device)
    rng = random.Random(int(cfg["seed"]) + 11)
    steps = int(cfg["training"]["warmup_steps"])
    batch_size = int(cfg["training"]["micro_batch_size"])
    overhead = init_overhead(method)
    start_time = time.time()

    for step in range(1, steps + 1):
        batch, _ = sample_batch(datasets["train_pool"], tokenizer, device, rng, batch_size)
        stats = train_on_batch(model, optimizer, scaler, batch, cfg, device)
        overhead["optimizer_steps"] += 1
        if step % int(cfg["training"].get("log_every", 20)) == 0 or step == 1:
            print(f"warmup step={step} loss={stats['loss']:.4f} active_tokens={stats['active_tokens']}")
        maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, step, force=False)

    maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, steps, force=True)
    overhead["wall_clock_seconds"] = time.time() - start_time
    save_checkpoint(model, tokenizer, cfg["paths"]["warmup_dir"], cfg, {"steps": steps})
    save_overhead(cfg, "warmup", overhead)
    print(f"saved warmup checkpoint to {cfg['paths']['warmup_dir']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    run(cfg)


if __name__ == "__main__":
    main()
