import argparse
import json
from pathlib import Path

try:
    from .data import load_processed_data
    from .modeling import (
        ensure_dirs,
        evaluate_dataset_loss,
        load_config,
        load_lora_model,
        method_checkpoint_dir,
        read_jsonl,
        write_csv,
        write_json,
    )
except ImportError:
    from data import load_processed_data
    from modeling import (
        ensure_dirs,
        evaluate_dataset_loss,
        load_config,
        load_lora_model,
        method_checkpoint_dir,
        read_jsonl,
        write_csv,
        write_json,
    )


METHODS = [
    "random_sft",
    "static_loss_sft",
    "static_gradient_sft",
    "adaptive_utility_sft",
    "adaptive_shuffled_scores",
    "oracle_gradient_sft",
]


def run(cfg):
    ensure_dirs(cfg)
    datasets = load_processed_data(cfg)
    rows = []
    details = {}
    for method in METHODS:
        ckpt = method_checkpoint_dir(cfg, method)
        if not Path(ckpt).exists():
            print(f"skipping {method}: checkpoint not found at {ckpt}")
            continue
        model, tokenizer, _ = load_lora_model(cfg, adapter_path=ckpt, train=False)
        metrics = evaluate_dataset_loss(
            model,
            datasets["test"],
            tokenizer,
            cfg,
            split_name="test",
            max_batches=int(cfg["evaluation"].get("max_test_batches", 0)),
            batch_size=int(cfg["evaluation"].get("batch_size", cfg["training"]["micro_batch_size"])),
        )
        row = {"method": method, "test_loss": metrics["loss"], "test_perplexity": metrics["perplexity"]}
        curve_path = Path(cfg["paths"]["curves_dir"]) / f"{method}.jsonl"
        row["validation_curve_points"] = len(read_jsonl(curve_path))
        rows.append(row)
        details[method] = {"test": metrics, "validation_curve": str(curve_path)}
        print(f"{method}: test_loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.2f}")
    write_json(Path(cfg["paths"]["output_dir"]) / "results.json", details)
    write_csv(Path(cfg["paths"]["output_dir"]) / "results.csv", rows)
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
