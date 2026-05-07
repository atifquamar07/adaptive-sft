import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

try:
    from .modeling import ensure_dirs, load_config, read_jsonl
except ImportError:
    from modeling import ensure_dirs, load_config, read_jsonl


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_results(cfg, plt):
    rows = _read_csv(Path(cfg["paths"]["output_dir"]) / "results.csv")
    if not rows:
        return
    order = [
        "random_sft",
        "static_loss_sft",
        "static_gradient_sft",
        "adaptive_shuffled_scores",
        "adaptive_utility_sft",
        "oracle_gradient_sft",
    ]
    by_method = {row["method"]: row for row in rows}
    methods = [method for method in order if method in by_method]
    methods.extend(row["method"] for row in rows if row["method"] not in methods)
    losses = [float(by_method[method]["test_loss"]) for method in methods]
    plt.figure(figsize=(9, 4))
    plt.bar(methods, losses)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("test SFT loss")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "test_loss_comparison.png", dpi=160)
    plt.close()


def _latest_curve_run(records):
    runs = []
    current = []
    last_step = None
    for record in records:
        if "step" not in record or "loss" not in record:
            continue
        step = int(record["step"])
        if current and last_step is not None and step < last_step:
            runs.append(current)
            current = []
        if current and step == last_step:
            current[-1] = record
        else:
            current.append(record)
        last_step = step
    if current:
        runs.append(current)
    return runs[-1] if runs else []


def _plot_curve_small_multiples(cfg, plt, curve_series):
    if not curve_series:
        return
    cols = 2
    rows = math.ceil(len(curve_series) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, max(3, rows * 2.6)))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, (name, records) in zip(axes, curve_series):
        ax.plot(
            [r["step"] for r in records],
            [r["loss"] for r in records],
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
        ax.set_title(name)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("validation SFT loss")
    for ax in axes[len(curve_series) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(cfg["paths"]["plots_dir"]) / "validation_loss_curves_by_method.png", dpi=160)
    plt.close(fig)


def _plot_curves(cfg, plt):
    order = [
        "random_sft",
        "static_loss_sft",
        "static_gradient_sft",
        "adaptive_shuffled_scores",
        "adaptive_utility_sft",
        "oracle_gradient_sft",
    ]
    curve_series = []
    for method in order:
        path = Path(cfg["paths"]["curves_dir"]) / f"{method}.jsonl"
        records = _latest_curve_run(read_jsonl(path))
        if records:
            curve_series.append((method, records))
    for path in sorted(Path(cfg["paths"]["curves_dir"]).glob("*.jsonl")):
        if path.stem == "warmup" or path.stem in order:
            continue
        records = _latest_curve_run(read_jsonl(path))
        if records:
            curve_series.append((path.stem, records))
    if not curve_series:
        return

    plt.figure(figsize=(9, 5))
    for name, records in curve_series:
        plt.plot(
            [r["step"] for r in records],
            [r["loss"] for r in records],
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
    plt.xlabel("optimizer step")
    plt.ylabel("validation SFT loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "validation_loss_curves.png", dpi=160)
    plt.close()
    _plot_curve_small_multiples(cfg, plt, curve_series)


def _plot_adaptive_logs(cfg, plt, method):
    path = Path(cfg["paths"]["logs_dir"]) / f"{method}_selected_batches.jsonl"
    records = read_jsonl(path)
    if not records:
        return
    series = [
        ("predicted_utility", "predicted_utility"),
        ("selected_mean_sft_loss", "selected_batch_mean_loss"),
        ("selected_mean_input_length", "selected_batch_mean_length"),
        ("selected_gradient_norm", "selected_batch_gradient_norm"),
    ]
    for key, name in series:
        values = [r.get(key) for r in records if key in r]
        if not values:
            continue
        steps = [r["step"] for r in records if key in r]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, values)
        plt.xlabel("optimizer step")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(Path(cfg["paths"]["plots_dir"]) / f"{method}_{name}.png", dpi=160)
        plt.close()


def _plot_oracle_logs(cfg, plt):
    path = Path(cfg["paths"]["logs_dir"]) / "oracle_gradient_sft_selected_batches.jsonl"
    records = read_jsonl(path)
    if not records:
        return
    plt.figure(figsize=(8, 4))
    plt.plot([r["step"] for r in records], [r["selected_true_utility"] for r in records])
    plt.xlabel("optimizer step")
    plt.ylabel("selected true gradient utility")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "oracle_gradient_sft_selected_true_utility.png", dpi=160)
    plt.close()


def _plot_utility_labels(cfg, plt):
    path = Path(cfg["paths"]["utility_labels"])
    if not path.exists():
        return
    records = torch.load(path, map_location="cpu")
    utilities = [float(r["utility"]) for r in records]
    if not utilities:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(utilities, bins=30)
    plt.xlabel("gradient-cosine utility")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "utility_label_histogram.png", dpi=160)
    plt.close()


def _plot_evaluator_dev(cfg, plt):
    rows = _read_csv(cfg["paths"]["evaluator_dev_predictions"])
    if not rows:
        return
    true = [float(row["true_utility"]) for row in rows]
    pred = [float(row["predicted_utility"]) for row in rows]
    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, s=14, alpha=0.8)
    plt.xlabel("true dev utility target")
    plt.ylabel("predicted dev utility target")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "evaluator_dev_predicted_vs_true.png", dpi=160)
    plt.close()


def _plot_overhead(cfg, plt):
    logs = []
    for path in sorted(Path(cfg["paths"]["logs_dir"]).glob("*_overhead.json")):
        import json

        logs.append(json.loads(path.read_text(encoding="utf-8")))
    if not logs:
        return
    methods = [row["method"] for row in logs]
    candidates = [float(row.get("candidate_batches_scored", 0)) for row in logs]
    gradients = [float(row.get("gradient_calls", 0)) for row in logs]
    x = range(len(methods))
    plt.figure(figsize=(10, 4))
    plt.bar(x, candidates, label="candidate batches scored")
    plt.bar(x, gradients, bottom=candidates, label="gradient calls")
    plt.xticks(list(x), methods, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "selection_overhead.png", dpi=160)
    plt.close()


def _plot_evaluator_imitation(cfg, plt):
    rows = _read_csv(Path(cfg["paths"]["output_dir"]) / "evaluator_imitation.csv")
    if not rows:
        return
    true = [float(row["true_utility"]) for row in rows]
    pred = [float(row["predicted_utility"]) for row in rows]
    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, s=12, alpha=0.6)
    plt.xlabel("true gradient utility")
    plt.ylabel("predicted utility")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "evaluator_imitation_predicted_vs_true.png", dpi=160)
    plt.close()

    selected_ranks = [int(row["true_rank"]) for row in rows if str(row.get("predicted_selected", "")).lower() == "true"]
    if selected_ranks:
        plt.figure(figsize=(7, 4))
        plt.hist(selected_ranks, bins=range(1, max(selected_ranks) + 2), align="left")
        plt.xlabel("true utility rank of predicted-selected batch")
        plt.ylabel("candidate sets")
        plt.tight_layout()
        plt.savefig(Path(cfg["paths"]["plots_dir"]) / "evaluator_imitation_selected_true_rank_histogram.png", dpi=160)
        plt.close()

    by_set = defaultdict(lambda: {"pred": [], "true": []})
    for row in rows:
        by_set[row["candidate_set"]]["pred"].append(float(row["predicted_utility"]))
        by_set[row["candidate_set"]]["true"].append(float(row["true_utility"]))
    pred_std = [float(torch.tensor(vals["pred"]).std(unbiased=False).item()) for vals in by_set.values()]
    true_std = [float(torch.tensor(vals["true"]).std(unbiased=False).item()) for vals in by_set.values()]
    plt.figure(figsize=(7, 4))
    plt.hist(pred_std, bins=20, alpha=0.7, label="predicted")
    plt.hist(true_std, bins=20, alpha=0.7, label="true")
    plt.xlabel("within-candidate-set score std")
    plt.ylabel("candidate sets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "evaluator_imitation_score_std_histogram.png", dpi=160)
    plt.close()


def _plot_gradient_signal(cfg, plt):
    rows = _read_csv(Path(cfg["paths"]["output_dir"]) / "gradient_signal_check.csv")
    if not rows:
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(
        [float(row["cosine_utility"]) for row in rows],
        [float(row["delta_val_loss"]) for row in rows],
        s=18,
        alpha=0.8,
    )
    plt.xlabel("gradient cosine utility")
    plt.ylabel("validation loss decrease after temporary step")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots_dir"]) / "gradient_signal_cosine_vs_delta_val.png", dpi=160)
    plt.close()


def run(cfg):
    ensure_dirs(cfg)
    import matplotlib.pyplot as plt

    _plot_results(cfg, plt)
    _plot_curves(cfg, plt)
    _plot_adaptive_logs(cfg, plt, "adaptive_utility_sft")
    _plot_adaptive_logs(cfg, plt, "adaptive_shuffled_scores")
    _plot_oracle_logs(cfg, plt)
    _plot_utility_labels(cfg, plt)
    _plot_evaluator_dev(cfg, plt)
    _plot_overhead(cfg, plt)
    _plot_evaluator_imitation(cfg, plt)
    _plot_gradient_signal(cfg, plt)
    print(f"plots written to {cfg['paths']['plots_dir']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
