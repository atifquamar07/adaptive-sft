import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .modeling import read_jsonl, write_csv, write_json
except ImportError:
    from modeling import read_jsonl, write_csv, write_json


METHOD_ORDER = [
    "random_sft",
    "static_loss_sft",
    "static_gradient_sft",
    "adaptive_shuffled_scores",
    "adaptive_utility_sft",
    "oracle_gradient_sft",
]

ADAPTIVE_METHODS = ["adaptive_utility_sft", "adaptive_shuffled_scores"]

ADAPTIVE_LOG_METRICS = [
    ("predicted_utility", "predicted utility"),
    ("selected_mean_sft_loss", "selected batch mean SFT loss"),
    ("selected_mean_input_length", "selected batch mean input length"),
    ("selected_mean_response_length", "selected batch mean response length"),
    ("selected_mean_token_entropy", "selected batch mean token entropy"),
    ("selected_gradient_norm", "selected batch gradient norm"),
    ("raw_predicted_score_mean", "raw predicted score mean"),
    ("raw_predicted_score_std", "raw predicted score std"),
    ("standardized_score_std", "standardized score std"),
    ("selection_entropy", "selection entropy"),
    ("selected_probability", "selected probability"),
    ("max_probability", "max probability"),
    ("score_gap_best_second", "score gap best minus second"),
    ("candidate_score_std", "candidate score std"),
    ("selected_predicted_rank", "selected rank by predicted score"),
    ("selected_evaluator_rank", "selected rank by evaluator score"),
    ("selected_true_rank", "selected rank by selection score"),
]

ORACLE_LOG_METRICS = [
    ("selected_true_utility", "selected true gradient utility"),
    ("selected_rank", "selected true utility rank"),
    ("candidate_utility_std", "candidate utility std"),
    ("candidate_utility_range", "candidate utility range"),
    ("gradient_calls_cumulative", "gradient calls cumulative"),
    ("train_loss", "train loss"),
    ("selected_gradient_loss", "selected gradient-scoring loss"),
]

OVERHEAD_METRICS = ["candidate_batches_scored", "gradient_calls", "optimizer_steps", "wall_clock_seconds"]


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path):
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _auc(records):
    points = sorted((int(r["step"]), float(r["loss"])) for r in records if "step" in r and "loss" in r)
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (s0, l0), (s1, l1) in zip(points[:-1], points[1:]):
        total += (s1 - s0) * (l0 + l1) / 2.0
    return float(total)


def _sem(values):
    values = list(values)
    if not values:
        return 0.0
    return float(np.std(values) / np.sqrt(len(values)))


def _method_sort_key(method):
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def _aggregate_step_rows(seed_dirs, rel_path_template, metrics, method=None):
    grouped = defaultdict(list)
    for seed_dir in seed_dirs:
        rel_path = rel_path_template.format(method=method) if method else rel_path_template
        for record in read_jsonl(seed_dir / rel_path):
            if "step" not in record:
                continue
            step = int(record["step"])
            for key, _label in metrics:
                value = _to_float(record.get(key))
                if value is not None:
                    grouped[(key, step)].append(value)
    rows = []
    for (key, step), values in sorted(grouped.items()):
        rows.append(
            {
                "method": method or "",
                "metric": key,
                "step": step,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "sem": _sem(values),
                "seeds": len(values),
            }
        )
    return rows


def _plot_step_metric(plt, plots_dir, rows, metric, ylabel, filename, title):
    metric_rows = [row for row in rows if row["metric"] == metric]
    if not metric_rows:
        return
    metric_rows = sorted(metric_rows, key=lambda row: int(row["step"]))
    steps = np.array([int(row["step"]) for row in metric_rows], dtype=float)
    mean = np.array([float(row["mean"]) for row in metric_rows], dtype=float)
    std = np.array([float(row["std"]) for row in metric_rows], dtype=float)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, mean, marker="o", markersize=2.5, linewidth=1.4)
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    plt.xlabel("optimizer step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=160)
    plt.close()


def _plot_validation_outputs(plt, plots_dir, final_rows, curve_rows):
    if curve_rows:
        by_method = defaultdict(list)
        for row in curve_rows:
            by_method[row["method"]].append(row)

        plt.figure(figsize=(10, 5))
        for method in sorted(by_method, key=_method_sort_key):
            rows = sorted(by_method[method], key=lambda r: int(r["step"]))
            steps = np.array([int(row["step"]) for row in rows], dtype=float)
            mean = np.array([float(row["mean_validation_loss"]) for row in rows], dtype=float)
            std = np.array([float(row["std_validation_loss"]) for row in rows], dtype=float)
            plt.plot(steps, mean, label=method, marker="o", markersize=3, linewidth=1.5)
            plt.fill_between(steps, mean - std, mean + std, alpha=0.15)
        plt.xlabel("optimizer step")
        plt.ylabel("validation SFT loss")
        plt.title("validation loss across seeds, mean +/- 1 std")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "seed_validation_loss_mean_std.png", dpi=160)
        plt.close()

        cols = 2
        methods = sorted(by_method, key=_method_sort_key)
        rows_count = int(np.ceil(len(methods) / cols))
        fig, axes = plt.subplots(rows_count, cols, figsize=(10, max(3, rows_count * 2.7)))
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for ax, method in zip(axes, methods):
            rows = sorted(by_method[method], key=lambda r: int(r["step"]))
            steps = np.array([int(row["step"]) for row in rows], dtype=float)
            mean = np.array([float(row["mean_validation_loss"]) for row in rows], dtype=float)
            std = np.array([float(row["std_validation_loss"]) for row in rows], dtype=float)
            ax.plot(steps, mean, marker="o", markersize=2.5, linewidth=1.4)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
            ax.set_title(method)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel("validation SFT loss")
        for ax in axes[len(methods) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "seed_validation_loss_by_method_mean_std.png", dpi=160)
        plt.close(fig)

    if final_rows:
        rows = sorted(final_rows, key=lambda r: _method_sort_key(r["method"]))
        methods = [row["method"] for row in rows]
        means = [float(row["mean_test_loss"]) for row in rows]
        stds = [float(row["std_test_loss"]) for row in rows]
        x = np.arange(len(methods))

        plt.figure(figsize=(10, 4))
        plt.bar(x, means, yerr=stds, capsize=4)
        plt.xticks(x, methods, rotation=25, ha="right")
        plt.ylabel("test SFT loss")
        plt.title("final test loss across seeds, mean +/- 1 std")
        plt.tight_layout()
        plt.savefig(plots_dir / "seed_test_loss_mean_std.png", dpi=160)
        plt.close()

        auc_means = [float(row["mean_auc_validation"]) for row in rows]
        auc_stds = [float(row["std_auc_validation"]) for row in rows]
        plt.figure(figsize=(10, 4))
        plt.bar(x, auc_means, yerr=auc_stds, capsize=4)
        plt.xticks(x, methods, rotation=25, ha="right")
        plt.ylabel("validation-loss AUC")
        plt.title("validation-loss AUC across seeds, mean +/- 1 std")
        plt.tight_layout()
        plt.savefig(plots_dir / "seed_validation_auc_mean_std.png", dpi=160)
        plt.close()


def _plot_overhead_outputs(seed_dirs, root_path, plt, plots_dir):
    raw_rows = []
    for seed_dir in seed_dirs:
        seed = seed_dir.name.removeprefix("seed_")
        for path in sorted((seed_dir / "logs").glob("*_overhead.json")):
            record = _read_json(path)
            if not record:
                continue
            row = {"seed": seed, "method": record.get("method", path.stem.removesuffix("_overhead"))}
            for metric in OVERHEAD_METRICS:
                value = _to_float(record.get(metric))
                if value is not None:
                    row[metric] = value
            raw_rows.append(row)

    if not raw_rows:
        return

    grouped = defaultdict(list)
    for row in raw_rows:
        for metric in OVERHEAD_METRICS:
            if metric in row:
                grouped[(row["method"], metric)].append(float(row[metric]))

    aggregate_rows = []
    for (method, metric), values in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "method": method,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "sem": _sem(values),
                "seeds": len(values),
            }
        )
    write_csv(root_path / "seed_overhead.csv", aggregate_rows)

    methods = sorted({row["method"] for row in raw_rows}, key=_method_sort_key)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    plot_metrics = ["candidate_batches_scored", "gradient_calls", "wall_clock_seconds"]
    for ax, metric in zip(axes, plot_metrics):
        means = []
        stds = []
        for method in methods:
            values = grouped.get((method, metric), [])
            means.append(float(np.mean(values)) if values else 0.0)
            stds.append(float(np.std(values)) if values else 0.0)
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.set_ylabel(metric)
    axes[-1].set_xticks(x, methods, rotation=25, ha="right")
    fig.suptitle("selection overhead across seeds, mean +/- 1 std")
    fig.tight_layout()
    fig.savefig(plots_dir / "seed_selection_overhead_mean_std.png", dpi=160)
    plt.close(fig)


def _plot_selection_log_outputs(seed_dirs, root_path, plt, plots_dir):
    all_rows = []
    for method in ADAPTIVE_METHODS:
        rows = _aggregate_step_rows(
            seed_dirs,
            "logs/{method}_selected_batches.jsonl",
            ADAPTIVE_LOG_METRICS,
            method=method,
        )
        all_rows.extend(rows)
        for key, label in ADAPTIVE_LOG_METRICS:
            _plot_step_metric(
                plt,
                plots_dir,
                rows,
                key,
                label,
                f"seed_{method}_{key}_mean_std.png",
                f"{method}: {label} across seeds",
            )

    oracle_rows = _aggregate_step_rows(
        seed_dirs,
        "logs/{method}_selected_batches.jsonl",
        ORACLE_LOG_METRICS,
        method="oracle_gradient_sft",
    )
    all_rows.extend(oracle_rows)
    for key, label in ORACLE_LOG_METRICS:
        _plot_step_metric(
            plt,
            plots_dir,
            oracle_rows,
            key,
            label,
            f"seed_oracle_gradient_sft_{key}_mean_std.png",
            f"oracle_gradient_sft: {label} across seeds",
        )

    if all_rows:
        write_csv(root_path / "seed_selection_logs.csv", all_rows)


def _plot_utility_label_outputs(seed_dirs, plt, plots_dir):
    by_seed = {}
    for seed_dir in seed_dirs:
        records = read_jsonl(seed_dir / "utility_labels.jsonl")
        values = [_to_float(record.get("utility")) for record in records]
        values = [value for value in values if value is not None]
        if values:
            by_seed[seed_dir.name] = values

    if not by_seed:
        return

    all_values = [value for values in by_seed.values() for value in values]
    plt.figure(figsize=(8, 4))
    bins = min(40, max(10, int(np.sqrt(len(all_values)))))
    for seed, values in by_seed.items():
        plt.hist(values, bins=bins, alpha=0.16, density=True, label=seed)
    plt.hist(all_values, bins=bins, histtype="step", linewidth=2.0, density=True, color="black", label="all seeds")
    plt.xlabel("gradient-cosine utility")
    plt.ylabel("density")
    plt.title("utility label distribution across seeds")
    if len(by_seed) <= 8:
        plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "seed_utility_label_histogram.png", dpi=160)
    plt.close()


def _plot_scatter_by_seed(plt, plots_dir, rows_by_seed, x_key, y_key, filename, xlabel, ylabel, title):
    if not rows_by_seed:
        return
    plt.figure(figsize=(5.5, 5.5))
    for seed, rows in rows_by_seed.items():
        x = [_to_float(row.get(x_key)) for row in rows]
        y = [_to_float(row.get(y_key)) for row in rows]
        points = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if not points:
            continue
        px, py = zip(*points)
        plt.scatter(px, py, s=10, alpha=0.45, label=seed)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(rows_by_seed) <= 8:
        plt.legend(markerscale=1.5)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=160)
    plt.close()


def _plot_evaluator_outputs(seed_dirs, root_path, plt, plots_dir):
    dev_rows = {}
    for seed_dir in seed_dirs:
        rows = _read_csv(seed_dir / "evaluator_dev_predictions.csv")
        if rows:
            dev_rows[seed_dir.name] = rows
    _plot_scatter_by_seed(
        plt,
        plots_dir,
        dev_rows,
        "true_utility",
        "predicted_utility",
        "seed_evaluator_dev_predicted_vs_true.png",
        "true dev utility target",
        "predicted dev utility target",
        "evaluator dev predictions across seeds",
    )

    imitation_rows = {}
    for seed_dir in seed_dirs:
        rows = _read_csv(seed_dir / "evaluator_imitation.csv")
        if rows:
            imitation_rows[seed_dir.name] = rows
    _plot_scatter_by_seed(
        plt,
        plots_dir,
        imitation_rows,
        "true_utility",
        "predicted_utility",
        "seed_evaluator_imitation_predicted_vs_true.png",
        "true gradient utility",
        "predicted utility",
        "evaluator imitation predictions across seeds",
    )

    selected_ranks = []
    pred_stds = []
    true_stds = []
    for seed, rows in imitation_rows.items():
        for row in rows:
            if _is_true(row.get("predicted_selected")):
                rank = _to_float(row.get("true_rank"))
                if rank is not None:
                    selected_ranks.append(int(rank))
        by_set = defaultdict(lambda: {"pred": [], "true": []})
        for row in rows:
            set_key = (seed, row.get("candidate_set"))
            pred = _to_float(row.get("predicted_utility"))
            true = _to_float(row.get("true_utility"))
            if pred is not None:
                by_set[set_key]["pred"].append(pred)
            if true is not None:
                by_set[set_key]["true"].append(true)
        for vals in by_set.values():
            if vals["pred"]:
                pred_stds.append(float(np.std(vals["pred"])))
            if vals["true"]:
                true_stds.append(float(np.std(vals["true"])))

    if selected_ranks:
        plt.figure(figsize=(7, 4))
        plt.hist(selected_ranks, bins=range(1, max(selected_ranks) + 2), align="left")
        plt.xlabel("true utility rank of predicted-selected batch")
        plt.ylabel("candidate sets across seeds")
        plt.title("evaluator imitation selected-rank histogram across seeds")
        plt.tight_layout()
        plt.savefig(plots_dir / "seed_evaluator_imitation_selected_true_rank_histogram.png", dpi=160)
        plt.close()

    if pred_stds or true_stds:
        plt.figure(figsize=(7, 4))
        if pred_stds:
            plt.hist(pred_stds, bins=20, alpha=0.7, label="predicted")
        if true_stds:
            plt.hist(true_stds, bins=20, alpha=0.7, label="true")
        plt.xlabel("within-candidate-set score std")
        plt.ylabel("candidate sets across seeds")
        plt.title("evaluator imitation score std across seeds")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "seed_evaluator_imitation_score_std_histogram.png", dpi=160)
        plt.close()

    metric_sources = {
        "evaluator_metrics": (
            "evaluator_metrics.json",
            ["dev_mse", "dev_spearman", "dev_ranking_accuracy", "dev_raw_spearman", "dev_raw_ranking_accuracy"],
        ),
        "evaluator_imitation": (
            "evaluator_imitation.json",
            [
                "top1_agreement",
                "top3_agreement",
                "mean_spearman",
                "pairwise_ranking_accuracy",
                "avg_true_rank_of_pred_selected",
            ],
        ),
    }
    summary_rows = []
    for source_name, (filename, metric_names) in metric_sources.items():
        grouped = defaultdict(list)
        for seed_dir in seed_dirs:
            record = _read_json(seed_dir / filename)
            if not record:
                continue
            for metric in metric_names:
                value = _to_float(record.get(metric))
                if value is not None:
                    grouped[metric].append(value)
        if not grouped:
            continue
        metrics = list(metric_names)
        means = [float(np.mean(grouped[metric])) if grouped.get(metric) else 0.0 for metric in metrics]
        stds = [float(np.std(grouped[metric])) if grouped.get(metric) else 0.0 for metric in metrics]
        x = np.arange(len(metrics))
        plt.figure(figsize=(10, 4))
        plt.bar(x, means, yerr=stds, capsize=4)
        plt.xticks(x, metrics, rotation=25, ha="right")
        plt.ylabel("metric value")
        plt.title(f"{source_name} across seeds, mean +/- 1 std")
        plt.tight_layout()
        plt.savefig(plots_dir / f"seed_{source_name}_mean_std.png", dpi=160)
        plt.close()
        for metric in metrics:
            values = grouped.get(metric, [])
            if values:
                summary_rows.append(
                    {
                        "source": source_name,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "sem": _sem(values),
                        "seeds": len(values),
                    }
                )
    if summary_rows:
        write_csv(root_path / "seed_evaluator_diagnostics.csv", summary_rows)


def _plot_gradient_signal_outputs(seed_dirs, root_path, plt, plots_dir):
    rows_by_seed = {}
    for seed_dir in seed_dirs:
        rows = _read_csv(seed_dir / "gradient_signal_check.csv")
        if rows:
            rows_by_seed[seed_dir.name] = rows
    _plot_scatter_by_seed(
        plt,
        plots_dir,
        rows_by_seed,
        "cosine_utility",
        "delta_val_loss",
        "seed_gradient_signal_cosine_vs_delta_val.png",
        "gradient cosine utility",
        "validation loss decrease after temporary step",
        "gradient signal sanity across seeds",
    )

    metrics = ["pearson_cosine_delta_val", "spearman_cosine_delta_val", "mean_cosine", "std_cosine", "mean_delta_val_loss"]
    grouped = defaultdict(list)
    for seed_dir in seed_dirs:
        record = _read_json(seed_dir / "gradient_signal_check.json")
        if not record:
            continue
        for metric in metrics:
            value = _to_float(record.get(metric))
            if value is not None:
                grouped[metric].append(value)
    if not grouped:
        return

    summary_rows = []
    for metric in metrics:
        values = grouped.get(metric, [])
        if values:
            summary_rows.append(
                {
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "sem": _sem(values),
                    "seeds": len(values),
                }
            )
    write_csv(root_path / "seed_gradient_signal_summary.csv", summary_rows)

    x = np.arange(len(metrics))
    means = [float(np.mean(grouped[metric])) if grouped.get(metric) else 0.0 for metric in metrics]
    stds = [float(np.std(grouped[metric])) if grouped.get(metric) else 0.0 for metric in metrics]
    plt.figure(figsize=(10, 4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, metrics, rotation=25, ha="right")
    plt.ylabel("metric value")
    plt.title("gradient signal summary across seeds, mean +/- 1 std")
    plt.tight_layout()
    plt.savefig(plots_dir / "seed_gradient_signal_summary_mean_std.png", dpi=160)
    plt.close()


def _plot_seed_outputs(root_path: Path, seed_dirs, final_rows, curve_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"seed plots skipped: {exc}")
        return

    plots_dir = root_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_validation_outputs(plt, plots_dir, final_rows, curve_rows)
    _plot_overhead_outputs(seed_dirs, root_path, plt, plots_dir)
    _plot_selection_log_outputs(seed_dirs, root_path, plt, plots_dir)
    _plot_utility_label_outputs(seed_dirs, plt, plots_dir)
    _plot_evaluator_outputs(seed_dirs, root_path, plt, plots_dir)
    _plot_gradient_signal_outputs(seed_dirs, root_path, plt, plots_dir)


def run(root: str, threshold=None):
    root_path = Path(root)
    seed_dirs = sorted(path for path in root_path.glob("seed_*") if path.is_dir())
    final_by_method = defaultdict(list)
    curve_by_method_step = defaultdict(list)
    auc_by_method = defaultdict(list)
    threshold_steps = defaultdict(list)

    for seed_dir in seed_dirs:
        for row in _read_csv(seed_dir / "results.csv"):
            final_by_method[row["method"]].append(float(row["test_loss"]))
        for curve in (seed_dir / "curves").glob("*.jsonl"):
            records = read_jsonl(curve)
            if not records:
                continue
            method = curve.stem
            if method == "warmup":
                continue
            auc_by_method[method].append(_auc(records))
            reached = None
            for record in records:
                step = int(record["step"])
                loss = float(record["loss"])
                curve_by_method_step[(method, step)].append(loss)
                if threshold is not None and reached is None and loss <= float(threshold):
                    reached = step
            if threshold is not None:
                threshold_steps[method].append(reached if reached is not None else None)

    final_rows = []
    for method, losses in sorted(final_by_method.items(), key=lambda item: _method_sort_key(item[0])):
        aucs = auc_by_method.get(method, [0.0])
        final_rows.append(
            {
                "method": method,
                "seeds": len(losses),
                "mean_test_loss": float(np.mean(losses)),
                "std_test_loss": float(np.std(losses)),
                "sem_test_loss": _sem(losses),
                "mean_auc_validation": float(np.mean(aucs)),
                "std_auc_validation": float(np.std(aucs)),
                "sem_auc_validation": _sem(aucs),
                "mean_steps_to_threshold": (
                    float(np.mean([x for x in threshold_steps.get(method, []) if x is not None]))
                    if threshold is not None and any(x is not None for x in threshold_steps.get(method, []))
                    else ""
                ),
            }
        )
    curve_rows = []
    for (method, step), losses in sorted(curve_by_method_step.items()):
        curve_rows.append(
            {
                "method": method,
                "step": step,
                "mean_validation_loss": float(np.mean(losses)),
                "std_validation_loss": float(np.std(losses)),
                "sem_validation_loss": _sem(losses),
                "seeds": len(losses),
            }
        )
    summary = {"seed_dirs": [str(path) for path in seed_dirs], "final": final_rows}
    write_json(root_path / "seed_summary.json", summary)
    write_csv(root_path / "seed_results.csv", final_rows)
    write_csv(root_path / "seed_curves.csv", curve_rows)
    _plot_seed_outputs(root_path, seed_dirs, final_rows, curve_rows)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/seeds")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    run(args.root, threshold=args.threshold)


if __name__ == "__main__":
    main()
