import argparse
import json
import math
from pathlib import Path

try:
    from .modeling import load_config, read_jsonl
except ImportError:
    from modeling import load_config, read_jsonl


def _read_json(path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _test_loss(results, method):
    row = (results or {}).get(method, {})
    return row.get("test", {}).get("loss")


def _adaptive_stats(cfg, method="adaptive_utility_sft"):
    rows = read_jsonl(Path(cfg["paths"]["logs_dir"]) / f"{method}_selected_batches.jsonl")
    if not rows:
        return {}
    entropies = [float(row.get("selection_entropy", 0.0)) for row in rows if "selection_entropy" in row]
    ranks = [float(row.get("selected_predicted_rank", row.get("selected_true_rank", 0))) for row in rows]
    probs = [float(row.get("selected_probability", 0.0)) for row in rows if "selected_probability" in row]
    k = int(rows[0].get("candidate_count", cfg["adaptive"].get("candidate_count", 1)))
    return {
        "candidate_count": k,
        "uniform_entropy": math.log(max(k, 1)),
        "mean_selection_entropy": sum(entropies) / max(len(entropies), 1),
        "mean_selected_rank": sum(ranks) / max(len(ranks), 1),
        "random_mean_rank": (k + 1) / 2.0,
        "mean_selected_probability": sum(probs) / max(len(probs), 1),
        "score_sources": sorted({str(row.get("score_source", "")) for row in rows}),
    }


def run(cfg):
    out = Path(cfg["paths"]["output_dir"])
    results = _read_json(out / "results.json") or {}
    imitation = _read_json(out / "evaluator_imitation.json") or {}
    signal = _read_json(out / "gradient_signal_check.json") or {}
    adaptive_stats = _adaptive_stats(cfg)

    random_loss = _test_loss(results, "random_sft")
    oracle_loss = _test_loss(results, "oracle_gradient_sft")
    adaptive_loss = _test_loss(results, "adaptive_utility_sft")
    shuffled_loss = _test_loss(results, "adaptive_shuffled_scores")
    k = int(imitation.get("candidate_count", cfg["adaptive"].get("candidate_count", 1)))

    lines = ["DIAGNOSTIC SUMMARY"]
    if oracle_loss is None or random_loss is None:
        lines.append("A. Oracle vs random: missing results.")
    else:
        lines.append(
            "A. Oracle vs random: "
            + ("yes" if oracle_loss < random_loss else "no")
            + f" (oracle={oracle_loss:.6f}, random={random_loss:.6f})"
        )
    if adaptive_loss is None or shuffled_loss is None:
        lines.append("B. Adaptive vs shuffled: missing results.")
    else:
        lines.append(
            "B. Adaptive vs shuffled: "
            + ("yes" if adaptive_loss < shuffled_loss else "no")
            + f" (adaptive={adaptive_loss:.6f}, shuffled={shuffled_loss:.6f})"
        )
    if imitation:
        top1 = float(imitation.get("top1_agreement", 0.0))
        random_top1 = 1.0 / max(k, 1)
        pairwise = float(imitation.get("pairwise_ranking_accuracy", 0.0))
        spearman = float(imitation.get("mean_spearman", 0.0))
        exceeds_random = top1 > random_top1 and pairwise > 0.55 and spearman > 0.0
        lines.append(
            "C. Evaluator imitation: "
            + ("yes" if exceeds_random else "no")
            + f" (top1={top1:.3f} vs random={random_top1:.3f}, "
            f"pairwise={pairwise:.3f}, spearman={spearman:.3f})"
        )
    else:
        lines.append("C. Evaluator imitation: missing diagnostic.")
    if adaptive_stats:
        non_uniform = (
            adaptive_stats["mean_selection_entropy"] < adaptive_stats["uniform_entropy"]
            and adaptive_stats["mean_selected_rank"] < adaptive_stats["random_mean_rank"]
        )
        lines.append(
            "D. Adaptive non-uniformity: "
            + ("yes" if non_uniform else "no")
            + f" (entropy={adaptive_stats['mean_selection_entropy']:.3f} vs uniform={adaptive_stats['uniform_entropy']:.3f}, "
            f"selected_rank={adaptive_stats['mean_selected_rank']:.2f} vs random={adaptive_stats['random_mean_rank']:.2f}, "
            f"score_sources={adaptive_stats['score_sources']})"
        )
    else:
        lines.append("D. Adaptive non-uniformity: missing adaptive logs.")
    if signal:
        lines.append(
            "Gradient signal sanity: "
            f"Pearson={signal.get('pearson_cosine_delta_val', 0.0):.3f}, "
            f"Spearman={signal.get('spearman_cosine_delta_val', 0.0):.3f}"
        )
    if bool(cfg.get("data", {}).get("enable_synthetic_noise", False)):
        lines.append("E. Noisy-pool stress test: enabled for this run; compare separation against clean runs.")
    else:
        lines.append("E. Noisy-pool stress test: not enabled for this run.")

    text = "\n".join(lines)
    (out / "diagnostic_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
