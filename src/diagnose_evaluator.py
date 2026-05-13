import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch

try:
    from .data import load_processed_data
    from .evaluator import load_trained_evaluator
    from .features import compute_batch_features, vectorize_features
    from .grad_utils import compute_lora_gradient, cosine_utility
    from .modeling import (
        batch_from_indices,
        ensure_dirs,
        load_config,
        load_lora_model,
        method_checkpoint_dir,
        sample_indices,
        write_csv,
        write_json,
    )
except ImportError:
    from data import load_processed_data
    from evaluator import load_trained_evaluator
    from features import compute_batch_features, vectorize_features
    from grad_utils import compute_lora_gradient, cosine_utility
    from modeling import (
        batch_from_indices,
        ensure_dirs,
        load_config,
        load_lora_model,
        method_checkpoint_dir,
        sample_indices,
        write_csv,
        write_json,
    )


def _rank_desc(values: List[float]) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _spearman(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return 0.0
    ra = np.array(_rank_desc(a), dtype=np.float64)
    rb = np.array(_rank_desc(b), dtype=np.float64)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _pairwise_accuracy(true_scores: List[float], pred_scores: List[float]) -> float:
    correct = 0
    total = 0
    for i in range(len(true_scores)):
        for j in range(i + 1, len(true_scores)):
            true_diff = true_scores[i] - true_scores[j]
            pred_diff = pred_scores[i] - pred_scores[j]
            if abs(true_diff) <= 1.0e-12:
                continue
            correct += int(np.sign(true_diff) == np.sign(pred_diff))
            total += 1
    return float(correct / max(total, 1))


def _adapter_path_for_checkpoint(cfg, checkpoint_method: str) -> str:
    if checkpoint_method == "warmup":
        return cfg["paths"]["warmup_dir"]
    return method_checkpoint_dir(cfg, checkpoint_method)


def run(cfg, checkpoint_method: str = None):
    ensure_dirs(cfg)
    datasets = load_processed_data(cfg)
    checkpoint_method = checkpoint_method or cfg.get("diagnostics", {}).get("evaluator_imitation_checkpoint_method", "warmup")
    model, tokenizer, device = load_lora_model(
        cfg,
        adapter_path=_adapter_path_for_checkpoint(cfg, checkpoint_method),
        train=True,
    )
    evaluator, names, mean, std, _ = load_trained_evaluator(cfg["paths"]["evaluator"])
    rng = random.Random(int(cfg["seed"]) + 801)
    sets = int(cfg.get("diagnostics", {}).get("evaluator_imitation_sets", 32))
    k = int(cfg.get("diagnostics", {}).get("candidate_count", cfg["adaptive"].get("candidate_count", 8)))
    batch_size = int(cfg["training"]["micro_batch_size"])
    val_batch_size = int(cfg.get("diagnostics", {}).get("val_batch_size", cfg["utility"].get("val_batch_size", batch_size)))
    total_steps = int(cfg["training"]["warmup_steps"]) + int(cfg["training"]["continuation_steps"])
    rows = []
    per_set = []

    for set_id in range(sets):
        val_indices = sample_indices(rng, len(datasets["utility_val"]), val_batch_size)
        val_batch = batch_from_indices(datasets["utility_val"], val_indices, tokenizer, device)
        val_grad, _ = compute_lora_gradient(model, val_batch, device, cfg)
        true_scores = []
        pred_scores = []
        for cand in range(k):
            indices = sample_indices(rng, len(datasets["train_pool"]), batch_size)
            batch = batch_from_indices(datasets["train_pool"], indices, tokenizer, device)
            feats = compute_batch_features(
                model,
                batch,
                device,
                cfg,
                current_step=int(cfg["training"]["warmup_steps"]),
                total_steps=total_steps,
                use_gradient_features=False,
            )
            x = torch.tensor([vectorize_features(feats, names)], dtype=torch.float32)
            with torch.no_grad():
                pred = float(evaluator((x - mean) / std).item())
            train_grad, _ = compute_lora_gradient(model, batch, device, cfg)
            true = cosine_utility(train_grad, val_grad)
            true_scores.append(float(true))
            pred_scores.append(float(pred))
            rows.append(
                {
                    "candidate_set": set_id,
                    "candidate": cand,
                    "true_utility": float(true),
                    "predicted_utility": float(pred),
                }
            )
        true_ranks = _rank_desc(true_scores)
        pred_ranks = _rank_desc(pred_scores)
        pred_best = int(np.argmax(pred_scores))
        true_best = int(np.argmax(true_scores))
        true_top3 = set(sorted(range(k), key=lambda i: true_scores[i], reverse=True)[: min(3, k)])
        for idx in range(k):
            rows[-k + idx]["true_rank"] = true_ranks[idx]
            rows[-k + idx]["predicted_rank"] = pred_ranks[idx]
            rows[-k + idx]["predicted_selected"] = idx == pred_best
            rows[-k + idx]["true_best"] = idx == true_best
        per_set.append(
            {
                "candidate_set": set_id,
                "top1_agreement": pred_best == true_best,
                "top3_agreement": pred_best in true_top3,
                "spearman": _spearman(true_scores, pred_scores),
                "pairwise_accuracy": _pairwise_accuracy(true_scores, pred_scores),
                "predicted_score_std": float(np.std(pred_scores)),
                "true_utility_std": float(np.std(true_scores)),
                "utility_gap_pred_selected_to_true_best": float(true_scores[true_best] - true_scores[pred_best]),
                "true_rank_of_pred_selected": int(true_ranks[pred_best]),
            }
        )

    summary = {
        "candidate_count": k,
        "sets": sets,
        "checkpoint_method": checkpoint_method,
        "random_top1": 1.0 / max(k, 1),
        "top1_agreement": float(np.mean([row["top1_agreement"] for row in per_set])),
        "top3_agreement": float(np.mean([row["top3_agreement"] for row in per_set])),
        "mean_spearman": float(np.mean([row["spearman"] for row in per_set])),
        "pairwise_ranking_accuracy": float(np.mean([row["pairwise_accuracy"] for row in per_set])),
        "predicted_score_std": float(np.mean([row["predicted_score_std"] for row in per_set])),
        "true_utility_std": float(np.mean([row["true_utility_std"] for row in per_set])),
        "avg_utility_gap_pred_selected_to_true_best": float(
            np.mean([row["utility_gap_pred_selected_to_true_best"] for row in per_set])
        ),
        "avg_true_rank_of_pred_selected": float(np.mean([row["true_rank_of_pred_selected"] for row in per_set])),
    }
    write_json(Path(cfg["paths"]["output_dir"]) / "evaluator_imitation.json", summary)
    write_csv(Path(cfg["paths"]["output_dir"]) / "evaluator_imitation.csv", rows)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-method", default=None)
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke), checkpoint_method=args.checkpoint_method)


if __name__ == "__main__":
    main()
