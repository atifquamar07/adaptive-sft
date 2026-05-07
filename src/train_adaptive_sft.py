import argparse
import random
import time
from pathlib import Path

import torch

try:
    from .data import load_processed_data
    from .evaluator import load_trained_evaluator
    from .features import compute_batch_features, vectorize_features
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
        set_seed,
        train_on_batch,
    )
except ImportError:
    from data import load_processed_data
    from evaluator import load_trained_evaluator
    from features import compute_batch_features, vectorize_features
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
        set_seed,
        train_on_batch,
    )


def _select_index(scores, rng: random.Random, selection: str, temperature: float) -> int:
    if selection == "top1":
        return int(torch.argmax(scores).item())
    if selection != "softmax":
        raise ValueError("selection must be 'softmax' or 'top1'.")
    probs = torch.softmax(scores / max(float(temperature), 1.0e-6), dim=0).cpu().numpy()
    return int(rng.choices(range(len(probs)), weights=probs, k=1)[0])


def _selection_probs(scores: torch.Tensor, selection: str, temperature: float) -> torch.Tensor:
    if selection == "top1":
        probs = torch.zeros_like(scores, dtype=torch.float32)
        probs[int(torch.argmax(scores).item())] = 1.0
        return probs
    return torch.softmax(scores / max(float(temperature), 1.0e-6), dim=0)


def _rank_of(scores: torch.Tensor, idx: int) -> int:
    value = float(scores[idx].item())
    return 1 + int((scores > value).sum().item())


def _score_gap(scores: torch.Tensor) -> float:
    if scores.numel() < 2:
        return 0.0
    top2 = torch.topk(scores, k=2).values
    return float((top2[0] - top2[1]).item())


def _resolved_score_source(cfg, bundle) -> str:
    adaptive_cfg = cfg["adaptive"]
    configured = adaptive_cfg.get("score_source", "auto")
    if configured not in {"auto", "evaluator", "middle-loss"}:
        raise ValueError("adaptive.score_source must be 'auto', 'evaluator', or 'middle-loss'.")
    if configured != "auto":
        return configured
    if not bool(adaptive_cfg.get("fallback_on_weak_evaluator", True)):
        return "evaluator"
    metrics = bundle.get("metrics", {})
    dev_spearman = float(metrics.get("dev_spearman", 0.0))
    dev_rank_acc = float(metrics.get("dev_ranking_accuracy", 0.0))
    min_spearman = float(adaptive_cfg.get("min_dev_spearman", 0.10))
    min_rank_acc = float(adaptive_cfg.get("min_dev_ranking_accuracy", 0.55))
    if dev_spearman < min_spearman or dev_rank_acc < min_rank_acc:
        return adaptive_cfg.get("fallback_score_source", "middle-loss")
    return "evaluator"


def _middle_loss_scores(candidates):
    losses = torch.tensor(
        [float(candidate["features"]["mean_sft_loss"]) for candidate in candidates],
        dtype=torch.float32,
    )
    target = torch.median(losses)
    return -torch.abs(losses - target), float(target.item())


def run(cfg, shuffled_scores: bool = False, selection: str = None):
    method = "adaptive_shuffled_scores" if shuffled_scores else "adaptive_utility_sft"
    ensure_dirs(cfg)
    reset_curve(cfg, method)
    set_seed(int(cfg["seed"]))
    datasets = load_processed_data(cfg)
    model, tokenizer, device = load_lora_model(cfg, adapter_path=cfg["paths"]["warmup_dir"], train=True)
    evaluator, names, mean, std, bundle = load_trained_evaluator(cfg["paths"]["evaluator"])
    optimizer = make_optimizer(cfg, model)
    scaler = make_scaler(device)
    rng = random.Random(int(cfg["seed"]) + (601 if shuffled_scores else 501))
    overhead = init_overhead(method)
    start_time = time.time()
    log_path = Path(cfg["paths"]["logs_dir"]) / f"{method}_selected_batches.jsonl"
    log_path.write_text("", encoding="utf-8")
    k = int(cfg["adaptive"].get("candidate_count", cfg["adaptive"].get("candidate_batches_k", 8)))
    total_steps = int(cfg["training"]["continuation_steps"])
    warmup_steps = int(cfg["training"]["warmup_steps"])
    experiment_total_steps = warmup_steps + total_steps
    selection = selection or cfg["adaptive"].get("selection", "softmax")
    use_grad_features = bool(cfg["adaptive"].get("use_gradient_features", False)) and "gradient_norm" in names
    score_source = _resolved_score_source(cfg, bundle)
    evaluator_metrics = bundle.get("metrics", {})
    if score_source != "evaluator":
        print(
            f"{method} using score_source={score_source} "
            f"(evaluator dev_spearman={float(evaluator_metrics.get('dev_spearman', 0.0)):.4f}, "
            f"dev_ranking_accuracy={float(evaluator_metrics.get('dev_ranking_accuracy', 0.0)):.4f})"
        )
    if bool(cfg["adaptive"].get("use_gradient_features", False)) and not use_grad_features:
        print("adaptive use_gradient_features=true ignored because evaluator was trained without gradient_norm.")

    for step in range(1, total_steps + 1):
        candidates = []
        feature_rows = []
        global_step = warmup_steps + step
        for cand in range(k):
            indices = sample_indices(rng, len(datasets["train_pool"]), int(cfg["training"]["micro_batch_size"]))
            batch = batch_from_indices(datasets["train_pool"], indices, tokenizer, device)
            feats = compute_batch_features(
                model,
                batch,
                device,
                cfg,
                current_step=global_step,
                total_steps=experiment_total_steps,
                use_gradient_features=use_grad_features,
            )
            candidates.append({"indices": indices, "batch": batch, "features": feats})
            feature_rows.append(vectorize_features(feats, names))
            overhead["candidate_batches_scored"] += 1
            if use_grad_features:
                overhead["gradient_calls"] += 1
        x = torch.tensor(feature_rows, dtype=torch.float32)
        with torch.no_grad():
            evaluator_scores = evaluator((x - mean) / std)
        middle_loss_target = None
        if score_source == "evaluator":
            scores = evaluator_scores
        elif score_source == "middle-loss":
            scores, middle_loss_target = _middle_loss_scores(candidates)
        else:
            raise ValueError(f"Unsupported adaptive score source: {score_source}")
        raw_scores = scores.clone()
        raw_score_mean = float(raw_scores.mean().item())
        raw_score_std = float(raw_scores.std(unbiased=False).item()) if raw_scores.numel() > 1 else 0.0
        if bool(cfg["adaptive"].get("standardize_candidate_scores", False)):
            scores = (raw_scores - raw_scores.mean()) / (raw_scores.std(unbiased=False) + 1.0e-8)
        standardized_score_std = float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0
        selection_scores = scores.clone()
        if shuffled_scores:
            perm = torch.randperm(selection_scores.numel())
            selection_scores = selection_scores[perm]
        probs = _selection_probs(selection_scores, selection, float(cfg["adaptive"]["temperature"]))
        selected_idx = _select_index(selection_scores, rng, selection, float(cfg["adaptive"]["temperature"]))
        selected = candidates[selected_idx]
        selected_score = float(scores[selected_idx].item())
        base_scores = scores.detach().cpu()
        raw_scores_cpu = raw_scores.detach().cpu()
        evaluator_scores_cpu = evaluator_scores.detach().cpu()
        score_std = float(base_scores.std(unbiased=False).item()) if base_scores.numel() > 1 else 0.0
        selected_true_rank = _rank_of(base_scores, selected_idx)
        selected_predicted_rank = _rank_of(raw_scores_cpu, selected_idx)
        selected_eval_score = float(evaluator_scores_cpu[selected_idx].item())
        selected_eval_rank = _rank_of(evaluator_scores_cpu, selected_idx)
        selected_probability = float(probs[selected_idx].detach().cpu().item())
        max_probability = float(probs.max().detach().cpu().item())
        selection_entropy = float((-(probs * probs.clamp_min(1.0e-12).log()).sum()).detach().cpu().item())
        stats = train_on_batch(model, optimizer, scaler, selected["batch"], cfg, device)
        overhead["optimizer_steps"] += 1
        record = {
            "method": method,
            "step": step,
            "global_step": int(global_step),
            "candidate_count": int(k),
            "selected_candidate": selected_idx,
            "predicted_utility": selected_eval_score,
            "selection_score": float(selection_scores[selected_idx].item()),
            "raw_predicted_score_mean": raw_score_mean,
            "raw_predicted_score_std": raw_score_std,
            "standardized_score_std": standardized_score_std,
            "selection_entropy": selection_entropy,
            "selected_probability": selected_probability,
            "max_probability": max_probability,
            "score_gap_best_second": _score_gap(base_scores),
            "candidate_score_mean": float(base_scores.mean().item()),
            "candidate_score_std": score_std,
            "candidate_score_min": float(base_scores.min().item()),
            "candidate_score_max": float(base_scores.max().item()),
            "evaluator_score": selected_eval_score,
            "selected_evaluator_rank": int(selected_eval_rank),
            "selected_predicted_rank": int(selected_predicted_rank),
            "selected_true_rank": int(selected_true_rank),
            "score_source": score_source,
            "configured_score_source": cfg["adaptive"].get("score_source", "auto"),
            "middle_loss_target": middle_loss_target,
            "evaluator_dev_spearman": float(evaluator_metrics.get("dev_spearman", 0.0)),
            "evaluator_dev_ranking_accuracy": float(evaluator_metrics.get("dev_ranking_accuracy", 0.0)),
            "train_loss": float(stats["loss"]),
            "selection": selection,
            "temperature": float(cfg["adaptive"]["temperature"]),
            "shuffled_scores": bool(shuffled_scores),
            **{f"selected_{key}": float(value) for key, value in selected["features"].items()},
        }
        append_jsonl(log_path, record)
        if step % int(cfg["training"].get("log_every", 20)) == 0 or step == 1:
            print(f"{method} step={step} train_loss={stats['loss']:.4f} pred={record['predicted_utility']:.4f}")
        maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, step)

    maybe_eval_curve(model, datasets["utility_val"], tokenizer, cfg, method, total_steps, force=True)
    overhead["wall_clock_seconds"] = time.time() - start_time
    save_checkpoint(model, tokenizer, method_checkpoint_dir(cfg, method), cfg, {"method": method, "selection": selection})
    save_overhead(cfg, method, overhead)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--selection", choices=["softmax", "top1"], default=None)
    parser.add_argument("--shuffled-scores", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config, smoke=args.smoke), shuffled_scores=args.shuffled_scores, selection=args.selection)


if __name__ == "__main__":
    main()
