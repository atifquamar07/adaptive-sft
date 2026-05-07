import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .features import feature_names, vectorize_features
    from .modeling import ensure_dirs, load_config, set_seed, write_json
except ImportError:
    from features import feature_names, vectorize_features
    from modeling import ensure_dirs, load_config, set_seed, write_json


class UtilityEvaluator(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _split_records(records: List[Dict[str, Any]], dev_fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    dev_size = max(1, int(round(len(shuffled) * dev_fraction)))
    dev = shuffled[:dev_size]
    train = shuffled[dev_size:]
    if not train:
        train, dev = shuffled, shuffled
    return train, dev


def _matrix(records: Sequence[Dict[str, Any]], names: Sequence[str], target_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x = [vectorize_features(record["features"], names) for record in records]
    y = [float(record[target_key]) for record in records]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _normalizer(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return mean, std


def _pairwise_loss(scores: torch.Tensor, targets: torch.Tensor, max_pairs: int) -> torch.Tensor:
    diffs = targets[:, None] - targets[None, :]
    mask = diffs.abs() > 1.0e-8
    pairs = mask.nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return F.mse_loss(scores, targets)
    if pairs.size(0) > max_pairs:
        perm = torch.randperm(pairs.size(0), device=pairs.device)[:max_pairs]
        pairs = pairs[perm]
    i, j = pairs[:, 0], pairs[:, 1]
    sign = torch.sign(targets[i] - targets[j])
    return F.softplus(-sign * (scores[i] - scores[j])).mean()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    rt = _rankdata(y_true)
    rp = _rankdata(y_pred)
    if np.std(rt) == 0 or np.std(rp) == 0:
        return 0.0
    return float(np.corrcoef(rt, rp)[0, 1])


def _ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray, seed: int, max_pairs: int = 20000) -> float:
    n = len(y_true)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    checks = min(max_pairs, total_pairs)
    correct = 0
    valid = 0
    for _ in range(checks):
        i, j = rng.choice(n, size=2, replace=False)
        true_diff = y_true[i] - y_true[j]
        if abs(true_diff) <= 1.0e-8:
            continue
        pred_diff = y_pred[i] - y_pred[j]
        correct += int(np.sign(true_diff) == np.sign(pred_diff))
        valid += 1
    return float(correct / max(valid, 1))


def _with_training_targets(records: Sequence[Dict[str, Any]], target_mode: str) -> List[Dict[str, Any]]:
    records = [dict(record) for record in records]
    if target_mode in {"utility", "raw_utility"}:
        for record in records:
            record["target_utility"] = float(record["utility"])
        return records
    if target_mode not in {"step_zscore_utility", "state_zscore_utility"}:
        raise ValueError("evaluator.target must be 'step_zscore_utility' or 'raw_utility'.")

    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_step.setdefault(int(record.get("step", 0)), []).append(record)
    for step_records in by_step.values():
        values = np.array([float(record["utility"]) for record in step_records], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std())
        if std < 1.0e-8:
            for record in step_records:
                record["target_utility"] = 0.0
        else:
            for record in step_records:
                record["target_utility"] = (float(record["utility"]) - mean) / std
    return records


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, seed: int) -> Dict[str, float]:
    return {
        "mse": float(np.mean((y_pred - y_true) ** 2)),
        "spearman": _spearman(y_true, y_pred),
        "ranking_accuracy": _ranking_accuracy(y_true, y_pred, seed),
    }


def _feature_correlations(records: Sequence[Dict[str, Any]], names: Sequence[str], target_key: str) -> Dict[str, float]:
    y = np.array([float(record[target_key]) for record in records], dtype=np.float64)
    result = {}
    if len(y) < 2 or float(np.std(y)) == 0.0:
        return {name: 0.0 for name in names}
    for name in names:
        x = np.array([float(record["features"].get(name, 0.0)) for record in records], dtype=np.float64)
        if float(np.std(x)) == 0.0:
            result[name] = 0.0
        else:
            result[name] = float(np.corrcoef(x, y)[0, 1])
    return result


def train_evaluator(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))
    records = load_records(cfg["paths"]["utility_labels"])
    if not records:
        raise RuntimeError("No utility-label records found.")
    use_grad = bool(cfg["utility"].get("use_gradient_features", False))
    names = records[0].get("feature_names") or feature_names(
        use_grad,
        bool(cfg.get("data", {}).get("feed_noise_feature_to_evaluator", False)),
    )
    target_mode = cfg["evaluator"].get("target", "step_zscore_utility")
    records = _with_training_targets(records, target_mode)
    train_records, dev_records = _split_records(records, float(cfg["evaluator"]["dev_fraction"]), int(cfg["seed"]))
    x_train, y_train = _matrix(train_records, names, "target_utility")
    x_dev, y_dev = _matrix(dev_records, names, "target_utility")
    mean, std = _normalizer(x_train)
    x_train_n = (x_train - mean) / std
    x_dev_n = (x_dev - mean) / std

    model = UtilityEvaluator(len(names), int(cfg["evaluator"]["hidden_size"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["evaluator"]["lr"]))
    batch_size = int(cfg["evaluator"]["batch_size"])
    max_pairs = int(cfg["evaluator"]["max_pairs_per_batch"])
    use_pairwise = cfg["evaluator"].get("loss", "pairwise") == "pairwise"

    for epoch in range(int(cfg["evaluator"]["epochs"])):
        perm = torch.randperm(x_train_n.size(0))
        for start in range(0, x_train_n.size(0), batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train_n[idx]
            yb = y_train[idx]
            scores = model(xb)
            loss = _pairwise_loss(scores, yb, max_pairs) if use_pairwise else F.mse_loss(scores, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_train = model(x_train_n).cpu().numpy()
        pred_dev = model(x_dev_n).cpu().numpy()
    true_dev = y_dev.cpu().numpy()
    true_train = y_train.cpu().numpy()
    raw_dev = np.array([float(record["utility"]) for record in dev_records], dtype=np.float64)
    target_train_metrics = _regression_metrics(true_train, pred_train, int(cfg["seed"]))
    target_dev_metrics = _regression_metrics(true_dev, pred_dev, int(cfg["seed"]))
    raw_dev_metrics = _regression_metrics(raw_dev, pred_dev, int(cfg["seed"]))
    metrics = {
        "target": target_mode,
        "dev_mse": target_dev_metrics["mse"],
        "dev_spearman": target_dev_metrics["spearman"],
        "dev_ranking_accuracy": target_dev_metrics["ranking_accuracy"],
        "dev_raw_mse": raw_dev_metrics["mse"],
        "dev_raw_spearman": raw_dev_metrics["spearman"],
        "dev_raw_ranking_accuracy": raw_dev_metrics["ranking_accuracy"],
        "train_mse": target_train_metrics["mse"],
        "train_spearman": target_train_metrics["spearman"],
        "train_ranking_accuracy": target_train_metrics["ranking_accuracy"],
        "feature_target_correlations": _feature_correlations(records, names, "target_utility"),
        "feature_raw_utility_correlations": _feature_correlations(records, names, "utility"),
        "num_train_records": len(train_records),
        "num_dev_records": len(dev_records),
        "feature_names": list(names),
    }

    bundle = {
        "state_dict": model.state_dict(),
        "feature_names": list(names),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "hidden_size": int(cfg["evaluator"]["hidden_size"]),
        "metrics": metrics,
    }
    Path(cfg["paths"]["evaluator"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cfg["paths"]["evaluator"])
    write_json(cfg["paths"]["evaluator_metrics"], metrics)

    with open(cfg["paths"]["evaluator_dev_predictions"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "raw_utility", "true_utility", "predicted_utility"],
        )
        writer.writeheader()
        for record, truth, pred in zip(dev_records, true_dev.tolist(), pred_dev.tolist()):
            writer.writerow(
                {
                    "step": int(record.get("step", 0)),
                    "raw_utility": float(record["utility"]),
                    "true_utility": truth,
                    "predicted_utility": pred,
                }
            )
    print(metrics)
    return metrics


def load_trained_evaluator(path: str):
    bundle = torch.load(path, map_location="cpu")
    model = UtilityEvaluator(len(bundle["feature_names"]), int(bundle["hidden_size"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    mean = torch.tensor(bundle["feature_mean"], dtype=torch.float32)
    std = torch.tensor(bundle["feature_std"], dtype=torch.float32)
    return model, bundle["feature_names"], mean, std, bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    train_evaluator(cfg)


if __name__ == "__main__":
    main()
