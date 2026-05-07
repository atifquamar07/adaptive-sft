from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

try:
    from .modeling import autocast_context, sft_loss_and_stats
except ImportError:
    from modeling import autocast_context, sft_loss_and_stats


BASE_FEATURE_NAMES = [
    "mean_sft_loss",
    "std_sft_loss",
    "mean_input_length",
    "max_input_length",
    "mean_response_length",
    "mean_token_entropy",
    "mean_label_confidence",
    "step_frac",
]
NOISE_FEATURE_NAMES = ["mean_is_synthetic_noise"]
GRADIENT_FEATURE_NAMES = ["gradient_norm"]


def feature_names(use_gradient_features: bool = False, use_noise_feature: bool = False) -> List[str]:
    names = list(BASE_FEATURE_NAMES)
    if use_noise_feature:
        names.extend(NOISE_FEATURE_NAMES)
    if use_gradient_features:
        names.extend(GRADIENT_FEATURE_NAMES)
    return names


def vectorize_features(features: Dict[str, float], names: Sequence[str]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in names]


def _entropy_and_confidence(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    if int(mask.sum().item()) == 0:
        return {"mean_token_entropy": 0.0, "mean_label_confidence": 0.0}
    entropy_sum = 0.0
    confidence_sum = 0.0
    token_count = 0
    chunk = 64
    for start in range(0, shift_logits.size(1), chunk):
        end = min(start + chunk, shift_logits.size(1))
        chunk_logits = shift_logits[:, start:end, :].float()
        chunk_labels = shift_labels[:, start:end]
        chunk_mask = mask[:, start:end]
        if int(chunk_mask.sum().item()) == 0:
            continue
        log_probs = F.log_softmax(chunk_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        safe_labels = chunk_labels.masked_fill(~chunk_mask, 0)
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        entropy_sum += float(entropy[chunk_mask].sum().detach().cpu())
        confidence_sum += float(token_log_probs.exp()[chunk_mask].sum().detach().cpu())
        token_count += int(chunk_mask.sum().detach().cpu())
    return {
        "mean_token_entropy": entropy_sum / max(token_count, 1),
        "mean_label_confidence": confidence_sum / max(token_count, 1),
    }


def compute_batch_features(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg: Dict[str, Any],
    current_step: int,
    total_steps: int,
    use_gradient_features: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        with autocast_context(device):
            loss, stats, logits = sft_loss_and_stats(model, batch, return_logits=True)
        per_example = stats["per_example_loss"].float()
        active_lengths = batch["labels"].ne(-100).sum(dim=1).float().detach().cpu()
        input_lengths = batch["attention_mask"].sum(dim=1).float().detach().cpu()
        ent_conf = _entropy_and_confidence(logits.detach(), batch["labels"])
        result = {
            "mean_sft_loss": float(loss.detach().cpu()),
            "std_sft_loss": float(per_example.std(unbiased=False).item()) if per_example.numel() > 1 else 0.0,
            "mean_input_length": float(input_lengths.mean().item()),
            "max_input_length": float(input_lengths.max().item()),
            "mean_response_length": float(active_lengths.mean().item()),
            "step_frac": float(current_step) / max(float(total_steps), 1.0),
            **ent_conf,
        }
        if bool(cfg.get("data", {}).get("feed_noise_feature_to_evaluator", False)):
            noise = batch.get("is_synthetic_noise")
            result["mean_is_synthetic_noise"] = float(noise.float().mean().detach().cpu()) if noise is not None else 0.0
    if use_gradient_features:
        try:
            from .grad_utils import compute_lora_gradient
        except ImportError:
            from grad_utils import compute_lora_gradient

        grad, _ = compute_lora_gradient(model, batch, device, cfg)
        result["gradient_norm"] = float(grad.norm().item())
    if was_training:
        model.train()
    return result
