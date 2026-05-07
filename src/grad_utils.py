from typing import Any, Dict, Tuple

import torch

try:
    from .modeling import autocast_context, sft_loss_and_stats
except ImportError:
    from modeling import autocast_context, sft_loss_and_stats


def flatten_trainable_grads(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32, device="cpu"))
        else:
            chunks.append(grad.detach().float().cpu().reshape(-1))
    if not chunks:
        raise RuntimeError("No trainable LoRA gradients found.")
    return torch.cat(chunks)


def compute_lora_gradient(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # Utility labels use gradient alignment over trainable LoRA parameters only.
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    with autocast_context(device):
        loss, stats, _ = sft_loss_and_stats(model, batch, return_logits=False)
    loss.backward()
    grad = flatten_trainable_grads(model)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return grad, stats


def cosine_utility(train_grad: torch.Tensor, val_grad: torch.Tensor) -> float:
    train_norm = train_grad.norm()
    val_norm = val_grad.norm()
    if train_norm.item() == 0.0 or val_norm.item() == 0.0:
        return 0.0
    return float(torch.dot(train_grad, val_grad) / (train_norm * val_norm))
