import torch
import torch.nn.functional as F


def smooth_l1_boundary_loss(
    predicted_span: torch.Tensor,
    target_span: torch.Tensor,
) -> torch.Tensor:
    predicted_span = predicted_span.float()
    target_span = target_span.float()
    if predicted_span.shape != target_span.shape or predicted_span.shape[-1] != 2:
        raise ValueError("Boundary spans must have the same shape [B, 2].")
    return F.smooth_l1_loss(predicted_span, target_span, reduction="none").sum(dim=-1).mean()


def diou_loss_1d(
    predicted_span: torch.Tensor,
    target_span: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    predicted_span = predicted_span.float()
    target_span = target_span.float()
    if predicted_span.shape != target_span.shape or predicted_span.shape[-1] != 2:
        raise ValueError("Boundary spans must have the same shape [B, 2].")

    predicted_start, predicted_end = predicted_span.unbind(dim=-1)
    target_start, target_end = target_span.unbind(dim=-1)
    intersection = torch.clamp(
        torch.minimum(predicted_end, target_end)
        - torch.maximum(predicted_start, target_start),
        min=0.0,
    )
    union = (
        torch.maximum(predicted_end, target_end)
        - torch.minimum(predicted_start, target_start)
        + float(eps)
    )
    iou = intersection / union
    predicted_center = 0.5 * (predicted_start + predicted_end)
    target_center = 0.5 * (target_start + target_end)
    enclosing_length = torch.maximum(predicted_end, target_end) - torch.minimum(
        predicted_start, target_start
    )
    loss = 1.0 - iou + (predicted_center - target_center).square() / (
        enclosing_length.square() + float(eps)
    )
    return loss.mean()
