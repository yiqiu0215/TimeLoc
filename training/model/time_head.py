"""Time distribution head components (ported from DisTime, made batch-safe).

These implement DisTime's continuous time modeling:
- TimeDec: MLP that decodes per-<TIME_STAMP> hidden states into start/end
  distribution logits over reg_max + 1 bins.
- Project: fixed projection that turns a distribution into its expectation (bin).
- distribution_focal_loss: DFL over continuous [0, reg_max] targets.
- diou_loss_1d: 1D Distance-IoU loss on [start, end] spans.

Unlike the original DisTime code (which hard-codes batch index 0), every
function here operates on flattened [N_stamp, ...] tensors so multiple samples
and multiple <TIME_STAMP> per sample are handled uniformly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDec(nn.Module):
    """MLP decoding hidden states at <TIME_STAMP> positions to time logits."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, feats):
        # feats: [N_stamp, input_dim] already gathered at <TIME_STAMP> positions
        x = feats
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TimeEnc(nn.Module):
    """MLP encoding a (start/end) Gaussian distribution into an LLM-hidden-size
    time embedding (ported from DisTime, batch-safe).

    Shared by the input side (frame sampling times) and the output side
    (``<TIME_STAMP>`` GT teacher-forcing). Input is always ``2 * (reg_max + 1)``
    so a single module serves both: a frame time is encoded as a degenerate
    interval ``[t, t]`` (its Gaussian duplicated), a GT span as
    ``gaussian(start) ⊕ gaussian(end)``.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x: [N, 2 * (reg_max + 1)] -> [N, hidden]
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def generate_gaussian_peaks(length, peaks, sigma):
    """Soft one-hot Gaussian distributions centered at ``peaks``.

    Args:
        length: number of bins (``reg_max + 1``).
        peaks: [N] continuous bin centers in ``[0, length - 1]``.
        sigma: Gaussian std (in bin units).

    Returns:
        [N, length] float32 distributions, each row sums to 1.
    """
    peaks = peaks.reshape(-1).float()
    x = torch.arange(length, device=peaks.device, dtype=torch.float32)  # [L]
    g = torch.exp(-0.5 * ((x[None, :] - peaks[:, None]) / float(sigma)) ** 2)  # [N, L]
    g = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return g


def _seconds_to_bins(t, duration, reg_max, max_offset=0.0):
    """Clamp seconds to ``[0, duration - max_offset]`` then map to ``[0, reg_max]``."""
    duration = max(float(duration), 1.0)
    t = t.float().clamp(min=0.0)
    t = torch.minimum(t, t.new_full(t.shape, duration - max_offset))
    return t / duration * reg_max


def encode_frame_times(time_enc, frame_times, duration, reg_max, sigma):
    """Encode per-frame sampling times into ``[T, hidden]`` time embeddings.

    Each frame time ``t`` is treated as a degenerate interval ``[t, t]``: its
    Gaussian over ``reg_max + 1`` bins is duplicated to width ``2 * (reg_max + 1)``
    so it matches the shared ``TimeEnc`` input.
    """
    dtype = next(time_enc.parameters()).dtype
    t_bin = _seconds_to_bins(frame_times, duration, reg_max, max_offset=0.0)  # [T]
    g = generate_gaussian_peaks(reg_max + 1, t_bin, sigma)  # [T, reg_max + 1]
    g = torch.cat([g, g], dim=-1).to(dtype)  # [T, 2 * (reg_max + 1)]
    return time_enc(g)


def encode_spans(time_enc, spans, duration, reg_max, sigma):
    """Encode ``[start, end]`` GT spans into ``[N, hidden]`` time embeddings.

    ``gaussian(start) ⊕ gaussian(end)`` -> width ``2 * (reg_max + 1)``. Clamping to
    ``duration - 1`` matches the DFL/decode mapping in the time head (self-consistent).
    """
    dtype = next(time_enc.parameters()).dtype
    spans = spans.reshape(-1, 2)
    s_bin = _seconds_to_bins(spans[:, 0], duration, reg_max, max_offset=1.0)  # [N]
    e_bin = _seconds_to_bins(spans[:, 1], duration, reg_max, max_offset=1.0)  # [N]
    gs = generate_gaussian_peaks(reg_max + 1, s_bin, sigma)  # [N, reg_max + 1]
    ge = generate_gaussian_peaks(reg_max + 1, e_bin, sigma)  # [N, reg_max + 1]
    g = torch.cat([gs, ge], dim=-1).to(dtype)  # [N, 2 * (reg_max + 1)]
    return time_enc(g)


class Project(nn.Module):
    """Fixed projection turning a (start/end) distribution into its expectation.

    Input logits are reshaped to [-1, reg_max + 1], softmaxed, then dotted with
    [0, 1, ..., reg_max] to obtain the expected bin. Output is [N_stamp, 2]
    (start_bin, end_bin).
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        # x: [N_stamp, 2 * (reg_max + 1)] -> [2 * N_stamp, reg_max + 1]
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        # Expectation over bins. NOTE: use elementwise mul + sum instead of
        # F.linear: under DeepSpeed ZeRO-3, torch.nn.functional.linear is
        # globally monkey-patched with a Function that assumes a 2D [out, in]
        # weight, which breaks the backward pass for our 1D [reg_max + 1]
        # projection buffer (grad_output.matmul(weight) shape mismatch).
        proj = self.project.to(x.dtype)  # [reg_max + 1]
        x = (x * proj).sum(dim=1).reshape(-1, 2)
        return x


def _reduce_loss(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def _weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        return _reduce_loss(loss, reduction)
    if reduction == "mean":
        return loss.sum() / avg_factor
    if reduction == "none":
        return loss
    raise ValueError('avg_factor can not be used with reduction="sum"')


def distribution_focal_loss(pred, label, weight=None, reduction="mean", avg_factor=None):
    """Distribution Focal Loss over continuous targets in [0, reg_max].

    Args:
        pred: [M, reg_max + 1] logits (M = 2 * N_stamp, start/end interleaved).
        label: [M] continuous bin targets in [0, reg_max].
    """
    disl = label.long()
    disr = disl + 1

    wl = disr.float() - label
    wr = label - disl.float()

    loss = F.cross_entropy(pred, disl, reduction="none") * wl + F.cross_entropy(
        pred, disr, reduction="none"
    ) * wr
    return _weight_reduce_loss(loss, weight, reduction, avg_factor)


def diou_loss_1d(input_spans, target_spans, reduction="none", eps=1e-6):
    """1D Distance-IoU loss on [start, end] spans (same scale on both inputs).

    Args:
        input_spans: [N, 2] predicted (start, end).
        target_spans: [N, 2] ground-truth (start, end).
    """
    input_spans = input_spans.float()
    target_spans = target_spans.float()

    lp, rp = input_spans[:, 0], input_spans[:, 1]
    lg, rg = target_spans[:, 0], target_spans[:, 1]

    lkis = torch.maximum(lp, lg)
    rkis = torch.minimum(rp, rg)

    zero = torch.zeros(1, device=input_spans.device, dtype=input_spans.dtype)
    overlap = torch.maximum(zero, rkis - lkis)
    iouk = overlap / ((rp - lp) + (rg - lg) - overlap).clamp(min=eps)

    # smallest enclosing interval
    lc = torch.minimum(lp, lg)
    rc = torch.maximum(rp, rg)
    len_c = rc - lc

    rho_p = 0.5 * (rp + lp)
    rho_g = 0.5 * (rg + lg)
    rho = rho_p - rho_g

    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        return loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    if reduction == "sum":
        return loss.sum()
    return loss
