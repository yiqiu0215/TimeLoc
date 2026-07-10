"""Attach DisTime-style continuous time distribution head to a TimeLens model.

Design note
-----------
TimeLens loads the base model via ``AutoModelForImageTextToText`` and trains it
with the standard HF ``Trainer`` under DeepSpeed ZeRO-3. Wrapping the model in a
plain ``nn.Module`` would break ZeRO-3 parameter consolidation and
``save_pretrained`` / ``from_pretrained``. So instead of a wrapper object, we
*attach* the head submodules (``time_dec``, ``time_project``) to the existing
``PreTrainedModel`` instance and replace its bound ``forward`` in place. The
model stays a ``PreTrainedModel``: head weights are saved alongside the LLM in
the same checkpoint with ``time_dec.*`` / ``time_project.*`` keys, and ZeRO-3 /
gradient checkpointing keep working unchanged.

Only the output side is modified (Stage 1): the base LM loss is computed exactly
as before, then DFL + 1D DIoU losses from the ``<TIME_STAMP>`` positions are
added. All computation is batch-safe (no hard-coded batch index 0).
"""

import types
import warnings

import torch

from training.model.time_head import (
    Project,
    TimeDec,
    TimeEnc,
    diou_loss_1d,
    distribution_focal_loss,
    encode_frame_times,
    encode_spans,
)


def attach_time_dist_head(
    model,
    time_token_id,
    reg_max=32,
    num_layers=3,
    lambda_dfl=1.0,
    lambda_iou=1.0,
    patch_forward=True,
    has_time_enc=False,
    frame_time_token_id=None,
    time_enc_sigma=1.0,
    time_enc_num_layers=3,
    time_enc_input=True,
    time_enc_teacher_forcing=True,
):
    """Attach a DisTime time head to ``model`` and patch its forward in place.

    Args:
        model: a loaded ``PreTrainedModel`` (Qwen2.5-VL / TimeLens).
        time_token_id: token id of the newly added ``<TIME_STAMP>`` special token.
        reg_max: number of distribution bins is ``reg_max + 1``.
        num_layers: depth of the ``TimeDec`` MLP.
        lambda_dfl / lambda_iou: auxiliary loss weights.
        patch_forward: if True (training), replace ``model.forward`` to add the
            time loss. Set False for inference/eval: ``generate`` introspects
            ``inspect.signature(model.forward)`` to decide which kwargs (e.g.
            ``position_ids``) to build, and a patched ``(*args, **kwargs)``
            signature breaks that, yielding ``position_ids=None``. Decoding uses
            ``decode_time_spans`` directly, so the patch is unnecessary at eval.
        has_time_enc: if True, also attach the shared ``TimeEnc`` module and
            register an embedding forward-hook that overwrites ``<FRAME_TIME>``
            (and, during training, ``<TIME_STAMP>``) rows with continuous time
            embeddings. The hook is registered regardless of ``patch_forward`` so
            that ``generate`` / ``decode_time_spans`` also inject frame times.
        frame_time_token_id: token id of ``<FRAME_TIME>`` (required if has_time_enc).
        time_enc_sigma: Gaussian std (in bin units) for the soft time targets.
        time_enc_num_layers: depth of the ``TimeEnc`` MLP.
        time_enc_input: if False, skip input frame-time injection (ablation).
        time_enc_teacher_forcing: if False, skip ``<TIME_STAMP>`` GT injection (ablation).

    Returns:
        The same ``model`` instance, now carrying the head (and patched forward
        when ``patch_forward`` is True).
    """
    hidden_size = model.config.hidden_size

    time_dec = TimeDec(
        input_dim=hidden_size,
        hidden_dim=hidden_size,
        output_dim=2 * (reg_max + 1),
        num_layers=num_layers,
    )
    time_dec.init_weights()
    time_project = Project(reg_max)

    # Match the base model dtype/device so DeepSpeed sees a homogeneous module.
    param = next(model.parameters())
    time_dec = time_dec.to(dtype=param.dtype, device=param.device)
    time_project = time_project.to(device=param.device)

    model.add_module("time_dec", time_dec)
    model.add_module("time_project", time_project)

    # Stash config on the instance for the patched forward to read.
    model._time_token_id = int(time_token_id)
    model._time_reg_max = int(reg_max)
    model._time_lambda_dfl = float(lambda_dfl)
    model._time_lambda_iou = float(lambda_iou)

    # TimeEnc (input-side frame-time encoding + output-side GT teacher-forcing).
    model._has_time_enc = bool(has_time_enc)
    model._time_enc_sigma = float(time_enc_sigma)
    model._time_enc_input = bool(time_enc_input)
    model._time_enc_teacher_forcing = bool(time_enc_teacher_forcing)
    model._frame_time_token_id = (
        int(frame_time_token_id) if frame_time_token_id is not None else None
    )
    # Per-batch state the embedding hook reads (set by the patched forward in
    # training and by decode_time_spans / generate stashing at eval).
    model._tenc_frame_times = None
    model._tenc_time_gt = None
    model._tenc_duration = None

    if has_time_enc:
        if frame_time_token_id is None:
            raise ValueError("has_time_enc=True requires frame_time_token_id.")
        time_enc = TimeEnc(
            input_dim=2 * (reg_max + 1),
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=time_enc_num_layers,
        )
        time_enc.init_weights()
        time_enc = time_enc.to(dtype=param.dtype, device=param.device)
        model.add_module("time_enc", time_enc)
        # Hook fires after the text embedding layer produces inputs_embeds, before
        # the base forward scatters vision features into <|video_pad|> positions.
        model.get_input_embeddings().register_forward_hook(_make_time_enc_hook(model))

    # Keep a handle to the original bound forward.
    model._orig_forward = model.forward

    if patch_forward:
        model.forward = types.MethodType(_time_dist_forward, model)
    return model


def _make_time_enc_hook(model):
    """Build a forward-hook (closure over ``model``) for ``get_input_embeddings()``."""

    def hook(module, inputs, output):
        return _inject_time_enc(model, inputs[0] if inputs else None, output)

    return hook


def _inject_time_enc(model, input_ids, embeds):
    """Overwrite <FRAME_TIME> / <TIME_STAMP> embedding rows with TimeEnc outputs.

    Runs inside the embedding forward-hook, so it sees the freshly embedded
    ``inputs_embeds`` [B, L, H]. Frame-time rows are always injected (prefill);
    <TIME_STAMP> rows are injected only when a GT is stashed (training
    teacher-forcing). Incremental decode steps carry neither token -> no-op.
    """
    if not getattr(model, "_has_time_enc", False) or input_ids is None:
        return embeds

    reg_max = model._time_reg_max
    sigma = model._time_enc_sigma
    frame_tid = model._frame_time_token_id
    stamp_tid = model._time_token_id
    frame_times = model._tenc_frame_times
    time_gt = model._tenc_time_gt
    duration = model._tenc_duration

    out = embeds
    cloned = False
    B = input_ids.size(0)

    def _dur(b):
        if duration is None:
            return 1.0
        return float(duration[b]) if hasattr(duration, "__getitem__") else float(duration)

    # (a) Frame-time injection (only present at prefill).
    if model._time_enc_input and frame_times is not None:
        for b in range(B):
            ft = frame_times[b] if b < len(frame_times) else None
            if ft is None:
                continue
            pos = (input_ids[b] == frame_tid).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            ft_t = torch.as_tensor(ft, device=embeds.device, dtype=torch.float32)
            if pos.numel() != ft_t.numel():
                warnings.warn(
                    f"<FRAME_TIME> count ({pos.numel()}) != frame_times "
                    f"({ft_t.numel()}) for sample {b}; using first "
                    f"{min(pos.numel(), ft_t.numel())}."
                )
                n = min(pos.numel(), ft_t.numel())
                pos, ft_t = pos[:n], ft_t[:n]
            emb = encode_frame_times(model.time_enc, ft_t, _dur(b), reg_max, sigma)
            if not cloned:
                out = out.clone()
                cloned = True
            out[b, pos] = emb.to(out.dtype)

    # (b) <TIME_STAMP> teacher-forcing (training only: GT stashed).
    if model._time_enc_teacher_forcing and time_gt is not None:
        for b in range(B):
            gt = time_gt[b] if b < len(time_gt) else None
            if gt is None or len(gt) == 0:
                continue
            pos = (input_ids[b] == stamp_tid).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            gt_t = torch.as_tensor(
                gt, device=embeds.device, dtype=torch.float32
            ).reshape(-1, 2)
            n = min(pos.numel(), gt_t.size(0))
            emb = encode_spans(model.time_enc, gt_t[:n], _dur(b), reg_max, sigma)
            if not cloned:
                out = out.clone()
                cloned = True
            out[b, pos[:n]] = emb.to(out.dtype)

    return out


def _time_dist_forward(self, *args, **kwargs):
    time_gt = kwargs.pop("time_gt", None)
    duration = kwargs.pop("duration", None)
    frame_times = kwargs.pop("frame_times", None)

    # input_ids / labels are always passed as kwargs by the HF Trainer collator.
    input_ids = kwargs.get("input_ids", None)
    labels = kwargs.get("labels", None)

    kwargs["output_hidden_states"] = True
    kwargs["return_dict"] = True

    # Stash per-batch time info for the embedding hook (frame-time injection +
    # <TIME_STAMP> GT teacher-forcing). Cleared right after the forward.
    if getattr(self, "_has_time_enc", False):
        self._tenc_frame_times = frame_times
        self._tenc_time_gt = time_gt
        self._tenc_duration = duration

    try:
        outputs = self._orig_forward(*args, **kwargs)
    finally:
        if getattr(self, "_has_time_enc", False):
            self._tenc_frame_times = None
            self._tenc_time_gt = None
            self._tenc_duration = None

    # If we cannot supervise the head (no labels / no time targets / no input_ids),
    # return the base outputs untouched.
    if (
        time_gt is None
        or duration is None
        or input_ids is None
        or labels is None
        or outputs.hidden_states is None
    ):
        return outputs

    lm_loss = outputs.loss
    hidden = outputs.hidden_states[-1]  # [B, L, H]

    reg_max = self._time_reg_max
    time_token_id = self._time_token_id

    # Predict-<TIME_STAMP>-from-previous-position mask, aligned with autoregressive
    # generation: position t is selected when token t+1 == <TIME_STAMP>.
    time_mask = input_ids[:, 1:] == time_token_id  # [B, L-1]
    time_mask = torch.cat(
        [time_mask, torch.zeros_like(time_mask[:, :1])], dim=1
    )  # [B, L]

    B = input_ids.size(0)
    feats = []
    target_spans = []
    dur_per_stamp = []
    for b in range(B):
        positions = time_mask[b].nonzero(as_tuple=True)[0]
        gt_b = time_gt[b] if time_gt[b] is not None else []
        n = min(len(positions), len(gt_b))
        if len(positions) != len(gt_b):
            warnings.warn(
                f"<TIME_STAMP> count ({len(positions)}) != gt span count "
                f"({len(gt_b)}) for sample {b}; supervising first {n}."
            )
        for k in range(n):
            feats.append(hidden[b, positions[k]])
            target_spans.append(gt_b[k])
            dur_per_stamp.append(duration[b])

    if len(feats) == 0:
        # Keep the head in the graph so DDP/ZeRO see its grads as used.
        dummy = hidden.new_zeros(1, hidden.size(-1))
        pred = self.time_dec(dummy)
        loss_dfl = (pred * 0).sum()
        loss_iou = (pred * 0).sum()
        total = lm_loss + self._time_lambda_dfl * loss_dfl + self._time_lambda_iou * loss_iou
        outputs.loss = total
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            warnings.warn(
                "[time-head] no <TIME_STAMP> in this batch (n_stamp=0): the time "
                "head is NOT being supervised. Check that the data pipeline uses "
                "the <TIME_STAMP> answer template."
            )
        return outputs

    feats = torch.stack(feats, dim=0)  # [N_stamp, H]
    target_spans = torch.tensor(
        target_spans, dtype=torch.float32, device=feats.device
    )  # [N_stamp, 2]
    dur = torch.stack(
        [d.to(feats.device) for d in dur_per_stamp]
    ).float()  # [N_stamp]
    dur = dur.clamp(min=1e-6)

    # DFL targets use the inclusive [0, duration] coordinate range.
    gt_clamp = torch.clamp(target_spans, min=0.0)
    gt_clamp = torch.minimum(gt_clamp, dur.unsqueeze(1))  # [N_stamp, 2]
    gt_bin = gt_clamp / dur.unsqueeze(1) * reg_max  # [N_stamp, 2]

    logits = self.time_dec(feats).float()  # [N_stamp, 2*(reg_max+1)]
    loss_dfl = distribution_focal_loss(
        logits.reshape(-1, reg_max + 1),
        gt_bin.reshape(-1),
    )

    pred_bin = self.time_project(logits)  # [N_stamp, 2]
    pred_sec = pred_bin / reg_max * dur.unsqueeze(1)
    loss_iou = diou_loss_1d(pred_sec, gt_clamp, reduction="none").mean()

    total = lm_loss + self._time_lambda_dfl * loss_dfl + self._time_lambda_iou * loss_iou
    outputs.loss = total

    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(
    #         f"[time-head] n_stamp={feats.shape[0]} "
    #         f"lm={lm_loss.item():.4f} dfl={loss_dfl.item():.4f} "
    #         f"iou={loss_iou.item():.4f} total={total.item():.4f}",
    #         flush=True,
    #     )
    return outputs


@torch.no_grad()
def decode_time_spans(
    model,
    input_ids,
    attention_mask,
    duration,
    time_token_id,
    frame_times=None,
    **vision_kwargs,
):
    """Decode [start, end] seconds at every <TIME_STAMP> position (inference).

    Runs a single forward pass with ``output_hidden_states=True`` over the given
    sequence (e.g. prompt + generated answer), gathers the hidden state at the
    position preceding each ``<TIME_STAMP>`` token, and projects the time head's
    distribution to seconds.

    Args:
        model: a model with ``time_dec`` / ``time_project`` attached.
        input_ids: [1, L] full sequence (prompt + generated answer).
        attention_mask: [1, L].
        duration: scalar seconds for this sample.
        time_token_id: id of ``<TIME_STAMP>``.
        frame_times: optional [T] frame sampling times (seconds). When TimeEnc is
            attached, these are stashed so the embedding hook injects frame-time
            embeddings during this forward (consistent with training). ``time_gt``
            is intentionally NOT stashed -> no GT leakage at inference.

    Returns:
        List[[start_sec, end_sec]] in generation order (possibly empty).
    """
    reg_max = model._time_reg_max
    if getattr(model, "_has_time_enc", False):
        model._tenc_frame_times = [frame_times] if frame_times is not None else None
        model._tenc_time_gt = None
        model._tenc_duration = [float(duration)]
    forward = getattr(model, "_orig_forward", model.forward)
    try:
        outputs = forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **vision_kwargs,
        )
    finally:
        if getattr(model, "_has_time_enc", False):
            model._tenc_frame_times = None
            model._tenc_time_gt = None
            model._tenc_duration = None
    hidden = outputs.hidden_states[-1]  # [1, L, H]

    time_mask = input_ids[:, 1:] == time_token_id
    time_mask = torch.cat([time_mask, torch.zeros_like(time_mask[:, :1])], dim=1)
    positions = time_mask[0].nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return []

    feats = hidden[0, positions]  # [N_stamp, H]
    logits = model.time_dec(feats).float()
    pred_bin = model.time_project(logits)  # [N_stamp, 2]
    dur = float(duration)
    pred_sec = (pred_bin / reg_max * dur).tolist()
    return [[float(s), float(e)] for s, e in pred_sec]


@torch.no_grad()
def generate_with_time_refinement(
    model,
    input_ids,
    attention_mask,
    duration,
    time_token_id,
    frame_times=None,
    max_new_tokens=512,
    eos_token_id=None,
    **vision_kwargs,
):
    """Greedy decode with online ``<TIME_STAMP>`` decode-to-re-encode.

    The first forward is a multimodal prefill. Later forwards use the native
    ``prepare_inputs_for_generation`` and KV cache. When the selected token is
    ``<TIME_STAMP>``, its preceding hidden state is decoded by TimeDec; the
    predicted continuous span is encoded by TimeEnc and used as
    ``inputs_embeds`` when that timestamp token is consumed on the next step.
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise AssertionError(
            "generate_with_time_refinement currently requires batch_size == 1."
        )
    if attention_mask.ndim != 2 or attention_mask.size(0) != 1:
        raise AssertionError(
            "generate_with_time_refinement currently requires batch_size == 1."
        )
    if not getattr(model, "_has_time_enc", False):
        raise ValueError("Online time refinement requires an attached TimeEnc.")

    reg_max = int(model._time_reg_max)
    if torch.is_tensor(duration):
        duration_value = float(duration.reshape(-1)[0].item())
    else:
        duration_value = float(duration)
    duration_value = max(duration_value, 1e-6)

    if eos_token_id is None:
        eos_candidates = [
            getattr(getattr(model, "generation_config", None), "eos_token_id", None),
            getattr(getattr(model, "config", None), "eos_token_id", None),
            getattr(
                getattr(getattr(model, "config", None), "text_config", None),
                "eos_token_id",
                None,
            ),
        ]
    else:
        eos_candidates = [eos_token_id]

    eos_ids = set()
    for candidate in eos_candidates:
        if candidate is None:
            continue
        if torch.is_tensor(candidate):
            candidate_ids = {int(x) for x in candidate.reshape(-1).tolist()}
        elif isinstance(candidate, (list, tuple, set)):
            candidate_ids = {int(x) for x in candidate}
        else:
            candidate_ids = {int(candidate)}
        if candidate_ids:
            eos_ids = candidate_ids
            break
    if not eos_ids:
        warnings.warn(
            "No EOS token id found in the explicit argument, generation_config, "
            "config, or config.text_config; decoding will stop only at "
            f"max_new_tokens={int(max_new_tokens)}.",
            RuntimeWarning,
        )

    generated_ids = input_ids.clone()
    predicted_spans = []
    pending_time_embedding = None
    prefill_vision_kwargs = {
        key: value
        for key, value in vision_kwargs.items()
        if key not in {"input_ids", "attention_mask", "past_key_values"}
    }

    # TimeEnc frame injection is scoped to the multimodal prefill only. No GT
    # is ever stashed during inference and all temporary state is cleared.
    model._tenc_frame_times = [frame_times] if frame_times is not None else None
    model._tenc_time_gt = None
    model._tenc_duration = [duration_value]
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            **prefill_vision_kwargs,
        )
    finally:
        model._tenc_frame_times = None
        model._tenc_time_gt = None
        model._tenc_duration = None

    past_key_values = outputs.past_key_values
    cache_position = torch.tensor(
        [input_ids.size(1)], device=input_ids.device, dtype=torch.long
    )
    prepare_state = {}
    rope_deltas = getattr(outputs, "rope_deltas", None)
    if rope_deltas is not None:
        prepare_state["rope_deltas"] = rope_deltas

    for _ in range(int(max_new_tokens)):
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token[:, None]], dim=1)

        if int(next_token.item()) == int(time_token_id):
            time_dec_dtype = next(model.time_dec.parameters()).dtype
            logits_time = model.time_dec(
                last_hidden.to(dtype=time_dec_dtype)
            ).float()
            pred_bin = model.time_project(logits_time).float().clamp(
                0.0, float(reg_max)
            )
            pred_sec = pred_bin / float(reg_max) * duration_value
            span = pred_sec[0].detach().float().tolist()
            predicted_spans.append([float(span[0]), float(span[1])])
            pending_time_embedding = encode_spans(
                model.time_enc,
                pred_sec,
                duration_value,
                reg_max,
                model._time_enc_sigma,
            )
            embedding_weight = model.get_input_embeddings().weight
            pending_time_embedding = pending_time_embedding.to(
                device=embedding_weight.device,
                dtype=embedding_weight.dtype,
            )

        # Decode the timestamp before checking EOS so a timestamp token that is
        # also terminal is still retained in predicted_spans.
        if int(next_token.item()) in eos_ids:
            break

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((1, 1))], dim=1
        )
        prepare_kwargs = {
            "input_ids": generated_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
            "cache_position": cache_position,
        }
        prepare_kwargs.update(prepare_state)
        try:
            model_inputs = model.prepare_inputs_for_generation(**prepare_kwargs)
        except TypeError as exc:
            # Older Transformers versions may not expose cache_position in the
            # preparation signature; let that method construct its own state.
            if "cache_position" not in str(exc):
                raise
            prepare_kwargs.pop("cache_position")
            model_inputs = model.prepare_inputs_for_generation(**prepare_kwargs)

        current_cache_position = model_inputs.get("cache_position")
        if pending_time_embedding is not None:
            model_inputs["inputs_embeds"] = pending_time_embedding[:, None, :]
            model_inputs.pop("input_ids", None)
            pending_time_embedding = None

        # Keep native position/cache fields intact. rope_deltas is retained as
        # model-specific generation state if the native method returns it.
        if "rope_deltas" in model_inputs:
            prepare_state["rope_deltas"] = model_inputs["rope_deltas"]

        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        rope_deltas = getattr(outputs, "rope_deltas", None)
        if rope_deltas is not None:
            prepare_state["rope_deltas"] = rope_deltas
        if current_cache_position is None:
            cache_position = cache_position + 1
        else:
            cache_position = current_cache_position[-1:] + 1

    return generated_ids, predicted_spans
