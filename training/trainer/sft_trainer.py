import os
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, TRAINER_STATE_NAME, PREFIX_CHECKPOINT_DIR, logger, ExportableState, SaveStrategy
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.modules.normalization.LayerNorm]
from training.train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

class QwenSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_loss_sums = {
            "ntp_loss": None,
            "diou_loss": None,
            "smooth_l1_loss": None,
        }
        self._task_loss_update_weight = 0.0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        task_losses = {}
        for name in self._task_loss_sums:
            value = (
                outputs.get(name)
                if isinstance(outputs, dict)
                else getattr(outputs, name, None)
            )
            if value is None:
                break
            task_losses[name] = value

        if len(task_losses) == len(self._task_loss_sums):
            accumulation_steps = (
                max(
                    int(
                        getattr(
                            self,
                            "current_gradient_accumulation_steps",
                            self.args.gradient_accumulation_steps,
                        )
                    ),
                    1,
                )
                if model.training
                else 1
            )
            ntp_scale = 1.0
            if (
                self.args.average_tokens_across_devices
                and self.model_accepts_loss_kwargs
                and num_items_in_batch is not None
            ):
                ntp_scale = (
                    self.accelerator.num_processes
                    if self.args.n_gpu <= 1
                    else self.args.n_gpu
                )

            normalized_losses = {
                "ntp_loss": task_losses["ntp_loss"] * ntp_scale,
                "diou_loss": task_losses["diou_loss"] / accumulation_steps,
                "smooth_l1_loss": (
                    task_losses["smooth_l1_loss"] / accumulation_steps
                ),
            }
            config = self.model.config
            loss = (
                config.lambda_ntp * normalized_losses["ntp_loss"]
                + config.lambda_diou * normalized_losses["diou_loss"]
                + config.lambda_reg * normalized_losses["smooth_l1_loss"]
            )

            if model.training:
                for name, value in normalized_losses.items():
                    value = value.detach().float()
                    current = self._task_loss_sums[name]
                    self._task_loss_sums[name] = (
                        value if current is None else current + value
                    )
                self._task_loss_update_weight += 1.0 / accumulation_steps
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self._task_loss_update_weight:
            for name, total in self._task_loss_sums.items():
                logs[name] = round(
                    float(
                        (
                            self._nested_gather(total).mean()
                            / self._task_loss_update_weight
                        ).item()
                    ),
                    4,
                )
                self._task_loss_sums[name] = None
            self._task_loss_update_weight = 0.0
        return super().log(logs, start_time=start_time)

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [n for n in decay_parameters if "bias" not in n]
            named_parameters = list(opt_model.named_parameters())
            visual_parameters = [
                n
                for n, _ in named_parameters
                if "visual" in n and "merger" not in n and self.args.vision_lr
            ]
            merger_parameters = [
                n for n, _ in named_parameters if "merger" in n and self.args.merger_lr
            ]
            is_time_refine = any("time_refine_head" in n for n, _ in named_parameters)
            time_refine_lr = getattr(self.args, "time_refine_lr", None)
            time_token_lr = getattr(self.args, "time_token_lr", None)
            time_refine_parameters = [
                n
                for n, _ in named_parameters
                if is_time_refine
                and time_refine_lr
                and ("time_refine_head" in n or "time_proj" in n)
            ]
            time_token_parameters = [
                n
                for n, _ in named_parameters
                if is_time_refine
                and time_token_lr
                and ("embed_tokens" in n or "embed_token" in n)
            ]
            special_groups = [
                (self.args.vision_lr, visual_parameters),
                (self.args.merger_lr, merger_parameters),
                (time_refine_lr, time_refine_parameters),
                (time_token_lr, time_token_parameters),
            ]
            special = {
                name for _, names in special_groups for name in names
            }

            def make_group(names, weight_decay, lr=None):
                parameters = [
                    parameter
                    for name, parameter in named_parameters
                    if name in names and parameter.requires_grad
                ]
                if not parameters:
                    return None
                group = {"params": parameters, "weight_decay": weight_decay}
                if lr:
                    group["lr"] = lr
                return group

            grps = []
            default_decay = make_group(
                {
                    name
                    for name, _ in named_parameters
                    if name in decay_parameters and name not in special
                },
                self.args.weight_decay,
            )
            default_no_decay = make_group(
                {
                    name
                    for name, _ in named_parameters
                    if name not in decay_parameters and name not in special
                },
                0.0,
            )
            for group in (default_decay, default_no_decay):
                if group is not None:
                    grps.append(group)
            for lr, names in special_groups:
                if lr and names:
                    for is_decay, weight_decay in (
                        (True, self.args.weight_decay),
                        (False, 0.0),
                    ):
                        group = make_group(
                            {
                                name
                                for name in names
                                if (name in decay_parameters) == is_decay
                            },
                            weight_decay,
                            lr=lr,
                        )
                        if group is not None:
                            grps.append(group)
            if not grps:
                raise ValueError("No trainable parameters were found for the optimizer.")
            opt_cls, opt_kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = opt_cls(grps, **opt_kw)
            if opt_cls.__name__ == "Adam8bit":
                import bitsandbytes
                m = bitsandbytes.optim.GlobalOptimManager.get_instance()
                for mod in opt_model.modules():
                    if isinstance(mod, nn.Embedding):
                        m.register_module_override(mod, "weight", {"optim_bits": 32})
        return self.optimizer
    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            ckpt = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            if self.hp_search_backend is None and trial is None:
                self.store_flos()
            out = os.path.join(self._get_output_dir(trial=trial), ckpt)
            self.save_model(out, _internal_call=True)
            torch.save(get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False), os.path.join(out, "non_lora_state_dict.bin"))
            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best = os.path.join(self._get_output_dir(trial=trial), f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}")
                if os.path.exists(best):
                    self.state.best_model_checkpoint = best
            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(out)
                self._save_scaler(out)
                self._save_rng_state(out)
            if self.args.should_save:
                for cb in [c for c in self.callback_handler.callbacks + [self.control] if isinstance(c, ExportableState)]:
                    s = self.state.stateful_callbacks
                    n = cb.__class__.__name__
                    s[n] = s.get(n, []) + [cb.state()] if isinstance(s.get(n), list) else cb.state()
                self.state.save_to_json(os.path.join(out, TRAINER_STATE_NAME))
            if self.args.push_to_hub:
                self._push_from_checkpoint(out)
        else:
            super()._save_checkpoint(model, trial)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target_model = model if model is not None else self.model
        base_directory = os.path.join(resume_from_checkpoint, "base_model")
        if (
            os.path.isdir(base_directory)
            and hasattr(target_model, "load_attached_base_checkpoint")
        ):
            target_model.load_attached_base_checkpoint(base_directory)
        non_lora_state_file = os.path.join(
            resume_from_checkpoint, "non_lora_state_dict.bin"
        )
        if os.path.isfile(non_lora_state_file):
            non_lora_state_dict = torch.load(
                non_lora_state_file,
                map_location="cpu",
                weights_only=True,
            )
            target_model.load_state_dict(non_lora_state_dict, strict=False)
