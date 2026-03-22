"""
train.py — GemmaSFTTrainer

繼承 HuggingFace Trainer，客製化：
1. create_optimizer()：為 vision_tower 和 multi_modal_projector 設定獨立學習率
2. _maybe_log_save_evaluate()：額外記錄各 param group 的實際 lr
3. _save_checkpoint()：Stage 1 無 LoRA，路徑走 super()
"""
import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
)


class GemmaSFTTrainer(Trainer):

    def create_optimizer(self):
        """
        建立 AdamW optimizer。

        若 training_args 有設定 vision_lr 或 projector_lr，
        會把對應模組的參數獨立成自己的 param group，使用指定 lr；
        其餘可訓練參數使用 training_args.learning_rate。

        每個模組再依是否需要 weight decay 分成兩個子 group（共 4 個 special group）。
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model

        # 決定哪些參數要 weight decay（LayerNorm 和 bias 不做）
        decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        # 特殊 lr 對應表
        lr_mapper: dict[str, float] = {}
        if self.args.projector_lr is not None:
            lr_mapper["multi_modal_projector"] = self.args.projector_lr
        if self.args.vision_lr is not None:
            lr_mapper["vision_tower"] = self.args.vision_lr

        special_names: set[str] = set()
        for keyword in lr_mapper:
            for n, _ in opt_model.named_parameters():
                if keyword in n:
                    special_names.add(n)

        # 一般參數（不在特殊 lr 名單內）
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n in decay_params and n not in special_names and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n not in decay_params and n not in special_names and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        # 特殊 lr 模組各自加兩個 group（有 / 無 weight decay）
        for keyword, lr in lr_mapper.items():
            module_names = {n for n, _ in opt_model.named_parameters() if keyword in n}
            optimizer_grouped_parameters.extend([
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_params and n in module_names and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_params and n in module_names and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ])

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        """覆寫以額外記錄各 param group 的實際 lr。"""
        super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch,
            ignore_keys_for_eval, start_time, learning_rate=learning_rate,
        )

        if self.control.should_log and self.optimizer is not None:
            logs = {}
            for i, pg in enumerate(self.optimizer.param_groups):
                name = pg.get("param_group_name", f"group_{i}")
                logs[f"lr_{name}"] = pg["lr"]
            self.log(logs)

    def _save_checkpoint(self, model, trial):
        """Stage 1 無 LoRA，直接走父類別邏輯。"""
        super()._save_checkpoint(model, trial)
