"""
uv run deepspeed stage1/main.py --deepspeed scripts/stage1.json [args...]
"""

import sys
import pathlib

# 確保 gemma_ft/ 在 sys.path，讓 DeepSpeed 子進程也能 import stage1.*
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)

from stage1.dataset import make_data_module
from stage1.forward import replace_forward
from stage1.train import GemmaSFTTrainer
from stage1.utils import _freeze_llm, _unfreeze_vision, _count_params, _log


@dataclass
class ModelArguments:
    model_id: str = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "HuggingFace model ID 或本地模型路徑"},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "訓練資料 JSON 路徑（list of {messages:[...]}）"}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "圖片根目錄；資料中的相對路徑會自動加上此前綴"},
    )


@dataclass
class Stage1TrainingArguments(TrainingArguments):
    vision_lr: Optional[float] = field(
        default=2e-5,
        metadata={"help": "Vision Tower 的學習率（None 表示使用 learning_rate）"},
    )
    projector_lr: Optional[float] = field(
        default=2e-5,
        metadata={"help": "Multi-Modal Projector 的學習率（None 表示使用 learning_rate）"},
    )

    # 序列長度
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "最大 token 序列長度，超過會被截斷"},
    )

    # Stage 1 固定不變的設定（由程式硬鎖，不接受 CLI 覆蓋）
    cache_dir: Optional[str] = field(default=None)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, Stage1TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    device = training_args.device

    # ── 替換 forward ──────────────────────────────────────────────────────────
    replace_forward()

    # ── 載入模型 ──────────────────────────────────────────────────────────────
    _log(f"載入模型：{model_args.model_id}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="eager",  # Stage 1 使用 eager 避免 flash-attn 問題
    )

    # ── Stage 1：凍結 LLM，解凍視覺模組 ─────────────────────────────────────
    _freeze_llm(model)
    _unfreeze_vision(model, compute_dtype, device)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    model.config.use_cache = False
    # 將 lr 存入 config，讓 optimizer 可以讀取
    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    trainable, total = _count_params(model)
    _log(f"可訓練參數：{trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 印出前 20 個可訓練模組名稱（快速確認哪些層有開放訓練）
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    _log(f"可訓練模組（前 20）：{trainable_names[:20]}")

    # ── Processor & Dataset ───────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_args.model_id)
    data_module = make_data_module(
        processor=processor,
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GemmaSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    # 若有 checkpoint 則續訓
    output_dir = pathlib.Path(training_args.output_dir)
    resume = bool(list(output_dir.glob("checkpoint-*")))
    _log(f"{'續訓' if resume else '全新訓練'} → {training_args.output_dir}")
    trainer.train(resume_from_checkpoint=resume)

    # store
    trainer.save_state()
    model.config.use_cache = True

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(training_args.output_dir)
    else:
        state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
        trainer._save(training_args.output_dir, state_dict=state_dict)
        trainer.model.config.save_pretrained(training_args.output_dir)

    _log("訓練完成，模型已儲存至", training_args.output_dir)


if __name__ == "__main__":
    train()