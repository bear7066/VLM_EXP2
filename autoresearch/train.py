import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

MODEL_ID            = "google/gemma-3-4b-it"
DATA_PATH           = "/path/to/your/data.json"
IMAGE_FOLDER        = "/path/to/your/images"   # set None if using absolute paths
OUTPUT_DIR          = "./output/gemma3_stage1"

NUM_TRAIN_EPOCHS            = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH              = 4096

LEARNING_RATE    = 1e-5
VISION_LR        = 2e-5
PROJECTOR_LR     = 2e-5
WEIGHT_DECAY     = 0.0
WARMUP_RATIO     = 0.03
LR_SCHEDULER     = "cosine"

LOGGING_STEPS    = 10
SAVE_STEPS       = 500
SAVE_TOTAL_LIMIT = 2

GRADIENT_CHECKPOINTING = True
DATALOADER_NUM_WORKERS = 4
BF16                   = True
TF32                   = False
REPORT_TO              = "none"

# ── Imports from gemma_ft ──────────────────────────────────────────────────────

import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)

from gemma_ft.stage1.dataset import make_data_module
from gemma_ft.stage1.forward import replace_forward
from gemma_ft.stage1.train   import GemmaSFTTrainer
from gemma_ft.stage1.utils   import _freeze_llm, _unfreeze_vision, _count_params, _log

# arguments
@dataclass
class Stage1TrainingArguments(TrainingArguments):
    vision_lr: Optional[float] = field(default=VISION_LR, metadata={"help": "LR for vision_tower"})
    projector_lr: Optional[float] = field(default=PROJECTOR_LR, metadata={"help": "LR for multi_modal_projector"})
    max_seq_length: int = field(default=MAX_SEQ_LENGTH, metadata={"help": "Max token sequence length"})
    cache_dir: Optional[str] = field(default=None)

# train
def train() -> None:
    parser = HfArgumentParser((Stage1TrainingArguments,))
    (training_args,) = parser.parse_args_into_dataclasses()

    compute_dtype = torch.bfloat16
    device = training_args.device
    replace_forward()
    
    _log(f"Loading model: {MODEL_ID}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="eager",
    )

    _freeze_llm(model)
    _unfreeze_vision(model, compute_dtype, device)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    model.config.use_cache = False
    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    trainable, total = _count_params(model)
    _log(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    _log(f"Trainable modules (first 20): {trainable_names[:20]}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    data_module = make_data_module(
        processor=processor,
        data_path=DATA_PATH,
        image_folder=IMAGE_FOLDER,
    )

    trainer = GemmaSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    output_dir = pathlib.Path(training_args.output_dir)
    resume = bool(list(output_dir.glob("checkpoint-*")))
    _log(f"{'Resuming' if resume else 'Starting fresh'} → {training_args.output_dir}")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_state()
    model.config.use_cache = True

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(training_args.output_dir)
    else:
        state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
        trainer._save(training_args.output_dir, state_dict=state_dict)
        trainer.model.config.save_pretrained(training_args.output_dir)

    _log("Training complete. Model saved to", training_args.output_dir)


if __name__ == "__main__":
    train()
