import torch
from typing import List


def _set_requires_grad(params, value: bool):
    for p in params:
        p.requires_grad = value


def _freeze_llm(model):
    """凍結 language_model 全部參數。"""
    _set_requires_grad(model.language_model.parameters(), False)


def _unfreeze_vision(model, compute_dtype, device):
    """解凍 vision_tower 與 multi_modal_projector。"""
    model.vision_tower.to(dtype=compute_dtype, device=device)
    _set_requires_grad(model.vision_tower.parameters(), True)
    _set_requires_grad(model.multi_modal_projector.parameters(), True)


def _count_params(model):
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _log(*args):
    print("[Stage1]", *args, flush=True)



# dataset.py
def _pad_sequence(sequences: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """右側 padding，回傳 [batch, max_len] tensor。"""
    max_len = max(s.size(0) for s in sequences)
    batch = sequences[0].new_full((len(sequences), max_len), padding_value)
    for i, seq in enumerate(sequences):
        batch[i, :seq.size(0)] = seq
    return batch