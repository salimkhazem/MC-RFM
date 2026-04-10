"""Metrics for classification experiments """

from __future__ import annotations

import numpy as np
import torch

#TODO : add more metrics here 

def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return float((preds == labels).float().mean().item())


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
    y = labels.detach().cpu().numpy().astype(np.int64)
    f1s: list[float] = []
    for cls in range(num_classes):
        tp = int(((preds == cls) & (y == cls)).sum())
        fp = int(((preds == cls) & (y != cls)).sum())
        fn = int(((preds != cls) & (y == cls)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s))

