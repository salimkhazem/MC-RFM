"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == labels).float().mean().item())


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
    y = labels.detach().cpu().numpy().astype(np.int64)
    f1_scores = []
    for cls in range(num_classes):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        if tp == 0 and (fp > 0 or fn > 0):
            f1_scores.append(0.0)
            continue
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        if prec + rec == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * prec * rec / (prec + rec))
    return float(np.mean(f1_scores))

