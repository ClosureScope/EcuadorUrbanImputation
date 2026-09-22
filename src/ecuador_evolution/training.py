from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .model import masked_huber_loss


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float | int]]


def fit_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    accumulation_steps: int = 1,
    loss_function: Callable[..., torch.Tensor] | None = None,
    validation_metric: Callable[..., torch.Tensor] | None = None,
    gradient_clip: float | None = None,
    warmup_fraction: float = 0.0,
) -> TrainingResult:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    updates_per_epoch = max(math.ceil(len(train_loader) / accumulation_steps), 1)
    total_updates = max(max_epochs * updates_per_epoch, 1)
    warmup_updates = int(total_updates * warmup_fraction)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max((step + 1) / warmup_updates, 1e-6)
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = train_count = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            focal, neighbors, neighbor_mask, target, target_mask, *extras = (item.to(device) for item in batch)
            prediction = model(focal, neighbors, neighbor_mask)
            loss = (
                masked_huber_loss(prediction, target, target_mask)
                if loss_function is None
                else loss_function(prediction, target, target_mask, *extras)
            )
            (loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_total += float(loss.detach().cpu())
            train_count += 1
        model.eval()
        validation_total = validation_count = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                focal, neighbors, neighbor_mask, target, target_mask, *extras = (item.to(device) for item in batch)
                prediction = model(focal, neighbors, neighbor_mask)
                metric = validation_metric or loss_function
                loss = (
                    masked_huber_loss(prediction, target, target_mask)
                    if metric is None
                    else metric(prediction, target, target_mask, *extras)
                )
                validation_total += float(loss.cpu())
                validation_count += 1
        validation_loss = validation_total / max(validation_count, 1)
        history.append(
            {"epoch": epoch, "train_loss": train_total / max(train_count, 1), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(model=model, best_epoch=best_epoch, best_validation_loss=best_loss, history=history)
