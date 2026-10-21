"""
Task-agnostic Trainer: train/val loop, device move, step_fn, epoch logging.

Step function contract
----------------------
step_fn(batch, model, device) -> (loss_tensor, metrics_dict | None)

- **batch**: Dict of tensors, already on `device` when passed to step_fn. The trainer
  moves the batch to device before each call. All float tensors in the batch should be
  float32 where the model expects float (e.g. elemental_strengths, interaction_vector);
  cast in the step_fn or via a dataset transform (ChartDataset returns float64 by default).
- **Train step**: Trainer sets model.train(), then calls step_fn. The step_fn must
  zero_grad, forward, compute loss, backward, and optimizer.step(). The optimizer is
  typically captured in the closure when building step_fn.
- **Val step**: Trainer sets model.eval(), then calls step_fn. The step_fn must only
  forward and return loss (no backward). step_fn can branch on model.training.
- **Return**: (loss_tensor, metrics_dict | None). loss_tensor is a scalar. metrics_dict
  is optional and can contain task-specific scalars (e.g. accuracy); they are averaged
  over batches and logged to TensorBoard per epoch.

Batch float32 casting
--------------------
ChartDataset and chart_collate_fn produce int64 and float64 tensors. Models and loss
functions often expect float32. Either (1) in the step_fn, cast float tensors to
float32 before passing to the model, or (2) use a dataset transform that converts
float arrays to float32 when building the dataset.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def _get_summary_writer_type():
    """Resolve SummaryWriter at runtime so log_dir is created when tensorboard is available."""
    try:
        from torch.utils.tensorboard import SummaryWriter as _SW
        return _SW
    except ImportError:
        return None


def move_batch_to_device(
    batch: Union[dict[str, torch.Tensor], Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    device: torch.device,
) -> Union[dict[str, torch.Tensor], Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    """Move a batch to device. Supports single dict or (inputs, targets) tuple."""
    if isinstance(batch, tuple) and len(batch) == 2:
        inputs, targets = batch
        return (
            {k: v.to(device, non_blocking=True) for k, v in inputs.items()},
            {k: v.to(device, non_blocking=True) for k, v in targets.items()},
        )
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# Step function type: (batch, model, device) -> (loss, optional_metrics)
# batch can be a single dict (legacy) or (inputs, targets) tuple
StepFn = Callable[
    [
        Union[dict[str, torch.Tensor], Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
        torch.nn.Module,
        torch.device,
    ],
    tuple[torch.Tensor, Optional[dict[str, float]]],
]


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Average per-batch metrics (same keys in each dict)."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: sum(m.get(k, 0.0) for m in metrics_list) / len(metrics_list)
        for k in keys
    }


class Trainer:
    """
    Task-agnostic trainer: train/val loop, scheduler, early stopping, checkpointing,
    and TensorBoard logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        step_fn: StepFn,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        early_stopping_patience: int = 5,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        best_metric_key: Optional[str] = None,
        best_metric_mode: str = "min",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.step_fn = step_fn
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.best_metric_key = best_metric_key
        self.best_metric_mode = best_metric_mode
        self._history: list[dict[str, float]] = []
        self._writer: Optional[Any] = None

    def _get_writer(self) -> Optional[Any]:
        if self._writer is not None:
            return self._writer
        if self.log_dir is None:
            return None
        SummaryWriter = _get_summary_writer_type()
        if SummaryWriter is None:
            return None
        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self.log_dir)
        return self._writer

    def _save_checkpoint(self, epoch: int, val_metric: float) -> None:
        if self.checkpoint_dir is None:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "best.pt")
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": val_metric,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)

    def run(self, num_epochs: int) -> float:
        """
        Run training for num_epochs. Returns best validation metric (e.g. loss).
        Uses early stopping when validation does not improve for
        early_stopping_patience consecutive epochs. Saves best checkpoint when
        validation improves. Logs train/val loss and optional metrics to TensorBoard.
        """
        self.model.to(self.device)
        use_metric = self.best_metric_key is not None
        if self.best_metric_mode == "min":
            best_val_metric = float("inf")
        else:
            best_val_metric = float("-inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            self.model.train()
            train_losses: list[float] = []
            train_metrics_list: list[dict[str, float]] = []
            for batch in self.train_loader:
                batch = move_batch_to_device(batch, self.device)
                loss, metrics = self.step_fn(batch, self.model, self.device)
                train_losses.append(loss.detach().item())
                if metrics is not None:
                    train_metrics_list.append(metrics)

            train_avg = sum(train_losses) / len(train_losses) if train_losses else 0.0
            train_metrics_agg = _aggregate_metrics(train_metrics_list)

            # Validation
            self.model.eval()
            val_losses: list[float] = []
            val_metrics_list: list[dict[str, float]] = []
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = move_batch_to_device(batch, self.device)
                    loss, metrics = self.step_fn(batch, self.model, self.device)
                    val_losses.append(loss.detach().item())
                    if metrics is not None:
                        val_metrics_list.append(metrics)

            val_avg = sum(val_losses) / len(val_losses) if val_losses else 0.0
            val_metrics_agg = _aggregate_metrics(val_metrics_list)

            if use_metric and self.best_metric_key and self.best_metric_key in val_metrics_agg:
                val_metric = val_metrics_agg[self.best_metric_key]
            else:
                val_metric = val_avg

            # Scheduler step (per epoch)
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_avg)
                else:
                    self.scheduler.step()

            # TensorBoard
            writer = self._get_writer()
            if writer is not None:
                writer.add_scalar("train/loss", train_avg, epoch)
                writer.add_scalar("val/loss", val_avg, epoch)
                for k, v in train_metrics_agg.items():
                    writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics_agg.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

            record = {"epoch": float(epoch), "train_loss": train_avg, "val_loss": val_avg}
            self._history.append(record)
            print(f"Epoch {epoch}/{num_epochs}  train_loss={train_avg:.6f}  val_loss={val_avg:.6f}")

            # Best checkpoint and early stopping (by val_metric: loss or composite)
            is_better = (self.best_metric_mode == "min" and val_metric < best_val_metric) or (
                self.best_metric_mode == "max" and val_metric > best_val_metric
            )
            if is_better:
                best_val_metric = val_metric
                patience_counter = 0
                self._save_checkpoint(epoch, best_val_metric)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        return best_val_metric

    @property
    def history(self) -> list[dict[str, float]]:
        """Per-epoch log: list of {epoch, train_loss, val_loss}."""
        return self._history
