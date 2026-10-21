"""
Training utilities for bazi_model.

- Trainer: task-agnostic train/val loop with scheduler, early stopping, checkpointing,
  TensorBoard logging; see trainer module docstring for step_fn contract and float32.
- move_batch_to_device: helper to move batch (dict of tensors) to device.
- StepFn: type alias for the training step function signature.
"""

from bazi_model.training.trainer import Trainer, StepFn, move_batch_to_device

__all__ = ["Trainer", "StepFn", "move_batch_to_device"]
