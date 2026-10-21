"""Unit tests for bazi_model.training.Trainer.

Covers: mock step_fn, 1 train + 1 val step (one epoch), checkpoint and log assertions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bazi_model.training import Trainer, move_batch_to_device


def _fake_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Minimal batch (dict of tensors) for step_fn."""
    return {"x": torch.randn(batch_size, 4)}


def _make_fake_loaders(
    train_batches: int = 1,
    val_batches: int = 1,
    batch_size: int = 2,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create DataLoaders that yield dict batches. Each 'batch' is a list of batch_size dicts; we need to collate."""
    # Use a dataset that returns dicts; collate by stacking tensors
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, num_batches: int, batch_size: int):
            self.num_batches = num_batches
            self.batch_size = batch_size

        def __len__(self) -> int:
            return self.num_batches * self.batch_size

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"x": torch.randn(4)}

    def collate(items: list) -> dict[str, torch.Tensor]:
        return {"x": torch.stack([x["x"] for x in items])}

    train_ds = FakeDataset(train_batches, batch_size)
    val_ds = FakeDataset(val_batches, batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    return train_loader, val_loader


def test_move_batch_to_device():
    """move_batch_to_device moves all tensors to device."""
    batch = _fake_batch(2)
    assert batch["x"].device.type == "cpu"
    out = move_batch_to_device(batch, torch.device("cpu"))
    assert out["x"].device.type == "cpu"
    out2 = move_batch_to_device(batch, torch.device("cpu"))
    assert "x" in out2 and out2["x"].shape == (2, 4)


def test_trainer_one_epoch_mock_step_fn(tmp_path: Path):
    """Run Trainer for 1 epoch with mock step_fn; assert history and optional checkpoint/log dir."""
    device = torch.device("cpu")
    model = torch.nn.Linear(4, 1)
    train_loader, val_loader = _make_fake_loaders(train_batches=1, val_batches=1, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    fixed_train_loss = 2.0
    fixed_val_loss = 1.5

    def step_fn(
        batch: dict[str, torch.Tensor],
        m: torch.nn.Module,
        dev: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        assert batch["x"].device == dev
        if m.training:
            optimizer.zero_grad()
            out = m(batch["x"])
            loss = out.pow(2).mean()
            loss.backward()
            optimizer.step()
            return torch.tensor(fixed_train_loss, device=dev), {"acc": 0.8}
        # val
        with torch.no_grad():
            out = m(batch["x"])
            loss = out.pow(2).mean()
        return torch.tensor(fixed_val_loss, device=dev), {"acc": 0.75}

    checkpoint_dir = tmp_path / "ckpt"
    log_dir = tmp_path / "logs"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=None,
        early_stopping_patience=5,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
    )

    best_val = trainer.run(num_epochs=1)

    # One epoch completed
    assert len(trainer.history) == 1
    record = trainer.history[0]
    assert record["epoch"] == 1.0
    assert "train_loss" in record and "val_loss" in record
    assert record["val_loss"] == fixed_val_loss
    assert best_val == fixed_val_loss

    # Best checkpoint saved (val improved from inf)
    assert (checkpoint_dir / "best.pt").exists()
    ckpt = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "best_metric" in ckpt
    assert ckpt["best_metric"] == fixed_val_loss

    # When TensorBoard is available, Trainer creates log_dir and writes events
    try:
        from torch.utils.tensorboard import SummaryWriter
        _ = SummaryWriter
        assert log_dir.exists()
    except ImportError:
        pass  # log_dir only created when SummaryWriter is available


def test_trainer_no_checkpoint_no_log_dir(tmp_path: Path):
    """Trainer with checkpoint_dir=None and log_dir=None does not create files."""
    device = torch.device("cpu")
    model = torch.nn.Linear(4, 1)
    train_loader, val_loader = _make_fake_loaders(1, 1, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step_fn(
        batch: dict[str, torch.Tensor],
        m: torch.nn.Module,
        dev: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        if m.training:
            optimizer.zero_grad()
            loss = m(batch["x"]).pow(2).mean()
            loss.backward()
            optimizer.step()
        else:
            loss = m(batch["x"]).pow(2).mean()
        return loss.detach(), None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=None,
        early_stopping_patience=10,
        log_dir=None,
        checkpoint_dir=None,
    )

    best_val = trainer.run(num_epochs=1)
    assert len(trainer.history) == 1
    assert best_val == trainer.history[0]["val_loss"]
    # No checkpoint or log dir under tmp_path (we didn't pass any)
    assert not (tmp_path / "best.pt").exists()


def test_trainer_early_stopping_stops_early(tmp_path: Path):
    """Trainer stops when validation does not improve for early_stopping_patience epochs."""
    device = torch.device("cpu")
    model = torch.nn.Linear(4, 1)
    train_loader, val_loader = _make_fake_loaders(1, 1, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Val loss constant and high so we never improve after first epoch
    def step_fn(
        batch: dict[str, torch.Tensor],
        m: torch.nn.Module,
        dev: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        if m.training:
            optimizer.zero_grad()
            loss = m(batch["x"]).pow(2).mean()
            loss.backward()
            optimizer.step()
            return loss.detach(), None
        return torch.tensor(100.0, device=dev), None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=None,
        early_stopping_patience=2,
        log_dir=None,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )

    best_val = trainer.run(num_epochs=10)
    # First epoch: val_loss 100, best saved. Epochs 2–3: no improvement; after epoch 3, patience 2 reached, stop.
    assert len(trainer.history) == 3
    assert best_val == 100.0
