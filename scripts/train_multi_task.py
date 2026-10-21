#!/usr/bin/env python3
"""
Train multiple natal tasks with one shared NatalEncoder (multi-task learning).

Builds one shared NatalEncoder and one head per task: ten-god, element strength,
interaction detection, useful god. Each step computes per-task losses, sums them
(optionally weighted), and backprops once. Logs per-task train/val metrics to
TensorBoard for comparison with sequential single-task training.

Cross-temporal is not included (requires DynamicEncoder and temporal_only=True
dataset); use scripts/train_cross_temporal.py for that task.

Usage:
  python scripts/train_multi_task.py --data data/charts.jsonl [options]
  python scripts/train_multi_task.py --data data/charts.jsonl --epochs 30 --log-dir runs/multi_task

Pretrain-then-multitask (element first, then all tasks; use --save_name to avoid overwriting current best):
  python scripts/train_element_strength.py --data data/charts.jsonl --epochs 20 --checkpoint_dir checkpoints
  python scripts/train_multi_task.py --data data/charts.jsonl --encoder_checkpoint checkpoints/element_strength_best.pt --epochs 30 --task_weights 1,3,1,1 --save_by composite --save_name multi_task_after_element_pretrain.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader, random_split

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.ten_god_head import TenGodHead
from bazi_model.tasks.element_regressor import ElementStrengthHead
from bazi_model.tasks.interaction_detector import InteractionHead
from bazi_model.tasks.useful_god_classifier import UsefulGodHead
from bazi_model.training import Trainer

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)


def _load_encoder_and_element_from_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    element_head: torch.nn.Module | None,
    device: torch.device,
    load_element_head: bool = True,
) -> None:
    """
    Load encoder (and optionally element head) from a checkpoint.

    Supports: (1) Multi-task format dict with keys 'encoder', 'element_head';
    (2) Element-only model state_dict with keys 'encoder.*' and 'head.*'.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        enc_sd = ckpt["encoder"]
        if hasattr(enc_sd, "keys"):
            encoder.load_state_dict(enc_sd, strict=True)
        if load_element_head and element_head is not None and "element_head" in ckpt:
            element_head.load_state_dict(ckpt["element_head"], strict=True)
        return
    # Element-only checkpoint: state_dict of ElementStrengthModel (encoder.*, head.*)
    enc_state = {k[len("encoder.") :]: v for k, v in ckpt.items() if k.startswith("encoder.")}
    if enc_state:
        encoder.load_state_dict(enc_state, strict=True)
    if load_element_head and element_head is not None:
        head_state = {k[len("head.") :]: v for k, v in ckpt.items() if k.startswith("head.")}
        if head_state:
            element_head.load_state_dict(head_state, strict=True)


def _compute_pos_weight(dataset: ChartDataset) -> torch.Tensor:
    """(1-p)/p per bit for BCEWithLogitsLoss pos_weight from multi-task targets."""
    all_vectors = []
    for i in range(len(dataset)):
        _, targets = dataset[i]
        all_vectors.append(targets["interaction_vector"])
    all_vectors = torch.stack(all_vectors)
    positive_rate = all_vectors.mean(dim=0)
    return (1 - positive_rate) / (positive_rate + 1e-8)


def _cast_inputs_float32(inputs: dict[str, torch.Tensor]) -> None:
    """In-place cast float tensors in inputs to float32."""
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].float()


class MultiTaskModel(torch.nn.Module):
    """Shared NatalEncoder + one head per natal task. Encoder sees only inputs (no targets)."""

    def __init__(
        self,
        encoder: NatalEncoder,
        ten_god_head: TenGodHead,
        element_head: ElementStrengthHead,
        interaction_head: InteractionHead,
        useful_god_head: UsefulGodHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.ten_god_head = ten_god_head
        self.element_head = element_head
        self.interaction_head = interaction_head
        self.useful_god_head = useful_god_head

    def forward(self, **input_kwargs) -> torch.Tensor:
        """Return natal_emb from inputs only."""
        return self.encoder(**input_kwargs)


def make_step_fn(
    optimizer: torch.optim.Optimizer,
    task_weights: dict[str, float] | None = None,
):
    """
    step_fn((inputs, targets), model, device). Encoder sees only inputs; losses use targets.
    targets has ten_god_ids, elemental_strengths, interaction_vector, useful_god_id.
    """

    if task_weights is None:
        task_weights = {"ten_god": 1.0, "element": 1.0, "interaction": 1.0, "useful_god": 1.0}

    def step_fn(
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model: MultiTaskModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        inputs, targets = batch
        _cast_inputs_float32(inputs)
        encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
        natal_emb = model.encoder(**encoder_kw)

        loss_tg, logits_tg = model.ten_god_head.forward_loss(natal_emb, targets["ten_god_ids"])
        loss_el, probs_el = model.element_head.forward_loss(natal_emb, targets["elemental_strengths"])
        loss_int, logits_int = model.interaction_head.forward_loss(targets, natal_emb)
        loss_ug, logits_ug = model.useful_god_head.forward_loss(natal_emb, targets["useful_god_id"])

        w_tg = task_weights.get("ten_god", 1.0)
        w_el = task_weights.get("element", 1.0)
        w_int = task_weights.get("interaction", 1.0)
        w_ug = task_weights.get("useful_god", 1.0)
        total_loss = w_tg * loss_tg + w_el * loss_el + w_int * loss_int + w_ug * loss_ug

        if model.training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        metrics = {
            "loss_ten_god": loss_tg.detach().item(),
            "loss_element": loss_el.detach().item(),
            "loss_interaction": loss_int.detach().item(),
            "loss_useful_god": loss_ug.detach().item(),
        }
        # Composite metric: mean of task metrics (for save_by composite)
        tg_ids = targets["ten_god_ids"].long()
        acc_tg = (logits_tg.argmax(dim=-1) == tg_ids).float().mean().item()
        ord_el = (
            (probs_el.argsort(dim=-1) == targets["elemental_strengths"].argsort(dim=-1))
            .all(dim=-1)
            .float()
            .mean()
            .item()
        )
        acc_int = (
            (logits_int.sigmoid() > 0.5).float() == targets["interaction_vector"]
        ).float().mean().item()
        ug_id = targets["useful_god_id"].view(-1).long()
        acc_ug = (logits_ug.argmax(dim=-1) == ug_id).float().mean().item()
        metrics["composite"] = (acc_tg + ord_el + acc_int + acc_ug) / 4.0
        return total_loss.detach(), metrics

    return step_fn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-task training: shared NatalEncoder + ten-god, element, interaction, useful-god heads."
    )
    parser.add_argument("--data", type=Path, default=Path("data/charts.jsonl"), help="Path to charts.jsonl")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation fraction (e.g. 0.1 for 80/10/10)")
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test fraction for held-out set (0 = no test split; use same as evaluate_phase1 for test metrics)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Encoder and head hidden size (must be divisible by nhead=4). Try 512 for more capacity.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of transformer layers in NatalEncoder. Try 3 for more capacity.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="multi_task_best.pt",
        help="Final checkpoint filename (e.g. multi_task_after_element_pretrain.pt to avoid overwriting current best).",
    )
    parser.add_argument("--log_dir", type=Path, default=None, help="TensorBoard log directory")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument(
        "--task_weights",
        type=str,
        default=None,
        help="Optional comma-separated weights: ten_god,element,interaction,useful_god (e.g. 1,1,1,1)",
    )
    parser.add_argument(
        "--save_by",
        choices=["loss", "composite"],
        default="loss",
        help="Checkpoint selection: 'loss' = lowest val loss (default); 'composite' = highest mean(ten_god_acc, ordinal, interaction_acc, useful_god_acc)",
    )
    parser.add_argument(
        "--ordinal_loss_weight",
        type=float,
        default=0.15,
        help="Element head: weight for pairwise ranking (ordinal) loss; 0 = MSE only. Higher (e.g. 0.25) can improve ordinal accuracy.",
    )
    parser.add_argument(
        "--ordinal_margin_min",
        type=float,
        default=0.0,
        help="Element head: only penalize pairs with true strength diff >= this (default 0). Use 0.02 to skip borderline pairs and focus on clear orderings.",
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained encoder (e.g. element-only checkpoint). Load encoder and optionally element head, then train all tasks.",
    )
    parser.add_argument(
        "--load_element_head",
        action="store_true",
        default=True,
        help="When using --encoder_checkpoint from element-only training, also load element head (default: True). Set --no-load_element_head to init element head from scratch.",
    )
    parser.add_argument("--no_load_element_head", action="store_false", dest="load_element_head")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task_weights = None
    if args.task_weights:
        parts = [float(x) for x in args.task_weights.split(",")]
        if len(parts) == 4:
            task_weights = {
                "ten_god": parts[0],
                "element": parts[1],
                "interaction": parts[2],
                "useful_god": parts[3],
            }

    dataset = ChartDataset(args.data, task_name="multi", temporal_only=False)
    n = len(dataset)
    n_val = max(1, int(n * args.val_ratio))
    if args.test_ratio > 0:
        n_test = int(n * args.test_ratio)
        n_train = n - n_val - n_test
        if n_train < 1 or n_test < 1:
            print("val_ratio + test_ratio too large for dataset size", file=sys.stderr)
            return 1
        train_ds, val_ds, _ = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        n_train = n - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )

    d = args.d_model
    if d % 4 != 0:
        print(f"d_model={d} must be divisible by 4 (nhead).", file=sys.stderr)
        return 1
    encoder = NatalEncoder(d_model=d, num_layers=args.num_layers)
    pos_weight = _compute_pos_weight(dataset)
    element_head = ElementStrengthHead(
        d_natal=d,
        n_elements=5,
        ordinal_loss_weight=args.ordinal_loss_weight,
        ordinal_margin_min=args.ordinal_margin_min,
    )
    model = MultiTaskModel(
        encoder=encoder,
        ten_god_head=TenGodHead(d_model=d),
        element_head=element_head,
        interaction_head=InteractionHead(d_in=d, d_out=64, pos_weight=pos_weight),
        useful_god_head=UsefulGodHead(input_dim=d, n_classes=6),
    )

    if args.encoder_checkpoint is not None:
        if args.d_model != 256:
            print(
                f"Warning: --encoder_checkpoint is for 256-d encoder; d_model={args.d_model}. Skipping load.",
                file=sys.stderr,
            )
        elif not args.encoder_checkpoint.exists():
            print(f"Encoder checkpoint not found: {args.encoder_checkpoint}", file=sys.stderr)
            return 1
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _load_encoder_and_element_from_checkpoint(
                args.encoder_checkpoint,
                encoder,
                element_head,
                device,
                load_element_head=args.load_element_head,
            )
            print(f"Loaded encoder (and element_head={args.load_element_head}) from {args.encoder_checkpoint}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    step_fn = make_step_fn(optimizer, task_weights=task_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric_key = "composite" if args.save_by == "composite" else None
    best_metric_mode = "max" if args.save_by == "composite" else "min"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=args.patience,
        log_dir=str(args.log_dir) if args.log_dir else None,
        checkpoint_dir=str(args.checkpoint_dir),
        best_metric_key=best_metric_key,
        best_metric_mode=best_metric_mode,
    )

    best_val_metric = trainer.run(num_epochs=args.epochs)
    if args.save_by == "composite":
        print(f"Best validation composite: {best_val_metric:.6f}")
    else:
        print(f"Best validation loss (sum): {best_val_metric:.6f}")

    best_pt = args.checkpoint_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    ckpt_path = args.checkpoint_dir / args.save_name
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "ten_god_head": model.ten_god_head.state_dict(),
            "element_head": model.element_head.state_dict(),
            "interaction_head": model.interaction_head.state_dict(),
            "useful_god_head": model.useful_god_head.state_dict(),
            "best_val_metric": best_val_metric,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
