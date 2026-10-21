#!/usr/bin/env python3
"""
Evaluate Phase 1 checkpoints on the validation or test set.

Loads task-specific or multi-task checkpoints, runs inference on the val split
(test_ratio=0) or the held-out test split (test_ratio>0; same seed/val_ratio/test_ratio as training), and computes:
- Ten-god: accuracy, confusion matrix (11 classes × 4 positions flattened).
- Element strength: MSE, MAE, ordinal accuracy (argsort agreement).
- Interaction: binary accuracy, micro/macro F1, F1 on common (0–29) vs rare (30–63) bits.
- Useful god: accuracy overall, accuracy on unambiguous (target != 5), confusion matrix (6 classes).
- Cross-temporal: binary accuracy and F1 on 61 binary dims (0–59, 65); MSE on continuous (60–64).

Usage:
  python scripts/evaluate_phase1.py --data data/charts.jsonl --checkpoint_dir checkpoints
  python scripts/evaluate_phase1.py --data data/charts.jsonl --use_multi_task  # natal tasks from multi_task_best.pt
  python scripts/evaluate_phase1.py --data data/charts.jsonl --output reports/phase1_metrics.json

Optional: pip install scikit-learn for sklearn.metrics (confusion_matrix, f1_score); otherwise a numpy fallback is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

try:
    from sklearn.metrics import confusion_matrix, f1_score
except ImportError:
    # Fallback if scikit-learn not installed (pip install scikit-learn recommended)
    def confusion_matrix(y_true, y_pred, labels=None):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        labels = np.asarray(labels)
        n = len(labels)
        cm = np.zeros((n, n), dtype=np.int64)
        for lt, lp in zip(y_true, y_pred):
            i = np.searchsorted(labels, lt)
            j = np.searchsorted(labels, lp)
            if i < n and j < n and labels[i] == lt and labels[j] == lp:
                cm[i, j] += 1
        return cm

    def f1_score(y_true, y_pred, average="macro", zero_division=0, labels=None):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        labels = np.asarray(labels)
        f1s = []
        for c in labels:
            tp = ((y_true == c) & (y_pred == c)).sum()
            fp = ((y_true != c) & (y_pred == c)).sum()
            fn = ((y_true == c) & (y_pred != c)).sum()
            if tp + fp + fn == 0:
                f1s.append(zero_division)
            else:
                prec = tp / (tp + fp) if (tp + fp) > 0 else zero_division
                rec = tp / (tp + fn) if (tp + fn) > 0 else zero_division
                f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else zero_division)
        f1s = np.array(f1s)
        if average == "macro":
            return float(f1s.mean())
        if average == "micro":
            tp_total = ((y_true == y_pred) & (np.isin(y_true, labels))).sum()
            fp_total = (np.isin(y_pred, labels) & (y_true != y_pred)).sum()
            fn_total = (np.isin(y_true, labels) & (y_pred != y_true)).sum()
            if tp_total + fp_total + fn_total == 0:
                return float(zero_division)
            prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else zero_division
            rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else zero_division
            return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else float(zero_division)
        if average == "binary":
            if len(labels) != 2:
                labels = [0, 1]
            pos = labels[1]
            tp = ((y_true == pos) & (y_pred == pos)).sum()
            fp = ((y_true != pos) & (y_pred == pos)).sum()
            fn = ((y_true == pos) & (y_pred != pos)).sum()
            if tp + fp + fn == 0:
                return float(zero_division)
            prec = tp / (tp + fp) if (tp + fp) > 0 else zero_division
            rec = tp / (tp + fn) if (tp + fn) > 0 else zero_division
            return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else float(zero_division)
        if average is None:
            return f1s
        return float(f1s.mean())

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.dynamic_encoder import DynamicEncoder
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.cross_temporal_detector import CrossTemporalHead
from bazi_model.tasks.element_regressor import ElementStrengthHead
from bazi_model.tasks.interaction_detector import InteractionHead
from bazi_model.tasks.ten_god_head import TenGodHead
from bazi_model.tasks.useful_god_classifier import UsefulGodHead

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)
_DYNAMIC_INPUT_KEYS = (
    "dayun_stem_id",
    "dayun_branch_id",
    "liunian_stem_id",
    "liunian_branch_id",
    "liuyue_stem_id",
    "liuyue_branch_id",
    "pillar_presence_mask",
)
# Interaction: common vs rare bit groups for F1 reporting.
# Index-based (default): per docs/binary_vector_and_positions_spec.md §2.1:
#   common = 0–29: stem combinations (present + is_pure), branch six (present + is_blocked),
#                  three harmony (full + half).
#   rare   = 30–63: stem/branch clashes, harms, punishments, destructions.
# Alternative: positive_rate = split by bit positive rate on val set (rare = rate < threshold).
INTERACTION_DIM = 64
INTERACTION_COMMON_END = 30  # 0..29
INTERACTION_RARE_START = 30  # 30..63
COMMON_BIT_INDICES = list(range(0, INTERACTION_COMMON_END))
RARE_BIT_INDICES = list(range(INTERACTION_RARE_START, INTERACTION_DIM))


def get_common_rare_indices_by_positive_rate(
    targets: np.ndarray,
    rare_threshold: float = 0.15,
) -> tuple[list[int], list[int]]:
    """
    Split 64 interaction bits into common (positive rate >= threshold) and rare (positive rate < threshold).

    Args:
        targets: (N, 64) binary targets from the evaluation set.
        rare_threshold: Bits with positive rate below this are "rare".

    Returns:
        (common_indices, rare_indices) in ascending order.
    """
    pos_rate = targets.mean(axis=0)
    rare = [i for i in range(INTERACTION_DIM) if pos_rate[i] < rare_threshold]
    common = [i for i in range(INTERACTION_DIM) if pos_rate[i] >= rare_threshold]
    return sorted(common), sorted(rare)


def _cast_inputs_float32(inputs: dict[str, torch.Tensor]) -> None:
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.float32)


def _get_eval_loader(
    data_path: Path,
    task_name: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int,
    temporal_only: bool = False,
) -> tuple[DataLoader, str]:
    """
    Return a DataLoader for the evaluation split and the split name ("val" or "test").

    When test_ratio > 0: train/val/test split (e.g. 80/10/10); evaluation is on the test set.
    When test_ratio == 0: train/val split; evaluation is on the val set (legacy behavior).
    Uses the same seed and ratios as training scripts for reproducible splits.
    """
    dataset = ChartDataset(data_path, task_name=task_name, temporal_only=temporal_only)
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    if test_ratio > 0:
        n_test = max(0, int(n * test_ratio))
        n_train = n - n_val - n_test
        if n_train < 1 or n_test < 1:
            raise ValueError(
                f"Split invalid: n={n}, val_ratio={val_ratio}, test_ratio={test_ratio} -> n_train={n_train}, n_test={n_test}. Use smaller val_ratio/test_ratio."
            )
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed),
        )
        eval_ds = test_ds
        eval_split = "test"
    else:
        n_train = n - n_val
        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        eval_ds = val_ds
        eval_split = "val"
    return (
        DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=chart_collate_fn_inputs_targets,
            num_workers=0,
        ),
        eval_split,
    )


def _print_metric(k: str, v) -> None:
    """Print a single metric key-value for console output."""
    if k == "confusion_matrix":
        print(f"  {k}: (see below)")
        for row in v:
            print("   ", row)
    elif k == "f1_per_class":
        print(f"  {k}: {v}")
    elif k == "ordinal_error_analysis" and isinstance(v, dict):
        print(f"  {k}:")
        for subk, subv in v.items():
            if isinstance(subv, float):
                print(f"    {subk}: {subv:.2f}")
            else:
                print(f"    {subk}: {subv}")
    elif k == "f1_per_segment" and isinstance(v, dict):
        print(f"  {k}:")
        for seg_name, f1_val in v.items():
            print(f"    {seg_name}: {f1_val:.4f}")
    else:
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")


def _compute_ordinal_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of samples where argsort(pred) equals argsort(true). pred/true (N, 5)."""
    if pred.size == 0:
        return 0.0
    pred_rank = np.argsort(np.argsort(pred, axis=1), axis=1)
    true_rank = np.argsort(np.argsort(true, axis=1), axis=1)
    return float(np.all(pred_rank == true_rank, axis=1).mean())


def _compute_ordinal_accuracy_relaxed(
    pred: np.ndarray, true: np.ndarray, margin_min: float = 0.02
) -> float:
    """
    Ordinal accuracy ignoring borderline pairs: sample is correct iff for every pair
    (i, j) with true[i] - true[j] >= margin_min we have pred[i] > pred[j].
    """
    if pred.size == 0:
        return 0.0
    n, k = pred.shape
    correct = np.ones(n, dtype=bool)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            margin = true[:, i] - true[:, j]
            clear = margin >= margin_min
            wrong_order = (pred[:, j] >= pred[:, i]) & clear
            correct &= ~wrong_order
    return float(correct.mean())


def _ordinal_error_analysis(
    pred: np.ndarray, true: np.ndarray, balanced_threshold: float = 0.05
) -> dict:
    """
    Analyze wrong-ordinal cases: how many have very balanced true strengths (spread <= threshold)?
    pred, true: (N, 5). Returns counts and fractions for reporting.
    """
    if pred.size == 0:
        return {
            "n_wrong_ordinal": 0,
            "n_balanced_wrong": 0,
            "n_unbalanced_wrong": 0,
            "pct_balanced_among_wrong": 0.0,
        }
    pred_rank = np.argsort(np.argsort(pred, axis=1), axis=1)
    true_rank = np.argsort(np.argsort(true, axis=1), axis=1)
    ordinal_correct = np.all(pred_rank == true_rank, axis=1)
    wrong_mask = ~ordinal_correct
    n_wrong = int(wrong_mask.sum())
    if n_wrong == 0:
        return {
            "n_wrong_ordinal": 0,
            "n_balanced_wrong": 0,
            "n_unbalanced_wrong": 0,
            "pct_balanced_among_wrong": 0.0,
        }
    # True spread per sample: max - min of the 5 element strengths
    true_spread = true.max(axis=1) - true.min(axis=1)
    balanced = true_spread <= balanced_threshold
    n_balanced_wrong = int((wrong_mask & balanced).sum())
    n_unbalanced_wrong = n_wrong - n_balanced_wrong
    return {
        "n_wrong_ordinal": n_wrong,
        "n_balanced_wrong": n_balanced_wrong,
        "n_unbalanced_wrong": n_unbalanced_wrong,
        "pct_balanced_among_wrong": 100.0 * n_balanced_wrong / n_wrong,
    }


# ---------- Ten-god ----------


def _eval_ten_god(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            _cast_inputs_float32(inputs)
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            for k in encoder_kw:
                encoder_kw[k] = encoder_kw[k].to(device)
            natal_emb = model.encoder(**encoder_kw)
            logits = model.head(natal_emb)
            pred = logits.argmax(dim=-1)
            target = targets["ten_god_ids"].to(device)
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    acc = (pred_flat == target_flat).mean()
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(11)))
    f1_macro = f1_score(target_flat, pred_flat, average="macro", zero_division=0)
    f1_per_class = f1_score(
        target_flat, pred_flat, average=None, labels=list(range(11)), zero_division=0
    )
    f1_per_class = np.atleast_1d(f1_per_class)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


# ---------- Element strength ----------


def _eval_element_strength(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            _cast_inputs_float32(inputs)
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            for k in encoder_kw:
                encoder_kw[k] = encoder_kw[k].to(device)
            natal_emb = model.encoder(**encoder_kw)
            probs = model.head(natal_emb)
            target = targets["elemental_strengths"].to(device)
            all_pred.append(probs.cpu().numpy())
            all_target.append(target.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_target, axis=0)
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    ordinal = _compute_ordinal_accuracy(pred, true)
    ordinal_relaxed = _compute_ordinal_accuracy_relaxed(pred, true, margin_min=0.02)
    ordinal_analysis = _ordinal_error_analysis(pred, true, balanced_threshold=0.05)
    return {
        "mse": mse,
        "mae": mae,
        "ordinal_accuracy": ordinal,
        "ordinal_accuracy_relaxed": ordinal_relaxed,
        "ordinal_error_analysis": ordinal_analysis,
    }


# ---------- Interaction ----------


def _eval_interaction(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    common_rare_mode: str = "index",
    rare_threshold: float = 0.15,
) -> dict:
    """
    Evaluate interaction head: binary accuracy, micro/macro F1, and F1 for common vs rare bits.

    common_rare_mode: "index" = common 0–29, rare 30–63 (per binary_vector spec);
                      "positive_rate" = split by bit positive rate on val set (rare = rate < rare_threshold).
    """
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            _cast_inputs_float32(inputs)
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            for k in encoder_kw:
                encoder_kw[k] = encoder_kw[k].to(device)
            natal_emb = model.encoder(**encoder_kw)
            logits = model.head(natal_emb)
            pred = (logits.sigmoid() >= 0.5).float()
            target = targets["interaction_vector"].to(device)
            if target.dtype != pred.dtype:
                target = target.to(pred.dtype)
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    binary_acc = (pred_flat == target_flat).mean()
    f1_micro = f1_score(target_flat, pred_flat, average="micro", zero_division=0)
    # Macro: per-bit F1 then average over 64 bits
    f1_per_bit = [
        f1_score(target[:, b], pred[:, b], average="binary", zero_division=0)
        for b in range(INTERACTION_DIM)
    ]
    f1_macro = float(np.mean(f1_per_bit))

    if common_rare_mode == "positive_rate":
        common_inds, rare_inds = get_common_rare_indices_by_positive_rate(
            target, rare_threshold=rare_threshold
        )
    else:
        common_inds = COMMON_BIT_INDICES
        rare_inds = RARE_BIT_INDICES

    def _f1_for_indices(indices: list[int]) -> float:
        if not indices:
            return 0.0
        t_sub = target[:, indices].reshape(-1)
        p_sub = pred[:, indices].reshape(-1)
        return float(f1_score(t_sub, p_sub, average="binary", zero_division=0))

    f1_common = _f1_for_indices(common_inds)
    f1_rare = _f1_for_indices(rare_inds)

    return {
        "binary_accuracy": float(binary_acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_common_bits": float(f1_common),
        "f1_rare_bits": float(f1_rare),
        "n_common_bits": len(common_inds),
        "n_rare_bits": len(rare_inds),
        "common_rare_mode": common_rare_mode,
    }


# ---------- Useful god ----------


def _eval_useful_god(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            _cast_inputs_float32(inputs)
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            for k in encoder_kw:
                encoder_kw[k] = encoder_kw[k].to(device)
            natal_emb = model.natal_encoder(**encoder_kw)
            logits = model.head(natal_emb)
            pred = logits.argmax(dim=-1)
            target = targets["useful_god_id"].squeeze(-1).to(device)
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    acc_all = (pred == target).mean()
    unamb = target != 5
    acc_unamb = (pred[unamb] == target[unamb]).mean() if unamb.any() else 0.0
    cm = confusion_matrix(target, pred, labels=list(range(6)))
    f1_macro = f1_score(target, pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc_all),
        "accuracy_unambiguous": float(acc_unamb),
        "n_unambiguous": int(unamb.sum()),
        "confusion_matrix": cm.tolist(),
        "f1_macro": float(f1_macro),
    }


# ---------- Cross-temporal ----------


# Cross-temporal 61 binary dims: layout in eval array is [0:60] then dim 65.
# Segment ranges (in the 61-dim array): natal×大运, natal×流年, natal×流月, 大运×流年, 大运×流月, 流年×流月, useful_god.
CROSS_TEMPORAL_SEGMENTS = [
    (0, 16, "natal×大运"),
    (16, 32, "natal×流年"),
    (32, 48, "natal×流月"),
    (48, 52, "大运×流年"),
    (52, 56, "大运×流月"),
    (56, 60, "流年×流月"),
    (60, 61, "useful_god_active"),
]


def _eval_cross_temporal(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    bi = model.head._binary_idx.cpu().numpy()
    all_pred_bin, all_target_bin = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            _cast_inputs_float32(inputs)
            natal_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            dynamic_kw = {k: inputs[k] for k in _DYNAMIC_INPUT_KEYS if k in inputs}
            for d in (natal_kw, dynamic_kw):
                for k in d:
                    d[k] = d[k].to(device)
            natal_emb = model.natal_encoder(**natal_kw)
            dynamic_emb = model.dynamic_encoder(**dynamic_kw)
            logits = model.head(natal_emb, dynamic_emb)
            pred_bin = (logits[:, bi].sigmoid() >= 0.5).float()
            target = targets["dynamic_interaction_vector"].to(device)
            if target.dtype != pred_bin.dtype:
                target = target.to(pred_bin.dtype)
            target_bin = target[:, bi]
            all_pred_bin.append(pred_bin.cpu().numpy())
            all_target_bin.append(target_bin.cpu().numpy())
    pred_bin = np.concatenate(all_pred_bin, axis=0)
    target_bin = np.concatenate(all_target_bin, axis=0)
    n_bits = pred_bin.shape[1]
    pred_flat = pred_bin.reshape(-1)
    target_flat = target_bin.reshape(-1)
    binary_acc = (pred_flat == target_flat).mean()
    f1_micro = f1_score(target_flat, pred_flat, average="micro", zero_division=0)
    # Macro: per-bit binary F1 then average (same as natal interaction)
    f1_per_bit = [
        f1_score(target_bin[:, b], pred_bin[:, b], average="binary", zero_division=0)
        for b in range(n_bits)
    ]
    f1_macro = float(np.mean(f1_per_bit))
    f1_per_segment = {}
    for start, end, name in CROSS_TEMPORAL_SEGMENTS:
        if end > n_bits:
            continue
        t_sub = target_bin[:, start:end].reshape(-1)
        p_sub = pred_bin[:, start:end].reshape(-1)
        f1_seg = float(f1_score(t_sub, p_sub, average="binary", zero_division=0))
        f1_per_segment[name] = f1_seg
    return {
        "binary_accuracy": float(binary_acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_per_segment": f1_per_segment,
    }


# ---------- Model building and checkpoint loading ----------


def _build_ten_god(
    device: torch.device,
    d_model: int = 256,
    num_layers: int = 2,
) -> torch.nn.Module:
    class TenGodModel(torch.nn.Module):
        def __init__(self, d: int, nlay: int):
            super().__init__()
            self.encoder = NatalEncoder(d_model=d, num_layers=nlay)
            self.head = TenGodHead(d_model=d)

        def forward(self, **kwargs):
            natal_emb = self.encoder(**kwargs)
            return self.head(natal_emb)

    return TenGodModel(d_model, num_layers).to(device)


def _build_element_strength(
    device: torch.device,
    d_model: int = 256,
    num_layers: int = 2,
) -> torch.nn.Module:
    class ElementStrengthModel(torch.nn.Module):
        def __init__(self, d: int, nlay: int):
            super().__init__()
            self.encoder = NatalEncoder(d_model=d, num_layers=nlay)
            self.head = ElementStrengthHead(d_natal=d, n_elements=5)

        def forward(self, **kwargs):
            natal_emb = self.encoder(**kwargs)
            return self.head(natal_emb)

    return ElementStrengthModel(d_model, num_layers).to(device)


def _build_interaction(
    device: torch.device,
    d_model: int = 256,
    num_layers: int = 2,
) -> torch.nn.Module:
    class InteractionModel(torch.nn.Module):
        def __init__(self, d: int, nlay: int):
            super().__init__()
            self.encoder = NatalEncoder(d_model=d, num_layers=nlay)
            self.head = InteractionHead(d_in=d, d_out=64, pos_weight=None)

        def forward(self, **kwargs):
            natal_emb = self.encoder(**kwargs)
            return self.head(natal_emb)

    return InteractionModel(d_model, num_layers).to(device)


def _build_useful_god(
    device: torch.device,
    d_model: int = 256,
    num_layers: int = 2,
) -> torch.nn.Module:
    class UsefulGodModel(torch.nn.Module):
        def __init__(self, d: int, nlay: int):
            super().__init__()
            self.natal_encoder = NatalEncoder(d_model=d, num_layers=nlay)
            self.head = UsefulGodHead(input_dim=d, n_classes=6, class_weight=None)

        def forward(self, **kwargs):
            natal_emb = self.natal_encoder(**kwargs)
            return self.head(natal_emb)

    return UsefulGodModel(d_model, num_layers).to(device)


def _build_cross_temporal(device: torch.device) -> torch.nn.Module:
    class CrossTemporalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.natal_encoder = NatalEncoder()
            self.dynamic_encoder = DynamicEncoder()
            self.head = CrossTemporalHead()

        def forward(self, inputs):
            natal_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            dynamic_kw = {k: inputs[k] for k in _DYNAMIC_INPUT_KEYS if k in inputs}
            natal_emb = self.natal_encoder(**natal_kw)
            dynamic_emb = self.dynamic_encoder(**dynamic_kw)
            return natal_emb, dynamic_emb

    return CrossTemporalModel().to(device)


def _load_ten_god_checkpoint(
    model: torch.nn.Module, ckpt_path: Path, device: torch.device
) -> bool:
    """Returns True if loaded successfully."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    try:
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        return True
    except (RuntimeError, KeyError):
        return False


def _load_element_strength_checkpoint(
    model: torch.nn.Module, ckpt_path: Path, device: torch.device
) -> bool:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        return True
    except (RuntimeError, KeyError):
        return False


def _load_interaction_checkpoint(
    model: torch.nn.Module, ckpt_path: Path, device: torch.device
) -> bool:
    """Load checkpoint. Returns True if loaded successfully, False if format incompatible."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "encoder" not in ckpt or "head" not in ckpt:
        return False
    try:
        model.encoder.load_state_dict(ckpt["encoder"], strict=True)
        # strict=False: checkpoint may contain _pos_weight buffer from training; not needed for eval
        model.head.load_state_dict(ckpt["head"], strict=False)
        return True
    except (RuntimeError, KeyError):
        return False


def _load_useful_god_checkpoint(
    model: torch.nn.Module, ckpt_path: Path, device: torch.device
) -> bool:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "model_state_dict" not in ckpt:
            return False
        # strict=False: checkpoint may contain head.class_weight buffer from training; not needed for eval
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return True
    except (RuntimeError, KeyError):
        return False


def _load_cross_temporal_checkpoint(
    model: torch.nn.Module, ckpt_path: Path, device: torch.device
) -> bool:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        for key in ("natal_encoder", "dynamic_encoder", "head"):
            if key not in ckpt:
                return False
        model.natal_encoder.load_state_dict(ckpt["natal_encoder"], strict=True)
        model.dynamic_encoder.load_state_dict(ckpt["dynamic_encoder"], strict=True)
        # strict=False so checkpoints trained with segment_weights/pos_weight buffers still load
        model.head.load_state_dict(ckpt["head"], strict=False)
        return True
    except (RuntimeError, KeyError):
        return False


def _infer_encoder_arch(encoder_state_dict: dict) -> tuple[int, int]:
    """Infer (d_model, num_layers) from encoder state_dict for old checkpoints."""
    num_layers = 2
    d_model = 256
    for key in encoder_state_dict:
        if ".layers." in key:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    num_layers = max(num_layers, int(parts[i + 1]) + 1)
                    break
        if "pillar_proj.weight" in key:
            # (d_model, pillar_dim)
            d_model = encoder_state_dict[key].shape[0]
    return d_model, num_layers


def _load_multi_task_natal_models(
    ckpt_path: Path,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "d_model" in ckpt and "num_layers" in ckpt:
        d_model = ckpt["d_model"]
        num_layers = ckpt["num_layers"]
    else:
        d_model, num_layers = _infer_encoder_arch(ckpt["encoder"])

    models = {}
    # Ten-god
    m_tg = _build_ten_god(device, d_model=d_model, num_layers=num_layers)
    m_tg.encoder.load_state_dict(ckpt["encoder"], strict=True)
    m_tg.head.load_state_dict(ckpt["ten_god_head"], strict=True)
    models["ten_god"] = m_tg
    # Element strength
    m_el = _build_element_strength(device, d_model=d_model, num_layers=num_layers)
    m_el.encoder.load_state_dict(ckpt["encoder"], strict=True)
    m_el.head.load_state_dict(ckpt["element_head"], strict=True)
    models["element_strength"] = m_el
    # Interaction
    m_int = _build_interaction(device, d_model=d_model, num_layers=num_layers)
    m_int.encoder.load_state_dict(ckpt["encoder"], strict=True)
    # strict=False: checkpoint may contain _pos_weight buffer from training; not needed for eval
    m_int.head.load_state_dict(ckpt["interaction_head"], strict=False)
    models["interaction"] = m_int
    # Useful god
    m_ug = _build_useful_god(device, d_model=d_model, num_layers=num_layers)
    m_ug.natal_encoder.load_state_dict(ckpt["encoder"], strict=True)
    m_ug.head.load_state_dict(ckpt["useful_god_head"], strict=True)
    models["useful_god"] = m_ug
    return models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 1 checkpoints on validation set."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/charts.jsonl"),
        help="Path to charts.jsonl",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing task checkpoints",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation fraction (must match training)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test fraction for held-out evaluation (e.g. 0.1 = 80/10/10 train/val/test). When > 0, metrics are reported on test set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for eval")
    parser.add_argument(
        "--use_multi_task",
        action="store_true",
        help="Load natal tasks from multi_task_best.pt instead of single-task checkpoints",
    )
    parser.add_argument(
        "--multi_task_checkpoint",
        type=Path,
        default=None,
        help="When --use_multi_task: path to .pt file (default: checkpoint_dir/multi_task_best.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON",
    )
    parser.add_argument(
        "--common_rare",
        type=str,
        default="index",
        choices=["index", "positive_rate"],
        help="Interaction common/rare split: 'index' (0-29 vs 30-63 per spec) or 'positive_rate' (by val-set bit rate)",
    )
    parser.add_argument(
        "--rare_threshold",
        type=float,
        default=0.15,
        help="When --common_rare=positive_rate, bits with positive rate < this are rare",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    eval_split_name = None
    if args.use_multi_task:
        mt_path = args.multi_task_checkpoint or (args.checkpoint_dir / "multi_task_best.pt")
        if not mt_path.exists():
            print(f"Multi-task checkpoint not found: {mt_path}", file=sys.stderr)
            return 1
        natal_models = _load_multi_task_natal_models(mt_path, device)
        for task_name, model in natal_models.items():
            loader, eval_split_name = _get_eval_loader(
                args.data,
                "multi",
                args.val_ratio,
                args.test_ratio,
                args.seed,
                args.batch_size,
            )
            if task_name == "ten_god":
                results["ten_god"] = _eval_ten_god(model, loader, device)
            elif task_name == "element_strength":
                results["element_strength"] = _eval_element_strength(model, loader, device)
            elif task_name == "interaction":
                base_metrics = _eval_interaction(
                    model,
                    loader,
                    device,
                    common_rare_mode="index",
                    rare_threshold=args.rare_threshold,
                )
                pr_metrics = _eval_interaction(
                    model,
                    loader,
                    device,
                    common_rare_mode="positive_rate",
                    rare_threshold=args.rare_threshold,
                )
                results["interaction"] = {
                    "index": base_metrics,
                    "positive_rate": pr_metrics,
                }
            elif task_name == "useful_god":
                results["useful_god"] = _eval_useful_god(model, loader, device)
    else:
        # Ten-god
        ckpt_tg = args.checkpoint_dir / "ten_god_best.pt"
        if ckpt_tg.exists():
            model = _build_ten_god(device)
            if _load_ten_god_checkpoint(model, ckpt_tg, device):
                loader, eval_split_name = _get_eval_loader(
                    args.data,
                    "ten_god",
                    args.val_ratio,
                    args.test_ratio,
                    args.seed,
                    args.batch_size,
                )
                results["ten_god"] = _eval_ten_god(model, loader, device)
            else:
                print(f"Skip ten_god: checkpoint format incompatible with current model")
        else:
            print(f"Skip ten_god: {ckpt_tg} not found")

        # Element strength
        ckpt_el = args.checkpoint_dir / "element_strength_best.pt"
        if ckpt_el.exists():
            model = _build_element_strength(device)
            if _load_element_strength_checkpoint(model, ckpt_el, device):
                loader, eval_split_name = _get_eval_loader(
                    args.data,
                    "element_strength",
                    args.val_ratio,
                    args.test_ratio,
                    args.seed,
                    args.batch_size,
                )
                results["element_strength"] = _eval_element_strength(model, loader, device)
            else:
                print(f"Skip element_strength: checkpoint format incompatible")
        else:
            print(f"Skip element_strength: {ckpt_el} not found")

        # Interaction
        ckpt_int = args.checkpoint_dir / "interaction_best.pt"
        if ckpt_int.exists():
            model = _build_interaction(device)
            if _load_interaction_checkpoint(model, ckpt_int, device):
                loader, eval_split_name = _get_eval_loader(
                    args.data,
                    "interaction",
                    args.val_ratio,
                    args.test_ratio,
                    args.seed,
                    args.batch_size,
                )
                base_metrics = _eval_interaction(
                    model,
                    loader,
                    device,
                    common_rare_mode="index",
                    rare_threshold=args.rare_threshold,
                )
                pr_metrics = _eval_interaction(
                    model,
                    loader,
                    device,
                    common_rare_mode="positive_rate",
                    rare_threshold=args.rare_threshold,
                )
                results["interaction"] = {
                    "index": base_metrics,
                    "positive_rate": pr_metrics,
                }
            else:
                print(f"Skip interaction: checkpoint format incompatible with current model")
        else:
            print(f"Skip interaction: {ckpt_int} not found")

        # Useful god
        ckpt_ug = args.checkpoint_dir / "useful_god_best.pt"
        if ckpt_ug.exists():
            model = _build_useful_god(device)
            if _load_useful_god_checkpoint(model, ckpt_ug, device):
                loader, eval_split_name = _get_eval_loader(
                    args.data,
                    "useful_god",
                    args.val_ratio,
                    args.test_ratio,
                    args.seed,
                    args.batch_size,
                )
                results["useful_god"] = _eval_useful_god(model, loader, device)
            else:
                print(f"Skip useful_god: checkpoint format incompatible")
        else:
            print(f"Skip useful_god: {ckpt_ug} not found")

    # Cross-temporal (always single-task; different data)
    ckpt_ct = args.checkpoint_dir / "cross_temporal_best.pt"
    if ckpt_ct.exists():
        model = _build_cross_temporal(device)
        if _load_cross_temporal_checkpoint(model, ckpt_ct, device):
            loader, eval_split_name = _get_eval_loader(
                args.data,
                "cross_temporal",
                args.val_ratio,
                args.test_ratio,
                args.seed,
                args.batch_size,
                temporal_only=True,
            )
            results["cross_temporal"] = _eval_cross_temporal(model, loader, device)
        else:
            print(f"Skip cross_temporal: checkpoint format incompatible")
    else:
        print(f"Skip cross_temporal: {ckpt_ct} not found")

    if results and eval_split_name is not None:
        results["eval_split"] = eval_split_name

    # Print and optionally save
    print("\n=== Phase 1 evaluation results ===\n")
    if results.get("eval_split"):
        print(f"Evaluated on: {results['eval_split']} set (val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, seed={args.seed})\n")
    for task, metrics in results.items():
        if task == "eval_split":
            continue
        print(f"--- {task} ---")
        if isinstance(metrics, dict) and "index" in metrics and "positive_rate" in metrics:
            for mode in ("index", "positive_rate"):
                print(f"  [{mode}]")
                for k, v in metrics[mode].items():
                    _print_metric(k, v)
        else:
            for k, v in (metrics.items() if isinstance(metrics, dict) else []):
                _print_metric(k, v)
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
