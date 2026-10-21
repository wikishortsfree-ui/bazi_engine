#!/usr/bin/env python3
"""
Compare Phase 1 model outputs with bazi_engine (rule-based) results for a single chart.

Uses the same inputs as run_phase1_inference: birth date/time, location, optional current-date
and gender. Builds the chart, gets engine ground truth from BaziChart.to_ground_truth(),
runs model inference, and prints a side-by-side comparison with match/mismatch.

Usage:
  python scripts/compare_phase1_with_engine.py --date "1990-05-15 14:30:00" --tz 8
  python scripts/compare_phase1_with_engine.py --date "1990-05-15 14:30:00" --tz 8 --current-date 2024-06-01 --gender 1
  python scripts/compare_phase1_with_engine.py --date "1990-05-15" --tz -8 --checkpoint_dir checkpoints --use_multi_task
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_scripts = Path(__file__).resolve().parent
for p in (_root, _scripts):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from bazi_engine.encodings import TEN_GOD_TO_ID
from bazi_engine.models import BaziChart
from bazi_engine.strength import ELEMENTS

# Reuse chart building, model loading, and inference from run_phase1_inference
from run_phase1_inference import (
    USEFUL_GOD_LABELS,
    _build_chart,
    _load_cross_temporal,
    _load_multi_task_models,
    _load_single_checkpoints,
    get_predictions,
)

ID_TO_TEN_GOD = {v: k for k, v in TEN_GOD_TO_ID.items()}
POSITION_LABELS = ("年", "月", "日", "时")


def _get_engine_truth(chart: BaziChart) -> dict:
    """Get bazi_engine ground truth for all tasks (same as to_ground_truth)."""
    engine = {}
    tg = chart.to_ground_truth("ten_god")
    engine["ten_god"] = tg["ten_god_ids"]
    el = chart.to_ground_truth("element_strength")
    engine["element_strength"] = np.array([el["elemental_strengths"][i] for i in range(5)], dtype=np.float64)
    ir = chart.to_ground_truth("interaction")
    engine["interaction"] = ir["interaction_vector"]
    ug = chart.to_ground_truth("useful_god")
    engine["useful_god"] = int(ug["useful_god_id"].item())
    if chart.temporal_context is not None and chart.cross_interactions is not None:
        try:
            ct = chart.to_ground_truth("cross_temporal")
            engine["cross_temporal"] = ct["dynamic_interaction_vector"].squeeze(0)
        except Exception:
            engine["cross_temporal"] = None
    else:
        engine["cross_temporal"] = None
    return engine


def _compare_ten_god(engine: np.ndarray, pred: np.ndarray) -> None:
    print("十神 (per position)")
    print("  Position   Engine (bazi_engine)   Model (phase1)   Match")
    print("  " + "-" * 58)
    all_match = True
    for i, pos in enumerate(POSITION_LABELS):
        e_id = int(engine[i])
        p_id = int(pred[i])
        e_name = ID_TO_TEN_GOD.get(e_id, f"id_{e_id}")
        p_name = ID_TO_TEN_GOD.get(p_id, f"id_{p_id}")
        match = "✓" if e_id == p_id else "✗"
        if e_id != p_id:
            all_match = False
        print(f"  {pos}柱       {e_name} ({e_id})              {p_name} ({p_id})       {match}")
    print(f"  Overall: {'All match' if all_match else 'Mismatch'}")
    print()


def _compare_element_strength(engine: np.ndarray, pred: np.ndarray) -> None:
    print("五行强度 (element fractions)")
    print("  Element   Engine (bazi_engine)   Model (phase1)   Diff")
    print("  " + "-" * 55)
    for i, e in enumerate(ELEMENTS):
        eng = float(engine[i])
        mod = float(pred[i])
        diff = mod - eng
        print(f"  {e}        {eng:.4f}                 {mod:.4f}           {diff:+.4f}")
    mae = float(np.abs(pred.astype(np.float64) - engine).mean())
    print(f"  MAE: {mae:.4f}")
    print()


def _compare_interaction(engine: np.ndarray, pred: np.ndarray) -> None:
    eng_bin = (engine >= 0.5).astype(np.float32)
    pred_bin = (pred >= 0.5).astype(np.float32)
    match = (eng_bin == pred_bin).sum()
    total = 64
    print("相互关系 (64-dim binary)")
    print(f"  Engine bits on: {int(eng_bin.sum())}  |  Model bits on: {int(pred_bin.sum())}")
    print(f"  Agreement: {match}/{total} ({100 * match / total:.1f}%)")
    if match < total:
        diff_eng_only = [i for i in range(64) if eng_bin[i] >= 0.5 and pred_bin[i] < 0.5]
        diff_pred_only = [i for i in range(64) if pred_bin[i] >= 0.5 and eng_bin[i] < 0.5]
        if diff_eng_only:
            print(f"  Engine=1, Model=0 (missed): bits {diff_eng_only[:20]}{'...' if len(diff_eng_only) > 20 else ''}")
        if diff_pred_only:
            print(f"  Engine=0, Model=1 (extra): bits {diff_pred_only[:20]}{'...' if len(diff_pred_only) > 20 else ''}")
    print()


def _compare_useful_god(engine: int, pred: int) -> None:
    e_label = USEFUL_GOD_LABELS[engine] if 0 <= engine < 6 else f"id_{engine}"
    p_label = USEFUL_GOD_LABELS[pred] if 0 <= pred < 6 else f"id_{pred}"
    match = "✓" if engine == pred else "✗"
    print("用神 (useful god)")
    print(f"  Engine (bazi_engine): {e_label} ({engine})")
    print(f"  Model (phase1):       {p_label} ({pred})  {match}")
    print()


def _compare_cross_temporal(
    engine: np.ndarray,
    pred: dict,
    binary_indices: np.ndarray,
) -> None:
    # engine: (66,) dynamic_interaction_vector
    # pred: {"binary": (61,), "continuous": (5,), "useful_active": float}
    pred_bin = pred["binary"]
    pred_cont = pred["continuous"]
    pred_active = pred["useful_active"]
    eng_bin = np.array([engine[i] for i in binary_indices], dtype=np.float64)
    eng_cont = engine[60:65]
    eng_active = engine[65]

    bin_match = (np.abs(pred_bin - eng_bin) < 0.5).sum()  # same after threshold
    pred_bin_hard = (pred_bin >= 0.5).astype(np.float64)
    eng_bin_hard = (eng_bin >= 0.5).astype(np.float64)
    agreement = (pred_bin_hard == eng_bin_hard).sum()
    print("跨时空 (cross-temporal, 大运/流年/流月)")
    print(f"  Binary dims (61): agreement {agreement}/61 ({100 * agreement / 61:.1f}%)")
    print("  Elemental delta (木火土金水):")
    print("    Engine:", [f"{eng_cont[i]:.3f}" for i in range(5)])
    print("    Model:", [f"{pred_cont[i]:.3f}" for i in range(5)])
    mae_cont = float(np.abs(np.array(pred_cont) - eng_cont).mean())
    print(f"    MAE: {mae_cont:.4f}")
    print(f"  useful_god_active: Engine={eng_active:.3f}  Model={pred_active:.3f}  Match={'✓' if (eng_active >= 0.5) == (pred_active >= 0.5) else '✗'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 model outputs with bazi_engine for a single chart."
    )
    parser.add_argument("--date", required=True, help="Birth datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--tz", type=int, default=8, help="Timezone offset in hours (default 8)")
    parser.add_argument("--lat", type=float, default=39.9, help="Latitude")
    parser.add_argument("--lng", type=float, default=116.4, help="Longitude")
    parser.add_argument("--current-date", default=None, help="Date for 大运/流年/流月 (YYYY-MM-DD)")
    parser.add_argument("--gender", type=int, default=1, choices=(0, 1), help="Gender: 0=female, 1=male (default 1)")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument("--use_multi_task", action="store_true", help="Load from multi_task_best.pt")
    args = parser.parse_args()

    try:
        chart = _build_chart(
            args.date,
            args.tz,
            args.lat,
            args.lng,
            args.current_date,
            args.gender,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.checkpoint_dir
    if not ckpt_dir.is_dir():
        print(f"Checkpoint dir not found: {ckpt_dir}", file=sys.stderr)
        return 1

    if args.use_multi_task:
        mt_path = ckpt_dir / "multi_task_best.pt"
        if not mt_path.exists():
            print(f"Multi-task checkpoint not found: {mt_path}", file=sys.stderr)
            return 1
        natal_models = _load_multi_task_models(mt_path, device)
    else:
        natal_models = _load_single_checkpoints(ckpt_dir, device)

    if not natal_models:
        print("No natal checkpoints loaded.", file=sys.stderr)
        return 1

    cross_temporal_model = _load_cross_temporal(ckpt_dir, device) if args.current_date else None

    engine = _get_engine_truth(chart)
    pred = get_predictions(chart, natal_models, cross_temporal_model, device)

    # Print chart summary
    tz_off = chart.birth_dt.utcoffset()
    tz_str = f"UTC{tz_off.total_seconds()/3600:+.0f}" if tz_off else "UTC"
    print("Chart:", chart.birth_dt.strftime("%Y-%m-%d %H:%M:%S"), tz_str, f"lat={chart.lat}, lng={chart.lng}")
    if chart.temporal_context:
        print("Current date (大运/流年/流月):", args.current_date)
    print()
    print("=" * 60)
    print("Comparison: bazi_engine (rule-based) vs Phase 1 models")
    print("=" * 60)
    print()

    if "ten_god" in pred:
        _compare_ten_god(engine["ten_god"], pred["ten_god"])
    if "element_strength" in pred:
        _compare_element_strength(engine["element_strength"], pred["element_strength"])
    if "interaction" in pred:
        _compare_interaction(engine["interaction"], pred["interaction"])
    if "useful_god" in pred:
        _compare_useful_god(engine["useful_god"], pred["useful_god"])
    if "cross_temporal" in pred and engine.get("cross_temporal") is not None:
        from bazi_model.tasks.cross_temporal_detector import BINARY_INDICES
        bi = np.array(BINARY_INDICES, dtype=np.int64)
        _compare_cross_temporal(engine["cross_temporal"], pred["cross_temporal"], bi)

    return 0


if __name__ == "__main__":
    sys.exit(main())
