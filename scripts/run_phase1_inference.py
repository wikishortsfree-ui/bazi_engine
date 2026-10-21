#!/usr/bin/env python3
"""
Run Phase 1 trained models on a single chart from birthdate, location, and optional current date.

Same inputs as the bazi_engine CLI: birth date/time, timezone (or lat/lng), optional current-date
and gender for 大运/流年/流月. Computes the chart then runs inference with loaded checkpoints
and prints model outputs (ten-god, element strength, interaction bits, useful god; cross-temporal
if --current-date is given).

Usage:
  python scripts/run_phase1_inference.py --date "1990-05-15 14:30:00" --tz 8
  python scripts/run_phase1_inference.py --date "1990-05-15 14:30:00" --tz 8 --current-date 2024-06-01 --gender 1
  python scripts/run_phase1_inference.py --date "1990-05-15" --tz -8 --lat 37.8 --lng -122.4 --checkpoint_dir checkpoints --use_multi_task
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch

from bazi_engine import BaziCalculator
from bazi_engine.encodings import TEN_GOD_TO_ID, USEFUL_GOD_TO_ID
from bazi_engine.models import BaziChart
from bazi_engine.strength import ELEMENTS
from bazi_engine.temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)
from bazi_model.encoders.dynamic_encoder import DynamicEncoder
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.cross_temporal_detector import CrossTemporalHead
from bazi_model.tasks.element_regressor import ElementStrengthHead
from bazi_model.tasks.interaction_detector import InteractionHead
from bazi_model.tasks.ten_god_head import TenGodHead
from bazi_model.tasks.useful_god_classifier import UsefulGodHead

DEFAULT_LAT, DEFAULT_LNG = 39.9, 116.4

# Reverse mappings for readable output
ID_TO_TEN_GOD = {v: k for k, v in TEN_GOD_TO_ID.items()}
ID_TO_USEFUL_GOD = {v: k for k, v in USEFUL_GOD_TO_ID.items()}
# Useful god head: 6 classes; 0–4 = 比劫,食伤,财,官杀,印; 5 = ambiguous
USEFUL_GOD_LABELS = ["比劫", "食伤", "财", "官杀", "印", "ambiguous"]

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


def _parse_date(s: str, tz_offset: int) -> datetime:
    s = s.strip()
    tz = timezone(timedelta(hours=tz_offset))
    if " " in s:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.replace(tzinfo=tz)


def _build_chart(
    birth_date: str,
    tz_offset: int,
    lat: float,
    lng: float,
    current_date: str | None,
    gender: int,
) -> BaziChart:
    """Build BaziChart; add temporal context and cross_interactions if current_date is set."""
    dt = _parse_date(birth_date, tz_offset)
    chart = BaziCalculator.from_datetime(dt, lat, lng)
    if current_date is None:
        return chart
    cur = datetime.strptime(current_date, "%Y-%m-%d").date()
    dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
    dayun = dayun_seq.get_period_for_year(cur.year)
    liunian = compute_liunian(cur.year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(cur.year, cur.month, day_master_stem=chart.day_pillar.stem.value)
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems = [
        chart.year_pillar.stem.value,
        chart.month_pillar.stem.value,
        chart.day_pillar.stem.value,
        chart.hour_pillar.stem.value,
    ]
    branches = [
        chart.year_pillar.branch.value,
        chart.month_pillar.branch.value,
        chart.day_pillar.branch.value,
        chart.hour_pillar.branch.value,
    ]
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
    return BaziChart(
        birth_dt=chart.birth_dt,
        solar_dt=chart.solar_dt,
        lat=chart.lat,
        lng=chart.lng,
        year_pillar=chart.year_pillar,
        month_pillar=chart.month_pillar,
        day_pillar=chart.day_pillar,
        hour_pillar=chart.hour_pillar,
        sect=chart.sect,
        temporal_context=ctx,
        cross_interactions=cross,
    )


def _arrays_to_tensors(features: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert numpy feature dict to torch tensors (long for int, float32 for float)."""
    out = {}
    for key, arr in features.items():
        if arr.dtype == np.int64 or (hasattr(arr.dtype, "kind") and arr.dtype.kind == "i"):
            out[key] = torch.from_numpy(arr).long()
        else:
            out[key] = torch.from_numpy(arr).float()
    return out


def _chart_to_batch(chart: BaziChart, task_name: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Convert chart to batched inputs (and optionally targets) for the given task."""
    inputs = chart.to_input_features(task_name)
    inputs_t = _arrays_to_tensors(inputs)
    batched = {k: v.unsqueeze(0) for k, v in inputs_t.items()}
    targets = chart.to_ground_truth(task_name) if task_name != "multi" else None
    if targets is not None:
        targets_t = _arrays_to_tensors(targets)
        batched_targets = {k: v.unsqueeze(0) for k, v in targets_t.items()}
        return batched, batched_targets
    return batched, None


def _cast_inputs_float32(inputs: dict[str, torch.Tensor]) -> None:
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.float32)


# ---------- Model build / load (mirror evaluate_phase1) ----------


def _infer_encoder_arch(encoder_state_dict: dict) -> tuple[int, int]:
    """Infer (d_model, num_layers) from encoder state_dict (e.g. multi-task checkpoint)."""
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
            d_model = encoder_state_dict[key].shape[0]
    return d_model, num_layers


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
            return self.head(natal_emb, dynamic_emb)

    return CrossTemporalModel().to(device)


def _load_multi_task_models(ckpt_path: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "d_model" in ckpt and "num_layers" in ckpt:
        d_model = int(ckpt["d_model"])
        num_layers = int(ckpt["num_layers"])
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
    m_int.head.load_state_dict(ckpt["interaction_head"], strict=False)
    models["interaction"] = m_int
    # Useful god
    m_ug = _build_useful_god(device, d_model=d_model, num_layers=num_layers)
    m_ug.natal_encoder.load_state_dict(ckpt["encoder"], strict=True)
    m_ug.head.load_state_dict(ckpt["useful_god_head"], strict=True)
    models["useful_god"] = m_ug
    return models


def _load_single_checkpoints(ckpt_dir: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    """Load task checkpoints; skip any that fail (e.g. different encoder layout)."""
    models = {}
    # Ten-god (checkpoint may contain head.class_weight buffer from training)
    p = ckpt_dir / "ten_god_best.pt"
    if p.exists():
        try:
            m = _build_ten_god(device)
            ckpt = torch.load(p, map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            m.load_state_dict(state, strict=False)
            models["ten_god"] = m
        except (RuntimeError, KeyError):
            pass
    # Element strength
    p = ckpt_dir / "element_strength_best.pt"
    if p.exists():
        try:
            m = _build_element_strength(device)
            ckpt = torch.load(p, map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            m.load_state_dict(state, strict=False)
            models["element_strength"] = m
        except (RuntimeError, KeyError):
            pass
    # Interaction (saved as encoder + head; encoder may differ from current NatalEncoder)
    p = ckpt_dir / "interaction_best.pt"
    if p.exists():
        try:
            m = _build_interaction(device)
            ckpt = torch.load(p, map_location=device, weights_only=True)
            if "encoder" in ckpt and "head" in ckpt:
                m.encoder.load_state_dict(ckpt["encoder"], strict=True)
                m.head.load_state_dict(ckpt["head"], strict=False)
                models["interaction"] = m
        except (RuntimeError, KeyError):
            pass
    # Useful god
    p = ckpt_dir / "useful_god_best.pt"
    if p.exists():
        try:
            m = _build_useful_god(device)
            ckpt = torch.load(p, map_location=device, weights_only=True)
            if "model_state_dict" in ckpt:
                m.load_state_dict(ckpt["model_state_dict"], strict=False)
                models["useful_god"] = m
        except (RuntimeError, KeyError):
            pass
    return models


def _load_cross_temporal(ckpt_dir: Path, device: torch.device) -> torch.nn.Module | None:
    p = ckpt_dir / "cross_temporal_best.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location=device, weights_only=True)
    for key in ("natal_encoder", "dynamic_encoder", "head"):
        if key not in ckpt:
            return None
    m = _build_cross_temporal(device)
    m.natal_encoder.load_state_dict(ckpt["natal_encoder"], strict=True)
    m.dynamic_encoder.load_state_dict(ckpt["dynamic_encoder"], strict=True)
    m.head.load_state_dict(ckpt["head"], strict=True)
    return m


# ---------- Print helpers ----------


def _print_chart_summary(chart: BaziChart) -> None:
    tz_off = chart.birth_dt.utcoffset()
    tz_str = f"UTC{tz_off.total_seconds()/3600:+.0f}" if tz_off else "UTC"
    print("八字命盘")
    print("━" * 40)
    print(f"出生时间: {chart.birth_dt.strftime('%Y-%m-%d %H:%M:%S')} ({tz_str})")
    print(f"地点: lat={chart.lat}, lng={chart.lng}")
    print("四柱:")
    for label, pillar in [
        ("年柱", chart.year_pillar),
        ("月柱", chart.month_pillar),
        ("日柱", chart.day_pillar),
        ("时柱", chart.hour_pillar),
    ]:
        tg = pillar.ten_god_stem.value
        print(f"  {label}: {pillar}  ({pillar.stem.element}{pillar.branch.element})  十神: {tg}")
    print()


def _print_ten_god_pred(pred_ids: np.ndarray, labels: tuple[str, ...] = ("年", "月", "日", "时")) -> None:
    print("模型输出 — 十神 (per position):")
    for i, pos in enumerate(labels):
        tid = int(pred_ids[i])
        name = ID_TO_TEN_GOD.get(tid, f"id_{tid}")
        print(f"  {pos}柱: {name} ({tid})")
    print()


def _print_element_strength_pred(pred: np.ndarray) -> None:
    print("模型输出 — 五行强度 (predicted fractions):")
    for i, e in enumerate(ELEMENTS):
        print(f"  {e}: {pred[i]:.4f}")
    print(f"  sum: {pred.sum():.4f}")
    print()


def _print_interaction_pred(pred_bits: np.ndarray) -> None:
    n = int(pred_bits.sum())
    on_bits = [i for i in range(64) if pred_bits[i] >= 0.5]
    print("模型输出 — 相互关系 (64-dim binary):")
    print(f"  predicted on: {n} bits")
    print(f"  bit indices: {on_bits[:30]}{'...' if len(on_bits) > 30 else ''}")
    print()


def _print_useful_god_pred(class_id: int) -> None:
    label = USEFUL_GOD_LABELS[class_id] if 0 <= class_id < 6 else f"id_{class_id}"
    print("模型输出 — 用神:")
    print(f"  {label} ({class_id})")
    print()


def _print_cross_temporal_pred(
    pred_bin: np.ndarray,
    pred_cont: np.ndarray,
    useful_active: float,
) -> None:
    n_bin = int((pred_bin >= 0.5).sum())
    print("模型输出 — 跨时空 (大运/流年/流月):")
    print(f"  binary dims (0–59, 65) on: {n_bin} / 61")
    print(f"  elemental_delta (木火土金水): {[f'{pred_cont[i]:.3f}' for i in range(5)]}")
    print(f"  useful_god_active: {useful_active:.3f}")
    print()


def get_predictions(
    chart: BaziChart,
    natal_models: dict[str, torch.nn.Module],
    cross_temporal_model: torch.nn.Module | None,
    device: torch.device,
) -> dict:
    """
    Run phase1 model inference on a chart. Returns dict of predictions (no printing).

    Keys: "ten_god" (np (4,)), "element_strength" (np (5,)), "interaction" (np (64,)),
    "useful_god" (int), "cross_temporal" (dict with "binary", "continuous", "useful_active") or None.
    Missing tasks are omitted from the dict.
    """
    task_name = "multi"
    inputs, _ = _chart_to_batch(chart, task_name)
    _cast_inputs_float32(inputs)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    out: dict = {}
    with torch.no_grad():
        if "ten_god" in natal_models:
            m = natal_models["ten_god"]
            m.eval()
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            natal_emb = m.encoder(**encoder_kw)
            logits = m.head(natal_emb)
            out["ten_god"] = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        if "element_strength" in natal_models:
            m = natal_models["element_strength"]
            m.eval()
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            natal_emb = m.encoder(**encoder_kw)
            out["element_strength"] = m.head(natal_emb).squeeze(0).cpu().numpy()
        if "interaction" in natal_models:
            m = natal_models["interaction"]
            m.eval()
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            natal_emb = m.encoder(**encoder_kw)
            logits = m.head(natal_emb)
            out["interaction"] = (logits.sigmoid() >= 0.5).float().squeeze(0).cpu().numpy()
        if "useful_god" in natal_models:
            m = natal_models["useful_god"]
            m.eval()
            encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
            natal_emb = m.natal_encoder(**encoder_kw)
            logits = m.head(natal_emb)
            out["useful_god"] = logits.argmax(dim=-1).squeeze().item()
        if cross_temporal_model is not None and chart.temporal_context is not None:
            inp_ct, _ = _chart_to_batch(chart, "cross_temporal")
            _cast_inputs_float32(inp_ct)
            for k, v in inp_ct.items():
                inp_ct[k] = v.to(device)
            m = cross_temporal_model
            m.eval()
            bi = m.head._binary_idx.cpu().numpy()
            natal_kw = {k: inp_ct[k] for k in _NATAL_INPUT_KEYS if k in inp_ct}
            dynamic_kw = {k: inp_ct[k] for k in _DYNAMIC_INPUT_KEYS if k in inp_ct}
            natal_emb = m.natal_encoder(**natal_kw)
            dynamic_emb = m.dynamic_encoder(**dynamic_kw)
            raw = m.head(natal_emb, dynamic_emb)
            out["cross_temporal"] = {
                "binary": raw[:, bi].sigmoid().squeeze(0).cpu().numpy(),
                "continuous": raw[:, 60:65].squeeze(0).cpu().numpy(),
                "useful_active": raw[:, 65].sigmoid().squeeze().item(),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 models on a single chart (birthdate, location, optional current date)."
    )
    parser.add_argument("--date", required=True, help="Birth datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--tz", type=int, default=8, help="Timezone offset in hours (default 8)")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude")
    parser.add_argument("--lng", type=float, default=DEFAULT_LNG, help="Longitude")
    parser.add_argument(
        "--current-date",
        default=None,
        help="Date for 大运/流年/流月 (YYYY-MM-DD); enables cross-temporal model",
    )
    parser.add_argument(
        "--gender",
        type=int,
        default=1,
        choices=(0, 1),
        help="Gender for 大运 direction: 0=female, 1=male (default 1)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing phase1 checkpoints",
    )
    parser.add_argument(
        "--use_multi_task",
        action="store_true",
        help="Load natal tasks from multi_task_best.pt",
    )
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

    # Load natal models
    if args.use_multi_task:
        mt_path = ckpt_dir / "multi_task_best.pt"
        if not mt_path.exists():
            print(f"Multi-task checkpoint not found: {mt_path}", file=sys.stderr)
            return 1
        natal_models = _load_multi_task_models(mt_path, device)
    else:
        natal_models = _load_single_checkpoints(ckpt_dir, device)

    if not natal_models:
        print("No natal checkpoints loaded. Place multi_task_best.pt or task-specific best.pt in checkpoint_dir.", file=sys.stderr)
        return 1

    cross_temporal_model = _load_cross_temporal(ckpt_dir, device) if args.current_date else None

    # Chart summary
    _print_chart_summary(chart)

    pred = get_predictions(chart, natal_models, cross_temporal_model, device)
    if "ten_god" in pred:
        _print_ten_god_pred(pred["ten_god"])
    if "element_strength" in pred:
        _print_element_strength_pred(pred["element_strength"])
    if "interaction" in pred:
        _print_interaction_pred(pred["interaction"])
    if "useful_god" in pred:
        _print_useful_god_pred(pred["useful_god"])
    if "cross_temporal" in pred:
        ct = pred["cross_temporal"]
        _print_cross_temporal_pred(ct["binary"], ct["continuous"], ct["useful_active"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
