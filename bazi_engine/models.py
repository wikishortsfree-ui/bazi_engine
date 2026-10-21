"""
Data models for bāzì (八字) chart results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from ._constants import NULL_PILLAR_ID, POSITION_LABELS
from .encodings import (
    BRANCH_TO_ID,
    ELEMENT_TO_ID,
    STEM_TO_ID,
    TEN_GOD_TO_ID,
    USEFUL_GOD_TO_ID,
    branch_polarity,
    stem_polarity,
)
from .enums import Branch, Pillar, Stem, TenGod
from .temporal import (
    CrossInteractionResult,
    DaYun,
    LiuNian,
    LiuYue,
    TemporalContext,
)
from .tables import get_hidden_stems, get_ten_god_for_stem
from .strength import (
    ELEMENTS,
    classify_day_master,
    compute_day_master_strength,
    compute_element_strengths,
)
from .interactions import InteractionResult, detect_all_interactions

from .elements import (
    ELEMENT_CONTROLS as _ELEMENT_CONTROLS,
    ELEMENT_CONTROLS_ME as _ELEMENT_CONTROLS_ME,
    ELEMENT_GENERATES as _ELEMENT_GENERATES,
    ELEMENT_GENERATES_ME as _ELEMENT_GENERATES_ME,
)
_ELEMENT_TO_ENGLISH = {"木": "wood", "火": "fire", "土": "earth", "金": "metal", "水": "water"}
_ENGLISH_TO_ELEMENT = {v: k for k, v in _ELEMENT_TO_ENGLISH.items()}
_HARMONY_ELEMENT_TO_WUXING = {"Water": "水", "Fire": "火", "Metal": "金", "Wood": "木"}


@dataclass
class BaziChart:
    """Four-pillar (八字) chart for a birth moment at a given location."""

    birth_dt: datetime  # original input datetime (timezone-aware)
    solar_dt: datetime  # true solar time used for calculation (naive)
    lat: float
    lng: float

    year_pillar: Pillar
    month_pillar: Pillar
    day_pillar: Pillar
    hour_pillar: Pillar

    # 流派: 1 = 陰曆, 2 = 盲派 (default). Affects day-master and related logic.
    sect: int = 2

    temporal_context: Optional[TemporalContext] = None
    cross_interactions: Optional[CrossInteractionResult] = None

    def to_feature_arrays(self) -> dict[str, np.ndarray]:
        """
        Export chart as raw feature arrays for encoder input.

        Returns dict with stem_ids, branch_ids, pillar_positions, stem_elements,
        branch_elements, polarities, ten_god IDs, interaction_vector (64),
        elemental_strengths (5), day_master_strength, useful_god_id. When
        temporal context is present, adds dayun/liunian/liuyue IDs,
        dynamic_interaction_vector, elemental_delta, useful_god_active,
        pillar_presence_mask.
        """
        stems, branches = self._stems_and_branches()
        result: dict[str, np.ndarray] = {}

        result["stem_ids"] = np.array([STEM_TO_ID[s] for s in stems], dtype=np.int64)
        result["branch_ids"] = np.array([BRANCH_TO_ID[b] for b in branches], dtype=np.int64)
        result["pillar_positions"] = np.array([0, 1, 2, 3], dtype=np.int64)
        result["stem_elements"] = np.array(
            [ELEMENT_TO_ID[p.stem.element] for p in (self.year_pillar, self.month_pillar, self.day_pillar, self.hour_pillar)],
            dtype=np.int64,
        )
        result["branch_elements"] = np.array(
            [ELEMENT_TO_ID[p.branch.element] for p in (self.year_pillar, self.month_pillar, self.day_pillar, self.hour_pillar)],
            dtype=np.int64,
        )
        result["stem_polarities"] = np.array(
            [stem_polarity(s) for s in stems], dtype=np.int64
        )
        result["branch_polarities"] = np.array(
            [branch_polarity(b) for b in branches], dtype=np.int64
        )
        result["ten_god_stem_ids"] = np.array(
            [
                TEN_GOD_TO_ID[self.year_pillar.ten_god_stem.value],
                TEN_GOD_TO_ID[self.month_pillar.ten_god_stem.value],
                TEN_GOD_TO_ID[self.day_pillar.ten_god_stem.value],
                TEN_GOD_TO_ID[self.hour_pillar.ten_god_stem.value],
            ],
            dtype=np.int64,
        )
        ten_god_branch_flat: list[int] = []
        dm_stem = self.day_pillar.stem
        for pillar in (self.year_pillar, self.month_pillar, self.day_pillar, self.hour_pillar):
            hiddens = get_hidden_stems(pillar.branch)
            for i in range(3):
                if i < len(hiddens):
                    tg = get_ten_god_for_stem(dm_stem, hiddens[i][0])
                    ten_god_branch_flat.append(TEN_GOD_TO_ID[tg.value])
                else:
                    ten_god_branch_flat.append(-1)
        result["ten_god_branch_ids"] = np.array(ten_god_branch_flat, dtype=np.int64)
        assert result["ten_god_branch_ids"].shape == (12,)

        natal_ir = detect_all_interactions(stems, branches)
        result["interaction_vector"] = natal_ir.to_binary_vector().astype(np.float64)
        assert result["interaction_vector"].shape == (64,)

        el_strengths = compute_element_strengths(stems, branches)
        result["elemental_strengths"] = np.array(
            [el_strengths[e] for e in ELEMENTS], dtype=np.float64
        )
        dm_pct = compute_day_master_strength(stems, branches)
        result["day_master_strength"] = np.array([dm_pct], dtype=np.float64)
        useful_god_dict = _compute_useful_god(dm_pct, self.day_pillar.stem.element)
        if useful_god_dict is not None and not useful_god_dict.get("is_ambiguous", True):
            primary = useful_god_dict.get("primary", "")
            useful_id = USEFUL_GOD_TO_ID.get(primary, -1)
        else:
            useful_id = -1
        result["useful_god_id"] = np.array([useful_id], dtype=np.int64)

        ctx = self.temporal_context
        if ctx is not None:
            result["dayun_stem_id"] = np.array(
                [STEM_TO_ID[ctx.dayun.stem] if ctx.dayun else -1], dtype=np.int64
            )
            result["dayun_branch_id"] = np.array(
                [BRANCH_TO_ID[ctx.dayun.branch] if ctx.dayun else -1], dtype=np.int64
            )
            result["liunian_stem_id"] = np.array(
                [STEM_TO_ID[ctx.liunian.stem] if ctx.liunian else -1], dtype=np.int64
            )
            result["liunian_branch_id"] = np.array(
                [BRANCH_TO_ID[ctx.liunian.branch] if ctx.liunian else -1], dtype=np.int64
            )
            result["liuyue_stem_id"] = np.array(
                [STEM_TO_ID[ctx.liuyue.stem] if ctx.liuyue else -1], dtype=np.int64
            )
            result["liuyue_branch_id"] = np.array(
                [BRANCH_TO_ID[ctx.liuyue.branch] if ctx.liuyue else -1], dtype=np.int64
            )
            result["pillar_presence_mask"] = np.array(
                [
                    1 if ctx.dayun else 0,
                    1 if ctx.liunian else 0,
                    1 if ctx.liuyue else 0,
                ],
                dtype=np.int64,
            )
        if ctx is not None and self.cross_interactions is not None:
            cr = self.cross_interactions
            dyn_vec = _dynamic_interaction_vector(cr)
            result["dynamic_interaction_vector"] = dyn_vec
            result["elemental_delta"] = np.array(
                [cr.elemental_delta[e] for e in ELEMENTS], dtype=np.float64
            )
            active = 1 if (cr.useful_god_activation == "active") else 0
            result["useful_god_active"] = np.array([active], dtype=np.int64)

        return result

    def to_input_features(self, task_name: str) -> dict[str, np.ndarray]:
        """
        Return only the minimal input features needed for a specific task.
        These are the raw chart structure the encoder must learn from.
        Never include ground truth targets (ten-gods, interactions, strengths, useful_god).

        Args:
            task_name: One of 'ten_god', 'element_strength', 'interaction', 'useful_god', 'cross_temporal'.

        Returns:
            Dictionary of numpy arrays with minimal input features only.
        """
        stems, branches = self._stems_and_branches()
        pillars = (self.year_pillar, self.month_pillar, self.day_pillar, self.hour_pillar)
        base: dict[str, np.ndarray] = {
            "stem_ids": np.array([STEM_TO_ID[s] for s in stems], dtype=np.int64),
            "branch_ids": np.array([BRANCH_TO_ID[b] for b in branches], dtype=np.int64),
            "stem_elements": np.array(
                [ELEMENT_TO_ID[p.stem.element] for p in pillars], dtype=np.int64
            ),
            "stem_polarities": np.array(
                [stem_polarity(s) for s in stems], dtype=np.int64
            ),
            "pillar_positions": np.array([0, 1, 2, 3], dtype=np.int64),
        }
        if task_name == "cross_temporal":
            if self.temporal_context is None:
                raise ValueError(
                    "cross_temporal task requires temporal_context, but chart has none"
                )
            ctx = self.temporal_context
            base["dayun_stem_id"] = np.array(
                [STEM_TO_ID[ctx.dayun.stem] if ctx.dayun else -1], dtype=np.int64
            )
            base["dayun_branch_id"] = np.array(
                [BRANCH_TO_ID[ctx.dayun.branch] if ctx.dayun else -1], dtype=np.int64
            )
            base["liunian_stem_id"] = np.array(
                [STEM_TO_ID[ctx.liunian.stem] if ctx.liunian else -1], dtype=np.int64
            )
            base["liunian_branch_id"] = np.array(
                [BRANCH_TO_ID[ctx.liunian.branch] if ctx.liunian else -1], dtype=np.int64
            )
            base["liuyue_stem_id"] = np.array(
                [STEM_TO_ID[ctx.liuyue.stem] if ctx.liuyue else -1], dtype=np.int64
            )
            base["liuyue_branch_id"] = np.array(
                [BRANCH_TO_ID[ctx.liuyue.branch] if ctx.liuyue else -1], dtype=np.int64
            )
            base["pillar_presence_mask"] = np.array(
                [
                    1 if ctx.dayun else 0,
                    1 if ctx.liunian else 0,
                    1 if ctx.liuyue else 0,
                ],
                dtype=np.int64,
            )
        elif task_name not in ("ten_god", "element_strength", "interaction", "useful_god", "multi"):
            raise ValueError(f"Unknown task_name: {task_name!r}")
        return base

    def to_ground_truth(self, task_name: str) -> dict[str, np.ndarray]:
        """
        Return ground truth targets for validation/training loss for a specific task.
        These are the answers the model must learn to predict.

        Args:
            task_name: One of 'ten_god', 'element_strength', 'interaction', 'useful_god', 'cross_temporal'.

        Returns:
            Dictionary of numpy arrays with ground truth targets.
        """
        stems, branches = self._stems_and_branches()
        pillars = (self.year_pillar, self.month_pillar, self.day_pillar, self.hour_pillar)
        if task_name == "ten_god":
            ten_god_ids = np.array(
                [TEN_GOD_TO_ID[p.ten_god_stem.value] for p in pillars],
                dtype=np.int64,
            )
            return {"ten_god_ids": ten_god_ids}
        if task_name == "element_strength":
            el_strengths = compute_element_strengths(stems, branches)
            elemental_strengths = np.array(
                [el_strengths[e] for e in ELEMENTS], dtype=np.float64
            )
            return {"elemental_strengths": elemental_strengths}
        if task_name == "interaction":
            natal_ir = detect_all_interactions(stems, branches)
            interaction_vector = natal_ir.to_binary_vector().astype(np.float64)
            return {"interaction_vector": interaction_vector}
        if task_name == "useful_god":
            dm_pct = compute_day_master_strength(stems, branches)
            useful_god_dict = _compute_useful_god(dm_pct, self.day_pillar.stem.element)
            if useful_god_dict is not None and not useful_god_dict.get("is_ambiguous", True):
                primary = useful_god_dict.get("primary", "")
                useful_id = USEFUL_GOD_TO_ID.get(primary, -1)
            else:
                useful_id = -1
            # 6-class head: 0-4 for types, 5 for ambiguous (-1)
            useful_god_id = 5 if useful_id == -1 else useful_id
            return {"useful_god_id": np.array([useful_god_id], dtype=np.int64)}
        if task_name == "cross_temporal":
            if self.cross_interactions is None:
                raise ValueError(
                    "cross_temporal task requires cross_interactions, but chart has none"
                )
            dynamic_interaction_vector = _dynamic_interaction_vector(self.cross_interactions)
            return {"dynamic_interaction_vector": dynamic_interaction_vector}
        if task_name == "multi":
            # All four natal targets for multi-task training
            tg = self.to_ground_truth("ten_god")
            el = self.to_ground_truth("element_strength")
            ir = self.to_ground_truth("interaction")
            ug = self.to_ground_truth("useful_god")
            return {**tg, **el, **ir, **ug}
        raise ValueError(f"Unknown task_name: {task_name!r}")

    def to_prompt_text(self) -> str:
        """Compact deterministic 简体中文 summary of the chart for hybrid prompts.

        Used as the structured-text block in the
        ``[virtual tokens] + structured text + question`` prompt format
        (see ``plans/hybrid-prompt-v1-and-lora_*.plan.md`` §3). Output shape::

            八字结构：
            年柱：庚申  月柱：丙午  日柱：壬寅  时柱：戊申
            十神：年=七杀(申=偏印)  月=偏财(午=劫财)  日主=壬水  时=七杀(申=偏印)
            五行：水50.0% 木18.0% 火15.0% 土10.0% 金7.0%
            日主：身强  用神：土(官杀)
            主要交互：子午冲、申子半合

        The format is deterministic for a given ``BaziChart`` so it can be
        cached / replayed across runs.
        """
        stems, branches = self._stems_and_branches()
        pillars = (self.year_pillar, self.month_pillar, self.day_pillar, self.hour_pillar)
        labels = ("年", "月", "日", "时")

        pillars_seg = "  ".join(
            f"{lbl}柱：{p.stem.value}{p.branch.value}" for lbl, p in zip(labels, pillars)
        )

        ten_gods_parts: list[str] = []
        for lbl, p in zip(labels, pillars):
            branch_tg = p.ten_god_branch[0].value if p.ten_god_branch else "—"
            if lbl == "日":
                ten_gods_parts.append(f"日主={p.stem.value}{p.stem.element}")
            else:
                ten_gods_parts.append(
                    f"{lbl}={p.ten_god_stem.value}({p.branch.value}={branch_tg})"
                )

        el_strengths = compute_element_strengths(stems, branches)
        sorted_elems = sorted(el_strengths.items(), key=lambda kv: kv[1], reverse=True)
        five_elems = " ".join(f"{e}{v * 100:.1f}%" for e, v in sorted_elems)

        classification = classify_day_master(stems, branches)
        dm_pct = compute_day_master_strength(stems, branches)
        ug = _compute_useful_god(dm_pct, self.day_pillar.stem.element)
        if ug is not None and not ug.get("is_ambiguous", True):
            ug_str = f"{ug['primary_element']}({ug['primary']})"
        else:
            ug_str = "中和"

        natal_ir = detect_all_interactions(stems, branches)
        interactions_summary = _summarize_interactions_short(natal_ir) or "无"

        lines = [
            "八字结构：",
            pillars_seg,
            "十神：" + "  ".join(ten_gods_parts),
            "五行：" + five_elems,
            f"日主：{classification}  用神：{ug_str}",
            "主要交互：" + interactions_summary,
        ]
        return "\n".join(lines)

    @property
    def year_str(self) -> str:
        return str(self.year_pillar)

    @property
    def month_str(self) -> str:
        return str(self.month_pillar)

    @property
    def day_str(self) -> str:
        return str(self.day_pillar)

    @property
    def hour_str(self) -> str:
        return str(self.hour_pillar)

    def _stems_and_branches(self) -> tuple[list[str], list[str]]:
        """Return (stems[4], branches[4]) for year, month, day, hour."""
        pillars = (
            self.year_pillar,
            self.month_pillar,
            self.day_pillar,
            self.hour_pillar,
        )
        stems = [p.stem.value for p in pillars]
        branches = [p.branch.value for p in pillars]
        return stems, branches

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the complete BaziChart to JSON string.

        Args:
            indent: JSON indentation level (default 2, use None for compact).

        Returns:
            JSON string with all chart data (metadata, natal_pillars, day_master,
            ten_gods, hidden_stems, elemental_strengths, useful_god, interactions, dynamic).
        """
        data = self._to_json_dict()
        return json.dumps(data, ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> BaziChart:
        """
        Deserialize a BaziChart from a JSON string produced by to_json().

        Reconstructs pillars from natal_pillars and ten_gods, and optionally
        temporal_context and cross_interactions from the dynamic section, so that
        the returned chart yields the same to_feature_arrays() as the original.
        """
        data = json.loads(json_str)
        metadata = data.get("metadata") or {}
        natal_pillars = data.get("natal_pillars") or {}
        ten_gods = data.get("ten_gods") or {}
        dynamic = data.get("dynamic")

        birth_str = metadata.get("birth_datetime", "")
        if not birth_str:
            raise ValueError("metadata.birth_datetime is required")
        birth_dt = datetime.fromisoformat(
            birth_str.replace("Z", "+00:00")
        )
        offset_hours = metadata.get("timezone_offset", 0)
        if birth_dt.tzinfo is None:
            birth_dt = birth_dt.replace(tzinfo=timezone(timedelta(hours=offset_hours)))
        lat = float(metadata.get("lat", 0.0))
        lng = float(metadata.get("lng", 0.0))
        solar_dt = birth_dt.replace(tzinfo=None)

        position_map = ("年", "月", "日", "時")
        pillar_keys = ("year", "month", "day", "hour")
        pillars_list = []
        for key, pos in zip(pillar_keys, position_map):
            np_dict = natal_pillars.get(key) or {}
            tg_stem = ten_gods.get(f"{key}_stem", "")
            tg_branch = ten_gods.get(f"{key}_branch") or []
            p = _deserialize_pillar(np_dict, tg_stem, tg_branch, pos)
            pillars_list.append(p)
        year_pillar, month_pillar, day_pillar, hour_pillar = pillars_list

        temporal_context = None
        cross_interactions = None
        if dynamic:
            temporal_context, cross_interactions = _parse_dynamic(dynamic)

        return cls(
            birth_dt=birth_dt,
            solar_dt=solar_dt,
            lat=lat,
            lng=lng,
            year_pillar=year_pillar,
            month_pillar=month_pillar,
            day_pillar=day_pillar,
            hour_pillar=hour_pillar,
            sect=int(metadata.get("sect", 2)),
            temporal_context=temporal_context,
            cross_interactions=cross_interactions,
        )

    def _to_json_dict(self) -> dict[str, Any]:
        """Build the full JSON-serializable dict for this chart."""
        stems, branches = self._stems_and_branches()
        tz = self.birth_dt.tzinfo
        offset_seconds = tz.utcoffset(self.birth_dt).total_seconds() if tz else 0
        timezone_offset = int(offset_seconds / 3600)

        metadata = {
            "birth_datetime": self.birth_dt.isoformat(),
            "timezone_offset": timezone_offset,
            "lat": self.lat,
            "lng": self.lng,
            "sect": self.sect,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "calculator_version": "0.1.0",
        }

        natal_pillars = {
            "year": _serialize_pillar(self.year_pillar),
            "month": _serialize_pillar(self.month_pillar),
            "day": _serialize_pillar(self.day_pillar),
            "hour": _serialize_pillar(self.hour_pillar),
        }

        day_master_stem = stems[2]
        day_master_element = self.day_pillar.stem.element
        strength_pct = compute_day_master_strength(stems, branches)
        classification = classify_day_master(stems, branches)
        day_master = {
            "stem": day_master_stem,
            "element": day_master_element,
            "strength_percentile": round(strength_pct, 4),
            "classification": classification,
        }

        ten_gods = {
            "year_stem": self.year_pillar.ten_god_stem.value,
            "month_stem": self.month_pillar.ten_god_stem.value,
            "day_stem": self.day_pillar.ten_god_stem.value,
            "hour_stem": self.hour_pillar.ten_god_stem.value,
            "year_branch": [tg.value for tg in self.year_pillar.ten_god_branch],
            "month_branch": [tg.value for tg in self.month_pillar.ten_god_branch],
            "day_branch": [tg.value for tg in self.day_pillar.ten_god_branch],
            "hour_branch": [tg.value for tg in self.hour_pillar.ten_god_branch],
        }

        dm_stem_enum = self.day_pillar.stem
        hidden_stems = {
            "year_branch": _hidden_stems_list(self.year_pillar.branch, dm_stem_enum),
            "month_branch": _hidden_stems_list(self.month_pillar.branch, dm_stem_enum),
            "day_branch": _hidden_stems_list(self.day_pillar.branch, dm_stem_enum),
            "hour_branch": _hidden_stems_list(self.hour_pillar.branch, dm_stem_enum),
        }

        el_strengths = compute_element_strengths(stems, branches)
        elemental_strengths = {_ELEMENT_TO_ENGLISH[e]: round(el_strengths[e], 4) for e in ELEMENTS}

        useful_god = _compute_useful_god(strength_pct, day_master_element)

        natal_interactions = detect_all_interactions(stems, branches)
        interactions = _serialize_interactions(natal_interactions)

        result = {
            "metadata": metadata,
            "natal_pillars": natal_pillars,
            "day_master": day_master,
            "ten_gods": ten_gods,
            "hidden_stems": hidden_stems,
            "elemental_strengths": elemental_strengths,
            "useful_god": useful_god,
            "interactions": interactions,
            "dynamic": _serialize_dynamic(self.temporal_context, self.cross_interactions),
        }
        return result


def _deserialize_pillar(
    np_dict: dict[str, Any],
    ten_god_stem_val: str,
    ten_god_branch_vals: list[Any],
    position: str,
) -> Pillar:
    """Build a Pillar from JSON natal_pillar + ten_god stem/branch."""
    stem_char = np_dict.get("stem", "")
    branch_char = np_dict.get("branch", "")
    if not stem_char or not branch_char:
        raise ValueError(f"Pillar must have stem and branch, got {np_dict!r}")
    stem = Stem(stem_char)
    branch = Branch(branch_char)
    ten_god_stem = TenGod(ten_god_stem_val) if ten_god_stem_val else TenGod.日主
    ten_god_branch = [
        TenGod(x) for x in ten_god_branch_vals if x
    ]
    return Pillar(
        stem=stem,
        branch=branch,
        position=position,
        ten_god_stem=ten_god_stem,
        ten_god_branch=ten_god_branch,
    )


def _parse_dynamic(
    dynamic: dict[str, Any],
) -> tuple[Optional[TemporalContext], Optional[CrossInteractionResult]]:
    """Parse dynamic section into TemporalContext and CrossInteractionResult."""
    ctx = _parse_temporal_context(dynamic)
    cross = _parse_cross_interactions(dynamic)
    return ctx, cross


def _parse_temporal_context(dynamic: dict[str, Any]) -> Optional[TemporalContext]:
    """Build TemporalContext from da_yun, liu_nian, liu_yue in dynamic."""
    dayun = None
    if "da_yun" in dynamic:
        dy = dynamic["da_yun"]
        dayun = DaYun(
            stem=dy.get("stem", ""),
            branch=dy.get("branch", ""),
            start_year=int(dy.get("start_year", 0)),
            end_year=int(dy.get("end_year", 0)),
            start_age=int(dy.get("start_age", 0)),
            end_age=int(dy.get("end_age", 0)),
            index=int(dy.get("index", 0)),
        )
    liunian = None
    if "liu_nian" in dynamic:
        ln = dynamic["liu_nian"]
        liunian = LiuNian(
            year=int(ln.get("year", 0)),
            stem=ln.get("stem", ""),
            branch=ln.get("branch", ""),
        )
    liuyue = None
    if "liu_yue" in dynamic:
        ly = dynamic["liu_yue"]
        liuyue = LiuYue(
            year=int(ly.get("year", 0)),
            month=int(ly.get("month", 0)),
            stem=ly.get("stem", ""),
            branch=ly.get("branch", ""),
            lunar_month=int(ly.get("lunar_month", 0)),
            year_stem=ly.get("year_stem", ""),
            year_branch=ly.get("year_branch", ""),
        )
    if dayun is None and liunian is None and liuyue is None:
        return None
    return TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)


def _parse_cross_interactions(
    dynamic: dict[str, Any],
) -> Optional[CrossInteractionResult]:
    """Build CrossInteractionResult from cross_natal_temporal_interactions and elemental_delta."""
    cross_data = dynamic.get("cross_natal_temporal_interactions")
    if not cross_data:
        return None
    elem_delta_raw = dynamic.get("elemental_delta") or {}
    elemental_delta = {
        _ENGLISH_TO_ELEMENT.get(k, k): float(v)
        for k, v in elem_delta_raw.items()
    }
    for e in ELEMENTS:
        if e not in elemental_delta:
            elemental_delta[e] = 0.0
    total_shift_magnitude = sum(abs(elemental_delta[e]) for e in ELEMENTS)

    def parse_natal_vs(ctx_key: str) -> tuple[list, list, list, list]:
        ctx = cross_data.get(ctx_key) or {}
        clashes = ctx.get("clashes") or []
        combinations = ctx.get("combinations") or []
        branch_combos = ctx.get("branch_combinations") or []
        harms = ctx.get("harms") or []

        def _with_pos(d: dict, extra: dict) -> dict:
            out = dict(extra)
            if d.get("natal_pos") is not None:
                out["natal_pos"] = d["natal_pos"]
            return out

        return (
            [_with_pos(r, {"item_a": r.get("stem_a"), "item_b": r.get("stem_b")}) for r in clashes],
            [_with_pos(r, {"stem_a": r.get("stem_a"), "stem_b": r.get("stem_b"), "result_element": r.get("result_element")}) for r in combinations],
            [_with_pos(r, {"branch_a": r.get("branch_a"), "branch_b": r.get("branch_b")}) for r in branch_combos],
            [_with_pos(r, {"branch_a": r.get("branch_a"), "branch_b": r.get("branch_b")}) for r in harms],
        )

    def parse_pair_vs(ctx_key: str) -> tuple[list, list, list, list]:
        ctx = cross_data.get(ctx_key) or {}
        clashes = ctx.get("clashes") or []
        combinations = ctx.get("combinations") or []
        branch_combos = ctx.get("branch_combinations") or []
        harms = ctx.get("harms") or []
        return (
            [{"item_a": r.get("stem_a"), "item_b": r.get("stem_b")} for r in clashes],
            combinations,
            [{"branch_a": r.get("branch_a"), "branch_b": r.get("branch_b")} for r in branch_combos],
            [{"item_a": r.get("branch_a"), "item_b": r.get("branch_b")} for r in harms],
        )

    nd_ss, nd_sc, nd_bc, nd_bf = parse_natal_vs("natal_vs_dayun")
    nn_ss, nn_sc, nn_bc, nn_bf = parse_natal_vs("natal_vs_liunian")
    ny_ss, ny_sc, ny_bc, ny_bf = parse_natal_vs("natal_vs_liuyue")
    dn_ss, dn_sc, dn_bc, dn_cl = parse_pair_vs("dayun_vs_liunian")
    dy_ly_ss, dy_ly_sc, dy_ly_bc, dy_ly_cl = parse_pair_vs("dayun_vs_liuyue")
    ln_ly_ss, ln_ly_sc, ln_ly_bc, ln_ly_cl = parse_pair_vs("liunian_vs_liuyue")

    useful_activation = None
    useful_analysis = None
    uga = dynamic.get("useful_god_activation")
    if uga is not None:
        useful_activation = "active" if uga.get("is_active") else "neutral"
        useful_analysis = uga.get("mechanism") or ""

    def any_interactions(*lists) -> bool:
        return any(len(lst) > 0 for lst in lists)

    return CrossInteractionResult(
        natal_dayun_stem_combinations=nd_sc,
        natal_dayun_stem_clashes=nd_ss,
        natal_dayun_branch_combinations=nd_bc,
        natal_dayun_branch_conflicts=nd_bf,
        natal_liunian_stem_combinations=nn_sc,
        natal_liunian_stem_clashes=nn_ss,
        natal_liunian_branch_combinations=nn_bc,
        natal_liunian_branch_conflicts=nn_bf,
        natal_liuyue_stem_combinations=ny_sc,
        natal_liuyue_stem_clashes=ny_ss,
        natal_liuyue_branch_combinations=ny_bc,
        natal_liuyue_branch_conflicts=ny_bf,
        natal_liuri_stem_combinations=[],
        natal_liuri_stem_clashes=[],
        natal_liuri_branch_combinations=[],
        natal_liuri_branch_conflicts=[],
        dayun_liunian_stem_combinations=dn_sc,
        dayun_liunian_stem_clashes=dn_ss,
        dayun_liunian_branch_combinations=dn_bc,
        dayun_liunian_branch_clashes=dn_cl,
        dayun_liuyue_stem_combinations=dy_ly_sc,
        dayun_liuyue_stem_clashes=dy_ly_ss,
        dayun_liuyue_branch_combinations=dy_ly_bc,
        dayun_liuyue_branch_clashes=dy_ly_cl,
        liunian_liuyue_stem_combinations=ln_ly_sc,
        liunian_liuyue_stem_clashes=ln_ly_ss,
        liunian_liuyue_branch_combinations=ln_ly_bc,
        liunian_liuyue_branch_clashes=ln_ly_cl,
        has_natal_dayun_interactions=any_interactions(nd_sc, nd_ss, nd_bc, nd_bf),
        has_natal_liunian_interactions=any_interactions(nn_sc, nn_ss, nn_bc, nn_bf),
        has_natal_liuyue_interactions=any_interactions(ny_sc, ny_ss, ny_bc, ny_bf),
        has_natal_liuri_interactions=False,
        elemental_delta=elemental_delta,
        total_shift_magnitude=total_shift_magnitude,
        useful_god_activation=useful_activation,
        useful_god_analysis=useful_analysis,
    )


# Cross-temporal interaction vector layout (66 dims): 60 binary + 5 elemental delta + 1 useful god.
# [0:16]   natal×大运 (4 natal × 4 types), [16:32] natal×流年, [32:48] natal×流月,
# [48:52]  大运×流年, [52:56] 大运×流月, [56:60] 流年×流月, [60:65] elemental_delta, [65] useful_god_active.
DYNAMIC_INTERACTION_DIM = 66


def _natal_vs_temporal_16_bits(
    sc: list,
    ss: list,
    bc: list,
    bf: list,
) -> np.ndarray:
    """16 bits: for each natal position 0..3, 4 bits (stem_comb, stem_clash, branch_comb, branch_conflict)."""
    vec = np.zeros(16, dtype=np.float64)
    for pos in range(4):
        if any(x.get("natal_pos") == pos for x in sc):
            vec[pos * 4 + 0] = 1.0
        if any(x.get("natal_pos") == pos for x in ss):
            vec[pos * 4 + 1] = 1.0
        if any(x.get("natal_pos") == pos for x in bc):
            vec[pos * 4 + 2] = 1.0
        if any(x.get("natal_pos") == pos for x in bf):
            vec[pos * 4 + 3] = 1.0
    return vec


def _pair_4_bits(sc: list, ss: list, bc: list, bf: list) -> np.ndarray:
    """4 bits: stem_comb, stem_clash, branch_comb, branch_conflict (any presence)."""
    vec = np.zeros(4, dtype=np.float64)
    if sc:
        vec[0] = 1.0
    if ss:
        vec[1] = 1.0
    if bc:
        vec[2] = 1.0
    if bf:
        vec[3] = 1.0
    return vec


def _dynamic_interaction_vector(cr: CrossInteractionResult) -> np.ndarray:
    """
    Flatten cross-context interactions + elemental delta + useful god into a 66-dim float64 vector.

    Layout: [0:16] natal×大运, [16:32] natal×流年, [32:48] natal×流月,
    [48:52] 大运×流年, [52:56] 大运×流月, [56:60] 流年×流月,
    [60:65] elemental_delta (5), [65] useful_god_activation (1).
    """
    vec = np.zeros(DYNAMIC_INTERACTION_DIM, dtype=np.float64)
    # Natal vs temporal (16 bits each)
    vec[0:16] = _natal_vs_temporal_16_bits(
        cr.natal_dayun_stem_combinations,
        cr.natal_dayun_stem_clashes,
        cr.natal_dayun_branch_combinations,
        cr.natal_dayun_branch_conflicts,
    )
    vec[16:32] = _natal_vs_temporal_16_bits(
        cr.natal_liunian_stem_combinations,
        cr.natal_liunian_stem_clashes,
        cr.natal_liunian_branch_combinations,
        cr.natal_liunian_branch_conflicts,
    )
    vec[32:48] = _natal_vs_temporal_16_bits(
        cr.natal_liuyue_stem_combinations,
        cr.natal_liuyue_stem_clashes,
        cr.natal_liuyue_branch_combinations,
        cr.natal_liuyue_branch_conflicts,
    )
    # Temporal vs temporal (4 bits each)
    vec[48:52] = _pair_4_bits(
        cr.dayun_liunian_stem_combinations,
        cr.dayun_liunian_stem_clashes,
        cr.dayun_liunian_branch_combinations,
        cr.dayun_liunian_branch_clashes,
    )
    vec[52:56] = _pair_4_bits(
        cr.dayun_liuyue_stem_combinations,
        cr.dayun_liuyue_stem_clashes,
        cr.dayun_liuyue_branch_combinations,
        cr.dayun_liuyue_branch_clashes,
    )
    vec[56:60] = _pair_4_bits(
        cr.liunian_liuyue_stem_combinations,
        cr.liunian_liuyue_stem_clashes,
        cr.liunian_liuyue_branch_combinations,
        cr.liunian_liuyue_branch_clashes,
    )
    # Elemental delta (5) and useful god activation (1)
    for i, e in enumerate(ELEMENTS):
        vec[60 + i] = cr.elemental_delta.get(e, 0.0)
    vec[65] = 1.0 if cr.useful_god_activation == "active" else 0.0
    return vec


def _serialize_pillar(pillar: Pillar) -> dict[str, str]:
    """Serialize one pillar to JSON object: stem, branch, element_stem, element_branch."""
    return {
        "stem": pillar.stem.value,
        "branch": pillar.branch.value,
        "element_stem": pillar.stem.element,
        "element_branch": pillar.branch.element,
    }


def _hidden_stems_list(branch: Branch, day_master_stem: Stem) -> list[dict[str, Any]]:
    """List of {stem, element, weight, ten_god} for hidden stems in branch."""
    out = []
    for hidden_stem, weight in get_hidden_stems(branch):
        tg = get_ten_god_for_stem(day_master_stem, hidden_stem)
        out.append({
            "stem": hidden_stem.value,
            "element": hidden_stem.element,
            "weight": round(weight, 2),
            "ten_god": tg.value,
        })
    return out


_USEFUL_GOD_THRESHOLDS = (0.15, 0.22, 0.28, 0.35, 0.42)


def _compute_useful_god(percentile: float, day_master_element: str) -> Optional[dict[str, Any]]:
    """
    Percentile-based useful-god heuristic.

    Maps day-master strength percentile to one of five useful-god categories
    (or ambiguous when balanced).  Thresholds chosen to spread labels across
    all five categories:

        < 0.15  → 印 (very weak, needs generation from 生我 element)
        0.15–0.22 → 比劫 (weak, needs peer-element support)
        0.22–0.28 → ambiguous / 中和
        0.28–0.35 → 财 (mildly strong, exhausted by 我克 element)
        0.35–0.42 → 食伤 (strong, drained by 我生 element)
        ≥ 0.42  → 官杀 (very strong, controlled by 克我 element)
    """
    t1, t2, t3, t4, t5 = _USEFUL_GOD_THRESHOLDS

    if t2 <= percentile < t3:
        return None  # ambiguous / 中和

    if percentile < t1:
        primary = "印"
        primary_element = _ELEMENT_GENERATES_ME.get(day_master_element)
        methodology = "身弱用印"
    elif percentile < t2:
        primary = "比劫"
        primary_element = day_master_element
        methodology = "身弱用比劫"
    elif percentile < t4:
        primary = "财"
        primary_element = _ELEMENT_CONTROLS.get(day_master_element)
        methodology = "身强用财"
    elif percentile < t5:
        primary = "食伤"
        primary_element = _ELEMENT_GENERATES.get(day_master_element)
        methodology = "身强用食伤"
    else:
        primary = "官杀"
        primary_element = _ELEMENT_CONTROLS_ME.get(day_master_element)
        methodology = "身强用官杀"

    if not primary_element:
        return None

    return {
        "primary": primary,
        "primary_element": primary_element,
        "favorable_elements": [primary_element],
        "unfavorable_elements": [],
        "methodology": methodology,
        "is_ambiguous": False,
    }


def _summarize_interactions_short(ir: InteractionResult) -> str:
    """Render an ``InteractionResult`` as a compact ``、``-joined Chinese summary.

    Skips interactions that have been blocked by other interactions
    (``is_blocked``, ``absorbed``, ``blocked_by_clash``) so the summary only
    surfaces the ones that are actually active in the chart.
    """
    parts: list[str] = []

    for d in ir.stem_combinations:
        if d.get("transformation_active"):
            parts.append(f"{d['stem_a']}{d['stem_b']}合化{d['result_element']}")
        elif d.get("is_pure", True):
            parts.append(f"{d['stem_a']}{d['stem_b']}合")

    for d in ir.branch_six_combinations:
        if d.get("is_blocked") or d.get("absorbed"):
            continue
        parts.append(f"{d['branch_a']}{d['branch_b']}六合")

    for d in ir.three_harmony:
        if d.get("blocked_by_clash"):
            continue
        label = "三合" if d.get("combo_type") == "full" else "半合"
        joined = "".join(d.get("present_branches", []))
        if joined:
            parts.append(f"{joined}{label}")

    for d in ir.stem_clashes:
        parts.append(f"{d['item_a']}{d['item_b']}冲")

    for d in ir.branch_clashes:
        parts.append(f"{d['item_a']}{d['item_b']}冲")

    for d in ir.harms:
        parts.append(f"{d['branch_a']}{d['branch_b']}害")

    seen_punishment: set[str] = set()
    for d in ir.punishments:
        t = d.get("punishment_type")
        if t and t not in seen_punishment:
            seen_punishment.add(t)
            parts.append(t)

    for d in ir.destructions:
        parts.append(f"{d['branch_a']}{d['branch_b']}破")

    return "、".join(parts)


def _serialize_interactions(ir: InteractionResult) -> dict[str, Any]:
    """Serialize InteractionResult to plan JSON structure."""
    def pos_labels(positions: list[int]) -> list[str]:
        return [POSITION_LABELS[p] for p in positions]

    stem_combinations = [
        {
            "positions": sorted([r["pos_a"], r["pos_b"]]),
            "position_labels": pos_labels(sorted([r["pos_a"], r["pos_b"]])),
            "stems": [r["stem_a"], r["stem_b"]],
            "result_element": r["result_element"],
            "is_pure": r.get("is_pure", True),
            "transformation_active": r.get("transformation_active", False),
        }
        for r in ir.stem_combinations
    ]

    stem_clashes = [
        {
            "positions": sorted([r["pos_a"], r["pos_b"]]),
            "position_labels": pos_labels(sorted([r["pos_a"], r["pos_b"]])),
            "stems": [r["item_a"], r["item_b"]],
        }
        for r in ir.stem_clashes
    ]

    branch_six_combinations = [
        {
            "positions": sorted([r["pos_a"], r["pos_b"]]),
            "position_labels": pos_labels(sorted([r["pos_a"], r["pos_b"]])),
            "branches": [r["branch_a"], r["branch_b"]],
            "result_element": r["result_element"],
            "is_blocked": r.get("is_blocked", False),
        }
        for r in ir.branch_six_combinations
    ]

    branch_clashes = [
        {
            "positions": sorted([r["pos_a"], r["pos_b"]]),
            "position_labels": pos_labels(sorted([r["pos_a"], r["pos_b"]])),
            "branches": [r["item_a"], r["item_b"]],
        }
        for r in ir.branch_clashes
    ]

    three_harmony = []
    for r in ir.three_harmony:
        elem = r.get("element")
        wuxing = _HARMONY_ELEMENT_TO_WUXING.get(elem, elem) if isinstance(elem, str) else elem
        three_harmony.append({
            "element": wuxing,
            "combo_type": r["combo_type"],
            "has_central": r.get("has_central", False),
            "present_branches": r["present_branches"],
            "positions": r["positions"],
            "blocked_by_clash": r.get("blocked_by_clash", False),
        })

    harms = [
        {
            "positions": sorted([r["pos_a"], r["pos_b"]]),
            "position_labels": pos_labels(sorted([r["pos_a"], r["pos_b"]])),
            "branches": [r["branch_a"], r["branch_b"]],
        }
        for r in ir.harms
    ]

    punishments = [
        {
            "punishment_type": r.get("punishment_type"),
            "positions": r.get("positions", []),
            "members_present": r.get("members_present", []),
        }
        for r in ir.punishments
    ]

    destructions = [
        {
            "positions": sorted([r["pos_a"], r["pos_b"]]),
            "position_labels": pos_labels(sorted([r["pos_a"], r["pos_b"]])),
            "branches": [r["branch_a"], r["branch_b"]],
            "is_weakened": r.get("is_weakened", False),
        }
        for r in ir.destructions
    ]

    return {
        "stem_combinations": stem_combinations,
        "stem_clashes": stem_clashes,
        "branch_six_combinations": branch_six_combinations,
        "branch_clashes": branch_clashes,
        "three_harmony": three_harmony,
        "harms": harms,
        "punishments": punishments,
        "destructions": destructions,
    }


def _serialize_dynamic(
    temporal_context: Optional[TemporalContext],
    cross_interactions: Optional[CrossInteractionResult],
) -> Optional[dict[str, Any]]:
    """Serialize dynamic (temporal) section when both context and cross_interactions are present."""
    if temporal_context is None or cross_interactions is None:
        return None
    out: dict[str, Any] = {}
    if temporal_context.dayun is not None:
        dy = temporal_context.dayun
        out["da_yun"] = {
            "stem": dy.stem,
            "branch": dy.branch,
            "start_year": dy.start_year,
            "start_age": dy.start_age,
            "end_year": dy.end_year,
            "end_age": dy.end_age,
        }
    if temporal_context.liunian is not None:
        ln = temporal_context.liunian
        out["liu_nian"] = {"year": ln.year, "stem": ln.stem, "branch": ln.branch}
    if temporal_context.liuyue is not None:
        ly = temporal_context.liuyue
        out["liu_yue"] = {
            "year": ly.year,
            "month": ly.month,
            "stem": ly.stem,
            "branch": ly.branch,
        }
    out["cross_natal_temporal_interactions"] = _serialize_cross_interactions(cross_interactions)
    out["elemental_delta"] = {
        _ELEMENT_TO_ENGLISH[e]: round(cross_interactions.elemental_delta[e], 4)
        for e in ELEMENTS
    }
    if cross_interactions.useful_god_activation is not None:
        out["useful_god_activation"] = {
            "is_active": cross_interactions.useful_god_activation == "active",
            "mechanism": cross_interactions.useful_god_analysis or "",
        }
    return out


def _serialize_cross_interactions(cr: CrossInteractionResult) -> dict[str, Any]:
    """Serialize CrossInteractionResult to nested dict by context.

    Natal×temporal dicts preserve ``natal_pos`` so the 16-bit per-position
    encoding can reconstruct which natal pillar is involved.
    """
    def _ser_stem_clash(items: list, *, with_natal_pos: bool) -> list:
        out = []
        for r in items:
            d: dict[str, Any] = {
                "stem_a": r.get("item_a", r.get("stem_a")),
                "stem_b": r.get("item_b", r.get("stem_b")),
            }
            if with_natal_pos and r.get("natal_pos") is not None:
                d["natal_pos"] = r["natal_pos"]
            out.append(d)
        return out

    def _ser_stem_comb(items: list, *, with_natal_pos: bool) -> list:
        out = []
        for r in items:
            d: dict[str, Any] = {
                "stem_a": r.get("stem_a"),
                "stem_b": r.get("stem_b"),
                "result_element": r.get("result_element"),
            }
            if with_natal_pos and r.get("natal_pos") is not None:
                d["natal_pos"] = r["natal_pos"]
            out.append(d)
        return out

    def _ser_branch(items: list, *, with_natal_pos: bool) -> list:
        out = []
        for r in items:
            d: dict[str, Any] = {
                "branch_a": r.get("branch_a", r.get("item_a")),
                "branch_b": r.get("branch_b", r.get("item_b")),
            }
            if with_natal_pos and r.get("natal_pos") is not None:
                d["natal_pos"] = r["natal_pos"]
            out.append(d)
        return out

    def _ser_natal_ctx(sc: list, ss: list, bc: list, bf: list) -> dict:
        return {
            "clashes": _ser_stem_clash(ss, with_natal_pos=True),
            "combinations": _ser_stem_comb(sc, with_natal_pos=True),
            "branch_combinations": _ser_branch(bc, with_natal_pos=True),
            "harms": _ser_branch(bf, with_natal_pos=True),
        }

    def _ser_pair_ctx(sc: list, ss: list, bc: list, cl: list) -> dict:
        return {
            "clashes": _ser_stem_clash(ss, with_natal_pos=False),
            "combinations": _ser_stem_comb(sc, with_natal_pos=False),
            "branch_combinations": _ser_branch(bc, with_natal_pos=False),
            "harms": _ser_branch(cl, with_natal_pos=False),
        }

    return {
        "natal_vs_dayun": _ser_natal_ctx(
            cr.natal_dayun_stem_combinations, cr.natal_dayun_stem_clashes,
            cr.natal_dayun_branch_combinations, cr.natal_dayun_branch_conflicts,
        ),
        "natal_vs_liunian": _ser_natal_ctx(
            cr.natal_liunian_stem_combinations, cr.natal_liunian_stem_clashes,
            cr.natal_liunian_branch_combinations, cr.natal_liunian_branch_conflicts,
        ),
        "natal_vs_liuyue": _ser_natal_ctx(
            cr.natal_liuyue_stem_combinations, cr.natal_liuyue_stem_clashes,
            cr.natal_liuyue_branch_combinations, cr.natal_liuyue_branch_conflicts,
        ),
        "dayun_vs_liunian": _ser_pair_ctx(
            cr.dayun_liunian_stem_combinations, cr.dayun_liunian_stem_clashes,
            cr.dayun_liunian_branch_combinations, cr.dayun_liunian_branch_clashes,
        ),
        "dayun_vs_liuyue": _ser_pair_ctx(
            cr.dayun_liuyue_stem_combinations, cr.dayun_liuyue_stem_clashes,
            cr.dayun_liuyue_branch_combinations, cr.dayun_liuyue_branch_clashes,
        ),
        "liunian_vs_liuyue": _ser_pair_ctx(
            cr.liunian_liuyue_stem_combinations, cr.liunian_liuyue_stem_clashes,
            cr.liunian_liuyue_branch_combinations, cr.liunian_liuyue_branch_clashes,
        ),
    }
