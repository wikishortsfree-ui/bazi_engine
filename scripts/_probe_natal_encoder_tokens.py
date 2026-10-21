"""Linear probe: do the natal encoder's per-pillar tokens preserve stem/branch identity?

Architecture context (see bazi_model/encoders/natal_encoder_mlx.py):
    encoder builds a 6-token sequence [CLS, P0, P1, P2, P3, aux], runs 3 transformer
    layers, then RETURNS ONLY CLS. We want to know whether the four per-pillar tokens
    P0..P3 (year, month, day, hour) carry recoverable stem and branch identity even
    though the encoder discards them.

What this script does:
    1. Loads the trained natal encoder (frozen) and runs it over phase2_val.jsonl charts.
    2. Captures the FULL 6-token output (we monkey-patch the return to grab all 6).
    3. For each token slot (CLS, P0, P1, P2, P3, aux), trains a one-vs-rest logistic
       regression to predict stem_id at each pillar position (10 classes) and branch_id
       at each pillar position (12 classes).
    4. Reports accuracy per (token slot, target).

What the result tells us:
    - If P_i predicts stem_id[i] / branch_id[i] at >=90%, the per-pillar tokens
      preserve identity and the right fix is to expose them to the projection.
    - If accuracy is ~10% (chance), the encoder squashed the info even from per-pillar
      tokens; we'd need encoder unfreeze or per-pillar auxiliary supervision instead.
    - If CLS itself predicts all four stem_ids well, the original architecture wasn't
      bottlenecked here at all and we should look elsewhere (e.g. projection capacity,
      probe wrapper mismatch, prompt format).

Run: python scripts/_probe_natal_encoder_tokens.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--charts", type=Path, default=_root / "data" / "phase2_val.jsonl")
    p.add_argument(
        "--natal_weights",
        type=Path,
        default=_root / "checkpoints" / "natal_encoder_mlx.npz",
    )
    p.add_argument("--max_charts", type=int, default=750)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _capture_all_tokens(encoder, natal_inputs):
    """Re-run the encoder forward but return all 6 tokens.

    NatalEncoderMLX.__call__ returns sequence[:, 0, :] (CLS only). We replicate the
    forward path and stop short of that slice so we get all 6 tokens.
    """
    import mlx.core as mx

    kw = {k: mx.array(v) for k, v in natal_inputs.items()}
    batch = kw["stem_ids"].shape[0]
    ten_god_padding = mx.full((batch, 4), -1)
    pillar_emb = encoder.pillar_embedding(
        stem_ids=kw["stem_ids"],
        branch_ids=kw["branch_ids"],
        ten_god_stem_ids=ten_god_padding,
        stem_elements=kw["stem_elements"],
        stem_polarities=kw["stem_polarities"],
        pillar_positions=kw["pillar_positions"],
    )
    pillars_proj = encoder.pillar_proj(pillar_emb)
    cls = mx.broadcast_to(encoder.cls_token, (batch, 1, encoder.d_model))
    aux = mx.broadcast_to(encoder.aux_token, (batch, 1, encoder.d_model))
    sequence = mx.concatenate([cls, pillars_proj, aux], axis=1)
    for layer in encoder.layers:
        sequence = layer(sequence)
    return np.asarray(sequence)  # (batch, 6, d_model)


def _load_charts(path: Path, limit: int):
    import json
    from bazi_engine.interactions import detect_all_interactions
    from bazi_engine.models import BaziChart

    from bazi_model.data.phase2_dataset import _natal_from_chart

    natal_list = []
    stem_ids_list = []
    branch_ids_list = []
    interaction_vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            chart_val = rec.get("chart")
            if chart_val is None:
                continue
            chart = BaziChart.from_json(json.dumps(chart_val, ensure_ascii=False))
            natal = _natal_from_chart(chart)
            natal_list.append(natal)
            stem_ids_list.append(np.asarray(natal["stem_ids"]))
            branch_ids_list.append(np.asarray(natal["branch_ids"]))
            stems, branches = chart._stems_and_branches()
            ir = detect_all_interactions(stems, branches)
            interaction_vectors.append(np.asarray(ir.to_binary_vector(), dtype=np.float32))
            if len(natal_list) >= limit:
                break

    keys = list(natal_list[0].keys())
    batched = {
        k: np.stack([n[k] for n in natal_list], axis=0) for k in keys
    }
    stem_ids = np.stack(stem_ids_list, axis=0)
    branch_ids = np.stack(branch_ids_list, axis=0)
    interaction_vec = np.stack(interaction_vectors, axis=0)  # (n, 64) binary
    return batched, stem_ids, branch_ids, interaction_vec


def main() -> int:
    args = _parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn", file=sys.stderr)
        return 1

    from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX

    print(f"Loading natal encoder from {args.natal_weights} ...")
    encoder = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
    encoder.load_weights(str(args.natal_weights))

    print(f"Loading charts from {args.charts} (limit={args.max_charts}) ...")
    natal_batched, stem_ids, branch_ids, interaction_vec = _load_charts(
        args.charts, args.max_charts
    )
    n_charts = stem_ids.shape[0]
    print(f"  {n_charts} charts loaded; interaction vector shape: {interaction_vec.shape}")

    print("Running encoder and capturing all 6 tokens ...")
    chunk = 64
    all_tokens = []
    for start in range(0, n_charts, chunk):
        end = min(start + chunk, n_charts)
        sub = {k: v[start:end] for k, v in natal_batched.items()}
        tokens = _capture_all_tokens(encoder, sub)  # (chunk, 6, 256)
        all_tokens.append(tokens)
    all_tokens = np.concatenate(all_tokens, axis=0)  # (n, 6, 256)
    print(f"  encoder output shape: {all_tokens.shape}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_charts)
    n_test = max(1, int(n_charts * args.test_fraction))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    print(f"  split: train={len(train_idx)}  test={len(test_idx)}")

    TOKEN_NAMES = ("CLS", "P_year", "P_month", "P_day", "P_hour", "aux")
    PILLAR_NAMES = ("year", "month", "day", "hour")

    print("\n=== Linear probe results ===")
    print(
        f"{'token':>8s}  {'target':>16s}  {'train_acc':>9s}  {'test_acc':>9s}  "
        f"{'chance':>7s}  flag"
    )

    rows = []
    for token_idx, token_name in enumerate(TOKEN_NAMES):
        X = all_tokens[:, token_idx, :]
        X_tr, X_te = X[train_idx], X[test_idx]
        for pillar_idx, pillar_name in enumerate(PILLAR_NAMES):
            for target_name, target_array, n_classes in (
                ("stem", stem_ids, 10),
                ("branch", branch_ids, 12),
            ):
                y = target_array[:, pillar_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                if len(set(y_tr.tolist())) < 2:
                    continue
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X_tr, y_tr)
                tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
                te_acc = accuracy_score(y_te, clf.predict(X_te))
                chance = 1.0 / n_classes
                flag = ""
                if te_acc >= 0.9:
                    flag = "  IDENTITY-PRESERVED"
                elif te_acc >= 0.5:
                    flag = "  partial"
                elif te_acc <= chance + 0.05:
                    flag = "  ~chance"
                target_label = f"{pillar_name}_{target_name}"
                print(
                    f"{token_name:>8s}  {target_label:>16s}  "
                    f"{tr_acc:9.3f}  {te_acc:9.3f}  {chance:7.3f}{flag}"
                )
                rows.append((token_name, target_label, tr_acc, te_acc, chance))

    print("\n=== Summary ===")
    by_token = {n: [] for n in TOKEN_NAMES}
    for tname, _tgt, _tr, te, _ch in rows:
        by_token[tname].append(te)
    for tname in TOKEN_NAMES:
        accs = by_token[tname]
        if not accs:
            continue
        print(f"  {tname:>8s}  mean test_acc over 8 targets: {np.mean(accs):.3f}")

    # ---- Interaction-vector probe ---------------------------------------
    # Predict each of the 64 binary interaction-vector bits independently
    # from each token slot. Report the mean per-bit accuracy on the test set,
    # AND the mean accuracy over only the "active" bits (positives), since
    # most bits are 0 in any single chart and a constant-zero predictor
    # would otherwise look strong.
    print("\n=== Interaction-vector probe (predict each of 64 bits) ===")
    bit_freq = interaction_vec.mean(axis=0)
    n_active_bits = int((bit_freq > 0).sum())
    print(
        f"  64 bits total; {n_active_bits} ever-active in this dataset; "
        f"mean positive rate = {bit_freq.mean():.3f}"
    )
    print(
        f"\n{'token':>8s}  {'mean_bit_acc':>13s}  {'mean_pos_recall':>16s}  "
        f"{'majority_baseline':>18s}"
    )
    for token_idx, token_name in enumerate(TOKEN_NAMES):
        X = all_tokens[:, token_idx, :]
        X_tr, X_te = X[train_idx], X[test_idx]
        bit_accs: list[float] = []
        bit_pos_recalls: list[float] = []
        majority_accs: list[float] = []
        for bit in range(interaction_vec.shape[1]):
            y = interaction_vec[:, bit].astype(np.int64)
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(set(y_tr.tolist())) < 2:
                # constant-bit; skip
                continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            bit_accs.append(accuracy_score(y_te, y_pred))
            pos_mask = y_te == 1
            if pos_mask.any():
                bit_pos_recalls.append(float((y_pred[pos_mask] == 1).mean()))
            # majority baseline = predict 0 always
            majority_accs.append(float((y_te == 0).mean()))
        if not bit_accs:
            continue
        print(
            f"{token_name:>8s}  {np.mean(bit_accs):13.3f}  "
            f"{np.mean(bit_pos_recalls):16.3f}  {np.mean(majority_accs):18.3f}"
        )

    print("\nKey diagonal (each pillar token predicting its own pillar's stem/branch):")
    for pillar_idx, pillar_name in enumerate(PILLAR_NAMES):
        tname = TOKEN_NAMES[pillar_idx + 1]  # P_year is at token slot 1
        for target_name in ("stem", "branch"):
            target_label = f"{pillar_name}_{target_name}"
            for r in rows:
                if r[0] == tname and r[1] == target_label:
                    print(f"  {tname} -> {target_label}: test_acc={r[3]:.3f}")
                    break

    return 0


if __name__ == "__main__":
    sys.exit(main())
