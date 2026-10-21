"""
Run t-SNE/UMAP embedding visualization for Phase 1 natal encoder (Task 1.5.2).

Extracts natal_emb from a trained encoder for 1k–5k charts, reduces to 2D with t-SNE
(and optionally UMAP), colors by dominant element, useful god, and day-master strength,
saves plots to reports/ and a short cluster-quality summary.

Usage (from repo root, with project venv):
  python scripts/run_embedding_visualization.py --checkpoint checkpoints/multi_task_best.pt
  python scripts/run_embedding_visualization.py --checkpoint checkpoints/ten_god_best.pt --max_samples 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bazi_engine.models import BaziChart
from bazi_engine.strength import compute_day_master_strength
from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.natal_encoder import NatalEncoder

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)

ELEMENT_LABELS = ("Wood 木", "Fire 火", "Earth 土", "Metal 金", "Water 水")
STRENGTH_LABELS = ("Weak", "Medium", "Strong")
UG_LABELS = ["比肩", "劫财", "食神", "伤官", "偏财", "Ambiguous"]
# Ten-god (day pillar): 11 classes per bazi_engine.encodings.TEN_GOD_TO_ID
TEN_GOD_DAY_LABELS = [
    "比肩", "劫财", "食神", "伤官", "偏财", "正财", "七杀", "正官", "偏印", "正印", "日主",
]
INTERACTION_COUNT_LABELS = ("Low (0–3)", "Medium (4–7)", "High (8+)")


def _infer_encoder_arch(encoder_state_dict: dict) -> tuple[int, int]:
    """Infer (d_model, num_layers) from encoder state_dict."""
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


def load_encoder_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load NatalEncoder from multi-task or ten-god checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Multi-task format: encoder state_dict + optional d_model, num_layers
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        enc_sd = ckpt["encoder"]
        # Prefer saved arch; infer from state_dict so 3-layer checkpoints load correctly
        d_model, num_layers = _infer_encoder_arch(enc_sd)
        if "d_model" in ckpt:
            d_model = ckpt["d_model"]
        if "num_layers" in ckpt:
            num_layers = ckpt["num_layers"]
        encoder = NatalEncoder(d_model=d_model, num_layers=num_layers)
        encoder.load_state_dict(enc_sd, strict=True)
        return encoder.to(device)

    # Ten-god style: full model state_dict (encoder + head)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    if not enc_state:
        raise ValueError(f"No encoder state in checkpoint {checkpoint_path}")
    d_model, num_layers = _infer_encoder_arch(enc_state)
    encoder = NatalEncoder(d_model=d_model, num_layers=num_layers)
    encoder.load_state_dict(enc_state, strict=True)
    return encoder.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="t-SNE/UMAP visualization of natal encoder embeddings (Task 1.5.2)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "multi_task_best.pt",
        help="Path to multi-task or ten-god checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "charts.jsonl",
        help="Path to charts.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "reports",
        help="Directory for plots and summary",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3000,
        help="Max charts for t-SNE/UMAP (1k–5k recommended)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for val split",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of data used as val (we take from val for viz)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--no_umap",
        action="store_true",
        help="Skip UMAP (only t-SNE) if umap-learn not installed",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"Data not found: {args.data}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load encoder
    encoder = load_encoder_from_checkpoint(args.checkpoint, device)
    encoder.eval()

    # Dataset and val split
    full_ds = ChartDataset(args.data, task_name="multi", temporal_only=False)
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )

    # Extract embeddings and labels (multi-task targets include ten_god_ids, interaction_vector)
    emb_list = []
    el_strengths_list = []
    ug_id_list = []
    ten_god_ids_list = []
    interaction_vec_list = []
    stem_ids_list = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            if len(emb_list) * args.batch_size >= args.max_samples:
                break
            encoder_kw = {k: inputs[k].to(device) for k in _NATAL_INPUT_KEYS if k in inputs}
            natal_emb = encoder(**encoder_kw)
            emb_list.append(natal_emb.cpu().numpy())
            el_strengths_list.append(targets["elemental_strengths"].numpy())
            ug_id_list.append(targets["useful_god_id"].numpy().squeeze(-1))
            ten_god_ids_list.append(targets["ten_god_ids"].numpy())
            interaction_vec_list.append(targets["interaction_vector"].numpy())
            stem_ids_list.append(inputs["stem_ids"].numpy())

    embeddings = np.concatenate(emb_list, axis=0)[: args.max_samples]
    elemental_strengths = np.concatenate(el_strengths_list, axis=0)[: args.max_samples]
    useful_god_id = np.concatenate(ug_id_list, axis=0)[: args.max_samples]
    ten_god_ids = np.concatenate(ten_god_ids_list, axis=0)[: args.max_samples]  # (n, 4)
    interaction_vector = np.concatenate(interaction_vec_list, axis=0)[: args.max_samples]  # (n, 64)
    stem_ids = np.concatenate(stem_ids_list, axis=0)[: args.max_samples]  # (n, 4)
    n_used = len(embeddings)
    print(f"Collected {n_used} embeddings: shape {embeddings.shape}")

    # Day-master strength per sample (by val_ds indices)
    with open(args.data, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    day_master_strengths = []
    for i in range(n_used):
        line_idx = val_ds.indices[i]
        chart = BaziChart.from_json(lines[line_idx])
        stems, branches = chart._stems_and_branches()
        dm = compute_day_master_strength(stems, branches)
        day_master_strengths.append(dm)
    day_master_strength = np.array(day_master_strengths, dtype=np.float64)
    print(f"day_master_strength: min={day_master_strength.min():.3f}, max={day_master_strength.max():.3f}")

    # Labels for plotting
    dominant_element = np.argmax(elemental_strengths, axis=1)
    q1, q2 = np.percentile(day_master_strength, [33.33, 66.67])
    strength_bin = np.zeros(n_used, dtype=int)
    strength_bin[day_master_strength <= q1] = 0
    strength_bin[(day_master_strength > q1) & (day_master_strength <= q2)] = 1
    strength_bin[day_master_strength > q2] = 2
    # Extra labels for additional visualizations
    # Day-pillar ten-god is always 日主 (10); use month pillar (index 1) for variety across 十神
    ten_god_month = ten_god_ids[:, 1].astype(int)  # month-pillar ten-god, 0–10 (10 = 日主 if ever at month)
    day_stem_element = (stem_ids[:, 2] // 2).astype(int)  # 0–4: Wood, Fire, Earth, Metal, Water
    interaction_count = interaction_vector.sum(axis=1).astype(int)
    interaction_count_bin = np.zeros(n_used, dtype=int)
    interaction_count_bin[interaction_count <= 3] = 0
    interaction_count_bin[(interaction_count >= 4) & (interaction_count <= 7)] = 1
    interaction_count_bin[interaction_count >= 8] = 2

    # t-SNE
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, max_iter=1000)
    coords_tsne = tsne.fit_transform(embeddings)
    print("t-SNE done. Shape:", coords_tsne.shape)

    # UMAP (optional)
    use_umap = not args.no_umap
    coords_umap = None
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=args.seed
        )
        coords_umap = reducer.fit_transform(embeddings)
        print("UMAP done. Shape:", coords_umap.shape)
    except ImportError:
        use_umap = False
        print("UMAP skipped (umap-learn not installed). Use pip install umap-learn for UMAP plots.")

    # Plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_fig(name: str, fig: plt.Figure) -> None:
        path = args.output_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # t-SNE: dominant element
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(ELEMENT_LABELS):
        mask = dominant_element == i
        ax.scatter(coords_tsne[mask, 0], coords_tsne[mask, 1], label=label, alpha=0.6, s=8)
    ax.set_title("Natal embeddings (t-SNE) — colored by dominant element")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    save_fig("embedding_tsne_dominant_element.png", fig)

    # t-SNE: useful god
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(6):
        mask = useful_god_id == i
        if mask.any():
            ax.scatter(
                coords_tsne[mask, 0], coords_tsne[mask, 1], label=UG_LABELS[i], alpha=0.6, s=8
            )
    ax.set_title("Natal embeddings (t-SNE) — colored by useful god (用神)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    save_fig("embedding_tsne_useful_god.png", fig)

    # t-SNE: day-master strength
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(STRENGTH_LABELS):
        mask = strength_bin == i
        ax.scatter(coords_tsne[mask, 0], coords_tsne[mask, 1], label=label, alpha=0.6, s=8)
    ax.set_title("Natal embeddings (t-SNE) — colored by day-master strength")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    save_fig("embedding_tsne_day_master_strength.png", fig)

    # t-SNE: ten-god (month pillar; day pillar is always 日主)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(11):
        mask = ten_god_month == i
        if mask.any():
            ax.scatter(
                coords_tsne[mask, 0], coords_tsne[mask, 1],
                label=TEN_GOD_DAY_LABELS[i], alpha=0.6, s=8,
            )
    ax.set_title("Natal embeddings (t-SNE) — colored by ten-god (month pillar 十神)")
    ax.legend(loc="best", fontsize=7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    save_fig("embedding_tsne_ten_god_month.png", fig)

    # t-SNE: day stem element (day master's element, not dominant chart element)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(ELEMENT_LABELS):
        mask = day_stem_element == i
        ax.scatter(coords_tsne[mask, 0], coords_tsne[mask, 1], label=label, alpha=0.6, s=8)
    ax.set_title("Natal embeddings (t-SNE) — colored by day-stem element (日主五行)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    save_fig("embedding_tsne_day_stem_element.png", fig)

    # t-SNE: interaction count (low / medium / high)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(INTERACTION_COUNT_LABELS):
        mask = interaction_count_bin == i
        ax.scatter(coords_tsne[mask, 0], coords_tsne[mask, 1], label=label, alpha=0.6, s=8)
    ax.set_title("Natal embeddings (t-SNE) — colored by interaction count (64-dim binary)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    save_fig("embedding_tsne_interaction_count.png", fig)

    if use_umap and coords_umap is not None:
        # Row 1: original three
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for i, label in enumerate(ELEMENT_LABELS):
            mask = dominant_element == i
            axes[0, 0].scatter(coords_umap[mask, 0], coords_umap[mask, 1], label=label, alpha=0.6, s=8)
        axes[0, 0].set_title("Dominant element")
        axes[0, 0].legend(loc="best", fontsize=6)
        for i in range(6):
            mask = useful_god_id == i
            if mask.any():
                axes[0, 1].scatter(
                    coords_umap[mask, 0], coords_umap[mask, 1], label=UG_LABELS[i], alpha=0.6, s=8
                )
        axes[0, 1].set_title("Useful god (用神)")
        axes[0, 1].legend(loc="best", fontsize=6)
        for i, label in enumerate(STRENGTH_LABELS):
            mask = strength_bin == i
            axes[0, 2].scatter(
                coords_umap[mask, 0], coords_umap[mask, 1], label=label, alpha=0.6, s=8
            )
        axes[0, 2].set_title("Day-master strength")
        axes[0, 2].legend(loc="best", fontsize=6)
        # Row 2: ten-god (month pillar), day-stem element, interaction count
        for i in range(11):
            mask = ten_god_month == i
            if mask.any():
                axes[1, 0].scatter(
                    coords_umap[mask, 0], coords_umap[mask, 1],
                    label=TEN_GOD_DAY_LABELS[i], alpha=0.6, s=8,
                )
        axes[1, 0].set_title("Ten-god month pillar (十神)")
        axes[1, 0].legend(loc="best", fontsize=6)
        for i, label in enumerate(ELEMENT_LABELS):
            mask = day_stem_element == i
            axes[1, 1].scatter(
                coords_umap[mask, 0], coords_umap[mask, 1], label=label, alpha=0.6, s=8
            )
        axes[1, 1].set_title("Day-stem element (日主五行)")
        axes[1, 1].legend(loc="best", fontsize=6)
        for i, label in enumerate(INTERACTION_COUNT_LABELS):
            mask = interaction_count_bin == i
            axes[1, 2].scatter(
                coords_umap[mask, 0], coords_umap[mask, 1], label=label, alpha=0.6, s=8
            )
        axes[1, 2].set_title("Interaction count")
        axes[1, 2].legend(loc="best", fontsize=6)
        plt.suptitle("Natal embeddings (UMAP)")
        plt.tight_layout()
        save_fig("embedding_umap_all.png", fig)

    # Cluster quality summary (placeholder for human notes)
    summary_path = args.output_dir / "embedding_visualization_summary.md"
    summary_lines = [
        "# Embedding visualization (Task 1.5.2)",
        "",
        f"- **Checkpoint:** {args.checkpoint.name}",
        f"- **Samples:** {n_used}",
        f"- **Plots:** t-SNE (6 panels) + UMAP (1 panel, 2×3 subplots) in `reports/`",
        "",
        "## Cluster quality notes (fill after inspecting plots)",
        "",
        "- **Dominant element:** Do Wood/Fire/Earth/Metal/Water form distinct regions? Note overlap or separation.",
        "- **Useful god (用神):** Do the 6 classes show separation? Ambiguous may spread; document if unambiguous types cluster.",
        "- **Day-master strength:** Weak/medium/strong gradient or separation?",
        "- **Ten-god (month pillar):** Do 十神 classes (比肩…日主) form regions? (Day pillar is always 日主.)",
        "- **Day-stem element:** Day master’s 五行 (Wood/Fire/Earth/Metal/Water) — distinct from dominant chart element.",
        "- **Interaction count:** Low / medium / high (0–3, 4–7, 8+ bits) — does chart “complexity” cluster?",
        "",
        "Conclusion: [ ] Meaningful clustering observed — encoder captures semantic structure.",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved summary template: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
