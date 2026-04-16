"""
visualize_results.py — Publication-quality figures for the Week 8 report.

Produces four figures from benchmark + training outputs:
  Fig 1  consistency_by_model.png        — overall Consistency Score per model (bar)
  Fig 2  consistency_by_question_type.png — score broken down by question type (grouped bar)
  Fig 3  passrate_by_intervention.png     — CF pass rate by intervention type (grouped bar)
  Fig 4  training_curves.png              — CE loss + pairwise consistency loss over epochs (line)

Usage:
    python src/analysis/visualize_results.py \
        --results results/metrics/llava_consistency_scores.json \
                  results/metrics/instructblip_consistency_scores.json \
        --training results/checkpoints/lora_llava/training_log.json \
        --output   results/figures/
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive; must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Paul Tol's colour-blind-safe palette
PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_results(paths: List[str]) -> Dict[str, Dict]:
    """
    Load one or more scored-consistency JSON files.
    Model name is inferred from the filename stem:
      llava_consistency_scores.json  →  "llava"
    """
    results: Dict[str, Dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        stem = Path(path).stem                       # e.g. "llava_consistency_scores"
        model_name = stem.split("_consistency")[0]   # → "llava"
        results[model_name] = data
    return results


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Shared axis styling
# ---------------------------------------------------------------------------

def _style_ax(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    ylim: tuple = (0.0, 1.05),
    pct: bool = True,
) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))


def _bar_label(ax: plt.Axes, bar, value: float, fontsize: int = 10) -> None:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{value:.3f}",
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Figure 1 — Overall Consistency Score per model
# ---------------------------------------------------------------------------

def fig_overall(results: Dict[str, Dict], output_dir: Path) -> None:
    models = list(results.keys())
    scores = [results[m]["dataset_consistency_score"] for m in models]

    fig, ax = plt.subplots(figsize=(max(5, len(models) * 1.8), 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]
    bars = ax.bar(
        [m.upper() for m in models], scores,
        color=colors, width=0.45, edgecolor="white", linewidth=1.2,
    )
    for bar, score in zip(bars, scores):
        _bar_label(ax, bar, score)

    ax.axhline(0.5, linestyle="--", color="#888888", linewidth=1.2, label="0.5 reference")
    ax.legend(fontsize=9, framealpha=0.6)
    _style_ax(ax, "Overall Consistency Score by Model", "Consistency Score")

    fig.tight_layout()
    save_fig(fig, output_dir / "fig1_consistency_by_model.png")


# ---------------------------------------------------------------------------
# Figure 2 — Consistency Score by question type
# ---------------------------------------------------------------------------

def fig_by_question_type(results: Dict[str, Dict], output_dir: Path) -> None:
    all_qtypes: set = set()
    for data in results.values():
        all_qtypes.update(data.get("breakdown", {}).get("by_question_type", {}).keys())
    qtypes = sorted(all_qtypes)
    if not qtypes:
        print("  No question-type breakdown — skipping Fig 2")
        return

    models = list(results.keys())
    n_m = len(models)
    x = np.arange(len(qtypes))
    width = 0.7 / n_m

    fig, ax = plt.subplots(figsize=(max(7, len(qtypes) * 2.0), 5))
    for i, model in enumerate(models):
        bd = results[model].get("breakdown", {}).get("by_question_type", {})
        scores = [bd.get(qt, {}).get("mean_consistency_score", 0.0) for qt in qtypes]
        ns     = [bd.get(qt, {}).get("num_families", 0) for qt in qtypes]
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, scores, width,
            label=model.upper(),
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white", linewidth=0.8,
        )
        for bar, score, n in zip(bars, scores, ns):
            if n:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"n={n}", ha="center", va="bottom",
                    fontsize=7, color="#555555",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([qt.replace("_", "\n") for qt in qtypes], fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, "Consistency Score by Question Type", "Mean Consistency Score")

    fig.tight_layout()
    save_fig(fig, output_dir / "fig2_consistency_by_question_type.png")


# ---------------------------------------------------------------------------
# Figure 3 — CF pass rate by intervention type
# ---------------------------------------------------------------------------

def fig_by_intervention(results: Dict[str, Dict], output_dir: Path) -> None:
    all_itypes: set = set()
    for data in results.values():
        all_itypes.update(data.get("breakdown", {}).get("by_intervention_type", {}).keys())
    itypes = sorted(all_itypes)
    if not itypes:
        print("  No intervention-type breakdown — skipping Fig 3")
        return

    models = list(results.keys())
    n_m = len(models)
    x = np.arange(len(itypes))
    width = 0.7 / n_m

    fig, ax = plt.subplots(figsize=(max(8, len(itypes) * 2.2), 5))
    for i, model in enumerate(models):
        bd = results[model].get("breakdown", {}).get("by_intervention_type", {})
        rates = [bd.get(it, {}).get("pass_rate", 0.0) for it in itypes]
        ns    = [bd.get(it, {}).get("num_counterfactuals", 0) for it in itypes]
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, rates, width,
            label=model.upper(),
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white", linewidth=0.8,
        )
        for bar, rate, n in zip(bars, rates, ns):
            if n:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"n={n}", ha="center", va="bottom",
                    fontsize=7, color="#555555",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([it.replace("_", "\n") for it in itypes], fontsize=9)
    ax.legend(fontsize=10)
    _style_ax(ax, "Counterfactual Pass Rate by Intervention Type", "Pass Rate")

    fig.tight_layout()
    save_fig(fig, output_dir / "fig3_passrate_by_intervention.png")


# ---------------------------------------------------------------------------
# Figure 4 — Training loss curves
# ---------------------------------------------------------------------------

def fig_training_curves(training_log_path: str, output_dir: Path) -> None:
    with open(training_log_path, encoding="utf-8") as f:
        log = json.load(f)

    import math
    epochs  = [e["epoch"] for e in log]
    ce_vals = [e["avg_ce_loss"] for e in log]
    pc_vals = [e["avg_pairwise_consistency_loss"] for e in log]

    # Filter out NaN CE values (happens when all answer tokens were masked)
    ce_epochs = [ep for ep, v in zip(epochs, ce_vals) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    ce_clean  = [v for v in ce_vals if v is not None and not (isinstance(v, float) and math.isnan(v))]

    fig, ax = plt.subplots(figsize=(7, 4))
    if ce_clean:
        ax.plot(ce_epochs, ce_clean, marker="o", color=PALETTE[0], linewidth=2,
                markersize=6, label="Cross-Entropy Loss (CE)")
    ax.plot(epochs, pc_vals, marker="s", color=PALETTE[1], linewidth=2,
            markersize=6, linestyle="--", label="Pairwise Consistency Loss (PC)")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Average Loss", fontsize=11)
    ax.set_title("Training Loss Curves — LoRA Fine-tuning (LLaVA-1.5)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(epochs)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8)

    fig.tight_layout()
    save_fig(fig, output_dir / "fig4_training_curves.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures from benchmark and training results."
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="One or more *_consistency_scores.json files (one per model)",
    )
    parser.add_argument(
        "--training", default=None,
        help="training_log.json produced by train_lora.py (optional)",
    )
    parser.add_argument("--output", default="results/figures/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)
    print(f"Models loaded: {list(results.keys())}")

    fig_overall(results, output_dir)
    fig_by_question_type(results, output_dir)
    fig_by_intervention(results, output_dir)

    if args.training:
        fig_training_curves(args.training, output_dir)
    else:
        print("  --training not provided; skipping Fig 4 (training curves)")

    print(f"\nAll figures written to: {output_dir}")


if __name__ == "__main__":
    main()
