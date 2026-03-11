#!/usr/bin/env python3
"""
AutismBench - Visualization.

Generate publication-quality charts from results JSON.

Usage:
    python visualization.py results/autism_bench_results_*.json
    python visualization.py results.json --save
"""

import json
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np


# ── Style ────────────────────────────────────────────────────────────────────

PALETTE = [
    "#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#3A0CA3",
    "#F77F00", "#06D6A0", "#EF476F", "#118AB2", "#073B4C",
]

def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── Data loading ─────────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


# ── Charts ───────────────────────────────────────────────────────────────────

def plot_leaderboard(results: dict, save_dir: str | None = None):
    models = results["models"]
    ranked = sorted(models.items(), key=lambda x: x[1]["total_score"], reverse=True)

    names = [short_name(m) for m, _ in ranked]
    scores = [d["total_score"] for _, d in ranked]
    validity = [d["validity_ratio"] * 100 for _, d in ranked]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.55)))

    bars = ax.barh(
        range(len(names)), scores,
        color=PALETTE[:len(names)], edgecolor="white", linewidth=0.5, height=0.7,
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11, fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlabel("Total Score", fontsize=12, fontweight="medium")

    for bar, score, val in zip(bars, scores, validity):
        ax.text(
            bar.get_width() + max(scores) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score}  ({val:.0f}% valid)",
            va="center", fontsize=9.5, color="#444444",
        )

    ax.set_xlim(0, max(scores) * 1.35)
    ax.set_title("AutismBench Leaderboard", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.4)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/leaderboard.png", dpi=200, bbox_inches="tight")
    return fig


def plot_difficulty_curve(results: dict, save_dir: str | None = None):
    models = results["models"]
    ranked = sorted(models.items(), key=lambda x: x[1]["total_score"], reverse=True)
    levels = sorted(int(l) for l in list(models.values())[0]["levels"].keys())

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (model, data) in enumerate(ranked):
        name = short_name(model)
        avg_scores = []
        for level in levels:
            lvl_data = data["levels"].get(str(level), {})
            avg_scores.append(lvl_data.get("avg_score", 0))

        ax.plot(
            levels, avg_scores,
            marker="o", markersize=5, linewidth=2.2,
            color=PALETTE[i % len(PALETTE)],
            label=name, alpha=0.9,
        )

    ax.set_xlabel("Constraint Level", fontsize=12, fontweight="medium")
    ax.set_ylabel("Average Score", fontsize=12, fontweight="medium")
    ax.set_title("Difficulty Curve", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(levels)
    ax.legend(
        loc="upper left", fontsize=9, framealpha=0.95,
        edgecolor="#CCCCCC", ncol=2,
    )
    ax.grid(axis="both", alpha=0.4)

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/difficulty_curve.png", dpi=200, bbox_inches="tight")
    return fig


def plot_heatmap(results: dict, save_dir: str | None = None):
    models = results["models"]
    ranked = sorted(models.items(), key=lambda x: x[1]["total_score"], reverse=True)
    names = [short_name(m) for m, _ in ranked]
    levels = sorted(int(l) for l in list(models.values())[0]["levels"].keys())

    matrix = []
    for _, data in ranked:
        row = []
        for level in levels:
            trials = data["levels"].get(str(level), {}).get("trials", [])
            if trials:
                tp = sum(t["passed"] for t in trials)
                tc = sum(t["total"] for t in trials)
                row.append(tp / tc * 100 if tc else 0)
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(9, len(levels) * 0.7), max(4, len(names) * 0.6)))

    sns.heatmap(
        matrix, ax=ax,
        xticklabels=[str(l) for l in levels],
        yticklabels=names,
        cmap="RdYlGn", vmin=0, vmax=100,
        annot=True, fmt=".0f", annot_kws={"size": 9},
        linewidths=1.5, linecolor="#FAFAFA",
        cbar_kws={"label": "Validity %", "shrink": 0.8},
    )

    ax.set_xlabel("Constraint Level", fontsize=12, fontweight="medium")
    ax.set_title("Validity Ratio (% constraints passed)", fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis="y", labelsize=11)

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/heatmap.png", dpi=200, bbox_inches="tight")
    return fig


def plot_perfect_rate(results: dict, save_dir: str | None = None):
    models = results["models"]
    ranked = sorted(models.items(), key=lambda x: x[1]["total_score"], reverse=True)
    levels = sorted(int(l) for l in list(models.values())[0]["levels"].keys())

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bar_width = 0.8 / len(ranked)
    x = np.arange(len(levels))

    for i, (model, data) in enumerate(ranked):
        name = short_name(model)
        rates = []
        for level in levels:
            trials = data["levels"].get(str(level), {}).get("trials", [])
            if trials:
                perfect = sum(1 for t in trials if t["perfect"])
                rates.append(perfect / len(trials) * 100)
            else:
                rates.append(0)

        ax.bar(
            x + i * bar_width, rates, bar_width,
            label=name, color=PALETTE[i % len(PALETTE)],
            edgecolor="white", linewidth=0.3, alpha=0.9,
        )

    ax.set_xlabel("Constraint Level", fontsize=12, fontweight="medium")
    ax.set_ylabel("Perfect Solve Rate (%)", fontsize=12, fontweight="medium")
    ax.set_title("Perfect Solve Rate by Level", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x + bar_width * len(ranked) / 2)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_ylim(0, 105)
    ax.legend(
        loc="upper right", fontsize=8, framealpha=0.95,
        edgecolor="#CCCCCC", ncol=2,
    )
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/perfect_rate.png", dpi=200, bbox_inches="tight")
    return fig


def plot_category_breakdown(results: dict, save_dir: str | None = None):
    from constraint_pool import get_constraint_by_id

    models = results["models"]
    ranked = sorted(models.items(), key=lambda x: x[1]["total_score"], reverse=True)

    categories = set()
    model_cat_stats = {}

    for model, data in ranked:
        cat_passed = {}
        cat_total = {}
        for lvl_data in data["levels"].values():
            for trial in lvl_data["trials"]:
                for r in trial["results"]:
                    c = get_constraint_by_id(r["constraint_id"])
                    if c:
                        cat = c["category"]
                        categories.add(cat)
                        cat_passed[cat] = cat_passed.get(cat, 0) + (1 if r["passed"] else 0)
                        cat_total[cat] = cat_total.get(cat, 0) + 1

        model_cat_stats[model] = {
            cat: (cat_passed.get(cat, 0) / cat_total[cat] * 100) if cat_total.get(cat) else 0
            for cat in categories
        }

    categories = sorted(categories)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bar_width = 0.8 / len(ranked)
    x = np.arange(len(categories))

    for i, (model, _) in enumerate(ranked):
        name = short_name(model)
        vals = [model_cat_stats[model].get(cat, 0) for cat in categories]
        ax.bar(
            x + i * bar_width, vals, bar_width,
            label=name, color=PALETTE[i % len(PALETTE)],
            edgecolor="white", linewidth=0.3, alpha=0.9,
        )

    ax.set_ylabel("Pass Rate (%)", fontsize=12, fontweight="medium")
    ax.set_title("Pass Rate by Constraint Category", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x + bar_width * len(ranked) / 2)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(
        loc="upper right", fontsize=8, framealpha=0.95,
        edgecolor="#CCCCCC", ncol=2,
    )
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/categories.png", dpi=200, bbox_inches="tight")
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutismBench - Visualization")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--save", action="store_true", help="Save plots to ./assets/")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory")
    parser.add_argument("--no-show", action="store_true", help="Don't show interactive plots")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    setup_style()
    results = load_results(args.results_file)

    save_dir = None
    if args.save:
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)

    print("Generating charts...")
    plot_leaderboard(results, save_dir)
    plot_difficulty_curve(results, save_dir)
    plot_heatmap(results, save_dir)
    plot_perfect_rate(results, save_dir)
    plot_category_breakdown(results, save_dir)

    if save_dir:
        print(f"Saved to {save_dir}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
