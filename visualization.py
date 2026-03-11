#!/usr/bin/env python3
"""
AutismBench - Visualization.

Generate interactive leaderboard and analysis from results JSON using Plotly.

Usage:
    python visualization.py results/autism_bench_results_*.json
    python visualization.py results.json --save
"""

import json
import argparse
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def build_dashboard(results: dict) -> go.Figure:
    """Build a single-page interactive dashboard with all charts."""
    models = results["models"]
    config = results["config"]

    ranked = sorted(models.items(), key=lambda x: x[1]["total_score"], reverse=True)
    names = [short_name(m) for m, _ in ranked]
    levels = sorted(int(l) for l in list(models.values())[0]["levels"].keys())

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Leaderboard (Total Score)",
            "Perfect Solve Rate (%)",
            "Validity Ratio by Level",
            "Difficulty Curve (Avg Score)",
            "Category Breakdown (Pass Rate %)",
            "Score Distribution",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "box"}],
        ],
    )

    # Color palette
    colors = px.colors.qualitative.Set2

    # 1. Leaderboard bar chart
    scores = [d["total_score"] for _, d in ranked]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    fig.add_trace(
        go.Bar(
            y=names, x=scores, orientation="h",
            marker_color=bar_colors,
            text=scores, textposition="outside",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 2. Perfect solve rate
    perf_ranked = sorted(ranked, key=lambda x: x[1]["perfect_solve_rate"], reverse=True)
    perf_names = [short_name(m) for m, _ in perf_ranked]
    perf_rates = [d["perfect_solve_rate"] * 100 for _, d in perf_ranked]
    fig.add_trace(
        go.Bar(
            y=perf_names, x=perf_rates, orientation="h",
            marker_color=[colors[i % len(colors)] for i in range(len(perf_names))],
            text=[f"{r:.1f}%" for r in perf_rates], textposition="outside",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 3. Validity heatmap
    matrix = []
    for _, data in ranked:
        row = []
        for level in levels:
            lvl_data = data["levels"].get(str(level), {})
            trials = lvl_data.get("trials", [])
            if trials:
                total_p = sum(t["passed"] for t in trials)
                total_c = sum(t["total"] for t in trials)
                ratio = total_p / total_c if total_c else 0
            else:
                ratio = 0
            row.append(round(ratio * 100, 1))
        matrix.append(row)

    fig.add_trace(
        go.Heatmap(
            z=matrix, x=[str(l) for l in levels], y=names,
            colorscale="RdYlGn", zmin=0, zmax=100,
            text=[[f"{v:.0f}%" for v in row] for row in matrix],
            texttemplate="%{text}", textfont={"size": 9},
            colorbar=dict(title="Valid %", x=0.46, len=0.3, y=0.5),
            showscale=True,
        ),
        row=2, col=1,
    )

    # 4. Difficulty curve
    for i, (model, data) in enumerate(ranked):
        name = short_name(model)
        avg_scores = []
        for level in levels:
            lvl_data = data["levels"].get(str(level), {})
            avg_scores.append(lvl_data.get("avg_score", 0))

        fig.add_trace(
            go.Scatter(
                x=levels, y=avg_scores, mode="lines+markers",
                name=name, marker=dict(size=5),
                line=dict(color=colors[i % len(colors)], width=2),
                legendgroup=name,
            ),
            row=2, col=2,
        )

    # 5. Category breakdown
    from constraint_pool import get_constraint_by_id

    categories = set()
    model_cat_stats = {}

    for model, data in ranked:
        cat_passed = {}
        cat_total = {}
        for lvl_data in data["levels"].values():
            for trial in lvl_data["trials"]:
                for r in trial["results"]:
                    cid = r["constraint_id"]
                    c = get_constraint_by_id(cid)
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
    for i, (model, _) in enumerate(ranked):
        name = short_name(model)
        vals = [round(model_cat_stats[model].get(cat, 0), 1) for cat in categories]
        fig.add_trace(
            go.Bar(
                x=categories, y=vals, name=name,
                marker_color=colors[i % len(colors)],
                legendgroup=name, showlegend=False,
            ),
            row=3, col=1,
        )

    # 6. Score distribution (box plot per model)
    for i, (model, data) in enumerate(ranked):
        name = short_name(model)
        all_scores = []
        for lvl_data in data["levels"].values():
            for trial in lvl_data["trials"]:
                all_scores.append(trial["score"])

        fig.add_trace(
            go.Box(
                y=all_scores, name=name,
                marker_color=colors[i % len(colors)],
                legendgroup=name, showlegend=False,
            ),
            row=3, col=2,
        )

    # Layout
    fig.update_layout(
        title=dict(
            text="AutismBench - Results Dashboard",
            font=dict(size=22),
        ),
        height=1400,
        width=1200,
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=0.35, xanchor="center", x=0.75,
            font=dict(size=10),
        ),
        barmode="group",
    )

    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="Level", row=2, col=2)
    fig.update_yaxes(title_text="Avg Score", row=2, col=2)
    fig.update_yaxes(title_text="Pass Rate %", row=3, col=1)
    fig.update_yaxes(title_text="Score", row=3, col=2)

    return fig


def main():
    parser = argparse.ArgumentParser(description="AutismBench - Visualization")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--save", action="store_true", help="Save as HTML")
    parser.add_argument("--output", type=str, default="dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    results = load_results(args.results_file)
    fig = build_dashboard(results)

    if args.save:
        fig.write_html(args.output, include_plotlyjs="cdn")
        print(f"Dashboard saved to {args.output}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
