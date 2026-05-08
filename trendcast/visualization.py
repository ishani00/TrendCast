from __future__ import annotations

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trendcast-mpl-cache"))
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd


THEME = {
    "dark_green": "#0A3323",
    "moss": "#839958",
    "beige": "#F7F4D5",
    "rosy_brown": "#D3968C",
    "midnight_green": "#105666",
    "ink": "#26413c",
    "soft_line": "#b9c3b4",
}

SENTIMENT_CMAP = LinearSegmentedColormap.from_list(
    "trendcast_sentiment",
    [THEME["rosy_brown"], THEME["beige"], THEME["moss"]],
)


def save_bar_chart(final_trends: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "final_trend_scores.png"
    ordered = final_trends.sort_values("final_score_with_novelty", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        ordered["topic"],
        ordered["final_score_with_novelty"],
        color=[THEME["dark_green"], THEME["rosy_brown"], THEME["moss"], THEME["midnight_green"]],
    )
    fig.patch.set_facecolor(THEME["beige"])
    ax.set_facecolor("#fbf8e8")
    ax.set_title("Final Trend Score by Topic")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.05, ordered["final_score_with_novelty"].max() + 0.1))
    ax.tick_params(axis="x", rotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME["soft_line"])
    ax.spines["bottom"].set_color(THEME["soft_line"])
    ax.tick_params(colors=THEME["midnight_green"])
    ax.xaxis.label.set_color(THEME["midnight_green"])
    ax.yaxis.label.set_color(THEME["midnight_green"])
    ax.title.set_color(THEME["dark_green"])

    for bar, value in zip(bars, ordered["final_score_with_novelty"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=THEME["midnight_green"],
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_quad_chart(final_trends: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "trend_quad_chart.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(THEME["beige"])
    ax.set_facecolor("#fbf8e8")

    bubble_sizes = final_trends["mention_count"] * 25
    scatter = ax.scatter(
        final_trends["novelty_score_norm"],
        final_trends["trend_score"],
        s=bubble_sizes,
        c=final_trends["avg_sentiment"],
        cmap=SENTIMENT_CMAP,
        alpha=0.78,
        edgecolors=THEME["ink"],
        linewidths=1.1,
    )

    for _, row in final_trends.iterrows():
        x_offset = -72 if row["novelty_score_norm"] > 0.82 else 8
        ax.annotate(
            row["topic"],
            (row["novelty_score_norm"], row["trend_score"]),
            xytext=(x_offset, 8),
            textcoords="offset points",
            fontsize=10,
            color=THEME["dark_green"],
        )

    ax.axhline(0.65, color=THEME["midnight_green"], linestyle="--", linewidth=1, alpha=0.55)
    ax.axvline(0.50, color=THEME["midnight_green"], linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Novelty Score (normalized)")
    ax.set_ylabel("Trend Score")
    ax.set_title("TrendCast Quad Chart")
    ax.text(0.76, 0.95, "Hot and new", fontsize=10, color=THEME["midnight_green"])
    ax.text(0.07, 0.95, "Established demand", fontsize=10, color=THEME["midnight_green"])
    ax.text(0.76, 0.08, "Early signal", fontsize=10, color=THEME["midnight_green"])
    ax.text(0.07, 0.08, "Lower priority", fontsize=10, color=THEME["midnight_green"])
    ax.spines["top"].set_color(THEME["ink"])
    ax.spines["right"].set_color(THEME["ink"])
    ax.spines["left"].set_color(THEME["ink"])
    ax.spines["bottom"].set_color(THEME["ink"])
    ax.tick_params(colors=THEME["dark_green"])
    ax.xaxis.label.set_color(THEME["dark_green"])
    ax.yaxis.label.set_color(THEME["dark_green"])
    ax.title.set_color(THEME["dark_green"])

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Average sentiment")
    colorbar.outline.set_edgecolor(THEME["soft_line"])
    colorbar.ax.yaxis.label.set_color(THEME["dark_green"])
    colorbar.ax.tick_params(colors=THEME["dark_green"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_network_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "topic_network.png"
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(THEME["beige"])
    ax.set_facecolor("#fbf8e8")
    graph = nx.Graph()

    for _, row in nodes_df.iterrows():
        graph.add_node(row["topic"], size=max(500, 2200 * float(row.get("pagerank", 0.1) + 0.1)))
    for _, row in edges_df.iterrows():
        graph.add_edge(row["source_topic"], row["target_topic"], weight=float(row["weight"]))

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No network data available", ha="center", va="center", color=THEME["midnight_green"])
        ax.axis("off")
    else:
        layout = nx.spring_layout(graph, seed=42, weight="weight")
        edge_widths = [2 + 6 * graph[u][v]["weight"] for u, v in graph.edges()]
        nx.draw_networkx_edges(graph, layout, width=edge_widths, alpha=0.45, edge_color=THEME["soft_line"], ax=ax)
        nx.draw_networkx_nodes(
            graph,
            layout,
            node_size=[graph.nodes[node]["size"] for node in graph.nodes()],
            node_color=THEME["midnight_green"],
            edgecolors=THEME["dark_green"],
            linewidths=1.5,
            alpha=0.85,
            ax=ax,
        )
        nx.draw_networkx_labels(graph, layout, font_size=11, font_color=THEME["dark_green"], ax=ax)
        ax.set_title("Topic Similarity Network")
        ax.axis("off")
        ax.title.set_color(THEME["dark_green"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_scenario_heatmap(scenario_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "scenario_heatmap.png"
    pivot = scenario_df.pivot(index="scenario", columns="topic", values="adjusted_multimodal_score")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(THEME["beige"])
    ax.set_facecolor("#fbf8e8")
    heatmap_cmap = LinearSegmentedColormap.from_list(
        "trendcast_heatmap",
        [THEME["rosy_brown"], THEME["beige"], THEME["moss"], THEME["midnight_green"]],
    )
    image = ax.imshow(pivot.values, aspect="auto", cmap=heatmap_cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Scenario Planning Impact Heatmap")
    ax.tick_params(colors=THEME["dark_green"])
    ax.title.set_color(THEME["dark_green"])

    for row_index in range(pivot.shape[0]):
        for col_index in range(pivot.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{pivot.iloc[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                color=THEME["dark_green"],
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Adjusted multimodal score")
    colorbar.outline.set_edgecolor(THEME["soft_line"])
    colorbar.ax.yaxis.label.set_color(THEME["dark_green"])
    colorbar.ax.tick_params(colors=THEME["dark_green"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
