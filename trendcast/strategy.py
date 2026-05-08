from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GENOME_AXES = {
    "innovation": ("ai", "digital", "future", "generated", "creative", "technology", "coding"),
    "tradition": ("repair", "craft", "ethical", "mindfulness", "meditation", "human", "slow"),
    "wellness": ("wellness", "mindfulness", "meditation", "focus", "stress", "calm", "health"),
    "community": ("community", "social", "collective", "culture", "shared", "consumer", "artists"),
    "expression": ("art", "music", "design", "visual", "creative", "fashion", "artist"),
}


def build_cultural_genome(enriched_df: pd.DataFrame, topic_summary_df: pd.DataFrame) -> pd.DataFrame:
    topic_text = (
        enriched_df.groupby("topic")["clean_text"]
        .apply(lambda values: " ".join(values.astype(str)))
        .reset_index()
    )

    rows = []
    for _, row in topic_text.iterrows():
        topic = row["topic"]
        text = row["clean_text"]
        token_count = max(len(text.split()), 1)

        axis_scores = {}
        for axis, keywords in GENOME_AXES.items():
            hits = sum(text.count(keyword) for keyword in keywords)
            axis_scores[axis] = min(1.0, hits / max(token_count * 0.04, 1.0))

        topic_row = topic_summary_df.loc[topic_summary_df["topic"] == topic].iloc[0]
        innovation_score = float(
            np.clip(
                0.6 * axis_scores["innovation"] + 0.25 * topic_row.get("visual_innovation_norm", 0.0) + 0.15 * topic_row.get("audio_novelty_norm", 0.0),
                0,
                1,
            )
        )
        tradition_score = float(np.clip(axis_scores["tradition"], 0, 1))
        wellness_score = float(np.clip(axis_scores["wellness"], 0, 1))
        community_score = float(np.clip(axis_scores["community"], 0, 1))
        expression_score = float(np.clip(axis_scores["expression"], 0, 1))

        rows.append(
            {
                "topic": topic,
                "innovation_score": innovation_score,
                "tradition_score": tradition_score,
                "wellness_score": wellness_score,
                "community_score": community_score,
                "expression_score": expression_score,
                "genome_signature": _signature_from_scores(
                    innovation_score,
                    tradition_score,
                    wellness_score,
                    community_score,
                    expression_score,
                ),
            }
        )
    return pd.DataFrame(rows)


def _signature_from_scores(
    innovation: float,
    tradition: float,
    wellness: float,
    community: float,
    expression: float,
) -> str:
    axes = {
        "innovation": innovation,
        "tradition": tradition,
        "wellness": wellness,
        "community": community,
        "expression": expression,
    }
    top_axes = sorted(axes.items(), key=lambda item: item[1], reverse=True)[:2]
    return " + ".join(axis.title() for axis, _ in top_axes)


SCENARIOS = {
    "AI Regulation Crackdown": {
        "innovation_score": -0.45,
        "tradition_score": 0.10,
        "wellness_score": 0.05,
        "community_score": 0.00,
        "expression_score": -0.08,
    },
    "Climate Activism Surge": {
        "innovation_score": 0.05,
        "tradition_score": 0.18,
        "wellness_score": 0.05,
        "community_score": 0.22,
        "expression_score": 0.05,
    },
    "Creator Economy Boom": {
        "innovation_score": 0.22,
        "tradition_score": -0.05,
        "wellness_score": 0.00,
        "community_score": 0.08,
        "expression_score": 0.25,
    },
    "Mental Health Priority Shift": {
        "innovation_score": 0.00,
        "tradition_score": 0.08,
        "wellness_score": 0.30,
        "community_score": 0.10,
        "expression_score": 0.02,
    },
    "Recession Shock": {
        "innovation_score": -0.10,
        "tradition_score": 0.14,
        "wellness_score": 0.06,
        "community_score": 0.10,
        "expression_score": -0.12,
    },
}


def build_scenario_plan(multimodal_df: pd.DataFrame, genome_df: pd.DataFrame) -> pd.DataFrame:
    merged = multimodal_df.merge(genome_df, on="topic", how="left")
    rows = []
    for _, row in merged.iterrows():
        for scenario_name, weights in SCENARIOS.items():
            impact = sum(float(row.get(axis, 0.0) or 0.0) * weight for axis, weight in weights.items())
            adjusted_score = float(np.clip(float(row["multimodal_final_score"]) + impact * 0.25, 0, 1.5))
            rows.append(
                {
                    "topic": row["topic"],
                    "scenario": scenario_name,
                    "impact_score": impact,
                    "adjusted_multimodal_score": adjusted_score,
                }
            )
    return pd.DataFrame(rows)


def write_scenario_brief(scenario_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "scenario_brief.md"
    lines = ["# TrendCast Scenario Planning Brief", ""]
    for scenario_name, frame in scenario_df.groupby("scenario"):
        best_row = frame.sort_values("adjusted_multimodal_score", ascending=False).iloc[0]
        lines.append(f"## {scenario_name}")
        lines.append(
            f"Top resilient topic: {best_row['topic']} "
            f"(adjusted score {best_row['adjusted_multimodal_score']:.2f})."
        )
        lines.append("")
    output_path.write_text("\n".join(lines))
    return output_path
