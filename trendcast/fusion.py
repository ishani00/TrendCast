from __future__ import annotations

import numpy as np
import pandas as pd


def _minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value - min_value == 0:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - min_value) / (max_value - min_value)


def build_multimodal_topic_summary(
    final_trends: pd.DataFrame,
    modal_summary: pd.DataFrame,
    network_nodes: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    deep_fusion_topics: pd.DataFrame,
) -> pd.DataFrame:
    merged = final_trends.merge(modal_summary, on="topic", how="left")
    merged = merged.merge(network_nodes, on="topic", how="left")
    merged = merged.merge(temporal_summary, on="topic", how="left")
    merged = merged.merge(deep_fusion_topics, on="topic", how="left")

    merged["visual_energy_norm"] = _minmax(merged["visual_energy"])
    merged["visual_innovation_norm"] = _minmax(merged["visual_innovation"])
    merged["audio_energy_norm"] = _minmax(merged["audio_energy_score"])
    merged["audio_novelty_norm"] = _minmax(merged["audio_novelty_signal"])
    merged["network_centrality_norm"] = _minmax(
        merged["weighted_degree"].fillna(0) + merged["pagerank"].fillna(0) + merged["betweenness"].fillna(0)
    )
    merged["forecast_growth_score"] = pd.to_numeric(
        merged["forecast_growth_score"], errors="coerce"
    ).fillna(0.5)
    merged["forecast_confidence"] = pd.to_numeric(
        merged["forecast_confidence"], errors="coerce"
    ).fillna(0.5)
    merged["deep_fusion_score"] = pd.to_numeric(
        merged["deep_fusion_score"], errors="coerce"
    ).fillna(0.5)

    merged["cross_modal_alignment"] = 1 - merged[
        ["trend_score", "visual_energy_norm", "audio_energy_norm"]
    ].std(axis=1).fillna(0)
    merged["cross_modal_alignment"] = merged["cross_modal_alignment"].clip(0, 1)

    merged["multimodal_final_score"] = (
        0.22 * merged["trend_score"]
        + 0.12 * merged["novelty_score_norm"]
        + 0.10 * merged["visual_energy_norm"]
        + 0.08 * merged["visual_innovation_norm"]
        + 0.11 * merged["audio_energy_norm"]
        + 0.07 * merged["audio_novelty_norm"]
        + 0.07 * merged["network_centrality_norm"]
        + 0.08 * merged["forecast_growth_score"]
        + 0.10 * merged["deep_fusion_score"]
        + 0.05 * merged["cross_modal_alignment"]
    )

    merged["multimodal_rank"] = merged["multimodal_final_score"].rank(
        ascending=False, method="dense"
    )
    return merged.sort_values("multimodal_final_score", ascending=False).reset_index(drop=True)
