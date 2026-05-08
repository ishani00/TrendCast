from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_deep_fusion_outputs(enriched_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if enriched_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    records = enriched_df.copy()
    feature_columns_numeric = [
        "sentiment",
        "engagement_norm",
        "visual_brightness",
        "visual_colorfulness",
        "visual_edge_density",
        "visual_energy",
        "visual_innovation",
        "audio_energy_score",
        "audio_novelty_signal",
        "audio_tempo_norm",
        "audio_genre_confidence",
    ]
    feature_columns_categorical = ["source", "topic", "audio_genre_label"]

    available_numeric = [column for column in feature_columns_numeric if column in records.columns]
    available_categorical = [column for column in feature_columns_categorical if column in records.columns]

    X = records[available_numeric + available_categorical].copy()
    for column in available_numeric:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0.0)
    for column in available_categorical:
        X[column] = X[column].fillna("unknown").astype(str)
    target = (
        0.55 * pd.to_numeric(records["engagement_norm"], errors="coerce").fillna(0.0)
        + 0.25 * ((pd.to_numeric(records["sentiment"], errors="coerce").fillna(0.0) + 1.0) / 2.0)
        + 0.20 * records["topic"].map(records.groupby("topic")["engagement_norm"].mean()).fillna(0.0)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), available_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), available_categorical),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    random_state=42,
                    max_iter=900,
                ),
            ),
        ]
    )

    model.fit(X, target)
    records["deep_fusion_score"] = np.clip(model.predict(X), 0.0, 1.5)

    topic_scores = (
        records.groupby("topic")
        .agg(
            deep_fusion_score=("deep_fusion_score", "mean"),
            deep_fusion_max=("deep_fusion_score", "max"),
            audio_genre_label=("audio_genre_label", lambda labels: labels.mode().iloc[0] if not labels.mode().empty else ""),
        )
        .reset_index()
    )
    return records, topic_scores
