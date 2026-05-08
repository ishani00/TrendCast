from __future__ import annotations

import re

import numpy as np
import pandas as pd
from textblob import TextBlob


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentiment(text: str) -> float:
    if not text:
        return 0.0
    return float(TextBlob(text).sentiment.polarity)


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["engagement"] = pd.to_numeric(prepared["engagement"], errors="coerce").fillna(0)
    prepared["text"] = prepared["text"].fillna("")
    prepared["clean_text"] = prepared["text"].apply(clean_text)
    prepared["sentiment"] = prepared["clean_text"].apply(get_sentiment)

    # Normalize within each source because Spotify, News, and Reddit engagement scales differ.
    source_max = prepared.groupby("source")["engagement"].transform("max").replace(0, 1)
    prepared["engagement_norm"] = prepared["engagement"] / source_max
    return prepared


def compute_source_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    source_summary = (
        prepared.groupby(["topic", "source"])
        .agg(
            mention_count=("text", "count"),
            avg_engagement=("engagement_norm", "mean"),
            total_engagement_raw=("engagement", "sum"),
            avg_sentiment=("sentiment", "mean"),
        )
        .reset_index()
    )

    source_summary["mention_score"] = (
        source_summary["mention_count"] / max(source_summary["mention_count"].max(), 1)
    )
    source_summary["engagement_score"] = (
        source_summary["avg_engagement"] / max(source_summary["avg_engagement"].max(), 1e-9)
    )
    source_summary["sentiment_score"] = (source_summary["avg_sentiment"] + 1) / 2
    source_summary["trend_score"] = (
        0.4 * source_summary["mention_score"]
        + 0.4 * source_summary["engagement_score"]
        + 0.2 * source_summary["sentiment_score"]
    )
    return source_summary


def compute_final_topic_scores(source_summary: pd.DataFrame) -> pd.DataFrame:
    final_trends = (
        source_summary.groupby("topic")
        .agg(
            mention_count=("mention_count", "sum"),
            avg_engagement=("avg_engagement", "mean"),
            total_engagement=("total_engagement_raw", "sum"),
            avg_sentiment=("avg_sentiment", "mean"),
            source_count=("source", "nunique"),
        )
        .reset_index()
    )

    final_trends["mention_score"] = final_trends["mention_count"] / max(
        final_trends["mention_count"].max(), 1
    )
    final_trends["engagement_score"] = final_trends["avg_engagement"] / max(
        final_trends["avg_engagement"].max(), 1e-9
    )
    final_trends["sentiment_score"] = (final_trends["avg_sentiment"] + 1) / 2
    final_trends["trend_score"] = (
        0.4 * final_trends["mention_score"]
        + 0.4 * final_trends["engagement_score"]
        + 0.2 * final_trends["sentiment_score"]
    )

    topic_distribution = final_trends["total_engagement"] / max(
        final_trends["total_engagement"].sum(), 1
    )
    baseline = np.ones(len(final_trends)) / max(len(final_trends), 1)
    novelty_score = topic_distribution * np.log(
        (topic_distribution + 1e-10) / (baseline + 1e-10)
    )
    final_trends["novelty_score"] = novelty_score

    novelty_range = final_trends["novelty_score"].max() - final_trends["novelty_score"].min()
    if novelty_range == 0:
        final_trends["novelty_score_norm"] = 0.5
    else:
        final_trends["novelty_score_norm"] = (
            final_trends["novelty_score"] - final_trends["novelty_score"].min()
        ) / novelty_range

    final_trends["final_score_with_novelty"] = (
        0.30 * final_trends["mention_score"]
        + 0.30 * final_trends["engagement_score"]
        + 0.20 * final_trends["sentiment_score"]
        + 0.20 * final_trends["novelty_score_norm"]
    )

    final_trends["quadrant"] = final_trends.apply(_label_quadrant, axis=1)
    return final_trends.sort_values("final_score_with_novelty", ascending=False).reset_index(drop=True)


def _label_quadrant(row: pd.Series) -> str:
    trend_cutoff = 0.65
    novelty_cutoff = 0.50
    if row["trend_score"] >= trend_cutoff and row["novelty_score_norm"] >= novelty_cutoff:
        return "High trend / High novelty"
    if row["trend_score"] >= trend_cutoff:
        return "High trend / Lower novelty"
    if row["novelty_score_norm"] >= novelty_cutoff:
        return "Emerging niche"
    return "Lower priority"
