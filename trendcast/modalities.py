from __future__ import annotations

from io import BytesIO

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


IMAGE_TIMEOUT = 10

GENRE_KEYWORDS = {
    "ambient": ("meditation", "sleep", "calm", "mindfulness", "healing", "zen"),
    "electronic": ("ai", "digital", "synth", "cyber", "future", "beat"),
    "pop": ("love", "dance", "dream", "summer", "heart", "radio"),
    "indie": ("lofi", "bedroom", "indie", "acoustic", "soft"),
    "classical": ("piano", "orchestra", "instrumental", "string", "sonata"),
    "hiphop": ("rap", "trap", "flow", "bars", "beat"),
}


def _stable_topic_seed(topic_name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(topic_name)) % (2**32 - 1)


def _topic_visual_defaults(topic: str) -> dict[str, float]:
    defaults = {
        "AI music": {
            "visual_brightness": 0.58,
            "visual_colorfulness": 0.62,
            "visual_edge_density": 0.60,
        },
        "Sustainable fashion": {
            "visual_brightness": 0.66,
            "visual_colorfulness": 0.48,
            "visual_edge_density": 0.42,
        },
        "Mindfulness": {
            "visual_brightness": 0.72,
            "visual_colorfulness": 0.34,
            "visual_edge_density": 0.24,
        },
        "Digital art": {
            "visual_brightness": 0.63,
            "visual_colorfulness": 0.82,
            "visual_edge_density": 0.74,
        },
    }
    if topic in defaults:
        return defaults[topic]

    seed = _stable_topic_seed(topic)
    return {
        "visual_brightness": round(0.48 + (seed % 24) / 100, 3),
        "visual_colorfulness": round(0.34 + ((seed // 3) % 40) / 100, 3),
        "visual_edge_density": round(0.28 + ((seed // 5) % 42) / 100, 3),
    }


def _topic_audio_defaults(topic: str) -> dict[str, float]:
    defaults = {
        "AI music": {"audio_energy_score": 0.76, "audio_novelty_signal": 0.69},
        "Sustainable fashion": {"audio_energy_score": 0.46, "audio_novelty_signal": 0.42},
        "Mindfulness": {"audio_energy_score": 0.24, "audio_novelty_signal": 0.30},
        "Digital art": {"audio_energy_score": 0.61, "audio_novelty_signal": 0.72},
    }
    if topic in defaults:
        return defaults[topic]

    seed = _stable_topic_seed(topic)
    return {
        "audio_energy_score": round(0.38 + (seed % 30) / 100, 3),
        "audio_novelty_signal": round(0.34 + ((seed // 7) % 34) / 100, 3),
    }


def _classify_audio_genre(row: pd.Series) -> tuple[str, float]:
    text = str(row.get("text", "") or "").lower()
    keyword_hits = {
        genre: sum(keyword in text for keyword in keywords)
        for genre, keywords in GENRE_KEYWORDS.items()
    }
    keyword_genre = max(keyword_hits, key=keyword_hits.get) if keyword_hits else "electronic"

    danceability = pd.to_numeric(row.get("audio_danceability"), errors="coerce")
    energy = pd.to_numeric(row.get("audio_energy"), errors="coerce")
    valence = pd.to_numeric(row.get("audio_valence"), errors="coerce")
    acousticness = pd.to_numeric(row.get("audio_acousticness"), errors="coerce")
    speechiness = pd.to_numeric(row.get("audio_speechiness"), errors="coerce")
    instrumentalness = pd.to_numeric(row.get("audio_instrumentalness"), errors="coerce")

    if pd.isna(danceability) or pd.isna(energy):
        if keyword_hits.get(keyword_genre, 0) > 0:
            return keyword_genre, 0.75
        topic_defaults = {
            "AI music": "electronic",
            "Digital art": "electronic",
            "Mindfulness": "ambient",
            "Sustainable fashion": "indie",
        }
        return topic_defaults.get(str(row.get("topic", "")), "electronic"), 0.55

    danceability = float(danceability)
    energy = float(energy)
    valence = float(0.5 if pd.isna(valence) else valence)
    acousticness = float(0.0 if pd.isna(acousticness) else acousticness)
    speechiness = float(0.0 if pd.isna(speechiness) else speechiness)
    instrumentalness = float(0.0 if pd.isna(instrumentalness) else instrumentalness)

    if acousticness > 0.65 and instrumentalness > 0.35:
        return "ambient", 0.82
    if speechiness > 0.33:
        return "hiphop", 0.78
    if energy > 0.72 and danceability > 0.58:
        return "electronic", 0.80
    if acousticness > 0.45 and energy < 0.45:
        return "indie", 0.73
    if instrumentalness > 0.55 and acousticness > 0.45:
        return "classical", 0.70
    if valence > 0.58 and danceability > 0.55:
        return "pop", 0.68
    return keyword_genre, 0.60 if keyword_hits.get(keyword_genre, 0) else 0.52


def _download_image_features(url: str) -> dict[str, float] | None:
    if not url or not url.startswith("http"):
        return None

    try:
        response = requests.get(url, timeout=IMAGE_TIMEOUT)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB").resize((128, 128))
    except Exception:
        return None

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    brightness = float(rgb.mean())

    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    saturation = np.divide(
        max_rgb - min_rgb,
        np.where(max_rgb == 0, 1, max_rgb),
    )
    avg_saturation = float(saturation.mean())

    rg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
    yb = np.abs(0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2])
    colorfulness = float(
        min(
            1.0,
            (
                np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
            )
            / 1.5,
        )
    )

    gray = rgb.mean(axis=2)
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float(min(1.0, (grad_x + grad_y) * 6.0))

    return {
        "visual_brightness": brightness,
        "visual_colorfulness": colorfulness,
        "visual_saturation": avg_saturation,
        "visual_edge_density": edge_density,
    }


def enrich_modalities(prepared_df: pd.DataFrame) -> pd.DataFrame:
    enriched = prepared_df.copy()
    image_cache: dict[str, dict[str, float] | None] = {}
    visual_rows: list[dict[str, float]] = []
    audio_rows: list[dict[str, float]] = []

    for _, row in enriched.iterrows():
        media_url = str(row.get("media_url", "") or "")
        topic = str(row.get("topic", ""))
        sentiment = float(row.get("sentiment", 0.0) or 0.0)

        if media_url not in image_cache:
            image_cache[media_url] = _download_image_features(media_url)
        image_features = image_cache[media_url]

        if image_features is None:
            defaults = _topic_visual_defaults(topic)
            image_features = {
                "visual_brightness": min(1.0, max(0.0, defaults["visual_brightness"] + sentiment * 0.10)),
                "visual_colorfulness": defaults["visual_colorfulness"],
                "visual_saturation": min(1.0, defaults["visual_colorfulness"] * 0.85 + 0.08),
                "visual_edge_density": defaults["visual_edge_density"],
            }

        visual_energy = (
            0.35 * image_features["visual_brightness"]
            + 0.35 * image_features["visual_colorfulness"]
            + 0.30 * image_features["visual_edge_density"]
        )
        visual_innovation = (
            0.45 * image_features["visual_colorfulness"]
            + 0.35 * image_features["visual_edge_density"]
            + 0.20 * (1 - abs(0.5 - image_features["visual_saturation"]))
        )
        visual_rows.append(
            {
                **image_features,
                "visual_energy": float(visual_energy),
                "visual_innovation": float(visual_innovation),
            }
        )

        danceability = pd.to_numeric(row.get("audio_danceability"), errors="coerce")
        energy = pd.to_numeric(row.get("audio_energy"), errors="coerce")
        valence = pd.to_numeric(row.get("audio_valence"), errors="coerce")
        acousticness = pd.to_numeric(row.get("audio_acousticness"), errors="coerce")
        instrumentalness = pd.to_numeric(row.get("audio_instrumentalness"), errors="coerce")
        speechiness = pd.to_numeric(row.get("audio_speechiness"), errors="coerce")
        tempo = pd.to_numeric(row.get("audio_tempo"), errors="coerce")

        if pd.isna(danceability) or pd.isna(energy) or pd.isna(valence):
            defaults = _topic_audio_defaults(topic)
            audio_energy_score = defaults["audio_energy_score"]
            audio_novelty_signal = defaults["audio_novelty_signal"]
            tempo_norm = defaults["audio_energy_score"]
        else:
            acousticness = float(0.0 if pd.isna(acousticness) else acousticness)
            instrumentalness = float(0.0 if pd.isna(instrumentalness) else instrumentalness)
            speechiness = float(0.0 if pd.isna(speechiness) else speechiness)
            tempo_norm = float(min(1.0, max(0.0, ((float(tempo or 120.0) - 60.0) / 120.0))))
            audio_energy_score = float(
                np.clip(
                    0.35 * float(energy)
                    + 0.25 * float(danceability)
                    + 0.20 * float(valence)
                    + 0.20 * tempo_norm,
                    0,
                    1,
                )
            )
            audio_novelty_signal = float(
                np.clip(
                    0.35 * (1 - acousticness)
                    + 0.30 * instrumentalness
                    + 0.20 * speechiness
                    + 0.15 * tempo_norm,
                    0,
                    1,
                )
            )

        genre_label, genre_confidence = _classify_audio_genre(row)
        audio_rows.append(
            {
                "audio_energy_score": float(audio_energy_score),
                "audio_novelty_signal": float(audio_novelty_signal),
                "audio_tempo_norm": float(tempo_norm),
                "audio_genre_label": genre_label,
                "audio_genre_confidence": genre_confidence,
            }
        )

    visual_df = pd.DataFrame(visual_rows)
    audio_df = pd.DataFrame(audio_rows)
    for col in visual_df.columns:
        enriched[col] = visual_df[col].values
    for col in audio_df.columns:
        enriched[col] = audio_df[col].values
    return enriched


def build_modal_topic_summary(enriched_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        enriched_df.groupby("topic")
        .agg(
            visual_brightness=("visual_brightness", "mean"),
            visual_colorfulness=("visual_colorfulness", "mean"),
            visual_edge_density=("visual_edge_density", "mean"),
            visual_energy=("visual_energy", "mean"),
            visual_innovation=("visual_innovation", "mean"),
            audio_energy_score=("audio_energy_score", "mean"),
            audio_novelty_signal=("audio_novelty_signal", "mean"),
            audio_tempo_norm=("audio_tempo_norm", "mean"),
            audio_genre_confidence=("audio_genre_confidence", "mean"),
            modality_coverage=("source", "nunique"),
        )
        .reset_index()
    )
    dominant_genres = (
        enriched_df.groupby(["topic", "audio_genre_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["topic", "count", "audio_genre_label"], ascending=[True, False, True])
        .drop_duplicates("topic")
        .rename(columns={"audio_genre_label": "dominant_audio_genre"})
    )
    return summary.merge(dominant_genres[["topic", "dominant_audio_genre"]], on="topic", how="left")


def build_audio_genre_summary(enriched_df: pd.DataFrame) -> pd.DataFrame:
    genre_counts = (
        enriched_df.groupby(["topic", "audio_genre_label"])
        .size()
        .reset_index(name="record_count")
    )
    total_counts = genre_counts.groupby("topic")["record_count"].transform("sum").replace(0, 1)
    genre_counts["share_within_topic"] = genre_counts["record_count"] / total_counts
    return genre_counts.sort_values(["topic", "share_within_topic"], ascending=[True, False]).reset_index(drop=True)


def build_topic_network(enriched_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_docs = (
        enriched_df.groupby("topic")["clean_text"]
        .apply(lambda values: " ".join(values.astype(str)))
        .reset_index()
    )

    if topic_docs.empty:
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(topic_docs["clean_text"])
    similarities = cosine_similarity(matrix)

    graph = nx.Graph()
    for topic in topic_docs["topic"]:
        graph.add_node(topic)

    candidate_edges: dict[tuple[str, str], float] = {}
    topics = topic_docs["topic"].tolist()
    for i, source_topic in enumerate(topics):
        ranked_targets = sorted(
            (
                (topics[j], float(similarities[i, j]))
                for j in range(len(topics))
                if j != i
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        for target_topic, weight in ranked_targets:
            if weight <= 0:
                continue
            edge_key = tuple(sorted((source_topic, target_topic)))
            candidate_edges[edge_key] = max(candidate_edges.get(edge_key, 0.0), weight)

    edges: list[dict] = []
    for (source_topic, target_topic), weight in candidate_edges.items():
        graph.add_edge(source_topic, target_topic, weight=weight)
        edges.append(
            {
                "source_topic": source_topic,
                "target_topic": target_topic,
                "weight": weight,
            }
        )

    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True) if graph.number_of_nodes() else {}

    nodes = []
    for topic in topic_docs["topic"]:
        weighted_degree = sum(data["weight"] for _, _, data in graph.edges(topic, data=True))
        nodes.append(
            {
                "topic": topic,
                "pagerank": float(pagerank.get(topic, 0.0)),
                "betweenness": float(betweenness.get(topic, 0.0)),
                "degree": int(graph.degree(topic)),
                "weighted_degree": float(weighted_degree),
            }
        )

    return pd.DataFrame(nodes), pd.DataFrame(edges)
