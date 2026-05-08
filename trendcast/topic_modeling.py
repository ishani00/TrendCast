from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    HAS_BERTOPIC = True
except Exception:
    BERTopic = None
    HDBSCAN = None
    UMAP = None
    HAS_BERTOPIC = False


def build_topic_model_outputs(prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if HAS_BERTOPIC:
        try:
            document_topics, topic_summary = _build_real_bertopic_outputs(prepared_df)
            return document_topics, topic_summary, "BERTopic"
        except Exception:
            pass

    document_topics, topic_summary = _build_bertopic_style_topics(prepared_df)
    return document_topics, topic_summary, "BERTopic-style fallback"


def _build_real_bertopic_outputs(prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    documents = prepared_df["clean_text"].fillna("").astype(str).tolist()
    if not documents:
        return pd.DataFrame(), pd.DataFrame()

    embeddings = _build_local_embeddings(documents)
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=2000,
    )
    n_neighbors = min(15, max(2, len(documents) - 1))
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=min(5, max(2, embeddings.shape[1])),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min(10, max(3, len(documents) // 20 or 3)),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probabilities = topic_model.fit_transform(documents, embeddings)

    doc_df = prepared_df.copy()
    doc_df["bertopic_topic_id"] = topics
    if probabilities is not None and len(probabilities):
        topic_probs = []
        for idx, topic_id in enumerate(topics):
            if topic_id == -1:
                topic_probs.append(float(np.max(probabilities[idx])) if probabilities.shape[1] else 0.0)
            else:
                topic_probs.append(float(probabilities[idx][topic_id]))
        doc_df["bertopic_probability"] = topic_probs
    else:
        doc_df["bertopic_probability"] = 1.0

    topic_info = topic_model.get_topic_info()
    rows = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        topic_docs = doc_df.loc[doc_df["bertopic_topic_id"] == topic_id]
        if topic_docs.empty:
            continue
        keywords = [word for word, _ in (topic_model.get_topic(topic_id) or [])[:8]]
        dominant_source = Counter(topic_docs["source"]).most_common(1)[0][0]
        rows.append(
            {
                "bertopic_topic_id": topic_id,
                "bertopic_name": str(row.get("Name", f"Topic {topic_id}")),
                "top_keywords": ", ".join(keywords),
                "document_count": int(len(topic_docs)),
                "mean_probability": float(topic_docs["bertopic_probability"].mean()),
                "dominant_source": dominant_source,
                "dominant_topic_label": Counter(topic_docs["topic"]).most_common(1)[0][0],
            }
        )

    topic_summary = pd.DataFrame(rows).sort_values("document_count", ascending=False).reset_index(drop=True)
    return doc_df, topic_summary


def _build_local_embeddings(documents: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=2000,
    )
    tfidf = vectorizer.fit_transform(documents)
    if tfidf.shape[1] <= 1:
        return tfidf.toarray()
    n_components = min(50, max(2, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(tfidf)


def _build_bertopic_style_topics(prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    documents = prepared_df["clean_text"].fillna("").astype(str).tolist()
    if not documents:
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=2000,
    )
    tfidf = vectorizer.fit_transform(documents)

    n_docs = len(documents)
    n_topics = min(8, max(2, n_docs // 15))
    n_topics = min(n_topics, n_docs)
    if n_topics <= 1:
        n_topics = 2 if n_docs >= 2 else 1

    embedding_dims = min(50, max(2, tfidf.shape[1] - 1)) if tfidf.shape[1] > 1 else 1
    if embedding_dims > 1:
        svd = TruncatedSVD(n_components=embedding_dims, random_state=42)
        embeddings = svd.fit_transform(tfidf)
    else:
        embeddings = tfidf.toarray()

    if n_topics == 1:
        labels = np.zeros(n_docs, dtype=int)
        topic_probabilities = np.ones(n_docs, dtype=float)
    else:
        model = KMeans(n_clusters=n_topics, random_state=42, n_init=20)
        labels = model.fit_predict(embeddings)
        centroid_similarity = cosine_similarity(embeddings, model.cluster_centers_)
        probs = _softmax(centroid_similarity)
        topic_probabilities = probs[np.arange(len(labels)), labels]

    document_topics = prepared_df.copy()
    document_topics["bertopic_topic_id"] = labels
    document_topics["bertopic_probability"] = topic_probabilities

    topic_summary = _build_ctfidf_topic_summary(document_topics)
    return document_topics, topic_summary


def _build_ctfidf_topic_summary(document_topics: pd.DataFrame) -> pd.DataFrame:
    group_docs = (
        document_topics.groupby("bertopic_topic_id")["clean_text"]
        .apply(lambda texts: " ".join(texts.astype(str)))
        .reset_index()
    )
    if group_docs.empty:
        return pd.DataFrame()

    count_vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=2000)
    counts = count_vectorizer.fit_transform(group_docs["clean_text"])
    counts_arr = counts.toarray().astype(float)
    tf = counts_arr / np.clip(counts_arr.sum(axis=1, keepdims=True), 1.0, None)
    doc_freq = np.clip((counts_arr > 0).sum(axis=0), 1.0, None)
    idf = np.log(1 + len(group_docs) / doc_freq)
    ctfidf = tf * idf
    vocab = np.array(count_vectorizer.get_feature_names_out())

    rows = []
    for idx, row in group_docs.iterrows():
        topic_id = int(row["bertopic_topic_id"])
        topic_weights = ctfidf[idx]
        top_indices = np.argsort(topic_weights)[::-1][:10]
        keywords = [vocab[i] for i in top_indices if topic_weights[i] > 0]
        topic_docs = document_topics.loc[document_topics["bertopic_topic_id"] == topic_id]
        topic_name = _build_topic_name(keywords)
        dominant_source = Counter(topic_docs["source"]).most_common(1)[0][0]
        rows.append(
            {
                "bertopic_topic_id": topic_id,
                "bertopic_name": topic_name,
                "top_keywords": ", ".join(keywords[:8]),
                "document_count": int(len(topic_docs)),
                "mean_probability": float(topic_docs["bertopic_probability"].mean()),
                "dominant_source": dominant_source,
                "dominant_topic_label": Counter(topic_docs["topic"]).most_common(1)[0][0],
            }
        )
    return pd.DataFrame(rows).sort_values("document_count", ascending=False).reset_index(drop=True)


def _build_topic_name(keywords: list[str]) -> str:
    if not keywords:
        return "General discussion"
    return " / ".join(word.title() for word in keywords[:3])


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), 1e-9, None)
