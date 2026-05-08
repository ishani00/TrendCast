from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from trendcast.config import Settings, TopicQuery


REQUEST_TIMEOUT = 20
REDDIT_RETRY_LIMIT = 3
REDDIT_RETRY_BASE_SECONDS = 2.0


class DataCollectionError(RuntimeError):
    """Raised when a live API collection step fails."""


def _safe_json_response(response: requests.Response) -> dict:
    try:
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        snippet = response.text[:400] if response is not None else ""
        raise DataCollectionError(f"{exc}. Response snippet: {snippet}") from exc
    except ValueError as exc:
        raise DataCollectionError("Received a non-JSON response from the API.") from exc


def _empty_status(source: str, status: str, note: str, record_count: int = 0) -> dict:
    return {
        "source": source,
        "status": status,
        "record_count": record_count,
        "note": note,
    }


def fetch_news_articles(
    topics: Iterable[TopicQuery],
    api_key: str,
    page_size: int = 20,
) -> pd.DataFrame:
    rows: list[dict] = []
    for topic in topics:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic.news_query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": api_key,
            },
            timeout=REQUEST_TIMEOUT,
        )
        data = _safe_json_response(response)

        for item in data.get("articles", []):
            title = str(item.get("title", "") or "")
            description = str(item.get("description", "") or "")
            rows.append(
                {
                    "date": str(item.get("publishedAt", ""))[:10],
                    "source": "NewsAPI",
                    "topic": topic.name,
                    "text": f"{title} {description}".strip(),
                    "engagement": max(len(title) / 10, 1),
                    "detail": str(item.get("url", "") or ""),
                    "media_url": str(item.get("urlToImage", "") or ""),
                    "source_item_id": str(item.get("url", "") or ""),
                    "modality_hint": "text+image",
                }
            )

    return pd.DataFrame(rows)


def _get_spotify_access_token(client_id: str, client_secret: str) -> str:
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=REQUEST_TIMEOUT,
    )
    data = _safe_json_response(response)
    token = data.get("access_token")
    if not token:
        raise DataCollectionError("Spotify authentication succeeded without an access token.")
    return token


def _fetch_spotify_audio_features(token: str, track_ids: list[str]) -> dict[str, dict]:
    if not track_ids:
        return {}

    headers = {"Authorization": f"Bearer {token}"}
    features_by_id: dict[str, dict] = {}
    for start in range(0, len(track_ids), 100):
        batch = track_ids[start : start + 100]
        response = requests.get(
            "https://api.spotify.com/v1/audio-features",
            headers=headers,
            params={"ids": ",".join(batch)},
            timeout=REQUEST_TIMEOUT,
        )
        try:
            data = _safe_json_response(response)
        except DataCollectionError:
            return {}
        for item in data.get("audio_features", []) or []:
            if item and item.get("id"):
                features_by_id[item["id"]] = item
    return features_by_id


def fetch_spotify_tracks(
    topics: Iterable[TopicQuery],
    client_id: str,
    client_secret: str,
    limit: int = 10,
) -> tuple[pd.DataFrame, str]:
    token = _get_spotify_access_token(client_id, client_secret)
    rows: list[dict] = []
    headers = {"Authorization": f"Bearer {token}"}
    topic_tracks: list[dict] = []
    track_ids: list[str] = []

    for topic in topics:
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params={"q": topic.spotify_query, "type": "track", "limit": limit},
            timeout=REQUEST_TIMEOUT,
        )
        data = _safe_json_response(response)
        tracks = data.get("tracks", {}).get("items", [])

        for item in tracks:
            track_id = str(item.get("id", "") or "")
            if track_id:
                track_ids.append(track_id)
            topic_tracks.append(
                {
                    "topic": topic.name,
                    "item": item,
                    "track_id": track_id,
                }
            )

    audio_features = _fetch_spotify_audio_features(token, track_ids)
    audio_features_note = (
        "Spotify tracks collected with audio features."
        if audio_features
        else "Spotify tracks collected, but audio-features access was unavailable."
    )

    for record in topic_tracks:
        item = record["item"]
        track_id = record["track_id"]
        artist_name = item.get("artists", [{}])[0].get("name", "Unknown artist")
        album_images = item.get("album", {}).get("images", []) or []
        primary_image = album_images[0]["url"] if album_images else ""
        feature_row = audio_features.get(track_id, {})
        rows.append(
            {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "source": "Spotify",
                "topic": record["topic"],
                "text": f"{item.get('name', '')} by {artist_name}".strip(),
                "engagement": item.get("popularity", 0),
                "detail": item.get("external_urls", {}).get("spotify", ""),
                "media_url": primary_image,
                "source_item_id": track_id,
                "preview_url": item.get("preview_url") or "",
                "artist_name": artist_name,
                "modality_hint": "text+audio+image",
                "audio_danceability": feature_row.get("danceability"),
                "audio_energy": feature_row.get("energy"),
                "audio_valence": feature_row.get("valence"),
                "audio_tempo": feature_row.get("tempo"),
                "audio_acousticness": feature_row.get("acousticness"),
                "audio_instrumentalness": feature_row.get("instrumentalness"),
                "audio_speechiness": feature_row.get("speechiness"),
            }
        )

    return pd.DataFrame(rows), audio_features_note


def fetch_reddit_posts(
    topics: Iterable[TopicQuery],
    user_agent: str,
    limit_per_search: int = 4,
    pause_seconds: float = 1.25,
    max_posts_per_topic: int = 18,
) -> tuple[pd.DataFrame, str, bool]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    headers = {"User-Agent": user_agent}
    successful_queries = 0
    rate_limited_queries = 0
    failed_queries = 0

    def _request_reddit(url: str, params: dict) -> dict | None:
        nonlocal successful_queries, rate_limited_queries, failed_queries

        last_error: str | None = None
        for attempt in range(REDDIT_RETRY_LIMIT):
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            try:
                data = _safe_json_response(response)
                successful_queries += 1
                return data
            except DataCollectionError as exc:
                last_error = str(exc)
                if response.status_code == 429:
                    rate_limited_queries += 1
                    retry_after_header = response.headers.get("Retry-After")
                    if retry_after_header:
                        try:
                            retry_after = max(float(retry_after_header), 0.0)
                        except ValueError:
                            retry_after = REDDIT_RETRY_BASE_SECONDS * (attempt + 1)
                    else:
                        retry_after = REDDIT_RETRY_BASE_SECONDS * (attempt + 1)
                    time.sleep(retry_after)
                    continue

                failed_queries += 1
                return None

        failed_queries += 1
        return None if last_error else None

    for topic in topics:
        topic_row_count_before = len(rows)
        subreddit_targets = topic.reddit_subreddits or ("*",)
        for term in topic.reddit_terms:
            for subreddit in subreddit_targets:
                is_sitewide = subreddit in {"*", "sitewide"}
                url = "https://www.reddit.com/search.json" if is_sitewide else f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": term,
                    "sort": "top",
                    "t": "month",
                    "limit": limit_per_search,
                }
                if not is_sitewide:
                    params["restrict_sr"] = 1
                data = _request_reddit(url, params)
                if data is None:
                    time.sleep(pause_seconds)
                    continue

                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})
                    post_id = str(post_data.get("id", ""))
                    if not post_id or post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)

                    thumbnail = post_data.get("thumbnail", "")
                    media_url = thumbnail if isinstance(thumbnail, str) and thumbnail.startswith("http") else ""
                    rows.append(
                        {
                            "date": datetime.fromtimestamp(
                                post_data.get("created_utc", 0), tz=timezone.utc
                            ).date().isoformat(),
                            "source": "Reddit",
                            "topic": topic.name,
                            "text": (
                                f"{post_data.get('title', '')} "
                                f"{post_data.get('selftext', '')}"
                            ).strip(),
                            "engagement": post_data.get("score", 0)
                            + post_data.get("num_comments", 0),
                            "detail": f"https://www.reddit.com{post_data.get('permalink', '')}",
                            "media_url": media_url,
                            "source_item_id": post_id,
                            "modality_hint": "text",
                        }
                    )
                time.sleep(pause_seconds)
                if len(rows) - topic_row_count_before >= max_posts_per_topic:
                    break
            if len(rows) - topic_row_count_before >= max_posts_per_topic:
                break

    note_parts = [f"Reddit posts collected from {successful_queries} successful queries."]
    if rate_limited_queries:
        note_parts.append(f"{rate_limited_queries} queries were rate-limited and skipped after retry.")
    if failed_queries:
        note_parts.append(f"{failed_queries} queries failed.")
    note = " ".join(note_parts)

    had_partial_failures = bool(rate_limited_queries or failed_queries)

    if not rows and had_partial_failures:
        raise DataCollectionError(note)

    return pd.DataFrame(rows), note, had_partial_failures


def _match_topic_from_text(text: str, topics: Iterable[TopicQuery]) -> str | None:
    lower_text = text.lower()
    for topic in topics:
        terms = {topic.name.lower(), topic.news_query.lower(), topic.spotify_query.lower(), *[term.lower() for term in topic.reddit_terms]}
        if any(term in lower_text for term in terms):
            return topic.name
    return None


def fetch_x_posts_from_csv(csv_path: str | Path, topics: Iterable[TopicQuery]) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise DataCollectionError(f"X/Twitter CSV not found at {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in raw.iterrows():
        text = str(row.get("text", "") or row.get("full_text", "") or "")
        topic = row.get("topic")
        if not topic or pd.isna(topic):
            topic = _match_topic_from_text(text, topics)
        if not topic:
            continue

        engagement = row.get("engagement")
        if pd.isna(engagement):
            engagement = 0.0
            for col in ("likes", "retweets", "reposts", "replies", "quotes", "bookmarks"):
                value = pd.to_numeric(row.get(col, 0), errors="coerce")
                if pd.isna(value):
                    value = 0.0
                engagement += float(value)

        rows.append(
            {
                "date": str(row.get("date", row.get("created_at", "")))[:10],
                "source": "X",
                "topic": str(topic),
                "text": text,
                "engagement": float(engagement or 0),
                "detail": str(row.get("url", row.get("tweet_url", "")) or ""),
                "media_url": str(row.get("media_url", row.get("image_url", "")) or ""),
                "source_item_id": str(row.get("id", row.get("tweet_id", "")) or ""),
                "modality_hint": "text+image",
            }
        )

    return pd.DataFrame(rows)


def fetch_x_posts_from_api(
    topics: Iterable[TopicQuery],
    bearer_token: str,
    max_results_per_topic: int = 25,
) -> pd.DataFrame:
    rows: list[dict] = []
    headers = {"Authorization": f"Bearer {bearer_token}"}

    for topic in topics:
        query_terms = topic.reddit_terms[:3] if topic.reddit_terms else (topic.name,)
        joined_terms = " OR ".join(f'"{term}"' for term in query_terms)
        query = f"({joined_terms}) lang:en -is:retweet"
        response = requests.get(
            "https://api.x.com/2/tweets/search/recent",
            headers=headers,
            params={
                "query": query,
                "max_results": min(100, max(10, max_results_per_topic)),
                "tweet.fields": "created_at,public_metrics,attachments",
                "expansions": "attachments.media_keys",
                "media.fields": "url,preview_image_url",
            },
            timeout=REQUEST_TIMEOUT,
        )
        data = _safe_json_response(response)

        media_lookup: dict[str, str] = {}
        for media in data.get("includes", {}).get("media", []) or []:
            media_key = str(media.get("media_key", "") or "")
            media_url = str(media.get("url", media.get("preview_image_url", "")) or "")
            if media_key and media_url:
                media_lookup[media_key] = media_url

        for item in data.get("data", []) or []:
            metrics = item.get("public_metrics", {}) or {}
            media_keys = item.get("attachments", {}).get("media_keys", []) or []
            media_url = media_lookup.get(media_keys[0], "") if media_keys else ""
            rows.append(
                {
                    "date": str(item.get("created_at", ""))[:10],
                    "source": "X",
                    "topic": topic.name,
                    "text": str(item.get("text", "") or ""),
                    "engagement": float(
                        metrics.get("like_count", 0)
                        + metrics.get("retweet_count", 0)
                        + metrics.get("reply_count", 0)
                        + metrics.get("quote_count", 0)
                    ),
                    "detail": f"https://x.com/i/web/status/{item.get('id', '')}",
                    "media_url": media_url,
                    "source_item_id": str(item.get("id", "") or ""),
                    "modality_hint": "text+image",
                }
            )
    return pd.DataFrame(rows)


def _stable_topic_seed(topic_name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(topic_name)) % (2**32 - 1)


def _proxy_spotify_audio(topic_name: str) -> dict[str, float]:
    defaults = {
        "AI music": {
            "audio_danceability": 0.61,
            "audio_energy": 0.76,
            "audio_valence": 0.48,
            "audio_tempo": 124.0,
            "audio_acousticness": 0.18,
            "audio_instrumentalness": 0.31,
            "audio_speechiness": 0.10,
        },
        "Sustainable fashion": {
            "audio_danceability": 0.55,
            "audio_energy": 0.47,
            "audio_valence": 0.58,
            "audio_tempo": 108.0,
            "audio_acousticness": 0.36,
            "audio_instrumentalness": 0.12,
            "audio_speechiness": 0.07,
        },
        "Mindfulness": {
            "audio_danceability": 0.35,
            "audio_energy": 0.24,
            "audio_valence": 0.67,
            "audio_tempo": 82.0,
            "audio_acousticness": 0.72,
            "audio_instrumentalness": 0.55,
            "audio_speechiness": 0.05,
        },
        "Digital art": {
            "audio_danceability": 0.57,
            "audio_energy": 0.63,
            "audio_valence": 0.42,
            "audio_tempo": 118.0,
            "audio_acousticness": 0.21,
            "audio_instrumentalness": 0.28,
            "audio_speechiness": 0.09,
        },
    }
    if topic_name in defaults:
        return defaults[topic_name]

    seed = _stable_topic_seed(topic_name)
    return {
        "audio_danceability": round(0.40 + (seed % 22) / 100, 3),
        "audio_energy": round(0.36 + ((seed // 3) % 32) / 100, 3),
        "audio_valence": round(0.38 + ((seed // 5) % 28) / 100, 3),
        "audio_tempo": float(88 + seed % 48),
        "audio_acousticness": round(0.18 + ((seed // 7) % 38) / 100, 3),
        "audio_instrumentalness": round(0.10 + ((seed // 11) % 42) / 100, 3),
        "audio_speechiness": round(0.04 + ((seed // 13) % 12) / 100, 3),
    }


def _build_generic_demo_profile(topic: TopicQuery) -> dict[str, object]:
    seed = _stable_topic_seed(topic.name)
    reference_term = topic.reddit_terms[0] if topic.reddit_terms else topic.news_query
    alternate_term = topic.reddit_terms[1] if len(topic.reddit_terms) > 1 else topic.spotify_query
    slug = topic.name.lower().replace(" ", "-")
    return {
        "base_engagement": {
            "NewsAPI": 4 + seed % 4,
            "Spotify": 34 + seed % 42,
            "Reddit": 120 + seed % 140,
            "X": 90 + seed % 120,
        },
        "records_per_source": 8 + seed % 5,
        "snippets": (
            f"online chatter is building around {topic.name.lower()}",
            f"communities connect {reference_term.lower()} with broader cultural shifts",
            f"creators and fans debate where {alternate_term.lower()} is headed next",
        ),
        "visual_url": f"demo://{slug}",
    }


def build_demo_dataset(topics: Iterable[TopicQuery]) -> pd.DataFrame:
    """Create deterministic sample data so the project can run without live APIs."""
    rng = np.random.default_rng(42)
    topic_profiles = {
        "AI music": {
            "base_engagement": {"NewsAPI": 5, "Spotify": 82, "Reddit": 220, "X": 170},
            "records_per_source": 14,
            "snippets": (
                "exciting breakthrough in AI music tools",
                "artists debate the future of generative sound design",
                "new release blends human vocals with machine composition",
            ),
            "visual_url": "demo://ai-music",
        },
        "Sustainable fashion": {
            "base_engagement": {"NewsAPI": 6, "Spotify": 45, "Reddit": 160, "X": 120},
            "records_per_source": 10,
            "snippets": (
                "thoughtful coverage of circular fashion and repair culture",
                "consumers respond positively to eco friendly design",
                "brands face pressure to prove sustainability claims",
            ),
            "visual_url": "demo://sustainable-fashion",
        },
        "Mindfulness": {
            "base_engagement": {"NewsAPI": 4, "Spotify": 72, "Reddit": 140, "X": 135},
            "records_per_source": 8,
            "snippets": (
                "calm discussion about mindfulness routines and focus",
                "listeners embrace meditation playlists for stress relief",
                "community members share helpful wellness habits",
            ),
            "visual_url": "demo://mindfulness",
        },
        "Digital art": {
            "base_engagement": {"NewsAPI": 5, "Spotify": 38, "Reddit": 260, "X": 210},
            "records_per_source": 12,
            "snippets": (
                "vibrant digital art communities celebrate new creators",
                "debate grows around creative coding and AI art authorship",
                "visual artists experiment with immersive digital canvases",
            ),
            "visual_url": "demo://digital-art",
        },
    }

    rows: list[dict] = []
    current_date = datetime.now(timezone.utc).date()
    for topic in topics:
        profile = topic_profiles.get(topic.name, _build_generic_demo_profile(topic))
        for source, base_engagement in profile["base_engagement"].items():
            for idx in range(profile["records_per_source"]):
                snippet = profile["snippets"][idx % len(profile["snippets"])]
                row = {
                    "date": (current_date - pd.Timedelta(days=int(rng.integers(0, 60)))).isoformat(),
                    "source": source,
                    "topic": topic.name,
                    "text": (
                        f"{topic.name} signal {idx + 1}: {snippet}. "
                        f"This sample item represents how {topic.name.lower()} is discussed online."
                    ),
                    "engagement": max(
                        1,
                        base_engagement
                        + int(rng.normal(0, base_engagement * 0.12 + 2)),
                    ),
                    "detail": f"demo://{source.lower()}",
                    "media_url": profile["visual_url"],
                    "source_item_id": f"{source.lower()}-{topic.name.lower().replace(' ', '-')}-{idx}",
                    "modality_hint": "text",
                }
                if source == "Spotify":
                    row["modality_hint"] = "text+audio+image"
                    row.update(_proxy_spotify_audio(topic.name))
                elif source in {"NewsAPI", "X"}:
                    row["modality_hint"] = "text+image"
                rows.append(row)

    return pd.DataFrame(rows)


def build_demo_source_status(topics: Iterable[TopicQuery]) -> pd.DataFrame:
    topic_count = len(tuple(topics))
    base_rows = []
    for source in ("NewsAPI", "Spotify", "Reddit", "X"):
        base_rows.append(
            _empty_status(
                source=source,
                status="demo",
                note=f"Demo records generated for {topic_count} topics.",
                record_count=1,
            )
        )
    return pd.DataFrame(base_rows)


def collect_live_data(
    settings: Settings,
    topics: Iterable[TopicQuery],
    news_page_size: int,
    spotify_limit: int,
    reddit_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    statuses: list[dict] = []

    if settings.news_api_key:
        try:
            news_df = fetch_news_articles(topics, settings.news_api_key, page_size=news_page_size)
            frames.append(news_df)
            statuses.append(
                _empty_status("NewsAPI", "loaded", "NewsAPI articles collected.", len(news_df))
            )
        except Exception as exc:
            statuses.append(_empty_status("NewsAPI", "failed", str(exc)))
    else:
        statuses.append(_empty_status("NewsAPI", "skipped", "NEWS_API_KEY not found."))

    if settings.spotify_client_id and settings.spotify_client_secret:
        try:
            spotify_df, spotify_note = fetch_spotify_tracks(
                topics,
                settings.spotify_client_id,
                settings.spotify_client_secret,
                limit=spotify_limit,
            )
            frames.append(spotify_df)
            statuses.append(
                _empty_status("Spotify", "loaded", spotify_note, len(spotify_df))
            )
        except Exception as exc:
            statuses.append(_empty_status("Spotify", "failed", str(exc)))
    else:
        statuses.append(
            _empty_status("Spotify", "skipped", "SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not found.")
        )

    try:
        reddit_df, reddit_note, reddit_partial = fetch_reddit_posts(
            topics,
            settings.reddit_user_agent,
            limit_per_search=reddit_limit,
        )
        frames.append(reddit_df)
        reddit_status = "loaded_partial" if reddit_partial else "loaded"
        statuses.append(_empty_status("Reddit", reddit_status, reddit_note, len(reddit_df)))
    except Exception as exc:
        statuses.append(_empty_status("Reddit", "failed", str(exc)))

    if settings.x_bearer_token:
        try:
            x_df = fetch_x_posts_from_api(topics, settings.x_bearer_token)
            frames.append(x_df)
            statuses.append(
                _empty_status(
                    "X",
                    "loaded",
                    "X recent-search API posts collected.",
                    len(x_df),
                )
            )
        except Exception as exc:
            statuses.append(_empty_status("X", "failed", str(exc)))
    elif settings.x_posts_csv:
        try:
            x_df = fetch_x_posts_from_csv(settings.x_posts_csv, topics)
            frames.append(x_df)
            statuses.append(
                _empty_status(
                    "X",
                    "loaded",
                    "X/Twitter posts loaded from CSV export.",
                    len(x_df),
                )
            )
        except Exception as exc:
            statuses.append(_empty_status("X", "failed", str(exc)))
    else:
        statuses.append(
            _empty_status(
                "X",
                "skipped",
                "Neither X_BEARER_TOKEN nor X_POSTS_CSV was provided.",
            )
        )

    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        raise DataCollectionError(
            "No live data was collected. Add API credentials or use demo mode."
        )

    return pd.concat(non_empty_frames, ignore_index=True), pd.DataFrame(statuses)
