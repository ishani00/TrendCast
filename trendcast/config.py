from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOPICS_FILE = PROJECT_ROOT / "config" / "topics.csv"
TOPIC_FILE_COLUMNS = (
    "name",
    "news_query",
    "spotify_query",
    "reddit_terms",
    "reddit_subreddits",
)


@dataclass(frozen=True)
class TopicQuery:
    name: str
    news_query: str
    spotify_query: str
    reddit_terms: tuple[str, ...]
    reddit_subreddits: tuple[str, ...]


DEFAULT_TOPICS: tuple[TopicQuery, ...] = (
    TopicQuery(
        name="AI music",
        news_query="AI music",
        spotify_query="AI music",
        reddit_terms=("AI music", "generative music", "music AI"),
        reddit_subreddits=("technology", "music", "artificial", "MachineLearning"),
    ),
    TopicQuery(
        name="Sustainable fashion",
        news_query="sustainable fashion",
        spotify_query="sustainable fashion",
        reddit_terms=("sustainable fashion", "ethical fashion", "slow fashion"),
        reddit_subreddits=("fashion", "femalefashionadvice", "malefashionadvice", "worldnews"),
    ),
    TopicQuery(
        name="Mindfulness",
        news_query="mindfulness",
        spotify_query="meditation",
        reddit_terms=("mindfulness", "meditation", "mental wellness"),
        reddit_subreddits=("Meditation", "mindfulness", "DecidingToBeBetter", "worldnews"),
    ),
    TopicQuery(
        name="Digital art",
        news_query="digital art",
        spotify_query="digital art",
        reddit_terms=("digital art", "creative coding", "AI art"),
        reddit_subreddits=("digitalart", "art", "technology", "datascience"),
    ),
)

TOPICS: tuple[TopicQuery, ...] = DEFAULT_TOPICS


@dataclass(frozen=True)
class Settings:
    news_api_key: str | None
    spotify_client_id: str | None
    spotify_client_secret: str | None
    reddit_user_agent: str
    x_posts_csv: str | None
    x_bearer_token: str | None


DEFAULT_REDDIT_USER_AGENT = "TrendCast academic capstone project"


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _split_multi_value(value: object) -> tuple[str, ...]:
    text = _normalize_text(value)
    if not text:
        return ()
    if "|" in text:
        parts = text.split("|")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = (text,)
    return tuple(part.strip() for part in parts if part.strip())


def _join_multi_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        parts = [_normalize_text(item) for item in value]
        return " | ".join(part for part in parts if part)
    return _normalize_text(value)


def _resolve_topics_path(path: str | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_TOPICS_FILE

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    return (PROJECT_ROOT / candidate).resolve()


def load_topics(path: str | Path | None = None) -> tuple[TopicQuery, ...]:
    topic_path = _resolve_topics_path(path)
    if not topic_path.exists():
        return DEFAULT_TOPICS

    with topic_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in TOPIC_FILE_COLUMNS if column not in fieldnames]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Topic config {topic_path} is missing required columns: {missing}")

        topics: list[TopicQuery] = []
        for row in reader:
            name = _normalize_text(row.get("name"))
            if not name:
                continue

            news_query = _normalize_text(row.get("news_query")) or name
            spotify_query = _normalize_text(row.get("spotify_query")) or news_query
            reddit_terms = _split_multi_value(row.get("reddit_terms")) or (name,)
            reddit_subreddits = _split_multi_value(row.get("reddit_subreddits")) or ("*",)
            topics.append(
                TopicQuery(
                    name=name,
                    news_query=news_query,
                    spotify_query=spotify_query,
                    reddit_terms=reddit_terms,
                    reddit_subreddits=reddit_subreddits,
                )
            )

    if not topics:
        raise ValueError(f"Topic config {topic_path} does not contain any valid topic rows.")

    return tuple(topics)


def topic_queries_to_rows(topics: tuple[TopicQuery, ...] | list[TopicQuery]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for topic in topics:
        rows.append(
            {
                "name": topic.name,
                "news_query": topic.news_query,
                "spotify_query": topic.spotify_query,
                "reddit_terms": _join_multi_value(topic.reddit_terms),
                "reddit_subreddits": _join_multi_value(topic.reddit_subreddits),
            }
        )
    return rows


def save_topics_file(rows: list[dict[str, object]], path: str | Path | None = None) -> Path:
    topic_path = _resolve_topics_path(path)
    topic_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        name = _normalize_text(row.get("name"))
        if not name:
            continue
        news_query = _normalize_text(row.get("news_query")) or name
        spotify_query = _normalize_text(row.get("spotify_query")) or news_query
        reddit_terms = _join_multi_value(row.get("reddit_terms")) or name
        reddit_subreddits = _join_multi_value(row.get("reddit_subreddits")) or "*"
        normalized_rows.append(
            {
                "name": name,
                "news_query": news_query,
                "spotify_query": spotify_query,
                "reddit_terms": reddit_terms,
                "reddit_subreddits": reddit_subreddits,
            }
        )

    if not normalized_rows:
        raise ValueError("Save aborted because at least one topic row with a name is required.")

    with topic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOPIC_FILE_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return topic_path


def load_env_file(path: str | Path = ".env") -> None:
    """Load a simple KEY=VALUE file without adding a dependency."""
    candidate_paths = []
    explicit_path = Path(path)
    candidate_paths.append(explicit_path)
    candidate_paths.append(Path.cwd() / ".env")
    candidate_paths.append(Path.cwd() / "outputs" / ".env")

    seen_paths: set[Path] = set()
    for env_path in candidate_paths:
        env_path = env_path.resolve()
        if env_path in seen_paths or not env_path.exists():
            continue
        seen_paths.add(env_path)

        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)


def load_settings() -> Settings:
    return Settings(
        news_api_key=os.getenv("NEWS_API_KEY"),
        spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", DEFAULT_REDDIT_USER_AGENT),
        x_posts_csv=os.getenv("X_POSTS_CSV"),
        x_bearer_token=os.getenv("X_BEARER_TOKEN"),
    )
