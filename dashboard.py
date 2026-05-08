from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from trendcast.config import (
    DEFAULT_TOPICS_FILE,
    TOPIC_FILE_COLUMNS,
    save_topics_file,
    load_topics,
    topic_queries_to_rows,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ACTIVE_SEARCH_TOPICS_FILE = OUTPUTS_DIR / "active_search_topics.csv"

UI_COLORS = {
    "bg_top": "#F7F4D5",
    "bg_mid": "#efe9c9",
    "bg_bottom": "#e6ddbc",
    "accent_glow": "rgba(211, 150, 140, 0.20)",
    "text_main": "#105666",
    "sidebar_top": "#0A3323",
    "sidebar_bottom": "#105666",
    "sidebar_text": "#F7F4D5",
    "input_bg": "rgba(247, 244, 213, 0.98)",
    "input_text": "#105666",
    "button_bg": "#D3968C",
    "button_bg_disabled": "#d7b4ac",
    "button_text": "#F7F4D5",
    "button_text_disabled": "#f4e7df",
    "button_hover": "#839958",
    "metric_bg": "rgba(247, 244, 213, 0.92)",
    "metric_border": "rgba(16, 86, 102, 0.16)",
    "hero_start": "#0A3323",
    "hero_mid": "#105666",
    "hero_end": "#D3968C",
    "hero_text": "#F7F4D5",
    "tag_bg": "#839958",
    "tag_text": "#F7F4D5",
}

CHART_COLORS = {
    "primary": "#0A3323",
    "secondary": "#D3968C",
    "tertiary": "#839958",
    "quaternary": "#105666",
    "plot_bg": "rgba(255,255,255,0.35)",
}

AUDIO_GENRE_COLORS = {
    "electronic": "#105666",
    "ambient": "#9bb7d4",
    "pop": "#D3968C",
    "hiphop": "#e6b8ad",
    "indie": "#839958",
    "classical": "#cdbb8c",
}


st.set_page_config(
    page_title="TrendCast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles() -> None:
    st.markdown(
        f"""
        <style>
            [data-testid="stHeader"] {{
                background: transparent;
            }}
            [data-testid="stToolbar"] {{
                display: none;
            }}
            [data-testid="collapsedControl"] {{
                display: none !important;
            }}
            [data-testid="stSidebarCollapseButton"] {{
                display: none !important;
            }}
            button[aria-label="Close sidebar"] {{
                display: none !important;
            }}
            button[title="Close sidebar"] {{
                display: none !important;
            }}
            .stApp {{
                background:
                    radial-gradient(circle at top left, {UI_COLORS["accent_glow"]}, transparent 28%),
                    linear-gradient(180deg, {UI_COLORS["bg_top"]} 0%, {UI_COLORS["bg_mid"]} 38%, {UI_COLORS["bg_bottom"]} 100%);
                color: {UI_COLORS["text_main"]};
            }}
            .block-container {{
                padding-top: 1.15rem;
                padding-bottom: 2rem;
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {UI_COLORS["sidebar_top"]} 0%, {UI_COLORS["sidebar_bottom"]} 100%);
            }}
            button[kind="header"][aria-label*="sidebar"],
            button[kind="header"][aria-label*="Sidebar"] {{
                display: none !important;
            }}
            section[data-testid="stSidebar"][aria-expanded="false"] {{
                min-width: 21rem !important;
                max-width: 21rem !important;
                transform: none !important;
            }}
            section[data-testid="stSidebar"][aria-expanded="false"] > div {{
                margin-left: 0 !important;
            }}
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] .stCaption,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
                color: {UI_COLORS["sidebar_text"]};
            }}
            [data-testid="stSidebar"] code {{
                color: {UI_COLORS["sidebar_text"]};
                background: rgba(248, 244, 237, 0.14);
                padding: 0.1rem 0.35rem;
                border-radius: 6px;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] input[type="text"] {{
                background: {UI_COLORS["input_bg"]};
                border-color: rgba(19, 58, 59, 0.18);
                color: {UI_COLORS["input_text"]} !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] input,
            [data-testid="stSidebar"] [data-baseweb="select"] div,
            [data-testid="stSidebar"] [data-baseweb="select"] span {{
                color: {UI_COLORS["input_text"]};
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] svg {{
                fill: {UI_COLORS["input_text"]};
                color: {UI_COLORS["input_text"]};
            }}
            [data-testid="stSidebar"] .stButton > button {{
                background: {UI_COLORS["button_bg"]};
                color: {UI_COLORS["button_text"]};
                border: 1px solid rgba(19, 58, 59, 0.22);
                font-weight: 700;
            }}
            [data-testid="stSidebar"] .stButton > button:hover {{
                border-color: {UI_COLORS["button_hover"]};
                color: {UI_COLORS["button_text"]};
            }}
            [data-testid="stSidebar"] .stButton > button p {{
                color: {UI_COLORS["button_text"]};
            }}
            [data-testid="stSidebar"] .stButton > button:disabled,
            [data-testid="stSidebar"] .stButton > button[disabled] {{
                background: {UI_COLORS["button_bg_disabled"]};
                color: {UI_COLORS["button_text_disabled"]};
            }}
            [data-testid="stSidebar"] [data-baseweb="tag"] {{
                background: {UI_COLORS["tag_bg"]};
                color: {UI_COLORS["tag_text"]};
            }}
            [data-testid="stSidebar"] [data-baseweb="tag"] span,
            [data-testid="stSidebar"] [data-baseweb="tag"] svg {{
                color: {UI_COLORS["tag_text"]};
                fill: {UI_COLORS["tag_text"]};
            }}
            div[data-testid="stMetric"] {{
                background: {UI_COLORS["metric_bg"]};
                border: 1px solid {UI_COLORS["metric_border"]};
                border-radius: 18px;
                padding: 0.9rem 1rem;
                box-shadow: 0 12px 30px rgba(35, 36, 31, 0.07);
            }}
            .trendcast-hero {{
                padding: 1.4rem 1.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, {UI_COLORS["hero_start"]} 0%, {UI_COLORS["hero_mid"]} 52%, {UI_COLORS["hero_end"]} 100%);
                color: {UI_COLORS["hero_text"]};
                box-shadow: 0 22px 40px rgba(23, 63, 66, 0.22);
                margin-bottom: 1.2rem;
            }}
            .trendcast-hero p {{
                margin: 0;
            }}
            .trendcast-eyebrow {{
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                opacity: 0.82;
            }}
            .trendcast-title {{
                font-family: "Iowan Old Style", "Palatino Linotype", serif;
                font-size: 2.45rem;
                line-height: 1.05;
                margin: 0.35rem 0 0.45rem 0;
            }}
            .trendcast-subtitle {{
                max-width: 52rem;
                font-size: 1rem;
                opacity: 0.95;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_topic_config() -> pd.DataFrame:
    topics = load_topics(DEFAULT_TOPICS_FILE)
    topic_frame = pd.DataFrame(topic_queries_to_rows(list(topics)))
    if topic_frame.empty:
        return pd.DataFrame(columns=TOPIC_FILE_COLUMNS)
    return topic_frame.reindex(columns=TOPIC_FILE_COLUMNS).fillna("")


def load_image_path(name: str) -> Path | None:
    path = OUTPUTS_DIR / name
    return path if path.exists() else None


def parse_search_topics(raw_value: str) -> list[str]:
    normalized = str(raw_value or "").replace("\n", ",")
    topics: list[str] = []
    seen: set[str] = set()
    for part in normalized.split(","):
        topic = part.strip()
        if not topic:
            continue
        key = topic.lower()
        if key in seen:
            continue
        seen.add(key)
        topics.append(topic)
    return topics


def build_search_topic_rows(
    search_terms: Iterable[str],
    saved_topics: pd.DataFrame,
    include_saved_topics: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for topic in search_terms:
        name = str(topic).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "name": name,
                "news_query": name,
                "spotify_query": name,
                "reddit_terms": name,
                "reddit_subreddits": "*",
            }
        )

    if include_saved_topics and not saved_topics.empty:
        for row in saved_topics.to_dict(orient="records"):
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "name": name,
                    "news_query": str(row.get("news_query", "")).strip(),
                    "spotify_query": str(row.get("spotify_query", "")).strip(),
                    "reddit_terms": str(row.get("reddit_terms", "")).strip(),
                    "reddit_subreddits": str(row.get("reddit_subreddits", "")).strip() or "*",
                }
            )

    return rows


def run_pipeline_request(mode: str, topics_file: Path) -> tuple[bool, str]:
    command = [
        sys.executable,
        "run_pipeline.py",
        "--mode",
        mode,
        "--topics-file",
        str(topics_file),
    ]
    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    success = result.returncode == 0
    message = result.stdout if success else (result.stderr or result.stdout)
    load_csv.clear()
    return success, message


def queue_feedback(level: str, summary: str, details: str = "") -> None:
    st.session_state["dashboard_feedback"] = {
        "level": level,
        "summary": summary,
        "details": details,
    }


def render_feedback() -> None:
    feedback = st.session_state.pop("dashboard_feedback", None)
    if not feedback:
        return

    level = feedback.get("level", "info")
    summary = feedback.get("summary", "")
    details = feedback.get("details", "")

    if level == "success":
        st.success(summary)
    elif level == "warning":
        st.warning(summary)
    elif level == "error":
        st.error(summary)
    else:
        st.info(summary)

    if details:
        st.code(details.strip(), language="bash")


def save_topic_config(editor_df: pd.DataFrame) -> tuple[bool, str]:
    normalized = editor_df.copy()
    for column in TOPIC_FILE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized = normalized.reindex(columns=TOPIC_FILE_COLUMNS).fillna("")

    rows = []
    for row in normalized.to_dict(orient="records"):
        if any(str(value).strip() for value in row.values()):
            rows.append({column: str(row.get(column, "")).strip() for column in TOPIC_FILE_COLUMNS})

    try:
        save_topics_file(rows, DEFAULT_TOPICS_FILE)
    except ValueError as exc:
        return False, str(exc)

    load_topic_config.clear()
    return True, f"Saved {len(rows)} topics to {DEFAULT_TOPICS_FILE}."


def run_saved_topics(mode: str) -> None:
    with st.spinner(f"Running saved topics in {mode} mode..."):
        success, message = run_pipeline_request(mode, DEFAULT_TOPICS_FILE)

    if success:
        st.session_state["active_topic_source"] = "saved"
        st.session_state.pop("visible_topics", None)
        queue_feedback("success", "Saved topic run completed.", message)
    else:
        queue_feedback("error", "Saved topic run failed.", message)
    st.rerun()


def run_search_topics(
    mode: str,
    search_input: str,
    saved_topics: pd.DataFrame,
    include_saved_topics: bool,
    fallback_to_saved_topics: bool,
) -> None:
    search_terms = parse_search_topics(search_input)
    if not search_terms:
        queue_feedback("warning", "Type at least one topic before running a live search.")
        st.rerun()

    rows = build_search_topic_rows(search_terms, saved_topics, include_saved_topics)
    save_topics_file(rows, ACTIVE_SEARCH_TOPICS_FILE)
    label = ", ".join(search_terms)

    with st.spinner(f"Searching APIs for {label}..."):
        success, message = run_pipeline_request(mode, ACTIVE_SEARCH_TOPICS_FILE)

    if success:
        st.session_state["active_topic_source"] = "search"
        st.session_state.pop("visible_topics", None)
        queue_feedback("success", f"Search completed for: {label}", message)
        st.rerun()

    if fallback_to_saved_topics:
        with st.spinner("Custom search failed, switching back to the saved topic list..."):
            fallback_success, fallback_message = run_pipeline_request(mode, DEFAULT_TOPICS_FILE)

        if fallback_success:
            st.session_state["active_topic_source"] = "saved"
            st.session_state.pop("visible_topics", None)
            queue_feedback(
                "warning",
                "Custom search did not return usable data, so the dashboard was reset to the saved topics.",
                f"Custom search output:\n{message}\n\nSaved topic output:\n{fallback_message}",
            )
            st.rerun()

        queue_feedback(
            "error",
            "Custom search failed, and the fallback saved-topic run also failed.",
            f"Custom search output:\n{message}\n\nSaved topic output:\n{fallback_message}",
        )
        st.rerun()

    queue_feedback("error", f"Custom search failed for: {label}", message)
    st.rerun()


def main() -> None:
    inject_custom_styles()
    st.session_state.setdefault("active_topic_source", "saved")

    st.markdown(
        """
        <div class="trendcast-hero">
            <p class="trendcast-eyebrow">TrendCast capstone</p>
            <p class="trendcast-subtitle">
                Combining signals from multiple platforms to understand real-world trends and uncover what’s gaining momentum.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_feedback()

    multimodal = load_csv("multimodal_topic_summary.csv")
    source_status = load_csv("source_status.csv")
    temporal = load_csv("temporal_forecasts.csv")
    scenarios = load_csv("scenario_planning.csv")
    bertopic_topics = load_csv("bertopic_topics.csv")
    audio_genres = load_csv("audio_genre_summary.csv")
    genome = load_csv("cultural_genome.csv")
    harvest_log = load_csv("harvest_log.csv")
    explainability = load_csv("explainability_summary.csv")
    topic_config = load_topic_config()

    all_topics = multimodal["topic"].dropna().tolist() if not multimodal.empty else []

    with st.sidebar:
        st.header("Controls")
        pipeline_mode = st.selectbox(
            "Pipeline mode",
            options=["live", "auto", "demo"],
            index=0,
            help="Live uses the real APIs, auto falls back if needed, and demo uses sample data only.",
        )

        st.subheader("Search Live Topics")
        st.caption("Type one or more topics separated by commas.")
        search_input = st.text_area(
            "Search any topic",
            value=st.session_state.get("quick_search_input", ""),
            placeholder="Examples: smart rings, K-pop fashion, AI agents",
            height=96,
            key="quick_search_input",
        )
        include_saved_topics = st.checkbox(
            "Also include my saved topics",
            value=False,
            help="Runs your custom search terms together with the topics from config/topics.csv.",
        )
        fallback_to_saved_topics = st.checkbox(
            "If search fails, switch back to saved topics",
            value=True,
            help="Keeps the dashboard usable if a custom search returns no live data.",
        )
        if st.button("Search APIs Now", use_container_width=True):
            run_search_topics(
                pipeline_mode,
                search_input,
                topic_config,
                include_saved_topics,
                fallback_to_saved_topics,
            )

        st.markdown("---")
        if all_topics:
            selected_topics = list(all_topics)
        else:
            selected_topics = []
            st.info("No results loaded yet. Run a search or the saved topic list.")

        st.markdown("---")
        st.subheader("Source Status")
        if source_status.empty:
            st.write("No source status available yet.")
        else:
            for _, row in source_status.iterrows():
                st.write(f"**{row['source']}**: {row['status']}")

    if multimodal.empty:
        st.info("No output files are loaded yet. Use the sidebar to search live topics or run the saved topic list.")
        return

    if not selected_topics:
        st.warning("Select at least one topic in the sidebar to populate the dashboard.")
        return

    filtered_multimodal = multimodal[multimodal["topic"].isin(selected_topics)].copy()
    filtered_temporal = temporal[temporal["topic"].isin(selected_topics)].copy()
    filtered_scenarios = scenarios[scenarios["topic"].isin(selected_topics)].copy()
    filtered_genome = genome[genome["topic"].isin(selected_topics)].copy()
    filtered_audio = audio_genres[audio_genres["topic"].isin(selected_topics)].copy()
    filtered_explainability = explainability[explainability["topic"].isin(selected_topics)].copy()

    if filtered_multimodal.empty:
        st.warning("No rows matched the selected topics in the current outputs.")
        return

    top_topic = filtered_multimodal.sort_values("multimodal_final_score", ascending=False).iloc[0]["topic"]
    latest_harvest = harvest_log["captured_at_utc"].iloc[-1] if not harvest_log.empty else "Not available"
    if not source_status.empty:
        sources_loaded = int(source_status["status"].isin(["loaded", "loaded_partial"]).sum())
    else:
        sources_loaded = 0
    total_records = int(filtered_multimodal["mention_count"].sum())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Top Topic", top_topic)
    metric_cols[1].metric("Sources Loaded", sources_loaded)
    metric_cols[2].metric("Total Records", total_records)
    metric_cols[3].metric("Last Harvest", latest_harvest[:19] if isinstance(latest_harvest, str) else latest_harvest)

    tabs = st.tabs(
        [
            "Overview",
            "Topic Manager",
            "Trends",
            "Forecasts",
            "Topics",
            "Audio & Fusion",
            "Scenarios",
            "Sources",
        ]
    )

    with tabs[0]:
        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            trend_chart = px.bar(
                filtered_multimodal.sort_values("multimodal_final_score", ascending=False),
                x="topic",
                y="multimodal_final_score",
                color="quadrant",
                title="Multimodal Final Score by Topic",
                text_auto=".2f",
                color_discrete_sequence=[
                    CHART_COLORS["primary"],
                    CHART_COLORS["secondary"],
                    CHART_COLORS["tertiary"],
                    CHART_COLORS["quaternary"],
                ],
            )
            trend_chart.update_layout(
                xaxis_title="",
                yaxis_title="Multimodal score",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=CHART_COLORS["plot_bg"],
            )
            st.plotly_chart(trend_chart, use_container_width=True)
        with col2:
            quad_path = load_image_path("trend_quad_chart.png")
            if quad_path:
                st.image(str(quad_path), caption="TrendCast Quad Chart", use_container_width=True)
            network_path = load_image_path("topic_network.png")
            if network_path:
                st.image(str(network_path), caption="Topic Similarity Network", use_container_width=True)

        st.subheader("Executive Summary")
        overview_cols = [
            "topic",
            "multimodal_rank",
            "multimodal_final_score",
            "trend_score",
            "novelty_score_norm",
            "dominant_audio_genre",
            "quadrant",
        ]
        st.dataframe(
            filtered_multimodal[overview_cols].sort_values("multimodal_rank"),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        st.subheader("Manage Canonical Topics")
        st.caption(
            "These are your saved fallback topics. Use `|` to separate multiple Reddit search terms "
            "or subreddit names in the last two columns."
        )

        edited_topics = st.data_editor(
            topic_config,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn(
                    "Topic name",
                    help="Canonical label used across all outputs.",
                    required=True,
                ),
                "news_query": st.column_config.TextColumn(
                    "News query",
                    help="Query string sent to NewsAPI.",
                ),
                "spotify_query": st.column_config.TextColumn(
                    "Spotify query",
                    help="Query string sent to Spotify search.",
                ),
                "reddit_terms": st.column_config.TextColumn(
                    "Reddit terms",
                    help="Separate multiple terms with `|`.",
                ),
                "reddit_subreddits": st.column_config.TextColumn(
                    "Reddit subreddits",
                    help="Use `*` for a broad sitewide Reddit search, or separate names with `|`.",
                ),
            },
            key="topic_editor",
        )

        action_cols = st.columns(3)
        if action_cols[0].button("Save Topic Config", use_container_width=True):
            success, message = save_topic_config(edited_topics)
            if success:
                st.success(message)
            else:
                st.error(message)
        if action_cols[1].button("Reload Topic Config", use_container_width=True):
            load_topic_config.clear()
            st.rerun()
        if action_cols[2].button("Save and Run Saved Topics", use_container_width=True):
            success, message = save_topic_config(edited_topics)
            if success:
                st.success(message)
                run_saved_topics(pipeline_mode)
            else:
                st.error(message)

        st.info(
            "Custom live searches do not overwrite this saved file. This tab is your stable fallback topic list."
        )
        st.code(str(DEFAULT_TOPICS_FILE), language="text")

    with tabs[2]:
        scatter = px.scatter(
            filtered_multimodal,
            x="novelty_score_norm",
            y="trend_score",
            size="mention_count",
            color="multimodal_final_score",
            hover_name="topic",
            title="Trend vs Novelty",
            color_continuous_scale=[
                UI_COLORS["hero_mid"],
                CHART_COLORS["quaternary"],
                CHART_COLORS["tertiary"],
                CHART_COLORS["secondary"],
            ],
        )
        scatter.update_layout(
            xaxis_title="Novelty score",
            yaxis_title="Trend score",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CHART_COLORS["plot_bg"],
        )
        st.plotly_chart(scatter, use_container_width=True)

        st.subheader("Detailed Trend Table")
        trend_cols = [
            "topic",
            "mention_count",
            "source_count",
            "trend_score",
            "novelty_score_norm",
            "final_score_with_novelty",
            "multimodal_final_score",
        ]
        st.dataframe(
            filtered_multimodal[trend_cols].sort_values("multimodal_final_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        forecast_chart = px.bar(
            filtered_temporal.sort_values("forecast_8w_mentions", ascending=False),
            x="topic",
            y=["forecast_4w_mentions", "forecast_8w_mentions"],
            barmode="group",
            title="Forecasted Mentions",
            color_discrete_sequence=[UI_COLORS["hero_mid"], CHART_COLORS["secondary"]],
        )
        forecast_chart.update_layout(
            xaxis_title="",
            yaxis_title="Forecasted mentions",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CHART_COLORS["plot_bg"],
        )
        st.plotly_chart(forecast_chart, use_container_width=True)

        confidence_chart = px.scatter(
            filtered_temporal,
            x="forecast_confidence",
            y="forecast_growth_score",
            size="forecast_8w_mentions",
            color="topic",
            hover_name="topic",
            title="Forecast Confidence vs Growth",
        )
        confidence_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CHART_COLORS["plot_bg"],
        )
        st.plotly_chart(confidence_chart, use_container_width=True)

        st.dataframe(filtered_temporal, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("BERTopic Topics")
        st.dataframe(bertopic_topics, use_container_width=True, hide_index=True)

        if not filtered_genome.empty:
            genome_long = filtered_genome.melt(
                id_vars=["topic", "genome_signature"],
                value_vars=[
                    "innovation_score",
                    "tradition_score",
                    "wellness_score",
                    "community_score",
                    "expression_score",
                ],
                var_name="dimension",
                value_name="score",
            )
            genome_chart = px.bar(
                genome_long,
                x="topic",
                y="score",
                color="dimension",
                barmode="group",
                title="Cultural Genome Dimensions",
            )
            genome_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=CHART_COLORS["plot_bg"],
            )
            st.plotly_chart(genome_chart, use_container_width=True)
            st.dataframe(filtered_genome, use_container_width=True, hide_index=True)

    with tabs[5]:
        col1, col2 = st.columns(2)
        with col1:
            audio_chart = px.bar(
                filtered_audio,
                x="topic",
                y="share_within_topic",
                color="audio_genre_label",
                title="Audio Genre Mix by Topic",
                barmode="stack",
                color_discrete_map=AUDIO_GENRE_COLORS,
            )
            audio_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=CHART_COLORS["plot_bg"],
            )
            st.plotly_chart(audio_chart, use_container_width=True)
        with col2:
            fusion_chart = px.bar(
                filtered_multimodal.sort_values("deep_fusion_score", ascending=False),
                x="topic",
                y=["deep_fusion_score", "audio_energy_score", "visual_energy"],
                barmode="group",
                title="Fusion / Audio / Visual Comparison",
                color_discrete_sequence=[
                    CHART_COLORS["primary"],
                    CHART_COLORS["secondary"],
                    CHART_COLORS["tertiary"],
                ],
            )
            fusion_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=CHART_COLORS["plot_bg"],
            )
            st.plotly_chart(fusion_chart, use_container_width=True)

        st.dataframe(
            filtered_multimodal[
                [
                    "topic",
                    "dominant_audio_genre",
                    "audio_genre_confidence",
                    "audio_energy_score",
                    "audio_novelty_signal",
                    "deep_fusion_score",
                    "deep_fusion_max",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[6]:
        scenario_heatmap_path = load_image_path("scenario_heatmap.png")
        if scenario_heatmap_path:
            st.image(str(scenario_heatmap_path), caption="Scenario Heatmap", use_container_width=True)

        scenario_chart = px.bar(
            filtered_scenarios,
            x="scenario",
            y="adjusted_multimodal_score",
            color="topic",
            barmode="group",
            title="Scenario Planning Outcomes",
        )
        scenario_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CHART_COLORS["plot_bg"],
        )
        st.plotly_chart(scenario_chart, use_container_width=True)
        st.dataframe(filtered_scenarios, use_container_width=True, hide_index=True)

    with tabs[7]:
        status_cols = st.columns(2)
        with status_cols[0]:
            st.subheader("Source Status")
            st.dataframe(source_status, use_container_width=True, hide_index=True)
        with status_cols[1]:
            st.subheader("Explainability")
            if not filtered_explainability.empty:
                for _, row in filtered_explainability.iterrows():
                    st.markdown(f"**{row['topic']}**")
                    st.write(row["explanation"])
                    st.caption(f"Drivers: {row['top_driver_1']}, {row['top_driver_2']}, {row['top_driver_3']}")
            else:
                st.info("No explainability rows available for the selected topics.")

        st.subheader("Harvest Log")
        st.dataframe(harvest_log, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
