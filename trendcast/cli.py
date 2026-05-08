from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from trendcast.analysis import compute_final_topic_scores, compute_source_summary, prepare_records
from trendcast.explainability import build_explainability_summary, write_explainability_report
from trendcast.forecasting import build_temporal_forecast
from trendcast.fusion import build_multimodal_topic_summary
from trendcast.fusion_network import build_deep_fusion_outputs
from trendcast.config import DEFAULT_TOPICS_FILE, load_env_file, load_settings, load_topics
from trendcast.data_sources import build_demo_dataset, build_demo_source_status, collect_live_data
from trendcast.modalities import (
    build_audio_genre_summary,
    build_modal_topic_summary,
    build_topic_network,
    enrich_modalities,
)
from trendcast.strategy import build_cultural_genome, build_scenario_plan, write_scenario_brief
from trendcast.topic_modeling import build_topic_model_outputs
from trendcast.visualization import (
    save_bar_chart,
    save_network_graph,
    save_quad_chart,
    save_scenario_heatmap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrendCast capstone pipeline")
    parser.add_argument(
        "--mode",
        choices=("auto", "demo", "live"),
        default="auto",
        help="Use live APIs, demo data, or auto-fallback behavior.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where CSVs and charts will be written.",
    )
    parser.add_argument("--news-page-size", type=int, default=20)
    parser.add_argument("--spotify-limit", type=int, default=10)
    parser.add_argument("--reddit-limit", type=int, default=6)
    parser.add_argument(
        "--topics-file",
        default="config/topics.csv",
        help=(
            "CSV file containing canonical topics and source queries. "
            f"Defaults to {DEFAULT_TOPICS_FILE.relative_to(DEFAULT_TOPICS_FILE.parent.parent)}."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Repeatedly rerun the pipeline on a polling interval for near-real-time harvesting.",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of watch-mode iterations. Use 0 to keep running until interrupted.",
    )
    return parser


def run_pipeline(args: argparse.Namespace, output_dir_override: Path | None = None) -> dict[str, Path | str]:
    load_env_file()
    settings = load_settings()
    topics = load_topics(args.topics_file)
    output_dir = output_dir_override or Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_mode = args.mode
    if args.mode == "demo":
        raw_df = build_demo_dataset(topics)
        source_status = build_demo_source_status(topics)
    else:
        try:
            raw_df, source_status = collect_live_data(
                settings=settings,
                topics=topics,
                news_page_size=args.news_page_size,
                spotify_limit=args.spotify_limit,
                reddit_limit=args.reddit_limit,
            )
            dataset_mode = "live"
        except Exception as exc:
            if args.mode == "live":
                raise
            print(f"Live data collection failed, switching to demo mode: {exc}")
            raw_df = build_demo_dataset(topics)
            source_status = build_demo_source_status(topics)
            dataset_mode = "demo"

    prepared_df = prepare_records(raw_df)
    enriched_df = enrich_modalities(prepared_df)
    fusion_records_df, deep_fusion_topics = build_deep_fusion_outputs(enriched_df)
    source_summary = compute_source_summary(prepared_df)
    final_trends = compute_final_topic_scores(source_summary)
    modal_summary = build_modal_topic_summary(enriched_df)
    audio_genre_summary = build_audio_genre_summary(enriched_df)
    bertopic_docs, bertopic_topics, topic_model_method = build_topic_model_outputs(prepared_df)
    network_nodes, network_edges = build_topic_network(enriched_df)
    timeline_df, temporal_summary = build_temporal_forecast(prepared_df)
    multimodal_summary = build_multimodal_topic_summary(
        final_trends=final_trends,
        modal_summary=modal_summary,
        network_nodes=network_nodes,
        temporal_summary=temporal_summary,
        deep_fusion_topics=deep_fusion_topics,
    )
    genome_df = build_cultural_genome(enriched_df, multimodal_summary)
    scenario_df = build_scenario_plan(multimodal_summary, genome_df)
    explainability_df = build_explainability_summary(multimodal_summary)
    bertopic_topics["topic_model_method"] = topic_model_method

    raw_csv = output_dir / "raw_combined.csv"
    prepared_csv = output_dir / "scored_records.csv"
    enriched_csv = output_dir / "multimodal_records.csv"
    source_csv = output_dir / "source_summary.csv"
    source_status_csv = output_dir / "source_status.csv"
    final_csv = output_dir / "final_topic_scores.csv"
    modal_csv = output_dir / "modal_topic_summary.csv"
    audio_genre_csv = output_dir / "audio_genre_summary.csv"
    bertopic_docs_csv = output_dir / "bertopic_document_topics.csv"
    bertopic_topics_csv = output_dir / "bertopic_topics.csv"
    fusion_records_csv = output_dir / "fusion_network_records.csv"
    deep_fusion_csv = output_dir / "fusion_network_topic_scores.csv"
    network_nodes_csv = output_dir / "topic_network_nodes.csv"
    network_edges_csv = output_dir / "topic_network_edges.csv"
    timeline_csv = output_dir / "daily_topic_timeseries.csv"
    forecast_csv = output_dir / "temporal_forecasts.csv"
    multimodal_csv = output_dir / "multimodal_topic_summary.csv"
    genome_csv = output_dir / "cultural_genome.csv"
    scenario_csv = output_dir / "scenario_planning.csv"
    explainability_csv = output_dir / "explainability_summary.csv"

    raw_df.to_csv(raw_csv, index=False)
    prepared_df.to_csv(prepared_csv, index=False)
    enriched_df.to_csv(enriched_csv, index=False)
    source_summary.to_csv(source_csv, index=False)
    source_status.to_csv(source_status_csv, index=False)
    final_trends.to_csv(final_csv, index=False)
    modal_summary.to_csv(modal_csv, index=False)
    audio_genre_summary.to_csv(audio_genre_csv, index=False)
    bertopic_docs.to_csv(bertopic_docs_csv, index=False)
    bertopic_topics.to_csv(bertopic_topics_csv, index=False)
    fusion_records_df.to_csv(fusion_records_csv, index=False)
    deep_fusion_topics.to_csv(deep_fusion_csv, index=False)
    network_nodes.to_csv(network_nodes_csv, index=False)
    network_edges.to_csv(network_edges_csv, index=False)
    timeline_df.to_csv(timeline_csv, index=False)
    temporal_summary.to_csv(forecast_csv, index=False)
    multimodal_summary.to_csv(multimodal_csv, index=False)
    genome_df.to_csv(genome_csv, index=False)
    scenario_df.to_csv(scenario_csv, index=False)
    explainability_df.to_csv(explainability_csv, index=False)

    bar_chart = save_bar_chart(final_trends, output_dir)
    quad_chart = save_quad_chart(final_trends, output_dir)
    network_chart = save_network_graph(network_nodes, network_edges, output_dir)
    scenario_heatmap = save_scenario_heatmap(scenario_df, output_dir)
    explainability_report = write_explainability_report(explainability_df, output_dir)
    scenario_brief = write_scenario_brief(scenario_df, output_dir)

    top_topic = multimodal_summary.iloc[0]["topic"] if not multimodal_summary.empty else "No topic"
    return {
        "mode": dataset_mode,
        "raw_csv": raw_csv,
        "prepared_csv": prepared_csv,
        "enriched_csv": enriched_csv,
        "source_csv": source_csv,
        "source_status_csv": source_status_csv,
        "final_csv": final_csv,
        "modal_csv": modal_csv,
        "audio_genre_csv": audio_genre_csv,
        "bertopic_docs_csv": bertopic_docs_csv,
        "bertopic_topics_csv": bertopic_topics_csv,
        "fusion_records_csv": fusion_records_csv,
        "deep_fusion_csv": deep_fusion_csv,
        "network_nodes_csv": network_nodes_csv,
        "network_edges_csv": network_edges_csv,
        "timeline_csv": timeline_csv,
        "forecast_csv": forecast_csv,
        "multimodal_csv": multimodal_csv,
        "genome_csv": genome_csv,
        "scenario_csv": scenario_csv,
        "explainability_csv": explainability_csv,
        "bar_chart": bar_chart,
        "quad_chart": quad_chart,
        "network_chart": network_chart,
        "scenario_heatmap": scenario_heatmap,
        "explainability_report": explainability_report,
        "scenario_brief": scenario_brief,
        "top_topic": top_topic,
        "topic_model_method": topic_model_method,
    }


def run_watch_mode(args: argparse.Namespace) -> dict[str, Path | str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    harvest_log_path = output_dir / "harvest_log.csv"
    latest_results: dict[str, Path | str] = {}
    import pandas as pd

    existing_rows = []
    if harvest_log_path.exists():
        existing_rows = pd.read_csv(harvest_log_path).to_dict(orient="records")

    iteration = 0
    while True:
        latest_results = run_pipeline(args, output_dir_override=output_dir)
        snapshot_time = datetime.now(timezone.utc).isoformat()
        existing_rows.append(
            {
                "iteration": iteration + 1,
                "captured_at_utc": snapshot_time,
                "mode": latest_results["mode"],
                "top_topic": latest_results["top_topic"],
                "topic_model_method": latest_results["topic_model_method"],
            }
        )
        pd.DataFrame(existing_rows).to_csv(harvest_log_path, index=False)
        print(
            f"Watch iteration {iteration + 1} captured at {snapshot_time}. "
            f"Top topic: {latest_results['top_topic']}"
        )
        iteration += 1
        if args.iterations > 0 and iteration >= args.iterations:
            break
        print(f"Sleeping for {max(1, args.poll_seconds)} seconds before the next harvest...")
        time.sleep(max(1, args.poll_seconds))

    latest_results["harvest_log_csv"] = harvest_log_path
    return latest_results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_watch_mode(args) if args.watch else run_pipeline(args)

    print(f"TrendCast pipeline completed in {results['mode']} mode.")
    print(f"Top-ranked topic: {results['top_topic']}")
    print(f"Saved final topic scores to: {results['final_csv']}")
    print(f"Saved bar chart to: {results['bar_chart']}")
    print(f"Saved quad chart to: {results['quad_chart']}")
    print(f"Saved multimodal topic summary to: {results['multimodal_csv']}")
    print(f"Saved scenario plan to: {results['scenario_csv']}")
    print(f"Topic modeling method used: {results['topic_model_method']}")
    if args.watch:
        print(f"Saved harvest log to: {results['harvest_log_csv']}")


if __name__ == "__main__":
    main()
