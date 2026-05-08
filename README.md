# TrendCast Capstone Project

TrendCast turns social and media signals into ranked topic trends. It collects data from NewsAPI, Spotify, Reddit, and optionally X/Twitter CSV exports, cleans the text, measures sentiment, computes trend and novelty scores, analyzes visual and audio signals, runs temporal forecasting, maps topics onto a cultural genome, simulates scenarios, and exports tables and charts for presentation use.

## What Changed From The Notebook

- Moved the notebook logic into reusable Python modules.
- Fixed topic consistency across sources so Reddit data is grouped into the same topic labels as News and Spotify.
- Normalized engagement within each source to avoid Reddit scores overpowering the other APIs.
- Added a demo mode so the full project still runs even without live API access.
- Added a quad chart image for capstone slides.
- Added multimodal analysis outputs for text, image, audio, network, forecasting, explainability, and scenario planning.
- Removed the need to hard-code credentials in code.

## Project Structure

```text
.
├── ChronoPredict_MVP.ipynb
├── README.md
├── config/
│   └── topics.csv
├── requirements.txt
├── run_pipeline.py
├── outputs/
└── trendcast/
    ├── __init__.py
    ├── analysis.py
    ├── cli.py
    ├── config.py
    ├── data_sources.py
    └── visualization.py
```

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

3. Put your API keys into `.env`.

Optional:
- set `X_BEARER_TOKEN` to use the official X recent-search API
- or set `X_POSTS_CSV` to a local X/Twitter export

Important: the notebook currently contains real-looking API credentials. Rotate those keys before turning this in or sharing the repo.

## Run It

Run with automatic fallback:

```bash
python3 run_pipeline.py --mode auto
```

Run with sample data only:

```bash
python3 run_pipeline.py --mode demo
```

Run with live APIs only:

```bash
python3 run_pipeline.py --mode live
```

Run in repeated harvesting mode:

```bash
python3 run_pipeline.py --mode live --watch --iterations 3 --poll-seconds 300
```

Run with an explicit topic file:

```bash
python3 run_pipeline.py --mode auto --topics-file config/topics.csv
```

## Outputs

The pipeline writes the following files into `outputs/`:

- `raw_combined.csv`: collected records before text processing
- `scored_records.csv`: row-level data with cleaned text, sentiment, and normalized engagement
- `source_summary.csv`: per-topic per-source scores
- `final_topic_scores.csv`: final ranked topic table
- `final_trend_scores.png`: bar chart of the final trend score
- `trend_quad_chart.png`: quad chart for presentation slides
- `multimodal_topic_summary.csv`: fused topic scores across text, visual, audio, network, and forecast signals
- `temporal_forecasts.csv`: Bayesian-style ensemble forecast summary by topic
- `cultural_genome.csv`: topic positioning across cultural dimensions
- `scenario_planning.csv`: scenario-adjusted topic scores
- `source_status.csv`: which sources loaded, failed, or were skipped
- `audio_genre_summary.csv`: explicit genre classification output for the audio branch
- `bertopic_document_topics.csv`: document-level BERTopic-style topic assignments
- `bertopic_topics.csv`: topic model summary with keywords and counts
- `fusion_network_topic_scores.csv`: neural-style multimodal fusion topic scores
- `harvest_log.csv`: snapshot log for watch-mode harvesting
- `topic_network.png`: topic similarity network
- `scenario_heatmap.png`: scenario planning heatmap
- `explainability_report.md`: human-readable driver explanations

## Methodology Summary

1. Collect topic-related mentions from NewsAPI, Spotify, Reddit, and optional X/Twitter CSV exports.
2. Clean the text by removing URLs, punctuation, and extra whitespace.
3. Compute sentiment using TextBlob polarity.
4. Normalize engagement scores inside each source.
5. Extract visual features from media thumbnails and audio features from Spotify.
6. Build a text trend score from mention volume, engagement, and sentiment.
7. Build a novelty score using KL-divergence against a uniform baseline.
8. Build topic similarity and network centrality from TF-IDF topic graphs.
9. Forecast near-term topic momentum with a Bayesian-style ensemble.
10. Fuse text, visual, audio, network, and forecast signals into a multimodal ranking.
11. Map topics into a cultural genome and simulate external scenarios.

Current topic modeling behavior:
- If the real `BERTopic` package is available, the project uses it.
- If not, it falls back to the local BERTopic-style implementation.

## Topic Management

Canonical topic definitions now live in `config/topics.csv` instead of a hard-coded Python constant.

- Edit `config/topics.csv` directly, or manage it from the Topic Manager tab in `dashboard.py`.
- The dashboard also supports ad hoc search: type any topic in the sidebar and run a live search without saving it permanently.
- The pipeline reads that CSV at startup.
- Use `|` to separate multiple values in `reddit_terms` and `reddit_subreddits`.
- Use `*` in `reddit_subreddits` if you want a broader sitewide Reddit search.
- If the CSV is missing, the project falls back to the original starter topics.

## Suggested Capstone Talking Points

- Cross-platform trend detection matters because a topic can behave differently in news, entertainment, and communities.
- Topic consistency matters, so each source is mapped back to the same canonical topic names.
- Source-specific normalization matters because Reddit scores are much larger in scale than headline proxies from NewsAPI.
- Novelty helps separate established trends from emerging opportunities.
- Demo mode keeps the project reproducible even when API rate limits or credentials become a problem.

## Next Fast Wins

- Continue polishing the Streamlit dashboard visuals and storytelling flow.
- Export the final score table directly into your slide deck or report.
- Add a time-series chart if you want to show trend movement over time.
