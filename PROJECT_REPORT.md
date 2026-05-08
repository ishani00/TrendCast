# TrendCast Full Project Report

Prepared from the current repository state on April 30, 2026.

## 1. Executive Summary

TrendCast is a multimodal trend-intelligence system designed to identify, rank, interpret, and monitor emerging cultural topics across heterogeneous public data sources. The project began as a notebook-based capstone prototype and was transformed into a modular Python application with a repeatable pipeline, a live/demo execution model, advanced analytics outputs, and a first-pass Streamlit dashboard.

The system collects signals from NewsAPI, Reddit, Spotify, and an optional X/Twitter branch; cleans and normalizes the data; computes trend and novelty scores; enriches the records with visual, audio, network, and forecasting features; performs exact BERTopic topic modeling when available; generates explainability, cultural genome, and scenario-planning outputs; and presents results through exported files and a local website.

At the time of this report, the current live run is operating successfully with NewsAPI, Spotify, and Reddit. X/Twitter support exists in the codebase but is currently inactive because no `X_BEARER_TOKEN` or `X_POSTS_CSV` has been provided. The current ad hoc live search run is centered on the topics `swimming`, `running`, `karate`, `gymnastics`, and `diving`, with `running` ranked first in both the base score table and the multimodal fusion output. The canonical saved topic set remains `AI music`, `Sustainable fashion`, `Mindfulness`, and `Digital art`.

TrendCast should be understood as a strong capstone-scale prototype rather than a production forecasting platform. Its value lies in combining multiple public signals into a transparent, scenario-aware decision-support workflow that goes beyond simple keyword counting.

## 2. Project Goal

The goal of TrendCast is to build a system that can answer a more useful question than “what is popular right now?” Specifically, the project aims to identify:

- which topics currently show strong cross-platform momentum,
- which topics are relatively novel compared with peers,
- which topics are likely to gain or lose momentum over time,
- which topics remain strong under different future scenarios, and
- why a topic ranked where it did.

This goal makes the project a trend-forecasting and decision-support system rather than a static descriptive dashboard.

## 3. Problem Statement

Emerging cultural signals are fragmented. News coverage, community discussion, creator behavior, and entertainment trends often evolve on different platforms and at different speeds. Looking at only one source can create a biased or incomplete picture. The project addresses this fragmentation by integrating multiple public sources into a unified analytics workflow.

The central project question is:

**How can a lightweight multimodal analytics system combine heterogeneous public signals into an interpretable forecast of which topics deserve attention now and which may become strategically important next?**

## 4. Objectives

The project objectives were:

- Build a modular analytics pipeline rather than leaving the work in a single notebook.
- Support both reproducible demo runs and real API-based live runs.
- Ingest data from heterogeneous sources with different structures and scales.
- Implement advanced analytics, including novelty detection, topic modeling, network analysis, forecasting, and multimodal fusion.
- Generate outputs that are useful for presentation, reporting, and decision-making.
- Provide a usable front end through a local interactive dashboard.

## 5. System Overview

TrendCast is organized into five layers:

1. **Configuration and topic definition**
2. **Data ingestion**
3. **Text and multimodal analytics**
4. **Forecasting, explanation, and strategy**
5. **Presentation and interaction**

The current codebase is centered on:

- `run_pipeline.py` as the main entry point
- `trendcast/` as the application package
- `config/topics.csv` as the canonical topic configuration
- `outputs/` as the exported artifact directory
- `dashboard.py` as the Streamlit website

## 6. Data Sources and Inputs

### 6.1 Primary Sources

TrendCast currently supports the following inputs:

- **NewsAPI**
  - Used for article-based media attention
  - Requires `NEWS_API_KEY`
- **Reddit public JSON/search**
  - Used for community discussion and grassroots signal capture
  - Uses `REDDIT_USER_AGENT`
- **Spotify search/track metadata**
  - Used for creator/entertainment platform signals
  - Requires `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`
- **X/Twitter**
  - Optional
  - Supports either `X_BEARER_TOKEN` or `X_POSTS_CSV`
  - Currently inactive in the user’s environment

### 6.2 Current Live Source Status

From the current `outputs/source_status.csv`:

| Source | Status | Record Count | Note |
|---|---:|---:|---|
| NewsAPI | loaded | 99 | NewsAPI articles collected |
| Spotify | loaded | 50 | Tracks collected, but audio-features access was unavailable |
| Reddit | loaded | 30 | Reddit posts collected from 5 successful queries |
| X | skipped | 0 | No bearer token or CSV provided |

### 6.3 Topic Inputs

TrendCast now supports two topic paths:

- **Saved canonical topics** in `config/topics.csv`
- **Ad hoc search topics** written to `outputs/active_search_topics.csv` by the dashboard

The canonical saved topics are:

- AI music
- Sustainable fashion
- Mindfulness
- Digital art

The current ad hoc live run used:

- swimming
- running
- karate
- gymnastics
- diving

This distinction matters because the project supports both reproducible benchmark topics and user-driven exploratory searches.

## 7. Project Structure

The main implementation files are:

- `trendcast/config.py`
  - topic and settings loading
- `trendcast/data_sources.py`
  - live and demo data collection
- `trendcast/analysis.py`
  - cleaning, sentiment, and base trend scoring
- `trendcast/modalities.py`
  - visual/audio/network feature engineering
- `trendcast/topic_modeling.py`
  - exact BERTopic with fallback logic
- `trendcast/forecasting.py`
  - temporal forecasting and Bayesian-style ensemble logic
- `trendcast/fusion.py`
  - multimodal topic fusion
- `trendcast/fusion_network.py`
  - neural-style deep fusion branch
- `trendcast/explainability.py`
  - explanation output generation
- `trendcast/strategy.py`
  - cultural genome and scenario planning
- `trendcast/visualization.py`
  - chart generation
- `trendcast/cli.py`
  - command-line orchestration and watch mode
- `dashboard.py`
  - Streamlit-based website/dashboard

## 8. Methodology

### 8.1 Data Ingestion

For each configured topic, TrendCast queries each enabled source and normalizes results into a unified tabular structure containing at minimum:

- topic
- source
- text
- date/time
- engagement proxy

This common schema makes it possible to aggregate heterogeneous data in a consistent way.

### 8.2 Text Cleaning

Text preprocessing includes:

- lowercasing
- URL removal
- punctuation stripping
- non-alphabetic character removal
- whitespace normalization
- missing-text handling

The cleaned text is stored for downstream scoring and topic modeling.

### 8.3 Sentiment Analysis

The pipeline uses TextBlob polarity to estimate sentiment. Each cleaned text record receives a continuous sentiment score. The topic-level sentiment score is later aggregated by source and then by topic.

### 8.4 Engagement Normalization

A major methodological issue in cross-source projects is scale mismatch. Reddit scores, Spotify popularity values, and headline-derived attention proxies are not directly comparable.

TrendCast addresses this by normalizing engagement **within each source** before topic-level aggregation. This design prevents a high-scale platform from dominating the ranking by magnitude alone.

### 8.5 Source-Level Trend Score

For each topic-source pair, the pipeline computes:

- mention count
- average normalized engagement
- total raw engagement
- average sentiment

It then derives a source-level score from normalized mention strength, engagement strength, and sentiment.

### 8.6 Topic-Level Trend Score

Source-level summaries are rolled up into topic-level scores. The final trend score is based on:

- mention score
- engagement score
- sentiment score

The current implementation uses a weighted structure of:

- 0.4 mention score
- 0.4 engagement score
- 0.2 sentiment score

This gives priority to actual activity and response strength while still allowing tone to matter.

### 8.7 Novelty Detection

TrendCast uses a KL-divergence-style novelty signal based on the current topic distribution relative to a uniform baseline. This produces:

- `novelty_score`
- `novelty_score_norm`

Novelty is meant to capture how unusually concentrated or distinctive a topic is relative to its peers in the same run.

Important interpretation note:

- It measures **relative novelty within the current topic set**
- It does **not** claim to measure global novelty across the whole internet

### 8.8 Topic Modeling

The project originally used a BERTopic-style local fallback. It now supports **exact BERTopic** when installed in the environment.

Current behavior:

- try exact `BERTopic` first
- fall back to the earlier local implementation if needed

Outputs:

- `bertopic_document_topics.csv`
- `bertopic_topics.csv`

These files add a latent topic-discovery branch on top of the canonical search-topic system. The BERTopic branch is used to enrich interpretation, not replace the configured top-level topics.

### 8.9 Visual and Audio Analysis

TrendCast adds prototype multimodal enrichment, including:

- visual brightness
- visual colorfulness
- visual edge density
- visual energy
- visual innovation
- audio energy score
- audio novelty signal
- tempo normalization
- audio genre classification

Audio branch outputs include:

- `audio_genre_summary.csv`

This branch is still prototype-level rather than a research-grade multimodal deep-learning stack, but it materially extends the system beyond text-only scoring.

### 8.10 Topic Network Analysis

The system builds a topic similarity network and computes network measures such as:

- PageRank
- betweenness
- degree
- weighted degree

This helps surface topics that are not only individually strong, but also structurally central to the broader conversation space.

Outputs:

- `topic_network_nodes.csv`
- `topic_network_edges.csv`
- `topic_network.png`

### 8.11 Temporal Forecasting

TrendCast aggregates mention activity into time series and forecasts near-term momentum using a three-part ensemble:

- Bayesian ridge regression
- Holt smoothing
- momentum extrapolation

The ensemble is adaptive rather than fixed-weight. The project estimates weights based on recent error behavior, which is why it is described as a **Bayesian-style ensemble** instead of a simple average.

Outputs:

- `daily_topic_timeseries.csv`
- `temporal_forecasts.csv`

### 8.12 Neural-Style Multimodal Fusion

The project includes a neural-style fusion branch in `trendcast/fusion_network.py`. It uses a lightweight ML regressor to combine structured multimodal inputs and produce:

- `deep_fusion_score`
- `fusion_network_topic_scores.csv`

This is then combined with trend, novelty, modality, network, and forecast information to produce:

- `multimodal_topic_summary.csv`

### 8.13 Explainability

To reduce black-box behavior, TrendCast produces:

- `explainability_summary.csv`
- `explainability_report.md`

These outputs identify:

- top drivers per topic
- risk flags
- plain-language explanations

### 8.14 Cultural Genome Mapping

The project maps topics onto a “cultural genome” using five interpretable axes:

- innovation
- tradition
- wellness
- community
- expression

Output:

- `cultural_genome.csv`

### 8.15 Scenario Planning

TrendCast simulates how topics behave under different future conditions, including:

- AI Regulation Crackdown
- Climate Activism Surge
- Creator Economy Boom
- Mental Health Priority Shift
- Recession Shock

Outputs:

- `scenario_planning.csv`
- `scenario_brief.md`
- `scenario_heatmap.png`

### 8.16 Watch Mode / Near-Real-Time Harvesting

The pipeline supports repeated reruns through:

- `--watch`
- `--poll-seconds`
- `--iterations`

This is not a production streaming backend, but it does provide basic near-real-time harvesting behavior.

Output:

- `harvest_log.csv`

## 9. Execution Modes

TrendCast supports four practical execution styles:

- `demo`
  - stable sample data
  - good for reproducibility
- `live`
  - real sources only
  - fails if required live collection fails
- `auto`
  - attempts live first
  - falls back to demo if needed
- `watch`
  - repeated reruns on a timer

This is one of the project’s biggest usability improvements over the original notebook.

## 10. Dashboard / Website

The project now includes a first Streamlit website in `dashboard.py`.

Current dashboard capabilities include:

- overview view
- trend charts
- forecast view
- BERTopic/topic view
- audio/fusion section
- scenario section
- source status section
- saved-topic runs
- ad hoc topic search through the APIs
- dashboard refresh controls

This turns the project from a static pipeline into an interactive local application.

## 11. Current Results

### 11.1 Current Live Topic Set

The current live search topic file is:

- `swimming`
- `running`
- `karate`
- `gymnastics`
- `diving`

### 11.2 Base Ranking

From `final_topic_scores.csv`, the current top-ranked topic is:

1. `running`
2. `diving`
3. `swimming`
4. `gymnastics`
5. `karate`

The highest base final score with novelty is currently `running` at approximately `0.906`.

### 11.3 Multimodal Ranking

From `multimodal_topic_summary.csv`, the multimodal ranking is:

1. `running`
2. `diving`
3. `karate`
4. `gymnastics`
5. `swimming`

The current multimodal top topic is therefore also `running`.

### 11.4 Forecasting Results

Current forecast outputs show:

- all five active topics have strong growth scores in the current run
- `running` has the strongest longer-horizon projected volume
- forecast confidence values are moderate to strong

Notable current forecast values:

- `running` forecast 8-week mentions: about `967.70`
- `swimming` forecast 8-week mentions: about `934.83`
- `gymnastics` forecast 8-week mentions: about `741.44`

### 11.5 Cultural Genome Results

Current genome signatures indicate:

- `running`: Innovation + Expression
- `diving`: Innovation + Expression
- `karate`: Innovation + Expression
- `gymnastics`: Innovation + Expression
- `swimming`: Innovation + Wellness

### 11.6 Scenario Results

Across the current run, the `Creator Economy Boom` scenario generally lifts topic scores, while `AI Regulation Crackdown` and `Recession Shock` create downward pressure for many topics.

### 11.7 Explainability Results

Current explanation summaries show:

- `running` is mainly driven by novelty, forecast growth, and trend score
- `diving` is strongly influenced by visual energy, forecast growth, and trend score
- `karate` is strongly influenced by audio energy, forecast growth, and trend score

This is useful because it clarifies that a topic’s rank is not coming from one factor alone.

## 12. Benchmark vs Current Live State

This project now supports two equally important modes of interpretation:

### Benchmark / canonical mode

The saved canonical topics in `config/topics.csv` are:

- AI music
- Sustainable fashion
- Mindfulness
- Digital art

These are the capstone’s original curated topics and are better for stable reporting and slide consistency.

### Current live exploratory mode

The current outputs reflect a user-driven ad hoc live search around sports-related topics:

- swimming
- running
- karate
- gymnastics
- diving

This is not a problem; it is actually evidence that the new search-driven design works. It just means the report should be explicit about whether it is talking about:

- the canonical benchmark topic set, or
- the latest exploratory live search run

## 13. Validation and Testing

The project has been validated through:

- successful end-to-end demo runs
- successful end-to-end live runs
- source-level status tracking
- BERTopic execution confirmation
- multimodal output generation
- watch-mode harvest logging
- dashboard-based review of outputs

Evidence of successful operation includes:

- populated `source_status.csv`
- populated `harvest_log.csv`
- populated `bertopic_topics.csv`
- generated figures such as the quad chart and scenario heatmap

## 14. Strengths of the Project

The strongest aspects of TrendCast are:

- modular architecture instead of notebook-only logic
- reproducible demo mode
- live API capability
- cross-source normalization
- novelty scoring
- exact BERTopic support
- temporal forecasting
- multimodal fusion
- explainability layer
- cultural genome layer
- scenario planning
- interactive website/dashboard

Taken together, these features make it a substantially stronger capstone than a simple scraper plus chart notebook.

## 15. Limitations

Despite the progress, several limitations remain:

### 15.1 X/Twitter Live Branch

X/Twitter is coded but not currently active in the user’s environment because neither:

- `X_BEARER_TOKEN`
- `X_POSTS_CSV`

has been provided.

### 15.2 Spotify Audio-Features Access

Spotify collection is working for search/track data, but advanced `audio-features` access is unavailable for the current app setup. The project handles this gracefully, but some richer audio descriptors are therefore missing in live runs.

### 15.3 Prototype-Level Multimodal Branches

The visual/audio/fusion system is meaningful and functional, but it is still a capstone prototype rather than a research-grade multimodal deep-learning architecture.

### 15.4 Novelty Interpretation

The novelty score is relative to the current topic batch. That is useful, but it should not be overclaimed as a universal measure of how novel a topic is globally.

### 15.5 Live Data Volatility

Live outputs change with:

- time
- API availability
- search topic selection
- platform content changes

That is normal for a live system, but it means reproducibility is best preserved through `demo` mode or a saved artifact snapshot.

## 16. Future Improvements

The next strongest improvements would be:

- activating live X/Twitter ingestion
- strengthening Spotify audio enrichment if policy access changes
- improving calibration of novelty against broader baselines
- expanding forecasting validation with longer historical windows
- improving UI polish and storytelling in the Streamlit website
- adding clearer “what changed since last run” comparisons in the dashboard
- packaging the dashboard for easier demo or deployment

## 17. Conclusion

TrendCast successfully evolved from a notebook experiment into a modular, multimodal trend-forecasting capstone application. It now supports:

- configurable topic sets
- live and demo execution
- cross-platform ingestion
- topic modeling
- temporal forecasting
- multimodal ranking
- explainability
- cultural interpretation
- scenario planning
- an interactive Streamlit interface

In its current form, the project is best described as a **working capstone prototype for multimodal cultural trend intelligence**. It is complete enough to demonstrate architecture, methodology, analytics depth, and user interaction, while still being honest about platform limitations and future opportunities.

## 18. Appendix: Important Files

### Code

- `run_pipeline.py`
- `dashboard.py`
- `trendcast/config.py`
- `trendcast/data_sources.py`
- `trendcast/analysis.py`
- `trendcast/modalities.py`
- `trendcast/topic_modeling.py`
- `trendcast/forecasting.py`
- `trendcast/fusion.py`
- `trendcast/fusion_network.py`
- `trendcast/explainability.py`
- `trendcast/strategy.py`
- `trendcast/visualization.py`

### Configuration

- `config/topics.csv`
- `.env`
- `.env.example`

### Current Outputs

- `outputs/source_status.csv`
- `outputs/final_topic_scores.csv`
- `outputs/multimodal_topic_summary.csv`
- `outputs/temporal_forecasts.csv`
- `outputs/cultural_genome.csv`
- `outputs/scenario_planning.csv`
- `outputs/explainability_summary.csv`
- `outputs/bertopic_topics.csv`
- `outputs/audio_genre_summary.csv`
- `outputs/fusion_network_topic_scores.csv`
- `outputs/harvest_log.csv`
- `outputs/final_trend_scores.png`
- `outputs/trend_quad_chart.png`
- `outputs/topic_network.png`
- `outputs/scenario_heatmap.png`

### Commands

Run demo:

```bash
python3 run_pipeline.py --mode demo
```

Run live:

```bash
python3 run_pipeline.py --mode live
```

Run auto fallback:

```bash
python3 run_pipeline.py --mode auto
```

Run near-real-time watch mode:

```bash
python3 run_pipeline.py --mode live --watch --iterations 0 --poll-seconds 300
```

Run the website:

```bash
streamlit run dashboard.py
```
