from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _safe_series_forecast(series: pd.Series, horizon: int) -> tuple[np.ndarray, dict[str, float]]:
    values = series.astype(float).to_numpy()
    if len(values) == 0:
        return np.zeros(horizon), {"bayesian": 1 / 3, "holt": 1 / 3, "momentum": 1 / 3}

    index = np.arange(len(values), dtype=float).reshape(-1, 1)
    future_index = np.arange(len(values), len(values) + horizon, dtype=float).reshape(-1, 1)

    model = BayesianRidge()
    model.fit(index, values)
    bayes_forecast = np.maximum(0.0, model.predict(future_index))

    if len(values) >= 4:
        try:
            holt_model = ExponentialSmoothing(
                values,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            holt_forecast = np.maximum(0.0, holt_model.forecast(horizon))
        except Exception:
            holt_forecast = np.repeat(values[-1], horizon)
    else:
        holt_forecast = np.repeat(values[-1], horizon)

    if len(values) >= 2:
        recent_growth = values[-1] - values[-2]
    else:
        recent_growth = 0.0
    momentum_forecast = np.maximum(
        0.0,
        np.array([values[-1] + recent_growth * step for step in range(1, horizon + 1)], dtype=float),
    )

    errors = {
        "bayesian": _rolling_backtest_error(values, "bayesian"),
        "holt": _rolling_backtest_error(values, "holt"),
        "momentum": _rolling_backtest_error(values, "momentum"),
    }
    precisions = {
        key: 1.0 / max(error, 1e-6)
        for key, error in errors.items()
    }
    precision_sum = sum(precisions.values())
    weights = {key: value / precision_sum for key, value in precisions.items()}
    ensemble = (
        weights["bayesian"] * bayes_forecast
        + weights["holt"] * holt_forecast
        + weights["momentum"] * momentum_forecast
    )
    return ensemble, weights


def _rolling_backtest_error(values: np.ndarray, method: str) -> float:
    if len(values) < 5:
        return 1.0

    errors = []
    for split in range(3, len(values)):
        train = pd.Series(values[:split])
        actual = values[split]
        forecast, _ = _simple_one_step_forecast(train, method)
        errors.append(abs(actual - forecast))
    return float(np.mean(errors)) if errors else 1.0


def _simple_one_step_forecast(series: pd.Series, method: str) -> tuple[float, float]:
    values = series.astype(float).to_numpy()
    if method == "momentum":
        if len(values) < 2:
            return float(values[-1]), 0.0
        return float(max(0.0, values[-1] + (values[-1] - values[-2]))), 0.0
    if method == "holt":
        if len(values) < 4:
            return float(values[-1]), 0.0
        model = ExponentialSmoothing(
            values,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = model.forecast(1)[0]
        return float(max(0.0, forecast)), 0.0

    model = BayesianRidge()
    index = np.arange(len(values), dtype=float).reshape(-1, 1)
    model.fit(index, values)
    forecast = model.predict(np.array([[len(values)]], dtype=float))[0]
    return float(max(0.0, forecast)), float(model.alpha_)


def build_temporal_forecast(prepared_df: pd.DataFrame, horizon_weeks: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        prepared_df.assign(day=prepared_df["date"].dt.date)
        .groupby(["topic", "day"])
        .agg(
            mentions=("text", "count"),
            engagement=("engagement", "sum"),
        )
        .reset_index()
    )

    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()

    timeline_rows: list[dict] = []
    summary_rows: list[dict] = []

    for topic, topic_frame in daily.groupby("topic"):
        topic_frame = topic_frame.sort_values("day")
        full_range = pd.date_range(topic_frame["day"].min(), topic_frame["day"].max(), freq="D")
        series = (
            topic_frame.set_index(pd.to_datetime(topic_frame["day"]))["mentions"]
            .reindex(full_range, fill_value=0)
            .astype(float)
        )

        rolling_short = series.rolling(7, min_periods=1).mean()
        rolling_long = series.rolling(21, min_periods=1).mean()
        momentum_7d = float(rolling_short.iloc[-1] - rolling_short.iloc[max(0, len(rolling_short) - 8)])
        base_level = float(max(series.mean(), 1e-6))

        weekly_series = series.resample("W").sum()
        ensemble_forecast, weights = _safe_series_forecast(weekly_series, horizon_weeks)
        forecast_4w = float(ensemble_forecast[:4].sum()) if len(ensemble_forecast) >= 4 else float(ensemble_forecast.sum())
        forecast_8w = float(ensemble_forecast.sum())
        historical_4w = float(weekly_series.tail(min(4, len(weekly_series))).sum())
        forecast_growth = (forecast_4w - historical_4w) / max(historical_4w, 1.0)
        confidence = float(min(1.0, 0.35 + 0.20 * math.log1p(len(weekly_series)) + 0.25 * max(weights.values())))

        for day, mentions, short_ma, long_ma in zip(full_range, series, rolling_short, rolling_long):
            timeline_rows.append(
                {
                    "topic": topic,
                    "day": day.date().isoformat(),
                    "mentions": float(mentions),
                    "rolling_7d_mentions": float(short_ma),
                    "rolling_21d_mentions": float(long_ma),
                }
            )

        summary_rows.append(
            {
                "topic": topic,
                "recent_daily_mean": float(series.tail(7).mean()),
                "momentum_7d": momentum_7d,
                "baseline_daily_mean": base_level,
                "forecast_4w_mentions": forecast_4w,
                "forecast_8w_mentions": forecast_8w,
                "forecast_growth_score": float(np.clip((forecast_growth + 1) / 2, 0, 1)),
                "forecast_confidence": confidence,
                "bayesian_weight": float(weights["bayesian"]),
                "holt_weight": float(weights["holt"]),
                "momentum_weight": float(weights["momentum"]),
            }
        )

    return pd.DataFrame(timeline_rows), pd.DataFrame(summary_rows)
