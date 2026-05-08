from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_explainability_summary(multimodal_df: pd.DataFrame) -> pd.DataFrame:
    component_columns = [
        "trend_score",
        "novelty_score_norm",
        "visual_energy_norm",
        "audio_energy_norm",
        "network_centrality_norm",
        "forecast_growth_score",
    ]

    rows = []
    for _, row in multimodal_df.iterrows():
        contributions = {column: float(row.get(column, 0.0) or 0.0) for column in component_columns}
        top_drivers = sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:3]
        risks = []
        if float(row.get("source_count", 0) or 0) < 2:
            risks.append("limited source diversity")
        if float(row.get("forecast_confidence", 0) or 0) < 0.55:
            risks.append("low forecast confidence")
        if float(row.get("avg_sentiment", 0) or 0) < 0:
            risks.append("negative sentiment")
        if float(row.get("cross_modal_alignment", 0) or 0) < 0.5:
            risks.append("weak modality alignment")

        rows.append(
            {
                "topic": row["topic"],
                "top_driver_1": top_drivers[0][0] if len(top_drivers) > 0 else "",
                "top_driver_2": top_drivers[1][0] if len(top_drivers) > 1 else "",
                "top_driver_3": top_drivers[2][0] if len(top_drivers) > 2 else "",
                "risk_flags": ", ".join(risks) if risks else "none",
                "explanation": _explain_topic(row, top_drivers, risks),
            }
        )
    return pd.DataFrame(rows)


def _explain_topic(row: pd.Series, top_drivers: list[tuple[str, float]], risks: list[str]) -> str:
    driver_text = ", ".join(driver.replace("_", " ") for driver, _ in top_drivers) or "balanced signals"
    risk_text = "; risks: " + ", ".join(risks) if risks else ""
    return (
        f"{row['topic']} ranks with strong support from {driver_text}. "
        f"Its fused score is {float(row.get('multimodal_final_score', 0.0)):.2f}, "
        f"with forecast confidence at {float(row.get('forecast_confidence', 0.0)):.2f}{risk_text}."
    )


def write_explainability_report(explainability_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "explainability_report.md"
    lines = ["# TrendCast Explainability Report", ""]
    for _, row in explainability_df.iterrows():
        lines.append(f"## {row['topic']}")
        lines.append(row["explanation"])
        lines.append("")
    output_path.write_text("\n".join(lines))
    return output_path
