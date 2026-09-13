"""Evaluate leakage-safe bias correction for archived operational GFS forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def run(forecast_path: Path, truth_path: Path, output: Path) -> pd.DataFrame:
    forecasts = pd.read_csv(forecast_path, parse_dates=["timestamp"])
    truth = pd.read_csv(truth_path, usecols=["timestamp", "temperature"], parse_dates=["timestamp"])
    frame = forecasts.merge(truth, on="timestamp", how="inner").sort_values("timestamp")
    calibration = frame[frame["timestamp"] < "2023-01-01"].copy()
    test = frame[frame["timestamp"] >= "2023-01-01"].copy()
    rows = []
    predictions = []
    for lead in range(1, 8):
        column = f"temperature_2m_previous_day{lead}"
        usable_calibration = calibration.dropna(subset=[column, "temperature"]).copy()
        usable_test = test.dropna(subset=[column, "temperature"]).copy()
        usable_calibration["month"] = usable_calibration["timestamp"].dt.month
        usable_calibration["hour"] = usable_calibration["timestamp"].dt.hour
        usable_test["month"] = usable_test["timestamp"].dt.month
        usable_test["hour"] = usable_test["timestamp"].dt.hour
        usable_calibration["residual"] = usable_calibration["temperature"] - usable_calibration[column]
        bias = usable_calibration.groupby(["month", "hour"])["residual"].mean()
        corrections = [bias.get((month, hour), usable_calibration["residual"].mean()) for month, hour in zip(usable_test["month"], usable_test["hour"])]
        usable_test["raw_prediction"] = usable_test[column]
        usable_test["corrected_prediction"] = usable_test[column] + np.asarray(corrections)
        corrected_calibration = usable_calibration[column] + [bias.loc[(m, h)] for m, h in zip(usable_calibration["month"], usable_calibration["hour"])]
        radius = float((usable_calibration["temperature"] - corrected_calibration).abs().quantile(0.9, interpolation="higher"))
        for model_column, name in [("raw_prediction", "gfs_raw"), ("corrected_prediction", "gfs_bias_corrected")]:
            error = usable_test["temperature"] - usable_test[model_column]
            rows.append(
                {
                    "model": name,
                    "lead_days": lead,
                    "observations": len(usable_test),
                    "mae": error.abs().mean(),
                    "rmse": np.sqrt((error**2).mean()),
                    "bias": error.mean(),
                    "interval_coverage": usable_test["temperature"].between(usable_test[model_column] - radius, usable_test[model_column] + radius).mean(),
                }
            )
        predictions.append(
            usable_test[["timestamp", "temperature", "raw_prediction", "corrected_prediction"]].assign(
                lead_days=lead, lower=usable_test["corrected_prediction"] - radius, upper=usable_test["corrected_prediction"] + radius
            )
        )
    metrics = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    raw = metrics[metrics["model"] == "gfs_raw"]["mae"].mean()
    corrected = metrics[metrics["model"] == "gfs_bias_corrected"]["mae"].mean()
    improvement = 100 * (1 - corrected / raw)
    (output / "REPORT.md").write_text(
        f"""# Archived GFS bias-correction report

Archived 1–7 day GFS forecasts are calibrated on 2022 and evaluated on 2023 against hourly ERA5.
The correction uses only mean residuals grouped by calendar month and hour of day.

- Raw mean MAE: {raw:.3f} °C
- Bias-corrected mean MAE: {corrected:.3f} °C
- Relative improvement: {improvement:.1f}%

This tests an operationally available forecast, unlike the historical-only GRU/LSTM experiment.
ERA5 remains the verification target and is not an IMD station observation.
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, default=ROOT / "data" / "raw" / "delhi_gfs_previous_runs_2022_2023.csv")
    parser.add_argument("--truth", type=Path, default=ROOT / "data" / "raw" / "delhi_era5_hourly_2000_2023.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "gfs-bias-correction")
    args = parser.parse_args()
    metrics = run(args.forecasts, args.truth, args.output)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
