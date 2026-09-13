from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"date", "temperature_mean", "temperature_min", "temperature_max"}
NUMERIC_WEATHER = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "precipitation_sum",
    "wind_speed_mean",
    "shortwave_radiation_sum",
]


def load_and_clean(path: Path, target: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    missing_columns = REQUIRED.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if target not in raw.columns:
        raise ValueError(f"Target {target!r} is not present")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    invalid_dates = int(raw["date"].isna().sum())
    raw = raw.dropna(subset=["date"]).sort_values("date")
    duplicates = int(raw["date"].duplicated().sum())
    raw = raw.drop_duplicates("date", keep="last").set_index("date")

    for column in set(NUMERIC_WEATHER).intersection(raw.columns):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    complete_index = pd.date_range(raw.index.min(), raw.index.max(), freq="D", name="date")
    missing_dates = int(len(complete_index.difference(raw.index)))
    frame = raw.reindex(complete_index)
    numeric = frame.select_dtypes(include=[np.number]).columns
    missing_before = {column: int(frame[column].isna().sum()) for column in numeric}
    imputed_flags = frame[numeric].isna().add_prefix("was_imputed_").astype(int)
    # Imputation must be causal: linear/time interpolation uses the next observation and would
    # leak future information into an earlier forecast origin. Carry only the last known value
    # across short gaps; leading and longer gaps remain missing and are excluded during fitting.
    frame[numeric] = frame[numeric].ffill(limit=3)
    frame = pd.concat([frame, imputed_flags], axis=1)

    impossible = {}
    for column in ["temperature_mean", "temperature_min", "temperature_max"]:
        impossible[column] = int(((frame[column] < -30) | (frame[column] > 60)).sum())
        frame.loc[(frame[column] < -30) | (frame[column] > 60), column] = np.nan
    order_violations = int((frame["temperature_min"] > frame["temperature_max"]).sum())

    report = {
        "rows_raw": int(len(raw)),
        "rows_daily_grid": int(len(frame)),
        "date_min": str(frame.index.min().date()),
        "date_max": str(frame.index.max().date()),
        "invalid_dates": invalid_dates,
        "duplicate_dates_removed": duplicates,
        "missing_dates_inserted": missing_dates,
        "missing_values_before_imputation": missing_before,
        "physically_implausible_values_removed": impossible,
        "min_greater_than_max_violations": order_violations,
        "remaining_target_missing": int(frame[target].isna().sum()),
    }
    return frame, report


def write_quality_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def synthetic_weather(start: str = "2015-01-01", end: str = "2023-12-31", seed: int = 42) -> pd.DataFrame:
    """Create deterministic Delhi-like daily data for smoke tests, never for reported findings."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    day = dates.dayofyear.to_numpy()
    annual = 25 + 10.5 * np.sin(2 * np.pi * (day - 95) / 365.25)
    monsoon = -2.5 * ((dates.month >= 7) & (dates.month <= 9)).astype(float)
    noise = np.zeros(len(dates))
    shocks = rng.normal(0, 1.8, len(dates))
    for i in range(1, len(dates)):
        noise[i] = 0.72 * noise[i - 1] + shocks[i]
    mean = annual + monsoon + noise
    spread = 6 + 1.5 * np.sin(2 * np.pi * (day - 40) / 365.25)
    frame = pd.DataFrame(
        {
            "date": dates,
            "temperature_mean": mean,
            "temperature_min": mean - spread + rng.normal(0, 0.5, len(dates)),
            "temperature_max": mean + spread + rng.normal(0, 0.5, len(dates)),
            "precipitation_sum": np.where((dates.month >= 7) & (dates.month <= 9), rng.gamma(1.2, 4, len(dates)), 0),
            "wind_speed_mean": np.maximum(0, rng.normal(9, 2, len(dates))),
            "shortwave_radiation_sum": np.maximum(0, 18 + 6 * np.sin(2 * np.pi * (day - 70) / 365.25) + rng.normal(0, 2, len(dates))),
            "source": "synthetic-smoke-test",
        }
    )
    return frame
