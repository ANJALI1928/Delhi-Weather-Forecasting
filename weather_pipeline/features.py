from __future__ import annotations

import numpy as np
import pandas as pd


def build_direct_features(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    lag_days: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    fourier_order: int,
) -> pd.DataFrame:
    """Features for target date t using observations available at origin t-horizon."""
    features = pd.DataFrame(index=frame.index)
    target_dates = frame.index
    day = target_dates.dayofyear.to_numpy()
    for order in range(1, fourier_order + 1):
        features[f"year_sin_{order}"] = np.sin(2 * np.pi * order * day / 365.25)
        features[f"year_cos_{order}"] = np.cos(2 * np.pi * order * day / 365.25)
    features["trend"] = np.arange(len(frame), dtype=float) / 365.25

    for lag in lag_days:
        features[f"target_lag_{lag}"] = frame[target].shift(horizon + lag)
    available_target = frame[target].shift(horizon)
    for window in rolling_windows:
        roll = available_target.rolling(window, min_periods=window)
        features[f"target_mean_{window}"] = roll.mean()
        features[f"target_std_{window}"] = roll.std()
        features[f"target_min_{window}"] = roll.min()
        features[f"target_max_{window}"] = roll.max()

    # Only lagged observed covariates are used. Current/future realized weather is excluded.
    covariates = [
        column
        for column in ["precipitation_sum", "wind_speed_mean", "shortwave_radiation_sum"]
        if column in frame.columns
    ]
    for column in covariates:
        features[f"{column}_at_origin"] = frame[column].shift(horizon)
        features[f"{column}_mean_7"] = frame[column].shift(horizon).rolling(7, min_periods=7).mean()
    return features


def append_future_dates(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    future = pd.date_range(frame.index.max() + pd.Timedelta(days=1), periods=days, freq="D", name="date")
    return frame.reindex(frame.index.append(future))
