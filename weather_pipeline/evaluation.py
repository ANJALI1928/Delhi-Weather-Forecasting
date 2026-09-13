from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .features import build_direct_features
from .models import arima_fourier_forecast_path, climatology, make_regressor, seasonal_naive


def origins_between(start: pd.Timestamp, end: pd.Timestamp, count: int) -> list[pd.Timestamp]:
    if end < start:
        return []
    candidates = pd.date_range(start, end, periods=count).round("D")
    return sorted(set(pd.Timestamp(value) for value in candidates))


def predict_origin(
    frame: pd.DataFrame,
    target: str,
    origin: pd.Timestamp,
    horizons: list[int],
    model: str,
    params: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history = frame.loc[:origin, target].dropna().astype(float)
    seasonal_differences = history.diff(365).abs().dropna()
    if seasonal_differences.empty:
        seasonal_differences = history.diff().abs().dropna()
    mase_scale = float(seasonal_differences.mean())
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise ValueError(f"Cannot calculate a positive MASE scale at origin={origin}")
    arima_path = None
    if model == "arima_fourier":
        arima_path = arima_fourier_forecast_path(
            frame,
            target,
            origin,
            max(horizons),
            params["order"],
            int(params["fourier_order"]),
        )
    for horizon in horizons:
        forecast_date = origin + pd.Timedelta(days=horizon)
        if forecast_date not in frame.index:
            continue
        actual = frame.at[forecast_date, target] if target in frame else np.nan
        if model == "seasonal_naive_365":
            prediction = seasonal_naive(frame.loc[:origin], target, forecast_date)
        elif model == "climatology":
            prediction = climatology(frame, target, forecast_date, origin, params["window"])
        elif model == "arima_fourier":
            prediction = float(arima_path[horizon - 1])
        else:
            features = build_direct_features(
                frame,
                target,
                horizon,
                tuple(params.get("lags", ())),
                tuple(params.get("windows", ())),
                int(params.get("fourier_order", 4)),
            )
            train_mask = (features.index <= origin) & frame[target].notna() & features.notna().all(axis=1)
            x_train = features.loc[train_mask].to_numpy(dtype=float)
            y_train = frame.loc[train_mask, target].to_numpy(dtype=float)
            x_pred = features.loc[[forecast_date]].to_numpy(dtype=float)
            if len(y_train) < 100 or np.isnan(x_pred).any():
                raise ValueError(f"Insufficient valid features for {model}, origin={origin}, h={horizon}")
            estimator = make_regressor(model, params, seed)
            estimator.fit(x_train, y_train)
            prediction = float(estimator.predict(x_pred)[0])
        rows.append(
            {
                "origin": origin,
                "forecast_date": forecast_date,
                "horizon": horizon,
                "model": model,
                "actual": float(actual) if pd.notna(actual) else np.nan,
                "prediction": prediction,
                "mase_scale": mase_scale,
                "params": json.dumps(params, sort_keys=True),
            }
        )
    return rows


def backtest(
    frame: pd.DataFrame,
    target: str,
    origins: list[pd.Timestamp],
    horizons: list[int],
    model: str,
    params: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for origin in origins:
        rows.extend(predict_origin(frame, target, origin, horizons, model, params, seed))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["error"] = result["actual"] - result["prediction"]
        result["absolute_error"] = result["error"].abs()
        result["squared_error"] = result["error"] ** 2
        result["scaled_absolute_error"] = result["absolute_error"] / result["mase_scale"]
    return result


def summarize_predictions(predictions: pd.DataFrame, interval_level: float | None = None) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    group_columns = ["model", "horizon"]
    summary = predictions.dropna(subset=["actual"]).groupby(group_columns).agg(
        observations=("actual", "size"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        bias=("error", "mean"),
        mase=("scaled_absolute_error", "mean"),
    ).reset_index()
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    if {"lower", "upper"}.issubset(predictions.columns):
        covered = predictions["actual"].between(predictions["lower"], predictions["upper"])
        enriched = predictions.assign(covered=covered, width=predictions["upper"] - predictions["lower"])
        interval = enriched.groupby(group_columns).agg(coverage=("covered", "mean"), mean_interval_width=("width", "mean")).reset_index()
        summary = summary.merge(interval, on=group_columns, how="left")
        summary["nominal_coverage"] = interval_level
    return summary


def add_conformal_intervals(predictions: pd.DataFrame, calibration: pd.DataFrame, level: float) -> pd.DataFrame:
    if not 0 < level < 1:
        raise ValueError("Conformal interval level must be strictly between 0 and 1")
    if calibration.empty or "absolute_error" not in calibration:
        raise ValueError("Non-empty calibration residuals are required for conformal intervals")
    result = predictions.copy()
    # Finite-sample split-conformal correction: ceil((n+1)*level)/n, clipped at 1.
    def conformal_radius(values: pd.Series) -> float:
        clean = values.dropna()
        if clean.empty:
            raise ValueError("Calibration residuals contain no finite errors")
        corrected = min(1.0, np.ceil((len(clean) + 1) * level) / len(clean))
        return float(clean.quantile(corrected, interpolation="higher"))

    global_quantile = conformal_radius(calibration["absolute_error"])
    radii = calibration.groupby(["model", "horizon"])["absolute_error"].apply(conformal_radius)
    result["interval_radius"] = [radii.get((m, h), global_quantile) for m, h in zip(result["model"], result["horizon"])]
    result["lower"] = result["prediction"] - result["interval_radius"]
    result["upper"] = result["prediction"] + result["interval_radius"]
    return result


def add_ensemble(
    predictions: pd.DataFrame,
    chosen_models: list[str],
    weights: dict[str, float] | None = None,
    name: str = "validation_ensemble",
) -> pd.DataFrame:
    selected = predictions[predictions["model"].isin(chosen_models)]
    if selected.empty:
        return predictions
    if weights is None:
        weights = {model: 1.0 for model in chosen_models}
    selected = selected.assign(model_weight=selected["model"].map(weights).fillna(0.0))
    if float(selected["model_weight"].sum()) <= 0:
        raise ValueError("Ensemble weights must contain at least one positive value")
    selected["weighted_prediction"] = selected["prediction"] * selected["model_weight"]
    aggregations: dict[str, tuple[str, str]] = {
        "actual": ("actual", "first"),
        "weighted_prediction": ("weighted_prediction", "sum"),
        "available_weight": ("model_weight", "sum"),
    }
    if "mase_scale" in selected:
        aggregations["mase_scale"] = ("mase_scale", "first")
    ensemble = selected.groupby(["origin", "forecast_date", "horizon"], as_index=False).agg(**aggregations)
    ensemble["prediction"] = ensemble.pop("weighted_prediction") / ensemble.pop("available_weight")
    ensemble["model"] = name
    ensemble["params"] = json.dumps({"members": chosen_models, "weights": weights}, sort_keys=True)
    ensemble["error"] = ensemble["actual"] - ensemble["prediction"]
    ensemble["absolute_error"] = ensemble["error"].abs()
    ensemble["squared_error"] = ensemble["error"] ** 2
    if "mase_scale" in ensemble:
        ensemble["scaled_absolute_error"] = ensemble["absolute_error"] / ensemble["mase_scale"]
    return pd.concat([predictions, ensemble[predictions.columns]], ignore_index=True)
