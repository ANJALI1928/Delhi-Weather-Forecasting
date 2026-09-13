from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_and_clean, write_quality_report
from .evaluation import add_conformal_intervals, add_ensemble, backtest, origins_between, predict_origin, summarize_predictions
from .features import append_future_dates
from .models import candidate_grid
from .reporting import write_plots


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _report(config: dict, quality: dict, leaderboard: pd.DataFrame, chosen: list[str], output: Path) -> None:
    best = leaderboard.iloc[0]
    text = f"""# Delhi weather forecasting report

## Experiment

- Target: `{config['target']}`
- Horizons: {config['horizons']} days
- Source range: {quality['date_min']} to {quality['date_max']}
- Final test length: {config['test_days']} days
- Model tuning: expanding-window rolling origins
- Interval calibration: later, disjoint rolling origins
- Primary selection metric: validation MAE averaged across horizons
- Prediction interval: {int(config['prediction_interval'] * 100)}% split-conformal absolute-residual interval

## Selection

The validation-selected ensemble contains: {', '.join(chosen)}.
The leading final-test entry is **{best['model']}**, with MAE **{best['mae']:.3f} °C** and
RMSE **{best['rmse']:.3f} °C** across evaluated origins and horizons.

Model selection was completed before inspecting final-test performance. Actual future weather
covariates were not used; regression models receive only lagged observations and known calendar
features.

## Outputs

- `data_quality.json`: input audit
- `tuning_results.csv`: every validation configuration
- `selected_models.json`: validation-selected parameters
- `tuning_predictions.csv`: out-of-sample predictions used only for model selection
- `calibration_predictions.csv`: later out-of-sample residuals used only for intervals
- `validation_predictions.csv`: compatibility alias of `calibration_predictions.csv`
- `test_predictions.csv`: untouched-period predictions and intervals
- `metrics_by_model_horizon.csv`: horizon-specific final metrics
- `leaderboard.csv`: final comparison averaged across horizons
- `future_forecast.csv`: forecasts beyond the last observation
- `plots/`: historical series, model leaderboard, and final forecast visualizations

## Interpretation cautions

Open-Meteo archive observations are appropriate for a reproducible demonstration but should not be
described as official IMD station measurements. With only five years of daily data, annual extremes
are sparsely represented and neural networks are not justified. Extend the history or train a global
multi-city/hourly model before adding LSTM, GRU, N-BEATS, or TFT models.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def run_pipeline(data_path: Path, config_path: Path, output: Path, fast: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    frame, quality = load_and_clean(data_path, config["target"])
    write_quality_report(quality, output / "data_quality.json")

    horizons = [int(value) for value in config["horizons"]]
    max_horizon = max(horizons)
    test_start = frame.index.max() - pd.Timedelta(days=int(config["test_days"]) - 1)
    development_end = test_start - pd.Timedelta(days=1)
    earliest_origin = frame.index.min() + pd.Timedelta(days=int(config["minimum_training_days"]))
    cv_end = development_end - pd.Timedelta(days=max_horizon)
    cv_count = min(4, int(config["cv_origins"])) if fast else int(config["cv_origins"])
    cv_origins = origins_between(earliest_origin, cv_end, cv_count)
    if len(cv_origins) < 4:
        raise ValueError("Not enough history for separate tuning and calibration periods; add data or reduce minimum_training_days")
    split = max(2, min(len(cv_origins) - 2, int(np.ceil(len(cv_origins) * 0.6))))
    tuning_origins = cv_origins[:split]
    calibration_origins = cv_origins[split:]

    test_origins = list(pd.date_range(test_start, frame.index.max() - pd.Timedelta(days=max_horizon), freq=f"{config['test_origin_stride']}D"))
    grid = candidate_grid(include_optional=not fast)
    tuning_frames = []
    selected: dict[str, dict] = {}
    validation_predictions = []

    for model, candidates in grid.items():
        if fast:
            candidates = candidates[:2]
        candidate_scores = []
        candidate_predictions = []
        for candidate_id, params in enumerate(candidates):
            predictions = backtest(frame, config["target"], tuning_origins, horizons, model, params, config["random_seed"])
            score = float(predictions["absolute_error"].mean())
            candidate_scores.append({"model": model, "candidate_id": candidate_id, "validation_mae": score, "params": json.dumps(_jsonable(params), sort_keys=True)})
            candidate_predictions.append(predictions)
        scores = pd.DataFrame(candidate_scores).sort_values("validation_mae")
        best_id = int(scores.iloc[0]["candidate_id"])
        selected[model] = candidates[best_id]
        tuning_frames.append(scores)
        validation_predictions.append(candidate_predictions[best_id])

    tuning = pd.concat(tuning_frames, ignore_index=True).sort_values(["validation_mae", "model"])
    tuning_predictions = pd.concat(validation_predictions, ignore_index=True)
    tuning_model_mae = tuning_predictions.groupby("model")["absolute_error"].mean().sort_values()
    chosen = list(tuning_model_mae.head(int(config["ensemble_size"])).index)
    inverse_error = 1.0 / tuning_model_mae.loc[chosen].clip(lower=1e-9)
    ensemble_weights = (inverse_error / inverse_error.sum()).to_dict()
    tuning_predictions = add_ensemble(tuning_predictions, chosen, ensemble_weights)

    # Calibration is chronologically after model selection. Reusing tuning residuals would make
    # interval coverage optimistic because the same errors were used to choose the models.
    calibration_frames = [
        backtest(frame, config["target"], calibration_origins, horizons, model, params, config["random_seed"])
        for model, params in selected.items()
    ]
    calibration = pd.concat(calibration_frames, ignore_index=True)
    calibration = add_ensemble(calibration, chosen, ensemble_weights)

    test_frames = [
        backtest(frame, config["target"], test_origins, horizons, model, params, config["random_seed"])
        for model, params in selected.items()
    ]
    test = pd.concat(test_frames, ignore_index=True)
    test = add_ensemble(test, chosen, ensemble_weights)
    test = add_conformal_intervals(test, calibration, float(config["prediction_interval"]))

    metrics = summarize_predictions(test, float(config["prediction_interval"]))
    baseline_mae = metrics.loc[metrics["model"] == "seasonal_naive_365", ["horizon", "mae"]].rename(
        columns={"mae": "seasonal_naive_mae"}
    )
    metrics = metrics.merge(baseline_mae, on="horizon", how="left")
    metrics["skill_vs_seasonal_naive_pct"] = 100 * (1 - metrics["mae"] / metrics["seasonal_naive_mae"])
    leaderboard = metrics.groupby("model", as_index=False).agg(
        observations=("observations", "sum"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        coverage=("coverage", "mean"),
        mean_interval_width=("mean_interval_width", "mean"),
        mase=("mase", "mean"),
        skill_vs_seasonal_naive_pct=("skill_vs_seasonal_naive_pct", "mean"),
    ).sort_values("mae")

    extended = append_future_dates(frame, max_horizon)
    future_rows = []
    final_origin = frame.index.max()
    for model, params in selected.items():
        future_rows.extend(predict_origin(extended, config["target"], final_origin, horizons, model, params, config["random_seed"]))
    future = pd.DataFrame(future_rows)
    future["error"] = np.nan
    future["absolute_error"] = np.nan
    future["squared_error"] = np.nan
    future["scaled_absolute_error"] = np.nan
    future = add_ensemble(future, chosen, ensemble_weights)
    future = add_conformal_intervals(future, calibration, float(config["prediction_interval"]))

    tuning.to_csv(output / "tuning_results.csv", index=False)
    tuning_predictions.to_csv(output / "tuning_predictions.csv", index=False)
    calibration.to_csv(output / "calibration_predictions.csv", index=False)
    # Backward-compatible alias for consumers of earlier project versions.
    calibration.to_csv(output / "validation_predictions.csv", index=False)
    test.to_csv(output / "test_predictions.csv", index=False)
    metrics.to_csv(output / "metrics_by_model_horizon.csv", index=False)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    future.to_csv(output / "future_forecast.csv", index=False)
    (output / "selected_models.json").write_text(
        json.dumps(
            {
                "models": _jsonable(selected),
                "ensemble_members": chosen,
                "ensemble_weights": ensemble_weights,
                "tuning_origins": [str(x.date()) for x in tuning_origins],
                "calibration_origins": [str(x.date()) for x in calibration_origins],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_plots(frame, config["target"], future, leaderboard, output)
    _report(config, quality, leaderboard, chosen, output)
    return {"leaderboard": leaderboard, "chosen": chosen, "output": output}
