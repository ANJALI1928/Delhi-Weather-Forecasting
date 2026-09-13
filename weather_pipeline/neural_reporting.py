from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def write_neural_report_and_plots(
    leaderboard: pd.DataFrame,
    metrics: pd.DataFrame,
    history: pd.DataFrame,
    metadata: dict,
    output: Path,
) -> None:
    best = leaderboard.iloc[0]
    weekly = leaderboard[leaderboard["model"] == "weekly_naive"].iloc[0]
    improvement = 100 * (1 - float(best["mae"]) / float(weekly["mae"]))
    report = f"""# Hourly Delhi neural forecasting report

## Data and split

- Source: Open-Meteo Historical Weather API, explicitly pinned to ERA5
- Coverage: 2000-01-01 through 2023-12-31 ({metadata['rows']:,} hourly observations)
- Input window: {metadata['config']['input_window_hours']} hours
- Horizons: {metadata['config']['horizons']} hours
- Training: through 2019; architecture selection: 2020–2021; untouched test: 2022–2023
- Training sequences: {metadata['train_sequences']:,}; validation: {metadata['validation_sequences']:,}; test: {metadata['test_sequences']:,}
- Compute device: {metadata['device']}

## Result

The leading model is **{best['model']}**, with mean horizon-level MAE **{best['mae']:.3f} °C** and
RMSE **{best['rmse']:.3f} °C**. This is a **{improvement:.1f}% MAE reduction** relative to the weekly
seasonal-naïve baseline ({weekly['mae']:.3f} °C). The selected configurations were
GRU `{metadata['winners']['gru']}` and LSTM `{metadata['winners']['lstm']}`.

The headline MAE equally averages the seven requested horizons. Use `metrics_by_horizon.csv` for
decisions at a specific lead time. Prediction intervals are split-conformal intervals calibrated on
2020–2021 residuals.

## Leakage controls

Every neural input sequence ends at its forecast origin. Realized weather after the origin is never
used as input. Feature scaling is fitted only on the training period. The architecture and hidden
size are selected before evaluating 2022–2023.

## Files

- `best_gru.pt`: deployable GRU checkpoint plus preprocessing metadata
- `tuning_results.csv`: validation comparison of candidate architectures
- `training_history.csv`: tuning and final-refit losses
- `validation_predictions.csv`: calibration-period predictions
- `test_predictions.csv`: untouched test predictions and 90% intervals
- `future_forecast.csv`: checkpoint-generated GRU forecasts after the last observation
- `metrics_by_horizon.csv`: horizon-specific MAE, RMSE, bias and coverage
- `leaderboard.csv`: averaged model comparison
- `run_metadata.json` and `data_quality.json`: reproducibility metadata

## Scope

ERA5 is a gridded reanalysis product, not an official IMD station record. These results establish
performance for reconstructing ERA5 temperature at Delhi's coordinates. Claims about station-level
forecasting require an IMD target series and a new untouched evaluation.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    cache = output / ".matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)

    ordered = leaderboard.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ordered["model"], ordered["mae"], color="#4c78a8")
    for position, value in enumerate(ordered["mae"]):
        ax.text(value + 0.025, position, f"{value:.2f}", va="center")
    ax.set(title="Hourly Delhi forecast comparison, 2022–2023", xlabel="Mean MAE across horizons (°C)", ylabel="")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "neural_model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model, group in metrics.groupby("model"):
        ax.plot(group["horizon"], group["mae"], marker="o", label=model)
    ax.set(xscale="log", title="Error by forecast horizon", xlabel="Forecast horizon (hours, log scale)", ylabel="MAE (°C)")
    ax.set_xticks(sorted(metrics["horizon"].unique()), labels=[str(x) for x in sorted(metrics["horizon"].unique())])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "mae_by_horizon.png", dpi=160)
    plt.close(fig)

    tuning_history = history[history["model"].isin([m for m in history["model"].unique() if "final_refit" not in m])]
    if not tuning_history.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model, group in tuning_history.groupby("model"):
            ax.plot(group["epoch"], group["validation_loss"], marker="o", label=model)
        ax.set(title="Architecture tuning history", xlabel="Epoch", ylabel="Validation MAE (standardized target)")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "training_history.png", dpi=160)
        plt.close(fig)
