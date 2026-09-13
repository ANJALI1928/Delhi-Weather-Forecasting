from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def write_plots(
    frame: pd.DataFrame,
    target: str,
    future: pd.DataFrame,
    leaderboard: pd.DataFrame,
    output: Path,
) -> list[Path]:
    cache = output / ".matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    history_path = plot_dir / "temperature_history.png"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame.index, frame[target], color="#1f77b4", linewidth=0.8)
    ax.set(title="Delhi daily mean temperature", xlabel="Date", ylabel="Temperature (°C)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(history_path, dpi=160)
    plt.close(fig)
    paths.append(history_path)

    leaderboard_path = plot_dir / "model_comparison.png"
    ordered = leaderboard.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["model"], ordered["mae"], color="#4c78a8")
    for index, value in enumerate(ordered["mae"]):
        ax.text(value + 0.02, index, f"{value:.2f}", va="center", fontsize=9)
    ax.set(title="Final-period model comparison", xlabel="Mean absolute error (°C)", ylabel="")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(leaderboard_path, dpi=160)
    plt.close(fig)
    paths.append(leaderboard_path)

    forecast_path = plot_dir / "future_forecast.png"
    model = "validation_ensemble" if "validation_ensemble" in set(future["model"]) else leaderboard.iloc[0]["model"]
    selected = future[future["model"] == model].sort_values("forecast_date")
    recent = frame[target].dropna().iloc[-120:]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(recent.index, recent, label="Observed", color="#4c78a8")
    ax.plot(selected["forecast_date"], selected["prediction"], "o-", label=model, color="#f58518")
    ax.fill_between(selected["forecast_date"], selected["lower"], selected["upper"], color="#f58518", alpha=0.2, label="90% interval")
    ax.set(title="Forecast beyond the last observation", xlabel="Date", ylabel="Temperature (°C)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(forecast_path, dpi=160)
    plt.close(fig)
    paths.append(forecast_path)
    return paths
