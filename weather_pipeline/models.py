from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        z = (x - self.mean_) / self.scale_
        self.y_mean_ = float(np.mean(y))
        yc = y - self.y_mean_
        identity = np.eye(z.shape[1])
        self.coef_ = np.linalg.solve(z.T @ z + self.alpha * identity, z.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.scale_
        return self.y_mean_ + z @ self.coef_


def optional_sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def optional_statsmodels_available() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


def make_regressor(model: str, params: dict[str, Any], seed: int):
    if model in {"harmonic_ridge", "autoregressive_ridge"}:
        return RidgeRegressor(alpha=params["alpha"])
    if model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=-1,
        )
    if model == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"],
            early_stopping=True,
            random_state=seed,
        )
    raise KeyError(model)


def candidate_grid(include_optional: bool = True) -> dict[str, list[dict[str, Any]]]:
    grid: dict[str, list[dict[str, Any]]] = {
        "seasonal_naive_365": [{}],
        "climatology": [{"window": 7}, {"window": 15}, {"window": 30}],
        "harmonic_ridge": [
            {"alpha": alpha, "fourier_order": order, "lags": (), "windows": ()}
            for alpha in (0.1, 1.0, 10.0)
            for order in (2, 4, 6)
        ],
        "autoregressive_ridge": [
            {"alpha": alpha, "fourier_order": order, "lags": lags, "windows": (3, 7, 14, 30)}
            for alpha in (0.1, 1.0, 10.0)
            for order in (2, 4)
            for lags in ((0, 1, 2, 3, 6, 13, 29), (0, 1, 2, 6, 13, 29, 364))
        ],
    }
    if include_optional and optional_sklearn_available():
        common = {"fourier_order": 4, "lags": (0, 1, 2, 3, 6, 13, 29, 364), "windows": (3, 7, 14, 30)}
        grid["random_forest"] = [
            {**common, "n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf, "max_features": 0.8}
            for trees in (300,)
            for depth in (6, 12)
            for leaf in (2, 5)
        ]
        grid["hist_gradient_boosting"] = [
            {**common, "learning_rate": rate, "max_iter": 300, "max_leaf_nodes": leaves, "l2_regularization": l2}
            for rate in (0.03, 0.08)
            for leaves in (15, 31)
            for l2 in (0.1,)
        ]
    if include_optional and optional_statsmodels_available():
        # Annual seasonality is represented with Fourier terms rather than a computationally
        # expensive 365-day seasonal state. ARIMA errors model the remaining serial dependence.
        grid["arima_fourier"] = [
            {"order": order, "fourier_order": fourier_order}
            for order in ((1, 0, 1), (2, 0, 1), (2, 1, 1))
            for fourier_order in (2, 4)
        ]
    return grid


def arima_fourier_forecast_path(
    frame: pd.DataFrame,
    target: str,
    origin: pd.Timestamp,
    max_horizon: int,
    order: tuple[int, int, int] | list[int],
    fourier_order: int,
) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    history = frame.loc[:origin, target].dropna().astype(float)

    def exogenous(index: pd.DatetimeIndex, offset: int = 0) -> np.ndarray:
        day = index.dayofyear.to_numpy()
        columns = []
        for harmonic in range(1, fourier_order + 1):
            columns.append(np.sin(2 * np.pi * harmonic * day / 365.25))
            columns.append(np.cos(2 * np.pi * harmonic * day / 365.25))
        columns.append((np.arange(len(index), dtype=float) + offset) / 365.25)
        return np.column_stack(columns)

    x_train = exogenous(history.index)
    future_index = pd.date_range(origin + pd.Timedelta(days=1), periods=max_horizon, freq="D")
    x_future = exogenous(future_index, len(history))
    fitted = SARIMAX(
        history.to_numpy(),
        exog=x_train,
        order=tuple(order),
        trend="c" if int(order[1]) == 0 else None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method="powell", disp=False, maxiter=100)
    if not bool(fitted.mle_retvals.get("converged", False)):
        raise RuntimeError(f"ARIMA-Fourier optimizer did not converge for order={tuple(order)}")
    return np.asarray(fitted.forecast(max_horizon, exog=x_future), dtype=float)


def seasonal_naive(frame: pd.DataFrame, target: str, forecast_date: pd.Timestamp) -> float:
    reference = forecast_date - pd.Timedelta(days=365)
    if reference in frame.index and pd.notna(frame.at[reference, target]):
        return float(frame.at[reference, target])
    history = frame.loc[frame.index < forecast_date, target].dropna()
    return float(history.iloc[-1])


def climatology(frame: pd.DataFrame, target: str, forecast_date: pd.Timestamp, cutoff: pd.Timestamp, window: int) -> float:
    history = frame.loc[frame.index <= cutoff, target].dropna()
    circular_distance = np.abs(history.index.dayofyear.to_numpy() - forecast_date.dayofyear)
    circular_distance = np.minimum(circular_distance, 366 - circular_distance)
    values = history.to_numpy()[circular_distance <= window]
    return float(np.mean(values)) if len(values) else float(history.mean())
