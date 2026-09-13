from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .neural_reporting import write_neural_report_and_plots


RAW_FEATURES = [
    "temperature",
    "relative_humidity",
    "dew_point",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed",
    "wind_direction",
    "shortwave_radiation",
]


@dataclass(frozen=True)
class RecurrentSpec:
    architecture: str
    hidden_size: int
    layers: int
    dropout: float


def load_hourly(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    missing = set(RAW_FEATURES + ["timestamp"]).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing hourly columns: {sorted(missing)}")
    frame = frame.set_index("timestamp")
    complete = pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="timestamp")
    inserted = len(complete.difference(frame.index))
    frame = frame.reindex(complete)
    for column in RAW_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_before = frame[RAW_FEATURES].isna().sum().astype(int).to_dict()
    # Keep preprocessing deployable at every historical origin. Bidirectional interpolation
    # would let a sequence see measurements recorded after a missing hour.
    frame[RAW_FEATURES] = frame[RAW_FEATURES].ffill(limit=6)
    remaining = int(frame[RAW_FEATURES].isna().sum().sum())
    if remaining:
        raise ValueError(f"Hourly data retain {remaining} missing numeric cells after limited interpolation")
    quality = {
        "rows": len(frame),
        "first_timestamp": str(frame.index.min()),
        "last_timestamp": str(frame.index.max()),
        "inserted_missing_hours": inserted,
        "missing_before_imputation": missing_before,
        "remaining_missing_cells": remaining,
        "source": str(frame["source"].dropna().iloc[0]) if "source" in frame else "unknown",
    }
    return frame, quality


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[[c for c in RAW_FEATURES if c != "wind_direction"]].copy()
    radians = np.deg2rad(frame["wind_direction"].to_numpy())
    result["wind_direction_sin"] = np.sin(radians)
    result["wind_direction_cos"] = np.cos(radians)
    hour = frame.index.hour.to_numpy()
    day = frame.index.dayofyear.to_numpy()
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["year_sin"] = np.sin(2 * np.pi * day / 365.25)
    result["year_cos"] = np.cos(2 * np.pi * day / 365.25)
    return result.astype("float32")


def valid_origins(
    index: pd.DatetimeIndex,
    window: int,
    max_horizon: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stride: int,
) -> np.ndarray:
    positions = np.arange(window - 1, len(index) - max_horizon, stride, dtype=np.int64)
    timestamps = index[positions]
    return positions[(timestamps >= start) & (timestamps <= end)]


def baseline_predictions(target: np.ndarray, origins: np.ndarray, horizons: list[int]) -> dict[str, np.ndarray]:
    horizon_array = np.asarray(horizons)
    actual = np.stack([target[origins + h] for h in horizon_array], axis=1)
    persistence = np.repeat(target[origins, None], len(horizons), axis=1)
    weekly = np.stack([target[origins + h - 168] for h in horizon_array], axis=1)
    return {"actual": actual, "persistence": persistence, "weekly_naive": weekly}


def require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install requirements-neural.txt") from exc


def make_dataset(features: np.ndarray, target: np.ndarray, origins: np.ndarray, window: int, horizons: list[int]):
    torch = require_torch()

    class WindowDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(origins)

        def __getitem__(self, item):
            origin = int(origins[item])
            x = features[origin - window + 1 : origin + 1]
            y = target[origin + np.asarray(horizons)]
            return torch.from_numpy(x), torch.from_numpy(y.astype("float32"))

    return WindowDataset()


def build_model(spec: RecurrentSpec, input_size: int, outputs: int):
    torch = require_torch()

    class RecurrentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            recurrent = torch.nn.GRU if spec.architecture == "gru" else torch.nn.LSTM
            self.recurrent = recurrent(
                input_size,
                spec.hidden_size,
                num_layers=spec.layers,
                batch_first=True,
                dropout=spec.dropout if spec.layers > 1 else 0.0,
            )
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(spec.hidden_size),
                torch.nn.Linear(spec.hidden_size, outputs),
            )

        def forward(self, x):
            sequence, _ = self.recurrent(x)
            return self.head(sequence[:, -1, :])

    return RecurrentModel()


def train_model(
    spec: RecurrentSpec,
    train_dataset,
    validation_dataset,
    input_size: int,
    outputs: int,
    config: dict,
    epochs: int,
    device,
) -> tuple[Any, list[dict[str, float]], float]:
    torch = require_torch()
    model = build_model(spec, input_size, outputs).to(device)
    generator = torch.Generator().manual_seed(int(config["random_seed"]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    loss_fn = torch.nn.L1Loss()
    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * len(x_batch)
            train_count += len(x_batch)
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for x_batch, y_batch in validation_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                loss = loss_fn(model(x_batch), y_batch)
                validation_total += float(loss.item()) * len(x_batch)
                validation_count += len(x_batch)
        train_loss = train_total / train_count
        validation_loss = validation_total / validation_count
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(config["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, history, best_loss


def predict(model, dataset, batch_size: int, device) -> np.ndarray:
    torch = require_torch()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            rows.append(model(x_batch.to(device)).cpu().numpy())
    return np.concatenate(rows)


def refit_fixed_epochs(
    spec: RecurrentSpec,
    dataset,
    input_size: int,
    outputs: int,
    config: dict,
    epochs: int,
    device,
) -> tuple[Any, list[dict[str, float]]]:
    torch = require_torch()
    model = build_model(spec, input_size, outputs).to(device)
    generator = torch.Generator().manual_seed(int(config["random_seed"]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    loss_fn = torch.nn.L1Loss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(x_batch)
            count += len(x_batch)
        history.append({"epoch": epoch, "train_loss": total / count})
    return model, history


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["absolute_error"] = (result["actual"] - result["prediction"]).abs()
    result["squared_error"] = (result["actual"] - result["prediction"]) ** 2
    result["error"] = result["actual"] - result["prediction"]
    metrics = result.groupby(["model", "horizon"]).agg(
        observations=("actual", "size"), mae=("absolute_error", "mean"), mse=("squared_error", "mean"), bias=("error", "mean")
    ).reset_index()
    metrics["rmse"] = np.sqrt(metrics.pop("mse"))
    return metrics


def to_prediction_frame(model: str, origins: np.ndarray, index: pd.DatetimeIndex, horizons: list[int], actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    rows = []
    for row, origin in enumerate(origins):
        for column, horizon in enumerate(horizons):
            rows.append(
                {
                    "model": model,
                    "origin": index[origin],
                    "forecast_time": index[origin + horizon],
                    "horizon": horizon,
                    "actual": float(actual[row, column]),
                    "prediction": float(predicted[row, column]),
                }
            )
    return pd.DataFrame(rows)


def add_intervals(test: pd.DataFrame, validation: pd.DataFrame, level: float) -> pd.DataFrame:
    calibration = validation.assign(absolute_error=(validation["actual"] - validation["prediction"]).abs())

    def radius(values: pd.Series) -> float:
        probability = min(1.0, np.ceil((len(values) + 1) * level) / len(values))
        return float(values.quantile(probability, interpolation="higher"))

    radii = calibration.groupby(["model", "horizon"])["absolute_error"].apply(radius)
    result = test.copy()
    result["interval_radius"] = [radii.loc[(m, h)] for m, h in zip(result["model"], result["horizon"])]
    result["lower"] = result["prediction"] - result["interval_radius"]
    result["upper"] = result["prediction"] + result["interval_radius"]
    return result


def add_neural_ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    neural = predictions[predictions["model"].isin(["gru", "lstm"])]
    ensemble = neural.groupby(["origin", "forecast_time", "horizon"], as_index=False).agg(
        actual=("actual", "first"), prediction=("prediction", "mean")
    )
    ensemble["model"] = "neural_ensemble"
    return pd.concat([predictions, ensemble[predictions.columns]], ignore_index=True)


def forecast_saved_gru(data_path: Path, output: Path) -> pd.DataFrame:
    """Forecast requested future horizons using the deployable GRU checkpoint."""
    torch = require_torch()
    checkpoint = torch.load(output / "best_gru.pt", map_location="cpu", weights_only=False)
    frame, _ = load_hourly(data_path)
    features_frame = feature_frame(frame)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype="float32")
    feature_std = np.asarray(checkpoint["feature_std"], dtype="float32")
    features = ((features_frame.to_numpy(dtype="float32") - feature_mean) / feature_std).astype("float32")
    window = int(checkpoint["window"])
    spec_data = checkpoint["spec"]
    spec = RecurrentSpec(
        architecture="gru",
        hidden_size=int(spec_data["hidden_size"]),
        layers=int(spec_data["layers"]),
        dropout=float(spec_data["dropout"]),
    )
    model = build_model(spec, features.shape[1], len(checkpoint["horizons"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        normalized = model(torch.from_numpy(features[-window:][None, :, :])).numpy()[0]
    values = normalized * float(checkpoint["target_std"]) + float(checkpoint["target_mean"])
    origin = frame.index.max()
    future = pd.DataFrame(
        {
            "model": "gru",
            "origin": origin,
            "forecast_time": [origin + pd.Timedelta(hours=int(h)) for h in checkpoint["horizons"]],
            "horizon": checkpoint["horizons"],
            "actual": np.nan,
            "prediction": values,
        }
    )
    validation = pd.read_csv(output / "validation_predictions.csv", parse_dates=["origin", "forecast_time"])
    metadata_path = output / "run_metadata.json"
    level = 0.9
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        level = float(metadata.get("config", {}).get("prediction_interval", level))
    future = add_intervals(future, validation[validation["model"] == "gru"], level)
    future.to_csv(output / "future_forecast.csv", index=False)
    return future


def run_neural_benchmark(data_path: Path, config_path: Path, output: Path, fast: bool = False) -> dict[str, Any]:
    torch = require_torch()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output.mkdir(parents=True, exist_ok=True)

    frame, quality = load_hourly(data_path)
    features_frame = feature_frame(frame)
    horizons = [int(h) for h in config["horizons"]]
    window = int(config["input_window_hours"])
    train_end = pd.Timestamp(config["train_end"])
    validation_end = pd.Timestamp(config["validation_end"])
    first_origin = frame.index[window - 1]
    train_origins = valid_origins(frame.index, window, max(horizons), first_origin, train_end - pd.Timedelta(hours=max(horizons)), int(config["training_stride"]))
    validation_origins = valid_origins(frame.index, window, max(horizons), train_end + pd.Timedelta(hours=1), validation_end - pd.Timedelta(hours=max(horizons)), int(config["evaluation_stride"]))
    test_origins = valid_origins(frame.index, window, max(horizons), validation_end + pd.Timedelta(hours=1), frame.index.max() - pd.Timedelta(hours=max(horizons)), int(config["evaluation_stride"]))
    if fast:
        train_origins = train_origins[-8000:]
        validation_origins = validation_origins[-1000:]
        test_origins = test_origins[-1000:]

    train_feature_rows = features_frame.index <= train_end
    feature_mean = features_frame.loc[train_feature_rows].mean().to_numpy(dtype="float32")
    feature_std = features_frame.loc[train_feature_rows].std().replace(0, 1).to_numpy(dtype="float32")
    features = ((features_frame.to_numpy(dtype="float32") - feature_mean) / feature_std).astype("float32")
    target = frame[config["target"]].to_numpy(dtype="float32")
    target_mean = float(frame.loc[:train_end, config["target"]].mean())
    target_std = float(frame.loc[:train_end, config["target"]].std())
    normalized_target = ((target - target_mean) / target_std).astype("float32")

    train_dataset = make_dataset(features, normalized_target, train_origins, window, horizons)
    validation_dataset = make_dataset(features, normalized_target, validation_origins, window, horizons)
    test_dataset = make_dataset(features, normalized_target, test_origins, window, horizons)
    specs = [
        RecurrentSpec("gru", 32, 1, 0.0),
        RecurrentSpec("gru", 64, 1, 0.0),
        RecurrentSpec("lstm", 32, 1, 0.0),
        RecurrentSpec("lstm", 64, 1, 0.0),
    ]
    if fast:
        specs = [RecurrentSpec("gru", 32, 1, 0.0), RecurrentSpec("lstm", 32, 1, 0.0)]

    tuning_rows = []
    trained = {}
    histories = []
    epochs = 1 if fast else int(config["tuning_epochs"])
    for spec in specs:
        model, history, validation_loss = train_model(
            spec, train_dataset, validation_dataset, features.shape[1], len(horizons), config, epochs, device
        )
        name = f"{spec.architecture}_h{spec.hidden_size}_l{spec.layers}"
        tuning_rows.append({"model": name, "validation_scaled_mae": validation_loss, **asdict(spec)})
        histories.extend([{"model": name, **row} for row in history])
        trained[name] = model

    tuning = pd.DataFrame(tuning_rows).sort_values("validation_scaled_mae")
    winners = {}
    for architecture in ("gru", "lstm"):
        winner = tuning[tuning["architecture"] == architecture].iloc[0]["model"]
        winners[architecture] = str(winner)

    # Keep original validation predictions for calibration, then refit each winning architecture
    # on all pre-test origins for a fair untouched-period evaluation.
    validation_models = {architecture: trained[winner] for architecture, winner in winners.items()}
    combined_origins = np.sort(np.concatenate([train_origins, validation_origins]))
    combined_dataset = make_dataset(features, normalized_target, combined_origins, window, horizons)
    final_models = {}
    final_epochs = 1 if fast else int(config["final_epochs"])
    for architecture, winner in winners.items():
        row = tuning[tuning["model"] == winner].iloc[0]
        spec = RecurrentSpec(
            architecture=architecture,
            hidden_size=int(row["hidden_size"]),
            layers=int(row["layers"]),
            dropout=float(row["dropout"]),
        )
        final_model, final_history = refit_fixed_epochs(
            spec, combined_dataset, features.shape[1], len(horizons), config, final_epochs, device
        )
        final_models[architecture] = final_model
        histories.extend([{"model": f"{architecture}_final_refit", "validation_loss": np.nan, **item} for item in final_history])

    baseline_validation = baseline_predictions(target, validation_origins, horizons)
    baseline_test = baseline_predictions(target, test_origins, horizons)
    validation_frames = []
    test_frames = []
    for name in ("persistence", "weekly_naive"):
        validation_frames.append(to_prediction_frame(name, validation_origins, frame.index, horizons, baseline_validation["actual"], baseline_validation[name]))
        test_frames.append(to_prediction_frame(name, test_origins, frame.index, horizons, baseline_test["actual"], baseline_test[name]))
    for architecture, winner in winners.items():
        validation_prediction = predict(validation_models[architecture], validation_dataset, int(config["batch_size"]), device) * target_std + target_mean
        test_prediction = predict(final_models[architecture], test_dataset, int(config["batch_size"]), device) * target_std + target_mean
        validation_frames.append(to_prediction_frame(architecture, validation_origins, frame.index, horizons, baseline_validation["actual"], validation_prediction))
        test_frames.append(to_prediction_frame(architecture, test_origins, frame.index, horizons, baseline_test["actual"], test_prediction))

    validation = pd.concat(validation_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    # Neural ensemble is fixed using validation-selected GRU and LSTM configurations.
    validation = add_neural_ensemble(validation)
    test = add_neural_ensemble(test)
    test = add_intervals(test, validation, float(config["prediction_interval"]))
    metrics = metrics_table(test)
    coverage_rows = test.assign(covered=test["actual"].between(test["lower"], test["upper"])).groupby(["model", "horizon"])["covered"].mean().rename("coverage").reset_index()
    metrics = metrics.merge(coverage_rows, on=["model", "horizon"])
    leaderboard = metrics.groupby("model", as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"), coverage=("coverage", "mean")).sort_values("mae")

    torch.save(
        {
            "state_dict": final_models["gru"].state_dict(),
            "spec": tuning[tuning["model"] == winners["gru"]].iloc[0].to_dict(),
            "feature_columns": list(features_frame.columns),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "horizons": horizons,
            "window": window,
        },
        output / "best_gru.pt",
    )
    history_frame = pd.DataFrame(histories)
    history_frame.to_csv(output / "training_history.csv", index=False)
    tuning.to_csv(output / "tuning_results.csv", index=False)
    validation.to_csv(output / "validation_predictions.csv", index=False)
    test.to_csv(output / "test_predictions.csv", index=False)
    metrics.to_csv(output / "metrics_by_horizon.csv", index=False)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    metadata = {
        "device": str(device),
        "winners": winners,
        "train_sequences": len(train_origins),
        "validation_sequences": len(validation_origins),
        "test_sequences": len(test_origins),
        "feature_columns": list(features_frame.columns),
        "config": config,
        **quality,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    forecast_saved_gru(data_path, output)
    write_neural_report_and_plots(leaderboard, metrics, history_frame, metadata, output)
    return {"leaderboard": leaderboard, "output": output, "winners": winners}


def rebuild_neural_reports(output: Path, data_path: Path) -> dict[str, Any]:
    """Rebuild derived ensemble metrics and reports without retraining saved networks."""
    validation = pd.read_csv(output / "validation_predictions.csv", parse_dates=["origin", "forecast_time"])
    test = pd.read_csv(output / "test_predictions.csv", parse_dates=["origin", "forecast_time"])
    validation = add_neural_ensemble(validation[validation["model"] != "neural_ensemble"])
    base_test_columns = ["model", "origin", "forecast_time", "horizon", "actual", "prediction"]
    test = add_neural_ensemble(test.loc[test["model"] != "neural_ensemble", base_test_columns])
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    quality_path = output / "data_quality.json"
    if quality_path.exists():
        for key, value in json.loads(quality_path.read_text(encoding="utf-8")).items():
            metadata.setdefault(key, value)
    level = float(metadata["config"]["prediction_interval"])
    test = add_intervals(test, validation, level)
    metrics = metrics_table(test)
    coverage = test.assign(covered=test["actual"].between(test["lower"], test["upper"])).groupby(
        ["model", "horizon"]
    )["covered"].mean().rename("coverage").reset_index()
    metrics = metrics.merge(coverage, on=["model", "horizon"])
    leaderboard = metrics.groupby("model", as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"), coverage=("coverage", "mean")
    ).sort_values("mae")
    history = pd.read_csv(output / "training_history.csv")
    validation.to_csv(output / "validation_predictions.csv", index=False)
    test.to_csv(output / "test_predictions.csv", index=False)
    metrics.to_csv(output / "metrics_by_horizon.csv", index=False)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    forecast_saved_gru(data_path, output)
    write_neural_report_and_plots(leaderboard, metrics, history, metadata, output)
    return {"leaderboard": leaderboard, "output": output, "winners": metadata["winners"]}
