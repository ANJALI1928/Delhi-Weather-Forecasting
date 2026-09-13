from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weather_pipeline.data import load_and_clean, synthetic_weather
from weather_pipeline.evaluation import add_ensemble, backtest, summarize_predictions
from weather_pipeline.features import build_direct_features
from weather_pipeline.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_short_gap_imputation_is_causal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "weather.csv"
            data = synthetic_weather("2020-01-01", "2020-01-10")
            data.loc[1, ["temperature_mean", "temperature_min", "temperature_max"]] = np.nan
            previous = float(data.loc[0, "temperature_mean"])
            following = float(data.loc[2, "temperature_mean"])
            data.to_csv(csv, index=False)
            frame, _ = load_and_clean(csv, "temperature_mean")
            self.assertEqual(float(frame.loc["2020-01-02", "temperature_mean"]), previous)
            self.assertNotEqual(float(frame.loc["2020-01-02", "temperature_mean"]), (previous + following) / 2)

    def test_weighted_ensemble_uses_declared_weights(self) -> None:
        base = pd.DataFrame(
            {
                "origin": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "forecast_date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
                "horizon": [1, 1],
                "model": ["a", "b"],
                "actual": [13.0, 13.0],
                "prediction": [10.0, 20.0],
                "params": ["{}", "{}"],
                "error": [3.0, -7.0],
                "absolute_error": [3.0, 7.0],
                "squared_error": [9.0, 49.0],
            }
        )
        result = add_ensemble(base, ["a", "b"], {"a": 0.75, "b": 0.25})
        prediction = result[result["model"] == "validation_ensemble"].iloc[0]["prediction"]
        self.assertEqual(float(prediction), 12.5)

    def test_mase_uses_origin_history_scale_not_test_baseline_error(self) -> None:
        data = synthetic_weather("2018-01-01", "2023-12-31").set_index("date")
        predictions = backtest(
            data,
            "temperature_mean",
            [pd.Timestamp("2022-06-01")],
            [1],
            "seasonal_naive_365",
            {},
            42,
        )
        summary = summarize_predictions(predictions)
        expected = float(predictions.iloc[0]["absolute_error"] / predictions.iloc[0]["mase_scale"])
        self.assertAlmostEqual(float(summary.iloc[0]["mase"]), expected)

    def test_direct_features_do_not_use_values_after_origin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = synthetic_weather("2018-01-01", "2023-12-31")
            csv = tmp_path / "weather.csv"
            data.to_csv(csv, index=False)
            frame, _ = load_and_clean(csv, "temperature_mean")
            horizon = 7
            features_before = build_direct_features(frame, "temperature_mean", horizon, (0, 1, 6), (7,), 2)
            target_date = pd.Timestamp("2022-06-15")
            origin = target_date - pd.Timedelta(days=horizon)
            modified = frame.copy()
            modified.loc[modified.index > origin, "temperature_mean"] = 9999
            features_after = build_direct_features(modified, "temperature_mean", horizon, (0, 1, 6), (7,), 2)
            pd.testing.assert_series_equal(features_before.loc[target_date], features_after.loc[target_date])

    def test_fast_pipeline_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = synthetic_weather("2016-01-01", "2023-12-31")
            csv = tmp_path / "weather.csv"
            data.to_csv(csv, index=False)
            config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            config["cv_origins"] = 4
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = tmp_path / "outputs"
            result = run_pipeline(csv, config_path, output, fast=True)
            self.assertFalse(result["leaderboard"].empty)
            for name in ["REPORT.md", "leaderboard.csv", "future_forecast.csv", "selected_models.json"]:
                self.assertTrue((output / name).exists())
            selection = json.loads((output / "selected_models.json").read_text(encoding="utf-8"))
            self.assertTrue(set(selection["tuning_origins"]).isdisjoint(selection["calibration_origins"]))
            self.assertAlmostEqual(sum(selection["ensemble_weights"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
