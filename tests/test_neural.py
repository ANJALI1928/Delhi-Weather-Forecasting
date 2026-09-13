from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weather_pipeline.neural import RecurrentSpec, add_neural_ensemble, build_model, valid_origins


class NeuralPipelineTests(unittest.TestCase):
    def test_origins_respect_window_horizon_and_time_bounds(self) -> None:
        index = pd.date_range("2020-01-01", periods=1000, freq="h")
        start = pd.Timestamp("2020-01-10")
        end = pd.Timestamp("2020-01-20")
        origins = valid_origins(index, window=24, max_horizon=72, start=start, end=end, stride=6)
        self.assertTrue((index[origins] >= start).all())
        self.assertTrue((index[origins] <= end).all())
        self.assertTrue((origins >= 23).all())
        self.assertTrue((origins + 72 < len(index)).all())

    def test_ensemble_is_exact_mean(self) -> None:
        base = pd.DataFrame(
            {
                "model": ["gru", "lstm"],
                "origin": ["2023-01-01", "2023-01-01"],
                "forecast_time": ["2023-01-02", "2023-01-02"],
                "horizon": [24, 24],
                "actual": [12.0, 12.0],
                "prediction": [10.0, 14.0],
            }
        )
        result = add_neural_ensemble(base)
        ensemble = result[result["model"] == "neural_ensemble"].iloc[0]
        self.assertEqual(ensemble["prediction"], 12.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_recurrent_model_output_shape(self) -> None:
        import torch

        model = build_model(RecurrentSpec("gru", 8, 1, 0.0), input_size=5, outputs=3)
        output = model(torch.zeros(4, 24, 5))
        self.assertEqual(tuple(output.shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
