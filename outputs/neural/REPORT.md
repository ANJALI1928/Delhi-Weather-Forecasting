# Hourly Delhi neural forecasting report

## Data and split

- Source: Open-Meteo Historical Weather API, explicitly pinned to ERA5
- Coverage: 2000-01-01 through 2023-12-31 (210,384 hourly observations)
- Input window: 168 hours
- Horizons: [1, 3, 6, 12, 24, 72, 168] hours
- Training: through 2019; architecture selection: 2020–2021; untouched test: 2022–2023
- Training sequences: 29,165; validation: 2,896; test: 2,892
- Compute device: cpu

## Result

The leading model is **neural_ensemble**, with mean horizon-level MAE **1.139 °C** and
RMSE **1.521 °C**. This is a **55.1% MAE reduction** relative to the weekly
seasonal-naïve baseline (2.536 °C). The selected configurations were
GRU `gru_h64_l1` and LSTM `lstm_h64_l1`.

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
