# Delhi weather forecasting report

## Experiment

- Target: `temperature_mean`
- Horizons: [1, 3, 7, 14] days
- Source range: 2019-01-01 to 2023-12-31
- Final test length: 365 days
- Validation: expanding-window rolling origins
- Primary selection metric: validation MAE averaged across horizons
- Prediction interval: 90% split-conformal absolute-residual interval

## Selection

The validation-selected ensemble contains: autoregressive_ridge, arima_fourier, harmonic_ridge.
The leading final-test entry is **autoregressive_ridge**, with MAE **1.538 °C** and
RMSE **1.843 °C** across evaluated origins and horizons.

Model selection was completed before inspecting final-test performance. Actual future weather
covariates were not used; regression models receive only lagged observations and known calendar
features.

## Outputs

- `data_quality.json`: input audit
- `tuning_results.csv`: every validation configuration
- `selected_models.json`: validation-selected parameters
- `validation_predictions.csv`: out-of-sample calibration predictions
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
