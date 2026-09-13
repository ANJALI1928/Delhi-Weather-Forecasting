# Archived GFS bias-correction report

Archived 1–7 day GFS forecasts are calibrated on 2022 and evaluated on 2023 against hourly ERA5.
The correction uses only mean residuals grouped by calendar month and hour of day.

- Raw mean MAE: 4.537 °C
- Bias-corrected mean MAE: 1.783 °C
- Relative improvement: 60.7%

This tests an operationally available forecast, unlike the historical-only GRU/LSTM experiment.
ERA5 remains the verification target and is not an IMD station observation.
