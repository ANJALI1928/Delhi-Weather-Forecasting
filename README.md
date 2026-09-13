# Delhi temperature forecasting benchmark

This project implements an end-to-end, leakage-safe forecasting workflow for daily Delhi
temperature. It assembles and audits data, engineers only forecast-time-available features, tunes
models through expanding-window validation, evaluates once on a final-year holdout, calibrates
prediction intervals, builds a validation-selected ensemble, and produces future forecasts.

## Current repository state

The downloaded datasets and project-local virtual environment were intentionally removed after the
original experiments. The compact files under `outputs/` preserve those earlier results, but they
predate the stricter causal-imputation, disjoint-calibration, weighted-ensemble, and formal-MASE
changes in the current code. Regenerate the data and outputs before quoting updated metrics.

## What is implemented

- Reproducible Open-Meteo archive download for 2019–2023
- Daily-grid validation, duplicate checks, missingness flags, and physical-range checks
- Direct 1-, 3-, 7-, and 14-day forecasting
- Seasonal-naïve and day-of-year climatology baselines
- Fourier-seasonal ridge regression and autoregressive ridge regression
- Optional random forest and histogram gradient boosting when scikit-learn is installed
- Expanding-window hyperparameter selection using validation MAE
- Chronologically separate residual calibration for honest conformal intervals
- Untouched final-year rolling test
- MAE, RMSE, formal in-sample-scaled MASE, seasonal-naïve skill, bias, interval coverage, and interval width by model and horizon
- Split-conformal 90% prediction intervals
- Inverse-validation-error weighted ensemble selected before calibration and testing
- Causal short-gap imputation that never reads a future measurement
- Machine-readable results plus a Markdown report

The default pipeline deliberately excludes LSTM/GRU models. Five years of daily observations are
too few for a credible standalone neural-network comparison. Add a longer hourly history or a
multi-city dataset before enabling a deep-learning track.

## Hourly GRU/LSTM benchmark

The neural track uses explicitly pinned ERA5 hourly data for 2000–2023, a 168-hour input window,
and joint forecasts at 1, 3, 6, 12, 24, 72, and 168 hours. It tunes GRU/LSTM hidden sizes using
2020–2021, refits the selected networks on all pre-2022 sequences, and evaluates once on 2022–2023.

```powershell
python -m pip install -r requirements-neural.txt
python download_hourly_data.py
python run_neural_forecasting.py
```

Use `--fast` for a one-epoch integration check. Full results are written to `outputs/neural`.

The operational extension downloads GFS forecasts archived at fixed 1–7 day lead times and tests a
month/hour bias correction calibrated on 2022 against an untouched 2023 period:

```powershell
python download_previous_runs.py
python run_bias_correction.py
```

## Run

Create and activate an isolated environment from PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python run_forecasting.py
```

For a quick dependency-light smoke benchmark:

```powershell
python run_forecasting.py --fast
```

Outputs are written to `outputs/latest`. Use `--data` to point to an IMD-derived CSV with the
canonical columns documented in `data/raw/SOURCE_NOTES.md`.

## Test

The tests use deterministic synthetic data and never report it as a real result:

```powershell
python -m unittest discover -s tests -v
```

The tests use Python's standard-library test runner and require no additional test dependency.
