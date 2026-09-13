# Data source notes

`download_data.py` retrieves daily historical weather from the Open-Meteo archive API for the
coordinates and dates in `config.json`. The API data are suitable for demonstrating a reproducible
forecasting workflow, but they are gridded/reanalysis-style data rather than an official Delhi
station record.

For a research-grade result, replace `delhi_weather_2019_2023.csv` with quality-controlled IMD
station observations and record the station identifier, coordinates, units, access date, missing
value codes, and any station moves. The pipeline accepts either source through its canonical column
names.

Future observed humidity, wind, pressure, and precipitation must not be used as predictors. The
pipeline uses only lagged observed covariates and deterministic calendar features. A production
version can add archived numerical-weather-prediction forecasts as separately labelled future
covariates.

`download_hourly_data.py` separately downloads explicitly pinned ERA5 hourly data in resumable
yearly chunks. Its default period is 2000–2023. This dataset is used only by the neural benchmark;
the input sequence ends at each forecast origin, so no realized future weather variables enter the
GRU or LSTM.
