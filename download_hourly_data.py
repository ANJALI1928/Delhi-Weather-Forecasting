"""Download consistent ERA5 hourly Delhi weather in resumable yearly chunks."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]


def fetch_year(config: dict, year: int, destination: Path, retries: int = 3) -> Path:
    if destination.exists():
        existing = pd.read_csv(destination, usecols=["timestamp"])
        expected_minimum = 24 * 365
        if len(existing) >= expected_minimum:
            return destination
    start = max(pd.Timestamp(config["start_date"]), pd.Timestamp(f"{year}-01-01"))
    end = min(pd.Timestamp(config["end_date"]), pd.Timestamp(f"{year}-12-31"))
    params = {
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "hourly": ",".join(VARIABLES),
        "timezone": config["timezone"],
        "models": config.get("model_source", "era5"),
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "delhi-neural-weather/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            hourly = payload.get("hourly")
            if not hourly:
                raise RuntimeError(f"No hourly data in response: {payload}")
            frame = pd.DataFrame(hourly).rename(
                columns={
                    "time": "timestamp",
                    "temperature_2m": "temperature",
                    "relative_humidity_2m": "relative_humidity",
                    "dew_point_2m": "dew_point",
                    "wind_speed_10m": "wind_speed",
                    "wind_direction_10m": "wind_direction",
                }
            )
            frame["source"] = f"open-meteo-{config.get('model_source', 'era5')}"
            frame["latitude"] = payload.get("latitude", config["latitude"])
            frame["longitude"] = payload.get("longitude", config["longitude"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination, index=False)
            return destination
        except Exception as exc:  # network retries are intentionally broad
            error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {year} after {retries} attempts") from error


def download(config_path: Path, output_path: Path, chunk_dir: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    start_year = pd.Timestamp(config["start_date"]).year
    end_year = pd.Timestamp(config["end_date"]).year
    paths = []
    for year in range(start_year, end_year + 1):
        path = fetch_year(config, year, chunk_dir / f"delhi_era5_hourly_{year}.csv")
        paths.append(path)
        print(f"Ready: {year} ({path.name})", flush=True)
    frames = [pd.read_csv(path, parse_dates=["timestamp"]) for path in paths]
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined):,} hourly rows to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "neural_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw" / "delhi_era5_hourly_2000_2023.csv")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "raw" / "hourly_chunks")
    args = parser.parse_args()
    download(args.config, args.output, args.chunks)


if __name__ == "__main__":
    main()
