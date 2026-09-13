"""Download and normalize daily Delhi weather data from the Open-Meteo archive."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "delhi_weather_2019_2023.csv"


def download(config_path: Path, output_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    daily = [
        "temperature_2m_mean",
        "temperature_2m_min",
        "temperature_2m_max",
        "precipitation_sum",
        "wind_speed_10m_mean",
        "shortwave_radiation_sum",
    ]
    params = {
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "daily": ",".join(daily),
        "timezone": config["timezone"],
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "delhi-weather-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "daily" not in payload:
        raise RuntimeError(f"Archive response did not contain daily data: {payload}")

    frame = pd.DataFrame(payload["daily"]).rename(
        columns={
            "time": "date",
            "temperature_2m_mean": "temperature_mean",
            "temperature_2m_min": "temperature_min",
            "temperature_2m_max": "temperature_max",
            "wind_speed_10m_mean": "wind_speed_mean",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["source"] = "open-meteo-archive"
    frame["latitude"] = payload.get("latitude", config["latitude"])
    frame["longitude"] = payload.get("longitude", config["longitude"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = download(args.config, args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
