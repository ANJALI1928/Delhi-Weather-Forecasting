"""Download archived GFS temperature forecasts at fixed 1–7 day lead times."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def download(output: Path, start: str = "2022-01-01", end: str = "2023-12-31") -> Path:
    variables = [f"temperature_2m_previous_day{lead}" for lead in range(1, 8)]
    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "Asia/Kolkata",
        "models": "gfs_seamless",
    }
    url = "https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "delhi-forecast-correction/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    if "hourly" not in payload:
        raise RuntimeError(f"No hourly archive returned: {payload}")
    frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "timestamp"})
    frame["source"] = "open-meteo-previous-runs-gfs"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} archived forecast rows to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw" / "delhi_gfs_previous_runs_2022_2023.csv")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2023-12-31")
    args = parser.parse_args()
    download(args.output, args.start, args.end)


if __name__ == "__main__":
    main()
