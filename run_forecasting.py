from __future__ import annotations

import argparse
from pathlib import Path

from weather_pipeline import run_pipeline


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe Delhi temperature forecasting benchmark")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "delhi_weather_2019_2023.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "latest")
    parser.add_argument("--fast", action="store_true", help="Run a small dependency-free smoke benchmark")
    args = parser.parse_args()
    result = run_pipeline(args.data, args.config, args.output, fast=args.fast)
    print(result["leaderboard"].to_string(index=False))
    print(f"\nOutputs: {result['output']}")


if __name__ == "__main__":
    main()
