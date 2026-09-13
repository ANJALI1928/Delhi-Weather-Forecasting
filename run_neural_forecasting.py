from __future__ import annotations

import argparse
from pathlib import Path

from weather_pipeline.neural import rebuild_neural_reports, run_neural_benchmark


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GRU and LSTM models on hourly Delhi ERA5 data")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "delhi_era5_hourly_2000_2023.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "neural_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "neural")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild ensemble metrics and plots without retraining")
    args = parser.parse_args()
    result = rebuild_neural_reports(args.output, args.data) if args.report_only else run_neural_benchmark(args.data, args.config, args.output, fast=args.fast)
    print(result["leaderboard"].to_string(index=False))
    print(f"Selected: {result['winners']}")
    print(f"Outputs: {result['output']}")


if __name__ == "__main__":
    main()
