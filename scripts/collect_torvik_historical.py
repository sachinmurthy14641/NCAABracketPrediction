"""Collect Bart Torvik T-Rank efficiency ratings for multiple seasons.

Loops through seasons 2015-2025, saves each year's raw data to:
    data/raw/torvik_{year}.csv

Then combines all years into:
    data/processed/torvik_all_seasons.csv

Usage::

    python scripts/collect_torvik_historical.py
    python scripts/collect_torvik_historical.py --start 2020 --end 2025
    python scripts/collect_torvik_historical.py --years 2023 2024 2025
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.torvik_collector import TorvikCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_START = 2015
DEFAULT_END = 2025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Bart Torvik T-Rank data for multiple seasons."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--years",
        nargs="+",
        type=int,
        metavar="YEAR",
        help="Explicit list of season years to collect (e.g. --years 2023 2024 2025).",
    )
    group.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START,
        help=f"First season year in range (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=DEFAULT_END,
        help=f"Last season year in range, inclusive (default: {DEFAULT_END}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Extra sleep in seconds between seasons to be polite (default: 3.0).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seasons whose raw CSV already exists.",
    )
    return parser.parse_args()


def collect_seasons(years: list[int], delay: float, skip_existing: bool) -> list[pd.DataFrame]:
    """Drive a single browser session across all requested seasons."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    collected: list[pd.DataFrame] = []
    failed: list[int] = []

    with TorvikCollector() as collector:
        for i, year in enumerate(years):
            raw_path = RAW_DATA_DIR / f"torvik_{year}.csv"

            if skip_existing and raw_path.exists():
                logger.info("Skipping %d — %s already exists.", year, raw_path.name)
                collected.append(pd.read_csv(raw_path))
                continue

            try:
                logger.info("Collecting season %d (%d/%d)…", year, i + 1, len(years))
                df = collector.fetch_season(year)
                df.to_csv(raw_path, index=False)
                logger.info("Saved %d rows → %s", len(df), raw_path)
                collected.append(df)
            except Exception as exc:
                logger.error("Failed to collect season %d: %s", year, exc)
                failed.append(year)

            if i < len(years) - 1:
                logger.debug("Sleeping %.1fs between seasons…", delay)
                time.sleep(delay)

    if failed:
        logger.warning("Failed seasons (consider re-running): %s", failed)

    return collected


def combine_and_save(frames: list[pd.DataFrame]) -> None:
    if not frames:
        logger.warning("No data collected — nothing to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "torvik_all_seasons.csv"
    combined.to_csv(out_path, index=False)
    logger.info(
        "Combined %d rows across %d seasons → %s",
        len(combined),
        combined["season"].nunique(),
        out_path,
    )


def main() -> None:
    args = parse_args()

    if args.years:
        years = sorted(set(args.years))
    else:
        if args.start > args.end:
            print(f"Error: --start ({args.start}) must be <= --end ({args.end}).")
            sys.exit(1)
        years = list(range(args.start, args.end + 1))

    logger.info("Seasons to collect: %s", years)

    frames = collect_seasons(years, delay=args.delay, skip_existing=args.skip_existing)
    combine_and_save(frames)

    logger.info("Done.")


if __name__ == "__main__":
    main()
