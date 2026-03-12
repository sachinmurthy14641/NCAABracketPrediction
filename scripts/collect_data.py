"""CLI script to process manually downloaded KenPom data into the standard format.

TODO: Replace with automated scraping via KenPomCollector once the 403 issue is resolved.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.preprocessors import add_derived_features, normalize_team_names

COLUMN_MAP = {
    "TeamName": "team",
    "AdjOE": "adj_off_eff",
    "AdjDE": "adj_def_eff",
    "AdjTempo": "adj_tempo",
    "AdjEM": "adj_em",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a manually downloaded KenPom CSV into the standard format."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season year to process (default: 2026).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path = RAW_DATA_DIR / f"kenpom_{args.season}.csv"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        sys.exit(1)

    df = pd.read_csv(raw_path)

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        print(f"Error: expected columns not found in CSV: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df = normalize_team_names(df)
    df = add_derived_features(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / f"kenpom_{args.season}_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
