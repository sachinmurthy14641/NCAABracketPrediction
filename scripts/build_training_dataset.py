"""Build the labeled matchup training dataset for model training.

Joins tournament game results with KenPom pre-tournament efficiency stats
to produce one row per game with both teams' features and the outcome.

Output: data/processed/training_data.csv

Usage::

    python scripts/build_training_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOURNAMENT_RESULTS_PATH = Path("data/historical/tournament_results_1997_2024.csv")
KENPOM_PATH = Path("data/processed/kenpom_pretourney_1997_2025.csv")
OUTPUT_PATH = Path("data/processed/training_data.csv")

KENPOM_FEATURES = ["adj_off_eff", "adj_def_eff", "adj_tempo", "adj_em", "efficiency_differential"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(TOURNAMENT_RESULTS_PATH)
    kenpom = pd.read_csv(KENPOM_PATH)
    return results, kenpom


def join_team_stats(results: pd.DataFrame, kenpom: pd.DataFrame) -> pd.DataFrame:
    """Join KenPom stats for team_a and team_b onto each game row."""
    # Prefix columns for each side
    kenpom_a = kenpom[["team", "season"] + KENPOM_FEATURES].rename(
        columns={c: f"a_{c}" for c in KENPOM_FEATURES} | {"team": "team_a"}
    )
    kenpom_b = kenpom[["team", "season"] + KENPOM_FEATURES].rename(
        columns={c: f"b_{c}" for c in KENPOM_FEATURES} | {"team": "team_b"}
    )

    df = results.merge(kenpom_a, on=["season", "team_a"], how="inner")
    df = df.merge(kenpom_b, on=["season", "team_b"], how="inner")
    return df


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add differential features from team_a's perspective."""
    df = df.copy()
    df["off_eff_advantage"]           = df["a_adj_off_eff"] - df["b_adj_def_eff"]
    df["def_eff_advantage"]           = df["b_adj_off_eff"] - df["a_adj_def_eff"]
    df["net_efficiency_edge"]         = df["off_eff_advantage"] - df["def_eff_advantage"]
    df["tempo_difference"]            = df["a_adj_tempo"] - df["b_adj_tempo"]
    df["overall_rating_diff"]         = df["a_adj_em"] - df["b_adj_em"]
    df["efficiency_differential_diff"]= df["a_efficiency_differential"] - df["b_efficiency_differential"]
    df["seed_diff"]                   = df["team_a_seed"] - df["team_b_seed"]
    return df


def mirror_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add mirrored rows (team_b perspective, winner=0) to balance the dataset.

    Without this every row has winner=1. Mirroring gives the model both
    perspectives and doubles training data.
    """
    flipped = df.copy()

    # Swap team identifiers and seeds
    flipped["team_a"], flipped["team_b"] = df["team_b"].copy(), df["team_a"].copy()
    flipped["team_a_seed"], flipped["team_b_seed"] = df["team_b_seed"].copy(), df["team_a_seed"].copy()
    flipped["score_a"], flipped["score_b"] = df["score_b"].copy(), df["score_a"].copy()
    flipped["winner"] = 0

    # Swap prefixed KenPom stats
    for feat in KENPOM_FEATURES:
        flipped[f"a_{feat}"], flipped[f"b_{feat}"] = df[f"b_{feat}"].copy(), df[f"a_{feat}"].copy()

    # Negate differential features
    for col in ["off_eff_advantage", "def_eff_advantage", "net_efficiency_edge",
                "tempo_difference", "overall_rating_diff", "efficiency_differential_diff", "seed_diff"]:
        flipped[col] = -df[col]

    return pd.concat([df, flipped], ignore_index=True)


def main() -> None:
    print("Loading data...")
    results, kenpom = load_data()
    print(f"  Tournament games : {len(results):,}")
    print(f"  KenPom records   : {len(kenpom):,}")

    print("Joining KenPom stats...")
    joined = join_team_stats(results, kenpom)
    dropped = len(results) - len(joined)
    if dropped:
        print(f"  WARNING: {dropped} games dropped (no KenPom match for one or both teams).")
    print(f"  Games after join : {len(joined):,}")

    print("Adding matchup features...")
    joined = add_matchup_features(joined)

    print("Mirroring rows for balanced dataset...")
    training = mirror_rows(joined)
    print(f"  Total rows (after mirror): {len(training):,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("TRAINING DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows      : {len(training):,}  ({len(training)//2:,} games × 2)")
    print(f"Seasons covered : {training['season'].min()}–{training['season'].max()}")
    print(f"Features        : {[c for c in training.columns if c not in ['team_a','team_b','season','winner','score_a','score_b']]}")
    print(f"Label balance   : {training['winner'].value_counts().to_dict()}")
    print(f"Output          : {OUTPUT_PATH}")

    print("\nMissing values per column:")
    missing = training.isnull().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "  None")

    print("\nSample rows:")
    cols = ["season", "team_a", "team_b", "team_a_seed", "team_b_seed",
            "net_efficiency_edge", "overall_rating_diff", "seed_diff", "winner"]
    print(training[cols].head(6).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
