"""Inspect features used in training_data.csv and the saved model.

Usage::

    python scripts/inspect_model_features.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# PlattModel must be importable for pickle to deserialize baseline_logistic_v1.pkl
from scripts.train_model import PlattModel  # noqa: F401

TRAINING_DATA_PATH = Path("data/processed/training_data.csv")
MODEL_PATH         = Path("outputs/models/baseline_logistic_v1.pkl")

METADATA_COLS = {"season", "team_a", "team_b", "score_a", "score_b"}
LABEL_COL     = "winner"


def main() -> None:
    df = pd.read_csv(TRAINING_DATA_PATH)

    all_cols      = list(df.columns)
    metadata_cols = [c for c in all_cols if c in METADATA_COLS]
    label_cols    = [c for c in all_cols if c == LABEL_COL]
    feature_cols  = [c for c in all_cols if c not in METADATA_COLS and c != LABEL_COL]

    print("=" * 60)
    print("TRAINING DATA COLUMNS")
    print("=" * 60)
    print(f"\nAll columns ({len(all_cols)}):")
    for c in all_cols:
        tag = "(metadata)" if c in METADATA_COLS else "(label)" if c == LABEL_COL else "(feature)"
        print(f"  {c:<40} {tag}")

    print(f"\nMetadata columns ({len(metadata_cols)}): {metadata_cols}")
    print(f"Label column     : {label_cols}")
    print(f"Feature columns  ({len(feature_cols)}): {feature_cols}")

    seed_features = [c for c in feature_cols if "seed" in c.lower()]
    print(f"\nSeed-related features ({len(seed_features)}): {seed_features}")

    print("\n" + "=" * 60)
    print("SAMPLE ROW (first training game)")
    print("=" * 60)
    row = df.iloc[0]
    print(f"\n  season  : {row['season']}")
    print(f"  team_a  : {row['team_a']}  (seed {row.get('team_a_seed', 'N/A')})")
    print(f"  team_b  : {row['team_b']}  (seed {row.get('team_b_seed', 'N/A')})")
    print(f"  winner  : {int(row['winner'])} ({'team_a' if row['winner'] == 1 else 'team_b'})")
    print("\n  Feature values:")
    for feat in feature_cols:
        print(f"    {feat:<40} {row[feat]:.4f}")

    print("\n" + "=" * 60)
    print("MODEL FEATURE LIST (from saved model)")
    print("=" * 60)
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        model_features = payload["features"]
        print(f"\nModel name : {payload['name']}")
        print(f"Features   ({len(model_features)}):")
        for feat in model_features:
            in_data = "✓" if feat in feature_cols else "✗ MISSING"
            print(f"  {feat:<40} {in_data}")

        extra = [c for c in feature_cols if c not in model_features]
        if extra:
            print(f"\nColumns in training data NOT used by model ({len(extra)}): {extra}")
        else:
            print("\nAll feature columns in training data are used by the model.")
    else:
        print(f"  Model not found at {MODEL_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    main()
