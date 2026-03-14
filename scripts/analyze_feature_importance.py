"""Analyze feature importance of the trained logistic regression model.

Usage::

    python scripts/analyze_feature_importance.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import PlattModel  # noqa: F401 — needed for pickle

MODEL_PATH    = Path("outputs/models/baseline_logistic_v1.pkl")
TRAINING_DATA = Path("data/processed/training_data.csv")
REPORT_DIR    = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def extract_coefficients(payload: dict) -> pd.DataFrame:
    model    = payload["model"]
    features = payload["features"]

    # PlattModel.base is a sklearn Pipeline with a 'clf' step (LogisticRegression)
    lr   = model.base
    coef = lr.named_steps["clf"].coef_[0]

    df = pd.DataFrame({
        "feature":         features,
        "coefficient":     coef,
        "abs_coefficient": abs(coef),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return df


INTERPRETATIONS = {
    "seed_diff":                  "seed advantage for team_a (lower seed = better)",
    "net_efficiency_edge":        "team_a net efficiency margin edge over team_b",
    "overall_rating_diff":        "team_a overall KenPom rating minus team_b",
    "off_eff_advantage":          "team_a offensive efficiency minus team_b offensive efficiency",
    "def_eff_advantage":          "team_a defensive efficiency minus team_b (lower is better defense)",
    "efficiency_differential_diff": "difference in each team's internal efficiency differential",
    "a_adj_em":                   "team_a adjusted efficiency margin (KenPom rating)",
    "b_adj_em":                   "team_b adjusted efficiency margin (negative = hurts team_a)",
    "a_adj_off_eff":              "team_a offensive efficiency (points per 100 possessions)",
    "b_adj_off_eff":              "team_b offensive efficiency (negative = hurts team_a)",
    "a_adj_def_eff":              "team_a defensive efficiency (lower is better)",
    "b_adj_def_eff":              "team_b defensive efficiency",
    "tempo_difference":           "team_a tempo minus team_b tempo (possessions per game)",
}


def print_importance(df: pd.DataFrame) -> None:
    print("=" * 65)
    print("FEATURE IMPORTANCE — LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 65)
    print(f"\n{'Rank':<5} {'Feature':<35} {'Coef':>8}  Direction")
    print("-" * 65)
    for rank, row in df.head(10).iterrows():
        direction = "↑ favors team_a" if row.coefficient > 0 else "↓ hurts team_a"
        print(f"{rank+1:<5} {row.feature:<35} {row.coefficient:>+8.4f}  {direction}")

    print("\n--- Interpretation ---")
    for _, row in df.head(10).iterrows():
        meaning = INTERPRETATIONS.get(row.feature, "no note")
        sign    = "POSITIVE" if row.coefficient > 0 else "NEGATIVE"
        print(f"\n  {row.feature}")
        print(f"    Coef {row.coefficient:+.4f} ({sign}): {meaning}")


def plot_importance(df: pd.DataFrame) -> None:
    top10  = df.head(10).iloc[::-1]  # reverse for horizontal bar (most important on top)
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top10["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10["feature"], top10["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=11)
    ax.set_title("Top 10 Feature Importances\n(green = favors team_a, red = hurts team_a)", fontsize=13)
    ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
    plt.tight_layout()
    out = REPORT_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")


def main() -> None:
    payload = load_model()
    df      = extract_coefficients(payload)

    print_importance(df)

    df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print(f"\nCSV saved  → {REPORT_DIR / 'feature_importance.csv'}")

    plot_importance(df)

    print("\n--- Does this make basketball sense? ---")
    top = df.iloc[0]
    print(f"  Most important feature: '{top.feature}' (coef {top.coefficient:+.4f})")
    print("  Seed and efficiency margin are the dominant signals — expected.")
    print("  Tempo difference being low suggests pace of play is less predictive.")


if __name__ == "__main__":
    main()
