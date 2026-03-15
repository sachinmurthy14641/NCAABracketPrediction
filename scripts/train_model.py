"""Train and evaluate NCAA tournament game prediction models.

Trains a Logistic Regression on the labeled matchup dataset using a
time-series 3-way split, calibrates on the validation season, and
evaluates on the held-out test season.

Split:
  - Train : seasons <= 2021  (original + mirrored rows)
  - Val   : season  == 2022  (both perspectives; used for Platt calibration)
  - Test  : season  == 2023  (both perspectives; final held-out evaluation)

Usage::

    python scripts/train_model.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAINING_DATA_PATH = Path("data/processed/training_data.csv")
MODELS_DIR         = Path("outputs/models")
REPORTS_DIR        = Path("outputs/reports")

FEATURE_COLS = [
    "off_eff_advantage",
    "def_eff_advantage",
    "net_efficiency_edge",
    "tempo_difference",
    "overall_rating_diff",
    "efficiency_differential_diff",
    "seed_diff",
    "a_adj_off_eff",
    "a_adj_def_eff",
    "a_adj_em",
    "b_adj_off_eff",
    "b_adj_def_eff",
    "b_adj_em",
]
LABEL_COL  = "winner"
SEASON_COL = "season"

TRAIN_MAX  = 2022
VAL_SEASON = 2023
TEST_SEASON = 2024


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with both mirrored perspectives each."""
    df = pd.read_csv(TRAINING_DATA_PATH)
    train_df = df[df[SEASON_COL] <= TRAIN_MAX].copy()
    val_df   = df[df[SEASON_COL] == VAL_SEASON].copy()
    test_df  = df[df[SEASON_COL] == TEST_SEASON].copy()
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_logistic_uncalibrated() -> Pipeline:
    """Logistic regression without internal calibration (we calibrate externally)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X: np.ndarray, y: np.ndarray, label: str) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "split":    label,
        "n_games":  len(y) // 2,
        "accuracy": accuracy_score(y, preds),
        "auc":      roc_auc_score(y, probs),
        "brier":    brier_score_loss(y, probs),
        "log_loss": log_loss(y, probs),
    }


def print_metrics(metrics: dict) -> None:
    print(f"\n  Split           : {metrics['split']}")
    print(f"  Unique games    : {metrics['n_games']}  (random baseline = 0.50)")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC         : {metrics['auc']:.4f}")
    print(f"  Brier Score     : {metrics['brier']:.4f}")
    print(f"  Log Loss        : {metrics['log_loss']:.4f}")


def save_calibration_plot(model, X_test: np.ndarray, y_test: np.ndarray) -> Path:
    """Save reliability diagram for the test set."""
    probs = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", label="Logistic Regression (Platt)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration plot — Test season {TEST_SEASON}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "calibration_plot_logistic_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


class PlattModel:
    """Thin wrapper: base pipeline → Platt logistic → calibrated probabilities."""

    def __init__(self, base, platt):
        self.base = base
        self.platt = platt

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)
        return self.platt.predict_proba(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pickle

    print("Loading data splits...")
    train_df, val_df, test_df = load_splits()
    print(f"  Train rows : {len(train_df):,}  (seasons ≤ {TRAIN_MAX})")
    print(f"  Val rows   : {len(val_df):,}  (season {VAL_SEASON})")
    print(f"  Test rows  : {len(test_df):,}  (season {TEST_SEASON})")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[LABEL_COL].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[LABEL_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[LABEL_COL].values

    # Step 1: fit base logistic on train
    print("\nFitting base Logistic Regression on train set...")
    base = make_logistic_uncalibrated()
    base.fit(X_train, y_train)

    # Step 2: Platt calibration using val set
    print("Calibrating with Platt scaling on val set (season 2022)...")
    val_probs_uncal = base.predict_proba(X_val)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(val_probs_uncal, y_val)
    model = PlattModel(base, platt)

    # Step 3: Evaluate
    val_metrics  = evaluate(model, X_val,  y_val,  f"Val  ({VAL_SEASON})")
    test_metrics = evaluate(model, X_test, y_test, f"Test ({TEST_SEASON})")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION — baseline_logistic_v1")
    print("=" * 60)
    print_metrics(val_metrics)
    print_metrics(test_metrics)

    # Step 4: Calibration plot
    plot_path = save_calibration_plot(model, X_test, y_test)
    print(f"\nCalibration plot saved: {plot_path}")

    # Step 5: Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "baseline_logistic_v2.pkl"
    meta_path  = MODELS_DIR / "baseline_logistic_v2_meta.csv"

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLS, "name": "baseline_logistic_v2"}, f)

    meta = pd.DataFrame([{
        "model":       "baseline_logistic_v2",
        "train_max":   TRAIN_MAX,
        "val_season":  VAL_SEASON,
        "test_season": TEST_SEASON,
        **{f"val_{k}":  v for k, v in val_metrics.items()  if k not in ("split", "n_games")},
        **{f"test_{k}": v for k, v in test_metrics.items() if k not in ("split", "n_games")},
        "n_test_games": test_metrics["n_games"],
        "features":    ",".join(FEATURE_COLS),
    }])
    meta.to_csv(meta_path, index=False)

    print(f"  Saved model : {model_path}")
    print(f"  Saved meta  : {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
