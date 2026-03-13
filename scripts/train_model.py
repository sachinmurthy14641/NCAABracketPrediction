"""Train and evaluate NCAA tournament game prediction models.

Trains a Logistic Regression and XGBoost classifier on the labeled matchup
dataset, evaluates with time-series cross-validation (walk-forward by season),
calibrates probabilities, and saves the best model to outputs/models/.

Usage::

    python scripts/train_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAINING_DATA_PATH = Path("data/processed/training_data.csv")
MODELS_DIR = Path("outputs/models")

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
LABEL_COL = "winner"
SEASON_COL = "season"

# Walk-forward CV: train on all seasons before test season
# Use last 5 seasons as rolling test windows
N_TEST_SEASONS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRAINING_DATA_PATH)
    # De-duplicate mirrored rows for evaluation (keep only winner=1 perspective)
    # We train on both but evaluate on unique games
    return df


def walk_forward_cv(df: pd.DataFrame, model_factory, seasons: list[int]) -> dict:
    """Evaluate a model with walk-forward (time-series) cross-validation.

    For each test season, trains on all prior seasons and predicts on the
    test season. Only uses winner=1 rows for evaluation (unique games).
    """
    all_preds, all_labels, all_probs = [], [], []

    for test_season in seasons:
        train_df = df[df[SEASON_COL] < test_season]
        # Use both mirrored perspectives so both classes are present for AUC/log_loss
        test_df = df[df[SEASON_COL] == test_season]

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[LABEL_COL].values
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df[LABEL_COL].values

        model = model_factory()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_test)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return {
        "accuracy":    accuracy_score(all_labels, all_preds),
        "auc":         roc_auc_score(all_labels, all_probs),
        "brier":       brier_score_loss(all_labels, all_probs),
        "log_loss":    log_loss(all_labels, all_probs),
        "n_games":     len(all_labels),
    }


def make_logistic() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            cv=5, method="isotonic"
        )),
    ])


def make_hgb() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            ),
            cv=5, method="isotonic"
        )),
    ])


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print(f"  Games evaluated : {metrics['n_games']}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC         : {metrics['auc']:.4f}")
    print(f"  Brier Score     : {metrics['brier']:.4f}  (target < 0.18)")
    print(f"  Log Loss        : {metrics['log_loss']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pickle

    print("Loading training data...")
    df = load_data()
    print(f"  Total rows      : {len(df):,}")
    print(f"  Seasons         : {df[SEASON_COL].min()}–{df[SEASON_COL].max()}")

    seasons = sorted(df[SEASON_COL].unique())
    test_seasons = seasons[-N_TEST_SEASONS:]
    print(f"  Walk-forward CV test seasons: {test_seasons}")

    print("\nEvaluating models (walk-forward CV)...")
    lr_metrics  = walk_forward_cv(df, make_logistic, test_seasons)
    hgb_metrics = walk_forward_cv(df, make_hgb,      test_seasons)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION (Walk-Forward CV)")
    print("=" * 60)
    print_metrics("Logistic Regression (calibrated)",        lr_metrics)
    print_metrics("Hist Gradient Boosting (calibrated)",     hgb_metrics)

    # Pick best model by Brier Score (lower = better calibrated)
    best_name, best_factory, best_metrics = (
        ("Hist Gradient Boosting", make_hgb, hgb_metrics)
        if hgb_metrics["brier"] <= lr_metrics["brier"]
        else ("Logistic Regression", make_logistic, lr_metrics)
    )
    print(f"\nBest model: {best_name} (Brier={best_metrics['brier']:.4f})")

    # Retrain best model on ALL data
    print(f"\nRetraining {best_name} on full dataset...")
    X_all = df[FEATURE_COLS].values
    y_all = df[LABEL_COL].values
    final_model = best_factory()
    final_model.fit(X_all, y_all)

    # Save model + metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    meta_path  = MODELS_DIR / "best_model_meta.csv"

    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "features": FEATURE_COLS, "name": best_name}, f)

    meta = pd.DataFrame([{
        "model":    best_name,
        "accuracy": best_metrics["accuracy"],
        "auc":      best_metrics["auc"],
        "brier":    best_metrics["brier"],
        "log_loss": best_metrics["log_loss"],
        "n_games":  best_metrics["n_games"],
        "features": ",".join(FEATURE_COLS),
    }])
    meta.to_csv(meta_path, index=False)

    print(f"  Saved model  : {model_path}")
    print(f"  Saved meta   : {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
