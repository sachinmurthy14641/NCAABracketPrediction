"""Train LR and LightGBM, compare performance, and analyze ensemble value.

Uses the same 3-way time split as train_model.py:
  Train  : seasons <= 2021
  Val    : season  == 2022  (Platt calibration for LR; early stopping for LightGBM)
  Test   : season  == 2023  (held-out evaluation)

Usage::

    python scripts/compare_models.py
"""

import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import PlattModel, FEATURE_COLS, LABEL_COL, SEASON_COL  # noqa: F401

TRAINING_DATA  = Path("data/processed/training_data.csv")
LR_MODEL_PATH  = Path("outputs/models/baseline_logistic_v1.pkl")
LGBM_PATH      = Path("outputs/models/lgbm_v1.pkl")
ENSEMBLE_PATH  = Path("outputs/models/ensemble_v1.pkl")
REPORTS_DIR    = Path("outputs/reports")

TRAIN_MAX   = 2021
VAL_SEASON  = 2022
TEST_SEASON = 2023


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_splits():
    df       = pd.read_csv(TRAINING_DATA)
    train_df = df[df[SEASON_COL] <= TRAIN_MAX]
    val_df   = df[df[SEASON_COL] == VAL_SEASON]
    test_df  = df[df[SEASON_COL] == TEST_SEASON]
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_lr() -> PlattModel:
    with open(LR_MODEL_PATH, "rb") as f:
        return pickle.load(f)["model"]


def train_lgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(name: str, y_true, probs) -> dict:
    preds = (probs >= 0.5).astype(int)
    return {
        "model":    name,
        "accuracy": accuracy_score(y_true, preds),
        "brier":    brier_score_loss(y_true, probs),
        "log_loss": log_loss(y_true, probs),
        "auc":      roc_auc_score(y_true, probs),
    }


def print_metrics_table(results: list[dict]) -> None:
    print(f"\n  {'Model':<20} {'Accuracy':>9} {'Brier':>8} {'LogLoss':>9} {'AUC':>8}")
    print(f"  {'-' * 58}")
    for r in results:
        print(f"  {r['model']:<20} {r['accuracy']:>9.4f} {r['brier']:>8.4f} "
              f"{r['log_loss']:>9.4f} {r['auc']:>8.4f}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_correlation(lr_probs, lgbm_probs) -> float:
    corr, pval = pearsonr(lr_probs, lgbm_probs)
    print(f"\n  Prediction correlation (Pearson r) : {corr:.4f}  (p={pval:.2e})")
    if corr > 0.95:
        print("  → Models very similar (r > 0.95) — ensemble unlikely to help much.")
    elif corr > 0.80:
        print("  → Moderate similarity (0.80 < r < 0.95) — ensemble may help at the margins.")
    else:
        print("  → Models diverse (r < 0.80) — ensemble likely helps.")
    return corr


def analyze_disagreements(test_df, lr_probs, lgbm_probs, y_test) -> None:
    diff = np.abs(lr_probs - lgbm_probs)
    threshold = 0.20
    mask = diff > threshold
    n = mask.sum()
    print(f"\n  Games with |LR - LightGBM| > {threshold}: {n}")
    if n == 0:
        print("  No major disagreements found.")
        return

    ens = 0.5 * lr_probs + 0.5 * lgbm_probs
    lr_correct   = ((lr_probs   >= 0.5) == y_test)
    lgbm_correct = ((lgbm_probs >= 0.5) == y_test)
    ens_correct  = ((ens        >= 0.5) == y_test)

    sub = test_df[mask].copy().reset_index(drop=True)
    print(f"\n  {'#':<4} {'Matchup':<40} {'LR':>7} {'LGBM':>7} {'Ens':>7} {'Actual':>7} {'LR✓':>5} {'GB✓':>5} {'En✓':>5}")
    print(f"  {'-' * 90}")
    for i, (_, row) in enumerate(sub.iterrows()):
        j      = np.where(mask)[0][i]
        matchup = f"{row.get('team_a','?')[:18]} vs {row.get('team_b','?')[:18]}"
        actual  = int(y_test[j])
        print(f"  {i+1:<4} {matchup:<40} {lr_probs[j]:>7.3f} {lgbm_probs[j]:>7.3f} "
              f"{ens[j]:>7.3f} {actual:>7}  "
              f"{'✓' if lr_correct[j] else '✗':>4}  "
              f"{'✓' if lgbm_correct[j] else '✗':>4}  "
              f"{'✓' if ens_correct[j] else '✗':>4}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_agreement(lr_probs, lgbm_probs, y_test) -> None:
    ens_correct = ((0.5 * lr_probs + 0.5 * lgbm_probs >= 0.5) == y_test)
    colors = ["#2ecc71" if c else "#e74c3c" for c in ens_correct]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(lr_probs, lgbm_probs, c=colors, alpha=0.7, edgecolors="none", s=60)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect agreement")
    ax.set_xlabel("Logistic Regression P(team_a wins)", fontsize=12)
    ax.set_ylabel("LightGBM P(team_a wins)", fontsize=12)
    ax.set_title("LR vs LightGBM Predictions\n(green = ensemble correct, red = both wrong)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out = REPORTS_DIR / "model_agreement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Scatter plot saved → {out}")


def plot_calibration(y_test, lr_probs, lgbm_probs, ens_probs) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for name, probs in [("LR (Platt)", lr_probs), ("LightGBM", lgbm_probs), ("Ensemble", ens_probs)]:
        pt, pp = calibration_curve(y_test, probs, n_bins=8, strategy="quantile")
        ax.plot(pp, pt, "o-", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration comparison — Test season {TEST_SEASON}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = REPORTS_DIR / "calibration_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration plot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data splits...")
    train_df, val_df, test_df = load_splits()
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[LABEL_COL].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[LABEL_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[LABEL_COL].values
    print(f"  Train {len(train_df):,} / Val {len(val_df):,} / Test {len(test_df):,} rows")

    # --- LR (load saved) ---
    print("\nLoading saved Logistic Regression...")
    lr_model  = load_lr()
    lr_probs  = lr_model.predict_proba(pd.DataFrame(X_test, columns=FEATURE_COLS))[:, 1]

    # --- LightGBM ---
    print("Training LightGBM...")
    lgbm_model  = train_lgbm(X_train, y_train, X_val, y_val)
    lgbm_probs  = lgbm_model.predict_proba(X_test)[:, 1]
    print(f"  Best iteration: {lgbm_model.best_iteration_}")

    # --- Ensemble ---
    ens_probs = 0.5 * lr_probs + 0.5 * lgbm_probs

    # --- Metrics ---
    print("\n" + "=" * 65)
    print("  TEST SET PERFORMANCE (season=2023)")
    print("=" * 65)
    results = [
        metrics("Logistic Regression", y_test, lr_probs),
        metrics("LightGBM",            y_test, lgbm_probs),
        metrics("Ensemble (50/50)",    y_test, ens_probs),
    ]
    print_metrics_table(results)

    # --- Correlation ---
    print("\n" + "=" * 65)
    print("  MODEL AGREEMENT ANALYSIS")
    print("=" * 65)
    corr = analyze_correlation(lr_probs, lgbm_probs)

    # --- Disagreements ---
    analyze_disagreements(test_df, lr_probs, lgbm_probs, y_test)

    # --- Plots ---
    print("\n" + "=" * 65)
    print("  SAVING PLOTS")
    print("=" * 65)
    plot_agreement(lr_probs, lgbm_probs, y_test)
    plot_calibration(y_test, lr_probs, lgbm_probs, ens_probs)

    # --- Recommendation ---
    print("\n" + "=" * 65)
    print("  RECOMMENDATION")
    print("=" * 65)
    lr_brier   = results[0]["brier"]
    lgbm_brier = results[1]["brier"]
    ens_brier  = results[2]["brier"]
    best_brier = min(lr_brier, lgbm_brier, ens_brier)

    if best_brier == ens_brier and ens_brier < min(lr_brier, lgbm_brier) - 0.002:
        rec = "Use ensemble (50/50) — meaningfully better Brier score."
    elif best_brier == lgbm_brier and lgbm_brier < lr_brier - 0.002:
        rec = "Use LightGBM only — clearly outperforms LR; ensemble adds noise."
    elif best_brier == lr_brier and lr_brier < lgbm_brier - 0.002:
        rec = "Use Logistic Regression only — LightGBM adds no value on this dataset size."
    else:
        rec = "Use either — performance is similar. LR is more interpretable; ensemble for robustness."
    print(f"\n  {rec}")

    # --- Save LightGBM ---
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    with open(LGBM_PATH, "wb") as f:
        pickle.dump({"model": lgbm_model, "features": FEATURE_COLS, "name": "lgbm_v1"}, f)
    print(f"\n  LightGBM saved → {LGBM_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
