"""Analyze LightGBM probability distribution and calibration on the 2023 test set.

Usage::

    python scripts/analyze_lgbm_calibration.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import PlattModel, FEATURE_COLS, LABEL_COL, SEASON_COL  # noqa: F401

LR_PATH       = Path("outputs/models/baseline_logistic_v1.pkl")
LGBM_PATH     = Path("outputs/models/lgbm_v1.pkl")
TRAINING_DATA = Path("data/processed/training_data.csv")
REPORTS_DIR   = Path("outputs/reports")
TEST_SEASON   = 2023


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_models_and_data():
    with open(LR_PATH, "rb") as f:
        lr_payload = pickle.load(f)
    with open(LGBM_PATH, "rb") as f:
        lgbm_payload = pickle.load(f)

    df      = pd.read_csv(TRAINING_DATA)
    test_df = df[df[SEASON_COL] == TEST_SEASON].copy().reset_index(drop=True)
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[LABEL_COL].values

    lr_probs   = lr_payload["model"].predict_proba(X_test)[:, 1]
    lgbm_probs = lgbm_payload["model"].predict_proba(X_test.values)[:, 1]

    return lr_probs, lgbm_probs, y_test


def calculate_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def calculate_mce(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = probs[mask].mean()
        mce  = max(mce, abs(acc - conf))
    return mce


def confidence_counts(probs: np.ndarray) -> dict:
    return {
        "very_high (>90%)":    (probs > 0.90).sum(),
        "high (80-90%)":       ((probs > 0.80) & (probs <= 0.90)).sum(),
        "medium (60-80%)":     ((probs > 0.60) & (probs <= 0.80)).sum(),
        "moderate (50-60%)":   ((probs > 0.50) & (probs <= 0.60)).sum(),
        "toss_up (40-50%)":    ((probs >= 0.40) & (probs <= 0.50)).sum(),
        "below_40% (<40%)":    (probs < 0.40).sum(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_histogram(lr_probs, lgbm_probs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    bins = np.arange(0, 1.05, 0.05)

    for ax, probs, name, color in [
        (axes[0], lr_probs,   "Logistic Regression", "#3498db"),
        (axes[1], lgbm_probs, "LightGBM",            "#2ecc71"),
    ]:
        ax.hist(probs, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Predicted P(team_a wins)", fontsize=11)
        ax.set_ylabel("Number of games", fontsize=11)
        ax.set_title(f"{name}\nProbability Distribution (test={TEST_SEASON})", fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        pct_extreme = ((probs > 0.80) | (probs < 0.20)).mean() * 100
        ax.text(0.5, 0.93, f"{pct_extreme:.0f}% predictions >80% or <20%",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out = REPORTS_DIR / "lgbm_prob_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Histogram saved → {out}")


def plot_calibration(y_test, lr_probs, lgbm_probs) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

    for name, probs, color, marker in [
        ("LR (Platt)", lr_probs,   "#3498db", "o"),
        ("LightGBM",  lgbm_probs, "#2ecc71", "s"),
    ]:
        pt, pp = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(pp, pt, f"{marker}-", color=color, lw=2, markersize=8, label=name)
        # Annotate sample counts per bin
        bins_q = np.quantile(probs, np.linspace(0, 1, 11))
        for lo, hi, tp, pp_val in zip(bins_q[:-1], bins_q[1:], pt, pp):
            n = ((probs >= lo) & (probs <= hi)).sum()
            ax.annotate(f"n={n}", (pp_val, tp), textcoords="offset points",
                        xytext=(6, 2), fontsize=7, color=color)

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives (actual win rate)", fontsize=12)
    ax.set_title(f"Calibration Curves — Test Season {TEST_SEASON}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = REPORTS_DIR / "lgbm_calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration curve saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lr_probs, lgbm_probs, y_test = load_models_and_data()

    # 1. Confidence breakdown
    section("CONFIDENCE LEVEL BREAKDOWN")
    lr_counts   = confidence_counts(lr_probs)
    lgbm_counts = confidence_counts(lgbm_probs)
    n = len(lgbm_probs)

    print(f"\n  {'Band':<22} {'LR':>8} {'LR%':>6}  {'LGBM':>6} {'LGBM%':>7}")
    print(f"  {'-' * 54}")
    for band in lr_counts:
        lr_n   = lr_counts[band]
        lgbm_n = lgbm_counts[band]
        print(f"  {band:<22} {lr_n:>8} {lr_n/n:>6.1%}  {lgbm_n:>6} {lgbm_n/n:>7.1%}")

    lr_over80   = sum(v for k, v in lr_counts.items() if ">90" in k or "80-90" in k)
    lgbm_over80 = sum(v for k, v in lgbm_counts.items() if ">90" in k or "80-90" in k)
    print(f"\n  LR   predictions >80% confidence : {lr_over80}/{n} ({lr_over80/n:.1%})")
    print(f"  LGBM predictions >80% confidence : {lgbm_over80}/{n} ({lgbm_over80/n:.1%})")
    if lgbm_over80 < lr_over80:
        print("  ✓  LightGBM is LESS overconfident than LR.")
    else:
        print("  ⚠  LightGBM is equally or MORE overconfident than LR.")

    # 2. Calibration metrics
    section("CALIBRATION METRICS")
    lr_ece   = calculate_ece(y_test, lr_probs)
    lgbm_ece = calculate_ece(y_test, lgbm_probs)
    lr_mce   = calculate_mce(y_test, lr_probs)
    lgbm_mce = calculate_mce(y_test, lgbm_probs)
    lr_brier   = brier_score_loss(y_test, lr_probs)
    lgbm_brier = brier_score_loss(y_test, lgbm_probs)

    print(f"\n  {'Metric':<30} {'LR':>10} {'LightGBM':>12}")
    print(f"  {'-' * 55}")
    print(f"  {'ECE (target < 0.05)':<30} {lr_ece:>10.4f} {lgbm_ece:>12.4f}")
    print(f"  {'MCE':<30} {lr_mce:>10.4f} {lgbm_mce:>12.4f}")
    print(f"  {'Brier Score':<30} {lr_brier:>10.4f} {lgbm_brier:>12.4f}")

    # 3. Plots
    section("GENERATING PLOTS")
    plot_histogram(lr_probs, lgbm_probs)
    plot_calibration(y_test, lr_probs, lgbm_probs)

    # 4. Summary verdict
    section("SUMMARY")
    lgbm_tossups = lgbm_counts["toss_up (40-50%)"] + lgbm_counts["moderate (50-60%)"]
    spread = "mostly extreme (overconfident)" if lgbm_over80 / n > 0.80 else "better spread than LR"

    if lgbm_ece < 0.05:
        cal_quality = "Well-calibrated (ECE < 0.05)"
        trading_rec = "Probabilities are reliable — ready for trading"
    elif lgbm_ece < 0.10:
        cal_quality = "Acceptable calibration (ECE 0.05–0.10)"
        trading_rec = "Probabilities are usable but treat with some caution"
    else:
        cal_quality = "Poorly calibrated (ECE > 0.10) — overconfident"
        trading_rec = "Needs recalibration before using for Kalshi trading"

    print(f"""
  LightGBM Calibration Analysis:
  ─────────────────────────────────────────────────
  Probability spread  : {spread}
  Toss-up games       : {lgbm_tossups} / {n} ({lgbm_tossups/n:.1%})
  Calibration quality : {cal_quality}
  ECE                 : {lgbm_ece:.4f}
  MCE                 : {lgbm_mce:.4f}
  Brier Score         : {lgbm_brier:.4f}
  ─────────────────────────────────────────────────
  Recommendation      : {trading_rec}
""")


if __name__ == "__main__":
    main()
