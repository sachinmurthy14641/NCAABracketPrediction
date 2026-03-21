"""
Calibrate the NCAA tournament prediction model using held-out tournament data.

The base LightGBM model is accurate in direction (~72% accuracy) but wildly
overconfident: it outputs 99%+ even for 1v9 matchups that historically resolve
at ~80%.  This is a known issue with tree-based models on small datasets.

Fix: fit an isotonic regression calibrator on held-out tournament seasons
(2023-2024) and wrap the model.  The calibrated model maps the raw 99% scores
to realistic tournament probabilities.

Usage:
    python scripts/calibrate_model.py                  # calibrate + save
    python scripts/calibrate_model.py --validate       # also run validation
    python scripts/calibrate_model.py --plot           # save calibration plot

Output:
    outputs/models/lightgbm_calibrated_2026.pkl
    outputs/reports/calibration_report_2026.txt
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_IN  = ROOT / "outputs" / "models" / "lightgbm_final_2026.pkl"
MODEL_OUT = ROOT / "outputs" / "models" / "lightgbm_calibrated_2026.pkl"
REPORT    = ROOT / "outputs" / "reports" / "calibration_report_2026.txt"
TRAIN_CSV = ROOT / "data" / "processed" / "training_data.csv"

# Seasons reserved purely for calibration (never used to train the model)
CAL_SEASONS  = [2023, 2024]
TEST_SEASONS = [2025]          # hold-out to check calibrated model hasn't overfit

FEATURE_COLS = [
    "off_eff_advantage", "def_eff_advantage", "net_efficiency_edge",
    "tempo_difference",  "overall_rating_diff", "efficiency_differential_diff",
    "seed_diff",
    "a_adj_off_eff", "a_adj_def_eff", "a_adj_em",
    "b_adj_off_eff", "b_adj_def_eff", "b_adj_em",
]


# ─── Data prep ───────────────────────────────────────────────────────────────

def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, cal_df, test_df) of UNIQUE games, one row per game.

    The raw training data has each game listed twice (both team orderings).
    We keep only rows where team_a is already the stronger team (a_adj_em >= b_adj_em),
    giving us one canonical row per game with the correct model input ordering.
    """
    df = pd.read_csv(TRAIN_CSV)

    # Deduplicate: keep only the row where stronger team is already team_a.
    # This halves the dataset to unique games and ensures correct feature orientation.
    unique = df[df["a_adj_em"] >= df["b_adj_em"]].copy()

    print(f"  Deduped {len(df)} rows → {len(unique)} unique games")

    train = unique[~unique["season"].isin(CAL_SEASONS + TEST_SEASONS)]
    cal   = unique[unique["season"].isin(CAL_SEASONS)]
    test  = unique[unique["season"].isin(TEST_SEASONS)]
    return train, cal, test


def get_X_y(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["winner"].values
    return X, y


# ─── Calibration ─────────────────────────────────────────────────────────────

class _IsotonicWrapper:
    """Wraps base model + fitted isotonic regression into a predict_proba interface."""
    def __init__(self, lgbm_model, iso):
        self._model = lgbm_model
        self._iso   = iso
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._model.predict_proba(X)[:, 1]
        p1 = self._iso.predict(raw)
        return np.column_stack([1 - p1, p1])


class _SklearnWrapper:
    """Minimal sklearn-compatible wrapper so CalibratedClassifierCV can wrap
    our already-fitted LightGBM model without refitting it."""

    def __init__(self, lgbm_model, feature_names):
        self._model        = lgbm_model
        self.feature_names = feature_names
        self.classes_      = np.array([0, 1])

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1 = self._model.predict_proba(X)[:, 1]
        return np.column_stack([1 - p1, p1])

    def fit(self, X, y):
        return self   # already fitted


def calibrate(base_model, X_cal, y_cal) -> CalibratedClassifierCV:
    """Fit an isotonic calibrator on top of the frozen base model."""
    wrapper = _SklearnWrapper(base_model, FEATURE_COLS)
    # cv=None means use the estimator as-is (prefit); sklearn >=1.2 dropped "prefit" string
    try:
        cal = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal.fit(X_cal, y_cal)
    except Exception:
        # Fallback for newer sklearn that removed "prefit" string
        from sklearn.isotonic import IsotonicRegression
        # Get raw probabilities and fit isotonic regression directly
        raw_wrap = _SklearnWrapper(base_model, FEATURE_COLS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_probs = raw_wrap.predict_proba(X_cal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs, y_cal)
        # Wrap in a simple callable object
        cal = _IsotonicWrapper(base_model, iso)
    return cal


# ─── Metrics ─────────────────────────────────────────────────────────────────

def evaluate(model, X, y, label: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "label":    label,
        "n":        len(y),
        "accuracy": float((preds == y).mean()),
        "brier":    float(brier_score_loss(y, probs)),
        "logloss":  float(log_loss(y, probs)),
        "auc":      float(roc_auc_score(y, probs)),
        "probs":    probs,
        "y":        y,
    }


def prob_histogram(probs: np.ndarray, label: str) -> str:
    """Show distribution of predicted probabilities."""
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    labels = ["0-10%","10-20%","20-30%","30-40%","40-50%",
              "50-60%","60-70%","70-80%","80-90%","90-100%"]
    counts, _ = np.histogram(probs, bins=bins)
    lines = [f"  {label} probability distribution:"]
    for lbl, cnt in zip(labels, counts):
        bar = "█" * int(cnt / max(counts) * 30)
        lines.append(f"    {lbl:>10}  {bar:<30}  {cnt:>4}")
    return "\n".join(lines)


# ─── Report ──────────────────────────────────────────────────────────────────

def build_report(raw_pkg, cal_model, train_df, cal_df, test_df) -> str:
    raw_model = raw_pkg["model"]

    # Evaluate on calibration set
    X_cal, y_cal   = get_X_y(cal_df)
    X_test, y_test = get_X_y(test_df)

    # Raw wrapper for fair comparison
    raw_wrapper = _SklearnWrapper(raw_model, FEATURE_COLS)

    r_cal  = evaluate(raw_wrapper, X_cal,  y_cal,  "Raw  (cal set 2023-24)")
    c_cal  = evaluate(cal_model,   X_cal,  y_cal,  "Cal  (cal set 2023-24)")
    r_test = evaluate(raw_wrapper, X_test, y_test, "Raw  (test set 2025)")
    c_test = evaluate(cal_model,   X_test, y_test, "Cal  (test set 2025)")

    lines = [
        "=" * 68,
        "  MODEL CALIBRATION REPORT — 2026 NCAA Tournament",
        f"  Base model:  {MODEL_IN.name}",
        f"  Cal seasons: {CAL_SEASONS}   Test season: {TEST_SEASONS}",
        "=" * 68,
        "",
        "  ACCURACY (direction only — calibration does not change picks)",
        f"  {'Metric':<30}  {'Raw':>8}  {'Calibrated':>11}  {'Δ':>6}",
        "  " + "-" * 60,
    ]

    for r, c in [(r_cal, c_cal), (r_test, c_test)]:
        tag = r["label"].split("(")[1].rstrip(")")
        lines.append(
            f"  Accuracy  ({tag}){'':<8}  {r['accuracy']:>8.1%}  {c['accuracy']:>11.1%}"
            f"  {c['accuracy']-r['accuracy']:>+6.1%}"
        )

    lines += [
        "",
        "  CALIBRATION QUALITY (lower = better)",
        f"  {'Metric':<30}  {'Raw':>8}  {'Calibrated':>11}  {'Δ':>6}",
        "  " + "-" * 60,
    ]
    for r, c in [(r_cal, c_cal), (r_test, c_test)]:
        tag = r["label"].split("(")[1].rstrip(")")
        lines.append(
            f"  Brier score ({tag}){'':<5}  {r['brier']:>8.4f}  {c['brier']:>11.4f}"
            f"  {c['brier']-r['brier']:>+6.4f}"
        )
        lines.append(
            f"  Log loss    ({tag}){'':<5}  {r['logloss']:>8.4f}  {c['logloss']:>11.4f}"
            f"  {c['logloss']-r['logloss']:>+6.4f}"
        )

    lines += ["", prob_histogram(r_cal["probs"],  "RAW (2023-24)")]
    lines += ["", prob_histogram(c_cal["probs"],  "CAL (2023-24)")]
    lines += ["", prob_histogram(c_test["probs"], "CAL (2025)")]

    # Spot-check: what does calibrated model say for sample matchups?
    lines += [
        "",
        "  SPOT CHECK — Sample Round 2 matchups (calibrated probabilities)",
        "  (compare these to the 99%+ values from the raw model)",
        f"  {'Matchup':<40}  {'Raw':>6}  {'Cal':>6}",
        "  " + "-" * 56,
    ]

    kp_df = pd.read_csv(ROOT / "data" / "processed" / "kenpom_2026_clean.csv")

    def get_kp(t):
        m = kp_df[kp_df["team"].str.lower().str.contains(t.lower(), regex=False)]
        return m.iloc[0] if len(m) else None

    def build_features(ta, sa, tb, sb):
        ra, rb = get_kp(ta), get_kp(tb)
        if ra is None or rb is None:
            return None
        # Enforce strong-team-first convention
        if float(rb["adj_em"]) > float(ra["adj_em"]):
            ra, rb = rb, ra
            sa, sb = sb, sa
        off_adv = float(ra["adj_off_eff"]) - float(rb["adj_def_eff"])
        def_adv = float(rb["adj_off_eff"]) - float(ra["adj_def_eff"])
        row = {
            "off_eff_advantage": off_adv,
            "def_eff_advantage": def_adv,
            "net_efficiency_edge": off_adv - def_adv,
            "tempo_difference": float(ra["adj_tempo"]) - float(rb["adj_tempo"]),
            "overall_rating_diff": float(ra["adj_em"]) - float(rb["adj_em"]),
            "efficiency_differential_diff": float(ra["efficiency_differential"]) - float(rb["efficiency_differential"]),
            "seed_diff": sa - sb,
            "a_adj_off_eff": float(ra["adj_off_eff"]),
            "a_adj_def_eff": float(ra["adj_def_eff"]),
            "a_adj_em":      float(ra["adj_em"]),
            "b_adj_off_eff": float(rb["adj_off_eff"]),
            "b_adj_def_eff": float(rb["adj_def_eff"]),
            "b_adj_em":      float(rb["adj_em"]),
        }
        return np.array([[row[f] for f in FEATURE_COLS]])

    spot_checks = [
        ("Duke(1)",      "Duke",      1,  "TCU(9)",       "TCU",       9),
        ("Florida(1)",   "Florida",   1,  "Iowa(9)",      "Iowa",      9),
        ("Iowa State(2)","Iowa State",2,  "Kentucky(7)",  "Kentucky",  7),
        ("Kansas(4)",    "Kansas",    4,  "St. John's(5)","St. John's",5),
        ("Arizona(1)",   "Arizona",   1,  "Utah State(9)","Utah State",9),
        ("Gonzaga(3)",   "Gonzaga",   3,  "San Diego St.","San Diego", 6),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label_a, ta, sa, label_b, tb, sb in spot_checks:
            X = build_features(ta, sa, tb, sb)
            if X is None:
                lines.append(f"  {label_a+' vs '+label_b:<40}  SKIP (team not found)")
                continue
            p_raw = float(raw_wrapper.predict_proba(X)[:, 1][0])
            p_cal = float(cal_model.predict_proba(X)[:, 1][0])
            # p_raw and p_cal are for the stronger team (after internal swap)
            # If ta was swapped internally, output is P(stronger wins) = P(lower seed)
            lines.append(
                f"  {label_a+' vs '+label_b:<40}  {p_raw:>5.1%}  {p_cal:>5.1%}"
            )

    lines += [
        "",
        "=" * 68,
        "  VERDICT",
        "=" * 68,
    ]
    brier_improvement = r_cal["brier"] - c_cal["brier"]
    if brier_improvement > 0.02:
        lines.append(f"  ✓ SIGNIFICANT improvement: Brier score -{brier_improvement:.4f}")
        lines.append("  ✓ Use outputs/models/lightgbm_calibrated_2026.pkl going forward")
    elif brier_improvement > 0:
        lines.append(f"  ~ Modest improvement: Brier score -{brier_improvement:.4f}")
    else:
        lines.append(f"  ✗ Calibration did not help: Brier score {brier_improvement:+.4f}")

    if c_test["accuracy"] >= r_test["accuracy"] - 0.01:
        lines.append("  ✓ No accuracy regression on held-out 2025 test season")
    else:
        lines.append(f"  ! Accuracy dropped on 2025: {r_test['accuracy']:.1%} → {c_test['accuracy']:.1%}")

    lines.append("")
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Print detailed metrics to stdout")
    parser.add_argument("--plot", action="store_true",
                        help="Save calibration curve plot (requires matplotlib)")
    args = parser.parse_args()

    print("Loading training data...")
    train_df, cal_df, test_df = load_split()
    print(f"  Train: {len(train_df)} rows  Cal: {len(cal_df)} rows  Test: {len(test_df)} rows")

    print(f"Loading base model from {MODEL_IN.name}...")
    with open(MODEL_IN, "rb") as f:
        raw_pkg = pickle.load(f)
    raw_model = raw_pkg["model"]

    X_cal, y_cal = get_X_y(cal_df)
    print(f"Fitting isotonic calibrator on {len(y_cal)} cal samples...")
    cal_model = calibrate(raw_model, X_cal, y_cal)

    # Build and print report
    print("Evaluating...")
    report = build_report(raw_pkg, cal_model, train_df, cal_df, test_df)
    print("\n" + report)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(f"Report saved → {REPORT}")

    # Save calibrated model bundle
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    calibrated_pkg = {
        "model":            cal_model,          # CalibratedClassifierCV wrapping LightGBM
        "base_model_pkl":   str(MODEL_IN),
        "features":         raw_pkg["features"],
        "name":             "lightgbm_calibrated_2026",
        "cal_seasons":      CAL_SEASONS,
        "test_seasons":     TEST_SEASONS,
        "calibration_method": "isotonic",
        "note": (
            "Use the same strong-team-first feature ordering as the base model. "
            "apply_flip_if_needed: put the team with higher adj_em as team_a "
            "before calling predict_proba, then flip if the original query was reversed."
        ),
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(calibrated_pkg, f)
    print(f"Calibrated model saved → {MODEL_OUT}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_wrapper = _SklearnWrapper(raw_model, FEATURE_COLS)
                X_test, y_test = get_X_y(test_df)
                raw_probs = raw_wrapper.predict_proba(X_cal)[:, 1]
                cal_probs = cal_model.predict_proba(X_cal)[:, 1]

            fig, ax = plt.subplots(figsize=(7, 6))
            for probs, label, color in [
                (raw_probs, "Raw LightGBM", "steelblue"),
                (cal_probs, "Isotonic calibrated", "darkorange"),
            ]:
                frac_pos, mean_pred = calibration_curve(y_cal, probs, n_bins=10)
                ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color)
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Calibration Curve — 2023-24 Tournament Games")
            ax.legend()
            plot_path = ROOT / "outputs" / "reports" / "calibration_curve_2026.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved → {plot_path}")
        except ImportError:
            print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
