"""Validate the production model on the 2025 tournament — final check before live trading.

The production model was trained on seasons ≤ 2022.
2023 was used for validation (Platt calibration).
2024 was the held-out test set.
2025 has NEVER been seen by the model in any form.

Usage::

    python scripts/validate_on_2025.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import PlattModel, FEATURE_COLS, LABEL_COL, SEASON_COL  # noqa: F401

MODEL_PATH    = Path("outputs/models/lightgbm_final_2026.pkl")
TRAINING_DATA = Path("data/processed/training_data.csv")
REPORTS_DIR   = Path("outputs/reports")

# Prior year results for comparison table
PRIOR_RESULTS = {
    2023: {"accuracy": 0.9846, "brier": 0.0148, "ece": 0.016,  "notes": "FDU 16-over-1 upset"},
    2024: {"accuracy": 1.0000, "brier": 0.0002, "ece": 0.001,  "notes": "Chalk year, no upsets"},
}


def section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def calculate_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - probs[mask].mean())
    return ece


def calculate_mce(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(y_true[mask].mean() - probs[mask].mean()))
    return mce


def confidence_band(prob: float) -> str:
    if prob > 0.80 or prob < 0.20:
        return ">80%"
    elif prob > 0.60 or prob < 0.40:
        return "60-80%"
    else:
        return "toss-up"


def load() -> tuple:
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    df      = pd.read_csv(TRAINING_DATA)
    test_df = df[df[SEASON_COL] == 2025].copy().reset_index(drop=True)
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[LABEL_COL].values
    return payload, test_df, X_test, y_test


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    payload, test_df, X_test, y_test = load()
    model    = payload["model"]
    probs    = model.predict_proba(X_test)[:, 1]
    preds    = (probs >= 0.5).astype(int)
    correct  = (preds == y_test)

    # Unique game view (winner==1 only, no mirrors)
    orig_mask   = y_test == 1
    orig_probs  = probs[orig_mask]
    orig_y      = y_test[orig_mask]
    orig_preds  = preds[orig_mask]
    orig_correct = correct[orig_mask]
    orig_df     = test_df[orig_mask].copy().reset_index(drop=True)

    # ----------------------------------------------------------------
    # 2025 Metrics
    # ----------------------------------------------------------------
    section("2025 VALIDATION METRICS")
    acc      = accuracy_score(y_test, preds)
    brier    = brier_score_loss(y_test, probs)
    ll       = log_loss(y_test, probs)
    auc      = roc_auc_score(y_test, probs)
    ece      = calculate_ece(y_test, probs)
    mce      = calculate_mce(y_test, probs)

    print(f"\n  Games evaluated  : {len(orig_y)} unique  ({len(y_test)} rows incl. mirrors)")
    print(f"  Correct          : {orig_correct.sum()}/{len(orig_correct)}")
    print(f"  Accuracy         : {acc:.4f} ({acc:.1%})")
    print(f"  Brier Score      : {brier:.4f}")
    print(f"  Log Loss         : {ll:.4f}")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  ECE              : {ece:.4f}")
    print(f"  MCE              : {mce:.4f}")

    # ----------------------------------------------------------------
    # Confidence breakdown
    # ----------------------------------------------------------------
    section("CONFIDENCE BREAKDOWN")
    bands = {">80%": [], "60-80%": [], "toss-up": []}
    for p, c in zip(orig_probs, orig_correct):
        bands[confidence_band(p)].append(int(c))

    print(f"\n  {'Band':<12} {'Games':>6}  {'Correct':>8}  {'Accuracy':>10}")
    print(f"  {'-'*42}")
    for band, results in bands.items():
        n = len(results)
        if n == 0:
            print(f"  {band:<12} {'0':>6}  {'—':>8}  {'—':>10}")
        else:
            print(f"  {band:<12} {n:>6}  {sum(results):>8}  {sum(results)/n:>9.1%}")

    # ----------------------------------------------------------------
    # Upset analysis
    # ----------------------------------------------------------------
    section("UPSET ANALYSIS (2025)")
    upsets = orig_df[orig_df["team_a_seed"] > orig_df["team_b_seed"]].copy()
    upsets.index = range(len(upsets))
    upset_probs   = orig_probs[orig_df["team_a_seed"].values > orig_df["team_b_seed"].values]
    upset_correct = orig_correct[orig_df["team_a_seed"].values > orig_df["team_b_seed"].values]

    print(f"\n  Upsets in 2025 (underdog won): {len(upsets)}")
    if len(upsets) > 0:
        print(f"  Model predicted correctly    : {upset_correct.sum()}/{len(upsets)}")
        print(f"\n  {'Matchup':<50} {'SeedDiff':>9} {'Prob':>7} {'Correct':>8}")
        print(f"  {'-'*78}")
        for i, row in upsets.iterrows():
            matchup  = f"{row['team_a']} (#{int(row['team_a_seed'])}) over {row['team_b']} (#{int(row['team_b_seed'])})"
            sd       = int(row["team_a_seed"] - row["team_b_seed"])
            p        = upset_probs[i]
            tick     = "✓" if upset_correct[i] else "✗"
            print(f"  {matchup:<50} {sd:>9} {p:>7.3f} {tick:>8}")

    # ----------------------------------------------------------------
    # Incorrect predictions
    # ----------------------------------------------------------------
    errors = [(i, row, orig_probs[i]) for i, row in orig_df.iterrows() if not orig_correct[i]]
    section(f"INCORRECT PREDICTIONS ({len(errors)} games)")
    if not errors:
        print("\n  ✓  Perfect — no incorrect predictions on 2025 test set.")
    else:
        for i, row, prob in errors:
            print(f"\n  {row['team_a']} (#{int(row['team_a_seed'])}) vs "
                  f"{row['team_b']} (#{int(row['team_b_seed'])})")
            print(f"    Model prob {prob:.3f} → predicted team_a wins, but team_b won")
            band = confidence_band(prob)
            note = "coin-flip" if band == "toss-up" else f"model was {band} confident — true upset"
            print(f"    Note: {note}")

    # ----------------------------------------------------------------
    # Multi-year comparison table
    # ----------------------------------------------------------------
    section("VALIDATION ACROSS ALL TEST YEARS")
    results_2025 = {"accuracy": acc, "brier": brier, "ece": ece,
                    "notes": f"{len(upsets)} upsets, {len(errors)} errors"}
    all_years = {**PRIOR_RESULTS, 2025: results_2025}

    print(f"\n  {'Year':<6} {'Tournament':<22} {'Accuracy':>9} {'Brier':>8} {'ECE':>8}  Notes")
    print(f"  {'-'*80}")
    for year, r in sorted(all_years.items()):
        print(f"  {year:<6} {r['notes']:<22} {r['accuracy']:>9.1%} {r['brier']:>8.4f} {r['ece']:>8.3f}")

    accs = [r["accuracy"] for r in all_years.values()]
    print(f"\n  Average accuracy : {np.mean(accs):.1%}")
    print(f"  Std deviation    : {np.std(accs):.1%}")
    trend = "improving ↑" if accs[-1] >= accs[0] else "declining ↓" if accs[-1] < accs[0] - 0.02 else "stable →"
    print(f"  Trend            : {trend}")

    # ----------------------------------------------------------------
    # Save detailed results
    # ----------------------------------------------------------------
    orig_df["predicted_prob"]   = orig_probs
    orig_df["predicted_winner"] = orig_preds
    orig_df["correct"]          = orig_correct.astype(int)
    orig_df["confidence_band"] = [confidence_band(p) for p in orig_probs]
    out_path = REPORTS_DIR / "2025_validation.csv"
    orig_df.to_csv(out_path, index=False)
    print(f"\n  Detailed results saved → {out_path}")

    # ----------------------------------------------------------------
    # Final recommendation
    # ----------------------------------------------------------------
    section("RECOMMENDATION FOR 2026 TRADING")
    avg_acc = np.mean(accs)
    if avg_acc >= 0.96:
        rec   = "Model extremely robust — HIGH confidence for 2026 trading"
        stars = "★★★★★"
    elif avg_acc >= 0.93:
        rec   = "Model good — realistic expectations, trade conservatively"
        stars = "★★★★☆"
    else:
        rec   = "Model needs improvement before trading — caution advised"
        stars = "★★★☆☆"

    print(f"\n  {stars}  {rec}")
    print(f"\n  Average test accuracy: {avg_acc:.1%} across {len(all_years)} unseen tournament years")
    print(f"  Production model    : outputs/models/lightgbm_final_2026.pkl")


if __name__ == "__main__":
    main()
