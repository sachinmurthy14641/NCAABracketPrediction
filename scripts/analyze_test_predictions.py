"""Analyze model predictions on the 2023 held-out test set in detail.

Usage::

    python scripts/analyze_test_predictions.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import PlattModel  # noqa: F401

MODEL_PATH    = Path("outputs/models/baseline_logistic_v1.pkl")
TRAINING_DATA = Path("data/processed/training_data.csv")
TEST_SEASON   = 2023

METADATA_COLS = {"season", "team_a", "team_b", "score_a", "score_b"}
LABEL_COL     = "winner"


def section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def load() -> tuple:
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    df = pd.read_csv(TRAINING_DATA)
    return payload, df


def get_features(payload: dict) -> list[str]:
    return payload["features"]


def build_test_set(df: pd.DataFrame, features: list[str]) -> tuple:
    test = df[df["season"] == TEST_SEASON].copy().reset_index(drop=True)
    # Keep all rows (original + mirrored) — this matches how the model was evaluated in train_model.py
    # Each unique game appears twice: once with winner=1 and once with winner=0
    X    = test[features]
    y    = test[LABEL_COL]
    return test, X, y


def predict(payload: dict, X: pd.DataFrame) -> tuple:
    model  = payload["model"]
    probs  = model.predict_proba(X)[:, 1]
    preds  = (probs >= 0.5).astype(int)
    return probs, preds


def confidence_band(prob: float) -> str:
    if prob > 0.80 or prob < 0.20:
        return "high"
    elif prob > 0.60 or prob < 0.40:
        return "medium"
    else:
        return "toss-up"


def analyze_confidence(test: pd.DataFrame, probs, preds, y) -> None:
    section("PREDICTION CONFIDENCE BREAKDOWN")
    bands = {"high": [], "medium": [], "toss-up": []}
    for i, (prob, pred, actual) in enumerate(zip(probs, preds, y)):
        band = confidence_band(prob)
        bands[band].append(int(pred == actual))

    print(f"\n  {'Band':<12} {'Games':>6}  {'Correct':>8}  {'Accuracy':>10}")
    print(f"  {'-' * 42}")
    for band in ["high", "medium", "toss-up"]:
        results = bands[band]
        n       = len(results)
        correct = sum(results)
        acc     = correct / n if n else float("nan")
        print(f"  {band:<12} {n:>6}  {correct:>8}  {acc:>9.1%}")


def analyze_upsets(test: pd.DataFrame, probs, preds, y) -> None:
    section("UPSET ANALYSIS")
    if "seed_diff" not in test.columns:
        print("  seed_diff not available — skipping upset analysis.")
        return

    # Only evaluate upsets on original games (winner==1) to avoid double-counting
    orig = test[test[LABEL_COL] == 1]
    upsets, non_upsets = [], []
    for i, row in orig.iterrows():
        # seed_diff = team_a_seed - team_b_seed; positive = team_a is lower seed (underdog)
        sd     = row.get("seed_diff", 0)
        actual = int(y.iloc[i])
        prob   = probs[i]
        pred   = int(preds[i])
        correct = pred == actual

        # Upset: team_a (winner=1) is actually the lower seed (higher seed number)
        is_upset = sd > 0  # team_a seed > team_b seed, yet team_a won (they're the underdog)
        if is_upset:
            upsets.append((i, row, prob, pred, actual, correct, sd))
        else:
            non_upsets.append((i, row, prob, pred, actual, correct, sd))

    print(f"\n  Total games   : {len(orig)}")
    print(f"  Upsets        : {len(upsets)}  (team_a had a higher seed number = underdog)")
    print(f"  Non-upsets    : {len(non_upsets)}")

    if upsets:
        correct_upsets = sum(1 for *_, c, _ in upsets if c)
        print(f"  Upset accuracy: {correct_upsets}/{len(upsets)} = {correct_upsets/len(upsets):.1%}")
        print(f"\n  {'#':<4} {'Matchup':<45} {'SeedDiff':>9} {'PredProb':>9} {'Correct':>8}")
        print(f"  {'-' * 78}")
        for i, row, prob, pred, actual, correct, sd in upsets[:10]:
            matchup = f"{row.get('team_a','?')} vs {row.get('team_b','?')}"
            tick    = "✓" if correct else "✗"
            print(f"  {i:<4} {matchup:<45} {sd:>9.0f} {prob:>9.3f} {tick:>8}")


def analyze_errors(test: pd.DataFrame, probs, preds, y, features: list[str]) -> None:
    section("INCORRECT PREDICTIONS (DETAILED)")
    errors = [(i, row, probs[i], int(preds[i]), int(y.iloc[i]))
              for i, row in test.iterrows()
              if preds[i] != y.iloc[i]]

    if not errors:
        print("\n  ✓  No incorrect predictions on test set.")
        return

    print(f"\n  {len(errors)} incorrect prediction(s) found:\n")
    for idx, (i, row, prob, pred, actual) in enumerate(errors, 1):
        team_a = row.get("team_a", "team_a")
        team_b = row.get("team_b", "team_b")
        print(f"  --- Error #{idx} ---")
        print(f"  Matchup   : {team_a} vs {team_b}")
        print(f"  Predicted : {'team_a wins' if pred == 1 else 'team_b wins'} (prob={prob:.3f})")
        print(f"  Actual    : {'team_a won' if actual == 1 else 'team_b won'}")
        print(f"  Margin    : {abs(prob - actual):.3f}")

        band = confidence_band(prob)
        if band == "toss-up":
            note = "Near coin-flip — model was appropriately uncertain."
        elif (pred == 1 and actual == 0) or (pred == 0 and actual == 1):
            note = "Model was confident but wrong — possible upset or outlier game."
        else:
            note = "Moderate confidence miss."
        print(f"  Analysis  : {note}")

        print(f"\n  Feature values:")
        print(f"  {'Feature':<35} {'Value':>10}")
        print(f"  {'-'*48}")
        for feat in features:
            print(f"  {feat:<35} {row.get(feat, float('nan')):>10.4f}")
        print()


def main() -> None:
    payload, df = load()
    features    = get_features(payload)

    test, X, y  = build_test_set(df, features)

    section(f"TEST SET OVERVIEW (season={TEST_SEASON})")
    print(f"\n  Games in test set (original only, winner==1): {len(test)}")
    print(f"  Features used: {len(features)}")

    probs, preds = predict(payload, X)
    correct      = (preds == y.values).sum()
    accuracy     = correct / len(y)

    print(f"\n  Overall accuracy : {correct}/{len(y)} = {accuracy:.1%}")

    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    print(f"  Brier score      : {brier_score_loss(y, probs):.4f}")
    print(f"  Log loss         : {log_loss(y, probs):.4f}")
    print(f"  ROC-AUC          : {roc_auc_score(y, probs):.4f}")

    analyze_confidence(test, probs, preds, y)
    analyze_upsets(test, probs, preds, y)
    analyze_errors(test, probs, preds, y, features)


if __name__ == "__main__":
    main()
