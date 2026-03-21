"""
Round 2 (Round of 32) Predictor — 2026 NCAA Tournament
=======================================================
Combines:
  1. Base LightGBM model (KenPom efficiency features)
  2. Momentum adjustments based on Round 1 results
     - Margin of victory (dominance / vulnerability)
     - Upset momentum boost
     - Struggle penalty for close calls

Usage:
    python scripts/predict_round2.py
    python scripts/predict_round2.py --csv        # save CSV output only
    python scripts/predict_round2.py --verbose    # print all debug info

Outputs:
    outputs/predictions/round2_predictions_2026.csv
    outputs/reports/round2_report_2026.txt

To reuse for Sweet 16, Elite 8, etc.:
    1. Update ROUND2_MATCHUPS with the new round's matchups
    2. Update R1_CSV path to point to that round's results CSV
    3. Re-run — all adjustment logic is round-agnostic
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path
from typing import Optional

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kalshi.strategy import NCAATradingStrategy
from scripts.adjust_for_momentum import (
    adjust_prediction,
    confidence_label,
    load_round_results,
)

# ─── Paths ───────────────────────────────────────────────────────────────────

R1_CSV      = ROOT / "data" / "processed" / "round1_results_2026.csv"
CSV_OUT     = ROOT / "outputs" / "predictions" / "round2_predictions_2026.csv"
REPORT_OUT  = ROOT / "outputs" / "reports"    / "round2_report_2026.txt"

# ─── Round 2 matchups ────────────────────────────────────────────────────────
# Format: (team_a, seed_a, team_b, seed_b, region, matchup_note)
#
# Bracket seeding logic (standard NCAA bracket):
#   1 vs 8/9 winner  |  4/5 vs 12/13 winner
#   3 vs 11/6 winner |  2 vs 10/7 winner
#
# Seeds reflect each team's original tournament seed (not their current round).

ROUND2_MATCHUPS: list[tuple[str, int, str, int, str, str]] = [
    # ── EAST ─────────────────────────────────────────────────────────────────
    ("Duke",        1,  "TCU",       9,  "East",    "1 vs 9-seed that beat Ohio St."),
    ("Houston",     2,  "VCU",      11,  "East",    "2 vs VCU (OT upset of UNC)"),
    ("Illinois",    3,  "Virginia", 10,  "East",    "3 vs 10-seed that beat St. Louis"),
    ("Texas Tech",  4,  "Alabama",  11,  "East",    "4 vs 11-seed"),
    # ── WEST ─────────────────────────────────────────────────────────────────
    ("Arizona",     1,  "Utah State",  9,  "West",  "1 vs 9-seed that beat Villanova"),
    ("Iowa State",  2,  "Kentucky",    7,  "West",  "2 vs 7-seed; Kentucky needed OT"),
    ("Gonzaga",     3,  "San Diego St.", 6, "West", "3 vs 6-seed"),
    ("Kansas",      4,  "St. John's",  5,  "West",  "4 vs 5; near-even seeds"),
    # ── MIDWEST ──────────────────────────────────────────────────────────────
    ("Michigan",    1,  "St. Louis",   9,  "Midwest", "1 vs 9-seed; St. Louis beat Georgia"),
    ("Marquette",   2,  "Nebraska",    4,  "Midwest", "2 vs 4-seed"),
    ("Ole Miss",    7,  "Vanderbilt",  8,  "Midwest", "7 vs 8; Ole Miss upset 5-seed"),
    ("Auburn",      4,  "Maryland",    5,  "Midwest", "Auburn survived close one"),
    # ── SOUTH ────────────────────────────────────────────────────────────────
    ("Florida",     1,  "Iowa",        9,  "South",  "1 vs 9-seed; Florida dominant"),
    ("Purdue",      2,  "Miami (Fla.)", 7, "South",  "2 vs 7-seed"),
    ("Tennessee",   3,  "Louisville",  6,  "South",  "3 vs 6-seed"),
    ("Arkansas",    5,  "High Point", 12,  "South",  "5 vs 12-seed Cinderella"),
]

# KenPom name overrides: bracket name → KenPom name
# (same logic as fill_bracket_2026.py)
KENPOM_ALIASES: dict[str, str] = {
    "Miami (Fla.)":   "Miami FL",
    "St. John's":     "St. John's",  # KenPom uses this spelling
    "Ole Miss":       "Mississippi",
    "VCU":            "Virginia Commonwealth",
    "St. Louis":      "Saint Louis",
    "San Diego St.":  "San Diego St.",
    "High Point":     "High Point",
    "Utah State":     "Utah St.",
}

# ─── Seed dictionary (all 68 teams) ──────────────────────────────────────────
SEEDS_2026: dict[str, int] = {
    # East
    "Duke": 1, "Ohio State": 8, "TCU": 9, "VCU": 11,
    "North Carolina": 6, "Houston": 2, "Illinois": 3,
    "Virginia": 10, "Kansas": 4, "Texas Tech": 4, "Alabama": 11,
    "St. John's": 5, "UConn": 2, "UCLA": 7, "Michigan State": 3,
    # West
    "Arizona": 1, "Villanova": 8, "Utah State": 9, "BYU": 6,
    "Texas": 11, "Iowa State": 2, "Gonzaga": 3, "Purdue": 2,
    "Kentucky": 7, "Arkansas": 4, "Wisconsin": 5, "High Point": 12,
    "San Diego St.": 6, "Kansas": 4, "Miami (Fla.)": 7,
    "Missouri": 10, "Queens": 15,
    # Midwest
    "Michigan": 1, "Georgia": 8, "St. Louis": 9, "Marquette": 2,
    "Nebraska": 4, "Auburn": 4, "Vanderbilt": 8, "Maryland": 5,
    "Tennessee": 3, "Ole Miss": 7, "Santa Clara": 10,
    # South
    "Florida": 1, "Clemson": 8, "Iowa": 9, "North Carolina": 2,
    "Louisville": 6, "Arkansas": 5, "High Point": 12,
    "Tennessee": 3, "Purdue": 2, "Miami (Fla.)": 7,
}


# ─── Prediction engine ───────────────────────────────────────────────────────

def kenpom_name(team: str) -> str:
    """Map bracket team name to KenPom name."""
    return KENPOM_ALIASES.get(team, team)


def run_predictions(verbose: bool = False) -> list[dict]:
    """
    Load the model and R1 results, then generate adjusted predictions
    for all Round 2 matchups.

    Returns a list of result dicts (one per game).
    """
    # Load base model
    strategy = NCAATradingStrategy()
    strategy.update_seeds(SEEDS_2026)

    # Load R1 momentum data (per-team adjustments)
    r1_data = load_round_results(R1_CSV)

    results: list[dict] = []

    for team_a, seed_a, team_b, seed_b, region, note in ROUND2_MATCHUPS:
        # ── Base model probability ────────────────────────────────────────────
        kp_a = kenpom_name(team_a)
        kp_b = kenpom_name(team_b)

        base_prob = strategy._predict_win_probability(
            kp_a, kp_b, seed_a=seed_a, seed_b=seed_b
        )

        if base_prob is None:
            # Try reverse lookup
            base_prob_rev = strategy._predict_win_probability(
                kp_b, kp_a, seed_a=seed_b, seed_b=seed_a
            )
            if base_prob_rev is not None:
                base_prob = 1.0 - base_prob_rev
            else:
                print(f"  [WARN] Could not predict {team_a} vs {team_b} — skipping")
                continue

        # ── Momentum adjustments ─────────────────────────────────────────────
        r1_a = r1_data.get(team_a) or r1_data.get(kp_a)
        r1_b = r1_data.get(team_b) or r1_data.get(kp_b)

        adj_a     = r1_a["adj"]     if r1_a else 0.0
        adj_b     = r1_b["adj"]     if r1_b else 0.0
        factors_a = r1_a["factors"] if r1_a else ["No R1 data"]
        factors_b = r1_b["factors"] if r1_b else ["No R1 data"]
        margin_a  = r1_a["margin"]  if r1_a else None
        margin_b  = r1_b["margin"]  if r1_b else None

        adjusted_prob = adjust_prediction(base_prob, adj_a, adj_b)

        # ── Confidence & metadata ─────────────────────────────────────────────
        delta            = adjusted_prob - base_prob
        conf             = confidence_label(adjusted_prob)
        base_winner      = team_a if base_prob >= 0.5 else team_b
        adjusted_winner  = team_a if adjusted_prob >= 0.5 else team_b
        pick_changed     = base_winner != adjusted_winner

        results.append({
            "region":           region,
            "team_a":           team_a,
            "seed_a":           seed_a,
            "team_b":           team_b,
            "seed_b":           seed_b,
            "note":             note,
            "base_prob_a":      base_prob,
            "adj_prob_a":       adjusted_prob,
            "delta":            delta,
            "adj_a":            adj_a,
            "adj_b":            adj_b,
            "factors_a":        "; ".join(factors_a),
            "factors_b":        "; ".join(factors_b),
            "margin_a":         margin_a,
            "margin_b":         margin_b,
            "base_winner":      base_winner,
            "adjusted_winner":  adjusted_winner,
            "pick_changed":     pick_changed,
            "confidence":       conf,
        })

        if verbose:
            print(f"  {team_a} vs {team_b}: base={base_prob:.1%} → adj={adjusted_prob:.1%} "
                  f"(Δ={delta:+.1%})  [{conf}]")

    return results


# ─── Output formatters ────────────────────────────────────────────────────────

def format_report(results: list[dict]) -> str:
    """Build a readable text report."""
    lines: list[str] = []

    lines += [
        "=" * 74,
        "  2026 NCAA TOURNAMENT — ROUND 2 PREDICTIONS",
        "  Base: LightGBM (KenPom efficiency features)",
        "  Adjustments: R1 margin of victory · upset momentum · struggle penalty",
        "=" * 74,
    ]

    current_region = None
    for r in results:
        if r["region"] != current_region:
            current_region = r["region"]
            lines += ["", f"{'─'*74}", f"  {current_region.upper()} REGION", f"{'─'*74}"]

        a, b = r["team_a"], r["team_b"]
        sa, sb = r["seed_a"], r["seed_b"]
        bp  = r["base_prob_a"]
        ap  = r["adj_prob_a"]
        d   = r["delta"]
        conf = r["confidence"]
        winner = r["adjusted_winner"]
        changed = "  ← PICK FLIPPED" if r["pick_changed"] else ""
        upset_flag = ""
        if r["adjusted_winner"] == b and sb > sa:
            upset_flag = f"  ★ UPSET PICK (#{sb} beats #{sa})"
        elif r["adjusted_winner"] == a and sa > sb:
            upset_flag = f"  ★ UPSET PICK (#{sa} beats #{sb})"

        lines.append("")
        lines.append(f"  #{sa} {a:<22}  vs  #{sb} {b}")
        lines.append(f"  {'─'*60}")
        lines.append(f"  Base model:   P({a} wins) = {bp:.1%}")
        lines.append(f"  Adjusted:     P({a} wins) = {ap:.1%}   (Δ = {d:+.1%})")
        lines.append(f"  Confidence:   {conf}")
        lines.append(f"  PICK:         {winner}{changed}{upset_flag}")

        # R1 context
        if r["margin_a"] is not None:
            lines.append(f"  {a} R1:  margin +{r['margin_a']}  →  {r['factors_a']}")
        if r["margin_b"] is not None:
            lines.append(f"  {b} R1:  margin +{r['margin_b']}  →  {r['factors_b']}")

    # Summary section
    lines += ["", "=" * 74, "  SUMMARY", "=" * 74, ""]

    # Games with biggest adjustments
    sorted_by_delta = sorted(results, key=lambda r: abs(r["delta"]), reverse=True)
    lines.append("  BIGGEST MOMENTUM SWINGS (base → adjusted):")
    for r in sorted_by_delta[:6]:
        a, b = r["team_a"], r["team_b"]
        lines.append(
            f"    #{r['seed_a']} {a:<18} vs #{r['seed_b']} {b:<18}  "
            f"Δ={r['delta']:+.1%}  ({r['base_prob_a']:.0%}→{r['adj_prob_a']:.0%})"
        )

    # Upset picks
    upset_picks = [
        r for r in results
        if (r["adjusted_winner"] == r["team_b"] and r["seed_b"] > r["seed_a"])
        or (r["adjusted_winner"] == r["team_a"] and r["seed_a"] > r["seed_b"])
    ]
    lines += ["", "  UPSET PICKS (lower seed favored after adjustments):"]
    if upset_picks:
        for r in upset_picks:
            w = r["adjusted_winner"]
            ws = r["seed_a"] if w == r["team_a"] else r["seed_b"]
            opp = r["team_b"] if w == r["team_a"] else r["team_a"]
            os_ = r["seed_b"] if w == r["team_a"] else r["seed_a"]
            lines.append(
                f"    ★ #{ws} {w} over #{os_} {opp}  "
                f"({r['adj_prob_a']:.1%} or {1-r['adj_prob_a']:.1%})"
            )
    else:
        lines.append("    None — base model favorites hold after adjustments.")

    # Toss-ups
    tossups = [r for r in results if r["confidence"] == "LOW (TOSS-UP)"]
    lines += ["", "  TOSS-UP GAMES (both teams within ~10%):"]
    if tossups:
        for r in tossups:
            lines.append(
                f"    #{r['seed_a']} {r['team_a']} vs #{r['seed_b']} {r['team_b']}"
                f"  →  {r['adj_prob_a']:.1%} / {1-r['adj_prob_a']:.1%}"
            )
    else:
        lines.append("    None — no true coin-flip games this round.")

    lines.append("")
    lines.append("=" * 74)
    return "\n".join(lines)


def save_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region", "team_a", "seed_a", "team_b", "seed_b",
        "base_prob_a", "adj_prob_a", "delta",
        "adj_a", "adj_b", "factors_a", "factors_b",
        "margin_a", "margin_b",
        "base_winner", "adjusted_winner", "pick_changed", "confidence", "note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  CSV  → {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NCAA 2026 Round 2 Predictor")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-game debug output during model scoring")
    parser.add_argument("--csv", action="store_true",
                        help="Save CSV output (always saves text report too)")
    args = parser.parse_args()

    print("Loading model and R1 results...")
    results = run_predictions(verbose=args.verbose)

    report = format_report(results)
    print("\n" + report)

    # Save text report
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report)
    print(f"\n  Report → {REPORT_OUT}")

    # Always save CSV
    save_csv(results, CSV_OUT)


if __name__ == "__main__":
    main()
