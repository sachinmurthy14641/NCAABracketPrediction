"""
Momentum adjustment layer for NCAA Tournament round-by-round predictions.

Applies post-model adjustments based on Round 1 (or any prior round) performance:
  - Margin of victory (dominance / vulnerability signals)
  - Upset momentum (teams that shocked higher seeds carry confidence)
  - Struggle penalty (favorites that barely beat heavy underdogs)

All adjustments are additive to the base probability, then clamped to [0.03, 0.97].
Functions are round-agnostic and can be reused for Sweet 16, Elite 8, etc.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R1_CSV = ROOT / "data" / "processed" / "round1_results_2026.csv"


# ─── Core adjustment functions ───────────────────────────────────────────────

def calculate_mov_adjustment(margin_of_victory: int, ot: bool = False) -> float:
    """
    Return a probability adjustment (+/-) based on margin of victory in prior round.

    Parameters
    ----------
    margin_of_victory : int
        Winning team's point margin (always positive; this is the winner's perspective).
    ot : bool
        True if the game went to overtime.

    Returns
    -------
    float
        Additive probability adjustment for the WINNING team.
    """
    if ot:
        # Won in OT: team was vulnerable; shows they can grind but weren't dominant
        return -0.05

    if margin_of_victory >= 30:
        return +0.05    # Dominant win → high confidence carry-over
    elif margin_of_victory >= 15:
        return +0.02    # Comfortable win → slight boost
    elif margin_of_victory >= 5:
        return 0.0      # Normal win → no adjustment
    else:
        # Won by <5 without OT: close game, slight vulnerability signal
        return -0.02


def calculate_upset_boost(winner_seed: int, loser_seed: int) -> float:
    """
    Return an additional boost for a team that upset a higher-seeded opponent.

    Only applies when winner_seed > loser_seed (winner was the underdog).

    Parameters
    ----------
    winner_seed : int   Seed of the team that won (the upset team).
    loser_seed  : int   Seed of the team that lost.

    Returns
    -------
    float
        Additive boost. Returns 0.0 if this was not an upset.
    """
    if winner_seed <= loser_seed:
        return 0.0  # Not an upset

    seed_gap = winner_seed - loser_seed  # e.g., 11 - 6 = 5

    if seed_gap >= 5:
        return +0.07    # Huge upset (e.g., 11 over 6, 12 over 5)
    elif seed_gap >= 3:
        return +0.05    # Clear upset (e.g., 10 over 7, 9 over 8)
    else:
        return +0.03    # Mild upset


def calculate_struggle_penalty(
    team_seed: int,
    opponent_seed: int,
    margin_of_victory: int,
    ot: bool = False,
) -> float:
    """
    Return a penalty if a favored team barely survived against a heavy underdog.

    Signals that the team may have underlying issues heading into the next round.

    Parameters
    ----------
    team_seed        : int   The team's seed (lower = better).
    opponent_seed    : int   The opponent's seed they just beat.
    margin_of_victory: int   How many points they won by.
    ot               : bool  Did the game go to overtime?

    Returns
    -------
    float
        Negative adjustment (penalty). Returns 0.0 if no penalty warranted.
    """
    seed_gap = opponent_seed - team_seed  # positive = team was favored

    if seed_gap <= 0:
        return 0.0  # Team was underdog — no struggle penalty

    # Near-loss against a massive underdog (e.g., 1-seed barely beat 16-seed)
    if seed_gap >= 10 and (margin_of_victory < 10 or ot):
        return -0.05
    elif seed_gap >= 7 and (margin_of_victory < 8 or ot):
        return -0.04
    elif seed_gap >= 5 and (margin_of_victory < 6 or ot):
        return -0.03
    elif seed_gap >= 3 and ot:
        return -0.02

    return 0.0


def total_adjustment(
    team_seed: int,
    opponent_seed: int,
    team_score: int,
    opp_score: int,
    ot: bool,
) -> tuple[float, list[str]]:
    """
    Compute the total momentum adjustment for a team based on its prior-round result.

    Returns both the numeric adjustment and a list of human-readable factor strings
    explaining the adjustments made.

    Parameters
    ----------
    team_seed      : int   The team's seed.
    opponent_seed  : int   The prior-round opponent's seed.
    team_score     : int   Points scored by this team.
    opp_score      : int   Points scored by the opponent.
    ot             : bool  Did the game go to overtime?

    Returns
    -------
    (float, list[str])
        Total additive adjustment, list of factor labels.
    """
    margin = team_score - opp_score
    if margin < 0:
        raise ValueError(
            f"Team score ({team_score}) must be >= opp score ({opp_score}) "
            "— only pass results for the winning team."
        )

    factors: list[str] = []
    adj = 0.0

    # 1. Margin of victory
    mov_adj = calculate_mov_adjustment(margin, ot=ot)
    adj += mov_adj
    if ot:
        factors.append(f"OT win (vulner.) {mov_adj:+.0%}")
    elif mov_adj > 0:
        factors.append(f"Dominant win +{margin}pts {mov_adj:+.0%}")
    elif mov_adj < 0:
        factors.append(f"Narrow win +{margin}pts {mov_adj:+.0%}")

    # 2. Upset bonus
    upset_adj = calculate_upset_boost(team_seed, opponent_seed)
    adj += upset_adj
    if upset_adj > 0:
        factors.append(f"Upset #{opponent_seed}→#{team_seed} {upset_adj:+.0%}")

    # 3. Struggle penalty
    struggle_adj = calculate_struggle_penalty(team_seed, opponent_seed, margin, ot)
    adj += struggle_adj
    if struggle_adj < 0:
        factors.append(f"Struggle vs #{opponent_seed} {struggle_adj:+.0%}")

    return adj, factors


def adjust_prediction(
    base_prob: float,
    team_a_adj: float,
    team_b_adj: float,
) -> float:
    """
    Apply momentum adjustments for both teams to the base model probability
    of team_a winning.

    Parameters
    ----------
    base_prob   : float   P(team_a wins) from the base model.
    team_a_adj  : float   Net momentum adjustment for team_a (+/-).
    team_b_adj  : float   Net momentum adjustment for team_b (+/-).

    Returns
    -------
    float
        Adjusted probability, clamped to [0.03, 0.97].
    """
    # Team B's positive momentum hurts team_a and vice versa
    adjusted = base_prob + team_a_adj - team_b_adj
    return max(0.03, min(0.97, adjusted))


def confidence_label(prob: float) -> str:
    """Return a confidence label based on win probability."""
    p = max(prob, 1 - prob)  # always look at the stronger team's prob
    if p >= 0.75:
        return "HIGH"
    elif p >= 0.60:
        return "MEDIUM"
    else:
        return "LOW (TOSS-UP)"


# ─── Data loader ─────────────────────────────────────────────────────────────

def load_round_results(csv_path: Path = R1_CSV) -> dict[str, dict]:
    """
    Load prior-round results CSV and return a per-team lookup dict.

    Returns
    -------
    dict[team_name -> {seed, opp_seed, team_score, opp_score, ot, upset, adj, factors}]
    """
    df = pd.read_csv(csv_path)
    df["ot"] = df["ot"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)
    df["upset"] = df["upset"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        team = row["team"]
        margin = int(row["team_score"]) - int(row["opp_score"])

        # Skip rows with score=0 (placeholder / unknown)
        if row["team_score"] == 0 and row["opp_score"] == 0:
            adj, factors = 0.0, ["No R1 data"]
        else:
            adj, factors = total_adjustment(
                team_seed=int(row["seed"]),
                opponent_seed=int(row["r1_opponent_seed"]),
                team_score=int(row["team_score"]),
                opp_score=int(row["opp_score"]),
                ot=bool(row["ot"]),
            )

        result[team] = {
            "seed":       int(row["seed"]),
            "opp_seed":   int(row["r1_opponent_seed"]),
            "team_score": int(row["team_score"]),
            "opp_score":  int(row["opp_score"]),
            "margin":     margin,
            "ot":         bool(row["ot"]),
            "upset":      bool(row["upset"]),
            "adj":        adj,
            "factors":    factors,
        }

    return result
