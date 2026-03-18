"""2026 NCAA Tournament Auction Draft Tool

Must be run AFTER fill_bracket_2026.py has completed successfully.
Bracket structure is imported directly from fill_bracket_2026.BRACKET_2026
so both scripts are guaranteed to use identical matchups and seeds.

Market-aware pricing model that blends seed-based market expectations (what
people WILL pay based on historical auction behavior) with our LightGBM
model's EFP signal (what teams are actually worth in your scoring system).

Scoring:
    Round wins  : R1=1, R2=2, Sweet16=3, Elite8=4, Final4=5, Champ=10
    Margin bonus: 30+ pts=3, 20-29=2, 10-19=1 (per win)
    Underdog    : +2 pts per win when team seed > opponent seed

Pricing model (--blend controls market-prior weight, default 0.55):
    market_prior  — seed-curve fit to last year's auction behavior
    model_value   — EFP proportional share of budget
    final_value   — blend × market_prior + (1-blend) × model_value

Value signals:
    BUY  — model likes team more than market will price them  (ratio > 1.20)
    FAIR — model and market roughly agree                     (ratio 0.80–1.20)
    SELL — market will overprice vs model expectation         (ratio < 0.80)

Outputs:
    outputs/predictions/auction_values_2026.csv
    outputs/predictions/optimal_lineup_200.json
    outputs/reports/auction_draft_cheatsheet.txt

Usage:
    python scripts/auction_draft_tool.py --budget 200 --sims 10000

Always run fill_bracket_2026.py first.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model import PlattModel  # noqa: F401 — needed by pickle to deserialize
from scripts.fill_bracket_2026 import BRACKET_2026  # single source of truth for bracket

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH    = ROOT / "outputs/models/lightgbm_final_2026.pkl"
KENPOM_PATH   = ROOT / "data/processed/kenpom_2026_clean.csv"
PROFILES_PATH = ROOT / "data/processed/tournament_team_profiles_2026.csv"
CHALK_PATH    = ROOT / "outputs/predictions/bracket_2026_chalk.json"
MOMENTUM_PATH = ROOT / "outputs/predictions/bracket_2026_momentum.json"
PRED_DIR      = ROOT / "outputs/predictions"
REPORTS_DIR   = ROOT / "outputs/reports"

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
ROUND_PTS   = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10}
ROUND_LABEL = {0: "FF", 1: "R1", 2: "R2", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}
MARGIN_BONUS = [(30, 3), (20, 2), (10, 1)]
UPSET_BONUS  = 2

# Simulation physics
MARGIN_STD   = 11.5
EM_TO_MARGIN = 0.44

# Feature columns must match training order exactly
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

# ---------------------------------------------------------------------------
# Market prior — unnormalized seed-to-price curve calibrated to last year's
# auction behavior (1-seeds averaged ~$120 in practice, proportionally
# commanding ~60% of bidding attention vs the field).
# These raw values are normalized to sum to $budget at runtime.
# ---------------------------------------------------------------------------
SEED_PRIOR_RAW: dict[int, float] = {
    1:  55.0,   # 1-seeds attract outsized bidding; ~60% of real auction spend
    2:  20.0,
    3:  10.0,
    4:   6.0,
    5:   4.0,
    6:   4.0,
    7:   3.0,
    8:   3.0,
    9:   3.0,
    10:  2.0,
    11:  2.0,
    12:  2.0,
    13:  1.0,
    14:  1.0,
    15:  1.0,
    16:  1.0,
}

# ---------------------------------------------------------------------------
# Team name normalization — bracket names → KenPom 2026 names
# ---------------------------------------------------------------------------
TO_KENPOM: dict[str, str] = {
    # Bracket name → KenPom 2026 name (only entries that differ)
    "UConn":                  "Connecticut",
    "Saint Mary's":           "St. Mary's",
    "Miami (Fla.)":           "Miami FL",
    "Miami (Ohio)":           "Miami OH",
    "Prairie View A&M":       "Prairie View A&M",
    "Long Island University":  "LIU",
    "Iowa State":             "Iowa St.",
    "Michigan State":         "Michigan St.",
    "Ohio State":             "Ohio St.",
    "Utah State":             "Utah St.",
    "Wright State":           "Wright St.",
    "Kennesaw State":         "Kennesaw St.",
    "Tennessee State":        "Tennessee St.",
    "North Dakota State":     "North Dakota St.",
}


def kp_name(name: str) -> str:
    return TO_KENPOM.get(name, name)


# ---------------------------------------------------------------------------
# Bracket — loaded from BRACKET_2026 (single source of truth)
# ---------------------------------------------------------------------------

SEEDS: dict[str, int] = dict(BRACKET_2026["seeds"])

FIRST_FOUR: list[tuple[str, int, str, int, str]] = [
    (ff["team_a"], ff["seed_a"], ff["team_b"], ff["seed_b"], ff["slot"])
    for ff in BRACKET_2026["first_four"]
]

REGIONS: dict[str, list[tuple[int, str]]] = dict(BRACKET_2026["regions"])

FINAL_FOUR_MATCHUPS: list[tuple[str, str]] = list(BRACKET_2026["final_four_matchups"])

TEAM_REGION: dict[str, str] = {}
for _region, _slots in REGIONS.items():
    for _seed, _team in _slots:
        if not _team.endswith("_winner"):
            TEAM_REGION[_team] = _region
for _ff in BRACKET_2026["first_four"]:
    TEAM_REGION[_ff["team_a"]] = _ff["region_dest"]
    TEAM_REGION[_ff["team_b"]] = _ff["region_dest"]


# ---------------------------------------------------------------------------
# Step 1: Dependency validation
# ---------------------------------------------------------------------------

def validate_dependencies() -> None:
    """Verify that fill_bracket_2026.py has been run successfully."""
    missing = []
    for path in [CHALK_PATH, MOMENTUM_PATH, PROFILES_PATH]:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError(
            "Run fill_bracket_2026.py first before generating auction values.\n"
            "Missing files:\n" + "\n".join(f"  - {p}" for p in missing)
        )
    print("  [✓] Bracket outputs verified — fill_bracket_2026.py has been run.")


# ---------------------------------------------------------------------------
# Step 2: Load momentum flags from tournament team profiles
# ---------------------------------------------------------------------------

def load_momentum_flags() -> dict[str, str]:
    """Load HOT/WARM/FLAT/COLD/ICY flags from tournament_team_profiles_2026.csv."""
    if not PROFILES_PATH.exists():
        print("  WARNING: tournament_team_profiles_2026.csv not found — momentum flags will be blank.")
        return {}
    profiles = pd.read_csv(PROFILES_PATH)
    flags: dict[str, str] = {}
    for _, row in profiles.iterrows():
        team = row.get("team", "")
        flag = row.get("momentum_flag", "")
        if team and flag:
            flags[str(team)] = str(flag)
    print(f"  [✓] Loaded momentum flags for {len(flags)} teams.")
    return flags


# ---------------------------------------------------------------------------
# KenPom helpers
# ---------------------------------------------------------------------------

def build_kenpom_index(kp_df: pd.DataFrame) -> dict:
    return {row["team"]: row for _, row in kp_df.iterrows()}


def get_kp_row(team: str, seed: int, kp_index: dict) -> pd.Series:
    lookup = kp_name(team)
    if lookup in kp_index:
        return kp_index[lookup]
    adj_em  = max(-5.0, 35.0 - (seed - 1) * 2.3)
    adj_off = 117.0 - (seed - 1) * 0.6
    adj_def = 99.0  + (seed - 1) * 0.8
    return pd.Series({
        "team": team, "adj_em": adj_em, "adj_off_eff": adj_off,
        "adj_def_eff": adj_def, "adj_tempo": 68.0, "efficiency_differential": adj_em,
    })


# ---------------------------------------------------------------------------
# Batched win-probability cache — single predict_proba call for all pairs
# ---------------------------------------------------------------------------

def precompute_probs(
    all_teams: list[tuple[str, int]],
    kp_index: dict,
    model,
) -> dict[tuple[str, str], float]:
    teams = list(all_teams)
    pair_index: list[tuple[str, str]] = []
    rows: list[dict] = []

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            ta, sa = teams[i]
            tb, sb = teams[j]
            ra = get_kp_row(ta, sa, kp_index)
            rb = get_kp_row(tb, sb, kp_index)
            off_adv = ra["adj_off_eff"] - rb["adj_def_eff"]
            def_adv = rb["adj_off_eff"] - ra["adj_def_eff"]
            rows.append({
                "off_eff_advantage":            off_adv,
                "def_eff_advantage":            def_adv,
                "net_efficiency_edge":          off_adv - def_adv,
                "tempo_difference":             ra["adj_tempo"] - rb["adj_tempo"],
                "overall_rating_diff":          ra["adj_em"] - rb["adj_em"],
                "efficiency_differential_diff": ra["efficiency_differential"] - rb["efficiency_differential"],
                "seed_diff":                    sa - sb,
                "a_adj_off_eff": ra["adj_off_eff"], "a_adj_def_eff": ra["adj_def_eff"],
                "a_adj_em": ra["adj_em"],
                "b_adj_off_eff": rb["adj_off_eff"], "b_adj_def_eff": rb["adj_def_eff"],
                "b_adj_em": rb["adj_em"],
            })
            pair_index.append((ta, tb))

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    probs = model.predict_proba(X)[:, 1]

    cache: dict[tuple[str, str], float] = {}
    for (ta, tb), p in zip(pair_index, probs):
        cache[(ta, tb)] = float(p)
        cache[(tb, ta)] = 1.0 - float(p)
    return cache


# ---------------------------------------------------------------------------
# Single game simulation
# ---------------------------------------------------------------------------

def sim_game(
    team_a: str, seed_a: int,
    team_b: str, seed_b: int,
    game_round: int,
    prob_cache: dict,
    kp_index: dict,
    rng: np.random.Generator,
) -> tuple[str, int, int, int, int, int]:
    p_a  = prob_cache.get((team_a, team_b), 0.5)
    won_a = rng.random() < p_a
    ra = get_kp_row(team_a, seed_a, kp_index)
    rb = get_kp_row(team_b, seed_b, kp_index)

    if won_a:
        winner, w_seed, l_seed = team_a, seed_a, seed_b
        em_diff = ra["adj_em"] - rb["adj_em"]
    else:
        winner, w_seed, l_seed = team_b, seed_b, seed_a
        em_diff = rb["adj_em"] - ra["adj_em"]

    exp_margin = max(0.0, em_diff * EM_TO_MARGIN)
    margin = max(1.0, float(rng.normal(exp_margin, MARGIN_STD)))
    base = ROUND_PTS[game_round]
    mb   = next((b for thresh, b in MARGIN_BONUS if margin >= thresh), 0)
    ub   = UPSET_BONUS if w_seed > l_seed else 0
    return winner, w_seed, base + mb + ub, base, mb, ub


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def run_one_tournament(
    ff_results: dict[str, tuple[str, int]],
    prob_cache: dict,
    kp_index: dict,
    rng: np.random.Generator,
) -> list[tuple[str, int, int, int, int]]:
    events: list[tuple[str, int, int, int, int]] = []

    for ta, sa, tb, sb, slot in FIRST_FOUR:
        winner, w_seed = ff_results[slot]
        l_seed = sb if winner == ta else sa
        ra = get_kp_row(ta, sa, kp_index)
        rb = get_kp_row(tb, sb, kp_index)
        em_diff = ra["adj_em"] - rb["adj_em"] if winner == ta else rb["adj_em"] - ra["adj_em"]
        exp_margin = max(0.0, em_diff * EM_TO_MARGIN)
        margin = max(1.0, float(rng.normal(exp_margin, MARGIN_STD)))
        mb = next((b for thresh, b in MARGIN_BONUS if margin >= thresh), 0)
        ub = UPSET_BONUS if w_seed > l_seed else 0
        events.append((winner, 0, ROUND_PTS[0], mb, ub))

    region_champs: dict[str, tuple[str, int]] = {}
    for region_name, slots in REGIONS.items():
        resolved = []
        for seed, team in slots:
            if team.startswith("__FF"):
                actual_team, actual_seed = ff_results[team]
                resolved.append((actual_seed, actual_team))
            else:
                resolved.append((seed, team))

        current = resolved
        for game_round in [1, 2, 3, 4]:
            next_round = []
            for i in range(0, len(current), 2):
                sa, ta = current[i]
                sb, tb = current[i + 1]
                winner, w_seed, _, base, mb, ub = sim_game(
                    ta, sa, tb, sb, game_round, prob_cache, kp_index, rng
                )
                events.append((winner, game_round, base, mb, ub))
                next_round.append((w_seed, winner))
            current = next_round

        champ_seed, champ = current[0]
        region_champs[region_name] = (champ, champ_seed)

    final_champs: list[tuple[str, int]] = []
    for reg_a, reg_b in FINAL_FOUR_MATCHUPS:
        ta, sa = region_champs[reg_a]
        tb, sb = region_champs[reg_b]
        winner, w_seed, _, base, mb, ub = sim_game(ta, sa, tb, sb, 5, prob_cache, kp_index, rng)
        events.append((winner, 5, base, mb, ub))
        final_champs.append((winner, w_seed))

    (ta, sa), (tb, sb) = final_champs
    winner, _, _, base, mb, ub = sim_game(ta, sa, tb, sb, 6, prob_cache, kp_index, rng)
    events.append((winner, 6, base, mb, ub))
    return events


# ---------------------------------------------------------------------------
# Monte Carlo main loop
# ---------------------------------------------------------------------------

def run_monte_carlo(n_sims: int, prob_cache: dict, kp_index: dict) -> dict[str, dict]:
    rng = np.random.default_rng(42)

    acc_round_wins:   dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    acc_base_pts:     dict[str, float]          = defaultdict(float)
    acc_margin_bonus: dict[str, float]          = defaultdict(float)
    acc_udog_bonus:   dict[str, float]          = defaultdict(float)

    for sim_i in range(n_sims):
        ff_results: dict[str, tuple[str, int]] = {}
        for ta, sa, tb, sb, slot in FIRST_FOUR:
            p_a = prob_cache.get((ta, tb), 0.5)
            ff_results[slot] = (ta, sa) if rng.random() < p_a else (tb, sb)

        events = run_one_tournament(ff_results, prob_cache, kp_index, rng)
        for team, rnd, base, mb, ub in events:
            acc_round_wins[team][rnd] += 1
            acc_base_pts[team]        += base
            acc_margin_bonus[team]    += mb
            acc_udog_bonus[team]      += ub

        if (sim_i + 1) % 2_000 == 0:
            print(f"  {sim_i + 1:,} / {n_sims:,} sims complete")

    result: dict[str, dict] = {}
    for team in SEEDS:
        rw = acc_round_wins[team]
        exp_base = acc_base_pts[team]     / n_sims
        exp_mb   = acc_margin_bonus[team] / n_sims
        exp_ub   = acc_udog_bonus[team]   / n_sims
        result[team] = {
            "p_ff":    rw.get(0, 0) / n_sims,
            "p_r1":    rw.get(1, 0) / n_sims,
            "p_r2":    rw.get(2, 0) / n_sims,
            "p_s16":   rw.get(3, 0) / n_sims,
            "p_e8":    rw.get(4, 0) / n_sims,
            "p_f4":    rw.get(5, 0) / n_sims,
            "p_champ": rw.get(6, 0) / n_sims,
            "exp_base": exp_base,
            "exp_mb":   exp_mb,
            "exp_ub":   exp_ub,
            "total_efp": exp_base + exp_mb + exp_ub,
        }
    return result


# ---------------------------------------------------------------------------
# Market-aware pricing model
# ---------------------------------------------------------------------------

def compute_market_aware_values(
    stats: dict,
    budget: float,
    blend: float = 0.55,
) -> dict[str, dict]:
    """Two-component blended pricing model.

    Component 1: market_prior  — seed-curve normalized to $budget
    Component 2: model_value   — EFP proportional share of $budget
    final_value = blend × market_prior + (1-blend) × model_value, re-normalized

    Returns per-team dict with pricing, signals, and draft guardrails.
    """
    # ── Component 1: market prior ──────────────────────────────────────────
    raw_prior = {t: SEED_PRIOR_RAW.get(s, 1.0) for t, s in SEEDS.items()}
    total_prior = sum(raw_prior.values())
    market_prior = {t: v / total_prior * budget for t, v in raw_prior.items()}

    # ── Component 2: model value ───────────────────────────────────────────
    total_efp = sum(s["total_efp"] for s in stats.values()) or 1.0
    model_value = {t: stats[t]["total_efp"] / total_efp * budget for t in stats}

    # ── Blended value (re-normalized so all 68 teams sum to $budget) ───────
    raw_blend = {
        t: blend * market_prior[t] + (1.0 - blend) * model_value[t]
        for t in stats
    }
    total_blend = sum(raw_blend.values())
    final_value = {t: max(1.0, v / total_blend * budget) for t, v in raw_blend.items()}

    # Re-normalize after floor
    total_floored = sum(final_value.values())
    final_value = {t: v / total_floored * budget for t, v in final_value.items()}

    # ── Per-team signals and bid guardrails ────────────────────────────────
    result: dict[str, dict] = {}
    for team in stats:
        mp = market_prior[team]
        mv = model_value[team]
        fv = final_value[team]
        vr = mv / mp if mp > 0 else 1.0

        if   vr > 1.20: signal = "BUY"
        elif vr < 0.80: signal = "SELL"
        else:           signal = "FAIR"

        result[team] = {
            "market_prior": round(mp, 2),
            "model_value":  round(mv, 2),
            "final_value":  round(fv, 2),
            "value_rating": round(vr, 3),
            "value_signal": signal,
            "target_bid":   round(max(1.0, mv * 0.90), 2),
            "max_bid":      round(max(1.0, mv * 1.20), 2),
            "avoid_above":  round(max(1.5, mv * 1.35), 2),
        }
    return result


# ---------------------------------------------------------------------------
# Value tier
# ---------------------------------------------------------------------------

def value_tier(v: float) -> str:
    if   v > 25:  return "Elite"
    elif v >= 15: return "Strong"
    elif v >= 8:  return "Value"
    elif v >= 3:  return "Lottery"
    return "Flier"


# ---------------------------------------------------------------------------
# Budget optimizer — two lineup strategies
# ---------------------------------------------------------------------------

def _greedy_picks(
    candidates: pd.DataFrame,
    budget: float,
    n_teams: int,
    boost_buys: bool = False,
) -> list[dict]:
    """Greedy pick by EFP/target_bid ratio. Optionally boost BUY signals."""
    work = candidates.copy()
    work["_score"] = work["Total_EFP"] / work["Target_Bid"]
    if boost_buys:
        work["_score"] *= work["Value_Signal"].map({"BUY": 1.25, "FAIR": 1.0, "SELL": 0.85})
    work = work.sort_values("_score", ascending=False)

    picks, remaining = [], budget
    for _, row in work.iterrows():
        if len(picks) >= n_teams:
            break
        cost = row["Target_Bid"]
        if cost <= remaining:
            picks.append({
                "team":          row["Team"],
                "seed":          int(row["Seed"]),
                "region":        row["Region"],
                "momentum_flag": row.get("Momentum_Flag", ""),
                "target_bid":    float(row["Target_Bid"]),
                "max_bid":       float(row["Max_Bid"]),
                "auction_value": float(row["Auction_Value"]),
                "total_efp":     round(float(row["Total_EFP"]), 2),
                "p_champ_pct":   round(float(row["P_Champ"]), 1),
                "p_f4_pct":      round(float(row["P_Final4"]), 1),
                "value_signal":  row["Value_Signal"],
            })
            remaining -= cost
    return picks


def optimize_lineup(df: pd.DataFrame, budget: float = 200.0, n_teams: int = 10) -> dict:
    """Build two lineup strategies using target_bid as cost.

    Anchor  : spend big on the best 1-seed, fill roster with highest value/$ picks
    Balanced: skip 1-seeds entirely, load up on BUY signals in seeds 2-8
    """
    # ── Anchor strategy ────────────────────────────────────────────────────
    seed1_df = df[df["Seed"] == 1].sort_values("Total_EFP", ascending=False)
    anchor_team = seed1_df.iloc[0]
    anchor_cost = anchor_team["Target_Bid"]
    rest = df[df["Team"] != anchor_team["Team"]].copy()
    anchor_rest = _greedy_picks(rest, budget - anchor_cost, n_teams - 1)
    anchor_picks = [{
        "team":          anchor_team["Team"],
        "seed":          int(anchor_team["Seed"]),
        "region":        anchor_team["Region"],
        "momentum_flag": anchor_team.get("Momentum_Flag", ""),
        "target_bid":    float(anchor_team["Target_Bid"]),
        "max_bid":       float(anchor_team["Max_Bid"]),
        "auction_value": float(anchor_team["Auction_Value"]),
        "total_efp":     round(float(anchor_team["Total_EFP"]), 2),
        "p_champ_pct":   round(float(anchor_team["P_Champ"]), 1),
        "p_f4_pct":      round(float(anchor_team["P_Final4"]), 1),
        "value_signal":  anchor_team["Value_Signal"],
    }] + anchor_rest

    # ── Balanced strategy ──────────────────────────────────────────────────
    no_seeds1 = df[df["Seed"] > 1].copy()
    balanced_picks = _greedy_picks(no_seeds1, budget, n_teams, boost_buys=True)

    def _summarize(picks):
        cost = round(sum(p["target_bid"] for p in picks), 2)
        return {
            "lineup":     picks,
            "total_cost": cost,
            "remaining":  round(budget - cost, 2),
            "total_efp":  round(sum(p["total_efp"] for p in picks), 2),
            "budget":     budget,
            "n_teams":    len(picks),
        }

    return {
        "anchor":   {"strategy": "Anchor — elite 1-seed + best value fill", **_summarize(anchor_picks)},
        "balanced": {"strategy": "Balanced — no 1-seeds, BUY-signal heavy", **_summarize(balanced_picks)},
    }


# ---------------------------------------------------------------------------
# Build output DataFrame
# ---------------------------------------------------------------------------

def build_output_df(
    stats: dict,
    pricing: dict,
    momentum_flags: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for team, seed in SEEDS.items():
        s = stats[team]
        p = pricing[team]
        rows.append({
            "Team":                    team,
            "Seed":                    seed,
            "Region":                  TEAM_REGION.get(team, ""),
            "Momentum_Flag":           momentum_flags.get(team, ""),
            "P_Champ":                 round(s["p_champ"] * 100, 1),
            "P_Final4":                round(s["p_f4"]    * 100, 1),
            "P_Elite8":                round(s["p_e8"]    * 100, 1),
            "P_Sweet16":               round(s["p_s16"]   * 100, 1),
            "Expected_Base_Pts":       round(s["exp_base"], 2),
            "Expected_Margin_Bonus":   round(s["exp_mb"],   2),
            "Expected_Underdog_Bonus": round(s["exp_ub"],   2),
            "Total_EFP":               round(s["total_efp"], 2),
            "Market_Prior":            p["market_prior"],
            "Model_Value":             p["model_value"],
            "Auction_Value":           p["final_value"],
            "Value_Signal":            p["value_signal"],
            "Target_Bid":              p["target_bid"],
            "Max_Bid":                 p["max_bid"],
            "Avoid_Above":             p["avoid_above"],
            "Value_Tier":              value_tier(p["final_value"]),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("Auction_Value", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "Rank", df.index + 1)
    return df


# ---------------------------------------------------------------------------
# Console output — full rankings table
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame, budget: float, n_sims: int, blend: float) -> None:
    print("\n" + "=" * 116)
    print(
        f"  2026 NCAA AUCTION DRAFT TOOL  "
        f"|  Budget: ${budget:.0f}  |  Sims: {n_sims:,}  |  Blend: {blend:.0%} market / {1-blend:.0%} model"
    )
    print("=" * 116)
    hdr = (
        f"  {'Rk':<4} {'Team':<22} {'Seed':>4} {'Region':<9}"
        f" {'Momentum':<8} {'P_Champ':>8} {'P_F4':>6} {'P_E8':>6} {'EFP':>7}"
        f" {'Mkt$':>7} {'Mdl$':>7} {'Val$':>7}"
        f" {'Sig':>5} {'Tgt$':>7} {'Max$':>7}  {'Tier'}"
    )
    print(hdr)
    print(f"  {'-' * 111}")

    last_tier = None
    for _, row in df.iterrows():
        tier = row["Value_Tier"]
        if tier != last_tier:
            if last_tier is not None:
                print(f"  {'·' * 111}")
            last_tier = tier
        sig_tag = {"BUY": "▲BUY", "SELL": "▼SEL", "FAIR": " ---"}.get(row["Value_Signal"], "    ")
        flag = str(row.get("Momentum_Flag", "")).ljust(7)
        print(
            f"  {int(row['Rank']):<4} {row['Team']:<22} #{int(row['Seed']):<3} {row['Region']:<9}"
            f" {flag:<8} {row['P_Champ']:>7.1f}% {row['P_Final4']:>5.1f}% {row['P_Elite8']:>5.1f}%"
            f" {row['Total_EFP']:>7.2f}"
            f" {row['Market_Prior']:>7.2f} {row['Model_Value']:>7.2f} {row['Auction_Value']:>7.2f}"
            f" {sig_tag:>5} {row['Target_Bid']:>7.2f} {row['Max_Bid']:>7.2f}  {tier}"
        )

    print(f"\n  Budget: ${budget:.0f}  |  Total auction values: ${df['Auction_Value'].sum():.2f}")


# ---------------------------------------------------------------------------
# Console output — lineup strategies
# ---------------------------------------------------------------------------

def print_lineups(lineups: dict) -> None:
    for key in ("anchor", "balanced"):
        strat = lineups[key]
        print("\n" + "=" * 86)
        print(f"  {strat['strategy'].upper()}")
        print("=" * 86)
        print(f"  {'Team':<22} {'Seed':>5} {'Region':<9} {'Momentum':<8} {'Tgt$':>7} {'Max$':>7} {'EFP':>7} {'P_Champ':>8} {'Sig':>5}")
        print(f"  {'-' * 82}")
        for p in strat["lineup"]:
            sig_tag = {"BUY": "▲BUY", "SELL": "▼SEL", "FAIR": " ---"}.get(p["value_signal"], "    ")
            flag = str(p.get("momentum_flag", "")).ljust(7)
            print(
                f"  {p['team']:<22} #{p['seed']:<4} {p['region']:<9}"
                f" {flag:<8} ${p['target_bid']:>6.2f} ${p['max_bid']:>6.2f} {p['total_efp']:>7.2f}"
                f" {p['p_champ_pct']:>7.1f}% {sig_tag:>5}"
            )
        print(
            f"\n  Roster cost (at target): ${strat['total_cost']:.2f} / ${strat['budget']:.0f}"
            f"  |  Remaining: ${strat['remaining']:.2f}"
            f"  |  Projected EFP: {strat['total_efp']:.2f}"
        )


# ---------------------------------------------------------------------------
# Step 5: Change detection — compare old vs new auction values
# ---------------------------------------------------------------------------

def detect_and_print_changes(
    df_new: pd.DataFrame,
    prev_path: Path,
) -> None:
    """Rename old CSV to _previous, then diff and print changes."""
    csv_path = PRED_DIR / "auction_values_2026.csv"

    # Rename current file to _previous before overwriting
    if csv_path.exists():
        csv_path.rename(prev_path)
        print(f"  Saved previous values → {prev_path.relative_to(ROOT)}")
    else:
        print("  No previous auction_values_2026.csv found — skipping change report.")
        return

    try:
        df_old = pd.read_csv(prev_path)
    except Exception:
        return

    # Merge on Team
    if "Team" not in df_old.columns or "Auction_Value" not in df_old.columns:
        return

    merged = df_new[["Team", "Total_EFP", "Auction_Value"]].merge(
        df_old[["Team", "Total_EFP", "Auction_Value"]].rename(
            columns={"Total_EFP": "Old_EFP", "Auction_Value": "Old_Value"}
        ),
        on="Team", how="inner",
    )
    merged["EFP_Delta"]   = merged["Total_EFP"]    - merged["Old_EFP"]
    merged["Value_Delta"] = merged["Auction_Value"] - merged["Old_Value"]
    changed = merged[merged["Value_Delta"].abs() > 0.01].sort_values(
        "Value_Delta", key=abs, ascending=False
    )

    if changed.empty:
        print("  [✓] Auction values unchanged from previous run.")
        return

    print("\n" + "=" * 72)
    print("  AUCTION VALUE CHANGES (Bracket Fix Impact)")
    print("=" * 72)
    print(f"  {'Team':<22} {'Old EFP':>8} {'New EFP':>8} {'Old $':>7} {'New $':>7} {'Δ$':>7}")
    print(f"  {'-' * 66}")
    for _, row in changed.iterrows():
        delta_str = f"{row['Value_Delta']:+.2f}"
        highlight = " ◄ LARGE" if abs(row["Value_Delta"]) > 2.0 else ""
        print(
            f"  {row['Team']:<22} {row['Old_EFP']:>8.2f} {row['Total_EFP']:>8.2f}"
            f" {row['Old_Value']:>7.2f} {row['Auction_Value']:>7.2f} {delta_str:>7}{highlight}"
        )
    print()


# ---------------------------------------------------------------------------
# Cheat sheet — organized by draft-day sections
# ---------------------------------------------------------------------------

def write_cheatsheet(
    df: pd.DataFrame,
    lineups: dict,
    budget: float,
    model_name: str,
    path: Path,
) -> None:
    W = 76
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "=" * W,
        "  2026 NCAA FANTASY AUCTION CHEAT SHEET",
        f"  Budget: ${budget:.0f}  |  Generated: {ts}",
        f"  Model: {model_name}",
        "  Bracket Source: NCAA.com official (corrected March 17 2026)",
        "=" * W,
        "",
    ]

    col_hdr = f"  {'Team':<22} {'Sd':>3} {'Region':<9} {'EFP':>6} {'Val$':>7} {'Tgt$':>7} {'Max$':>7} {'Champ%':>7} {'Sig':>5} {'Momentum'}"
    col_div = f"  {'-' * 73}"

    def _section(title: str, mask) -> list[str]:
        subset = df[mask].copy()
        if subset.empty:
            return []
        out = ["", f"{'─'*W}", f"  {title}", f"{'─'*W}", col_hdr, col_div]
        for _, r in subset.iterrows():
            sig  = {"BUY": "▲BUY", "SELL": "▼SEL", "FAIR": " ---"}.get(r["Value_Signal"], "    ")
            flag = str(r.get("Momentum_Flag", ""))
            out.append(
                f"  {r['Team']:<22} #{int(r['Seed']):<2} {r['Region']:<9}"
                f" {r['Total_EFP']:>6.2f} ${r['Auction_Value']:>6.2f}"
                f" ${r['Target_Bid']:>6.2f} ${r['Max_Bid']:>6.2f}"
                f" {r['P_Champ']:>6.1f}% {sig:>5}  {flag}"
            )
        return out

    # Section 1 — Elite Anchors (seeds 1-2)
    lines += _section(
        "ELITE ANCHORS  (seeds 1-2 — expect heavy bidding)",
        df["Seed"].isin([1, 2])
    )

    # Section 2 — Value Targets (BUY, seeds 3-8)
    lines += _section(
        "VALUE TARGETS  (BUY signal, seeds 3-8 — model > market)",
        df["Value_Signal"].eq("BUY") & df["Seed"].between(3, 8)
    )

    # Section 3 — Fades (SELL signal)
    lines += _section(
        "FADES  (SELL signal — market overpays; let others have them)",
        df["Value_Signal"].eq("SELL")
    )

    # Section 4 — Fliers (seeds 12+, BUY)
    buy_fliers = df[df["Seed"].ge(12) & df["Value_Signal"].eq("BUY")]
    if not buy_fliers.empty:
        lines += _section(
            "FLIERS  (seeds 12+ with BUY signal — cheap lottery tickets)",
            df["Seed"].ge(12) & df["Value_Signal"].eq("BUY")
        )

    # Section 4b — All fliers
    lines += _section(
        "ALL FLIERS  (seeds 12+ — full cheap-team reference list)",
        df["Seed"].ge(12)
    )

    # Section 5 — Budget optimizer
    for key in ("anchor", "balanced"):
        strat = lineups[key]
        lines += [
            "",
            "=" * W,
            f"  BUDGET OPTIMIZER — {strat['strategy'].upper()}",
            "=" * W,
            f"  {'Team':<22} {'Sd':>3} {'Region':<9} {'Momentum':<8} {'Tgt$':>7} {'Max$':>7} {'EFP':>6} {'Champ%':>7} {'Sig':>5}",
            f"  {'-' * 72}",
        ]
        for p in strat["lineup"]:
            sig  = {"BUY": "▲BUY", "SELL": "▼SEL", "FAIR": " ---"}.get(p["value_signal"], "    ")
            flag = str(p.get("momentum_flag", "")).ljust(7)
            lines.append(
                f"  {p['team']:<22} #{p['seed']:<2} {p['region']:<9}"
                f" {flag:<8} ${p['target_bid']:>6.2f}"
                f" ${p['max_bid']:>6.2f} {p['total_efp']:>6.2f} {p['p_champ_pct']:>6.1f}% {sig:>5}"
            )
        lines += [
            "",
            f"  Total cost (target bids): ${strat['total_cost']:.2f} / ${strat['budget']:.0f}"
            f"  |  Remaining: ${strat['remaining']:.2f}",
            f"  Projected EFP: {strat['total_efp']:.2f} pts",
        ]

    lines.append("")
    path.write_text("\n".join(lines))
    print(f"  Cheat sheet  → {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2026 NCAA Auction Draft Tool")
    p.add_argument("--budget",   type=float, default=200.0,  help="Auction budget (default: 200)")
    p.add_argument("--sims",     type=int,   default=10_000, help="Monte Carlo iterations (default: 10000)")
    p.add_argument("--n-teams",  type=int,   default=10,     help="Roster size (default: 10)")
    p.add_argument("--blend",    type=float, default=0.55,
                   help="Weight on market prior [0-1] (default: 0.55). "
                        "1.0 = pure market seed-curve, 0.0 = pure model EFP.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Validate dependencies ────────────────────────────────────────
    print("Validating dependencies...")
    validate_dependencies()

    # ── Step 2: Load momentum flags ───────────────────────────────────────────
    print("Loading momentum flags...")
    momentum_flags = load_momentum_flags()

    # ── Load model + KenPom ──────────────────────────────────────────────────
    print(f"Loading model:  {MODEL_PATH.relative_to(ROOT)}")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    model      = payload["model"]
    model_name = f"{payload.get('name', 'LightGBM')} ({len(payload.get('features', FEATURE_COLS))} features)"

    print(f"Loading KenPom: {KENPOM_PATH.relative_to(ROOT)}")
    kp_df    = pd.read_csv(KENPOM_PATH)
    kp_index = build_kenpom_index(kp_df)

    all_teams = list(SEEDS.items())
    n_pairs   = len(all_teams) * (len(all_teams) - 1) // 2
    print(f"\nPrecomputing {n_pairs:,} pairwise win probabilities ({len(all_teams)} teams)...")
    prob_cache = precompute_probs(all_teams, kp_index, model)
    print(f"  Cached {len(prob_cache):,} directed probabilities.")

    print("\n  FIRST FOUR WIN PROBABILITIES")
    print(f"  {'-' * 52}")
    for ta, sa, tb, sb, slot in FIRST_FOUR:
        p = prob_cache.get((ta, tb), 0.5)
        print(f"  {ta} (#{sa}) vs {tb} (#{sb})  →  {p:.1%} / {1-p:.1%}")

    # ── Steps 2-3: Monte Carlo + EFP ─────────────────────────────────────────
    print(f"\nRunning {args.sims:,} tournament simulations...")
    stats = run_monte_carlo(args.sims, prob_cache, kp_index)

    # ── Step 4: Auction values ────────────────────────────────────────────────
    print(f"\nComputing market-aware auction values (blend={args.blend:.0%} market / {1-args.blend:.0%} model)...")
    pricing = compute_market_aware_values(stats, args.budget, blend=args.blend)

    df = build_output_df(stats, pricing, momentum_flags)

    print_table(df, args.budget, args.sims, args.blend)

    lineups = optimize_lineup(df, budget=args.budget, n_teams=args.n_teams)
    print_lineups(lineups)

    # ── Step 5: Change report ─────────────────────────────────────────────────
    prev_path = PRED_DIR / "auction_values_2026_previous.csv"
    detect_and_print_changes(df, prev_path)

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    csv_path  = PRED_DIR / "auction_values_2026.csv"
    json_path = PRED_DIR / "optimal_lineup_200.json"
    txt_path  = REPORTS_DIR / "auction_draft_cheatsheet.txt"

    df.to_csv(csv_path, index=False)
    print(f"  Rankings CSV  → {csv_path.relative_to(ROOT)}")

    json_path.write_text(json.dumps(lineups, indent=2))
    print(f"  Lineup JSON   → {json_path.relative_to(ROOT)}")

    write_cheatsheet(df, lineups, args.budget, model_name, txt_path)


if __name__ == "__main__":
    main()
