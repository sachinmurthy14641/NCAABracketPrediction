"""2026 NCAA Tournament Auction Value Generator

Runs a Monte Carlo bracket simulation using the production LightGBM model
and 2026 KenPom efficiency data to compute expected fantasy points per team.
Scales results to a user-defined auction budget.

Scoring system:
  Round wins : First Four=1, R1=1, R2=2, Sweet16=3, Elite8=4, Final4=5, Champ=10
  Margin bonus (per win): 30+ pts margin=3, 20-29=2, 10-19=1
  Upset bonus            : +2 pts when lower seed (higher number) wins

Usage::

    python scripts/auction_values.py
    python scripts/auction_values.py --budget 200 --sims 50000
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import PlattModel  # noqa: F401 — needed for pickle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH  = Path("outputs/models/lightgbm_final_2026.pkl")
KENPOM_PATH = Path("data/processed/kenpom_2026_clean.csv")
OUTPUT_PATH = Path("outputs/reports/auction_values_2026.csv")

# ---------------------------------------------------------------------------
# Scoring system
# ---------------------------------------------------------------------------
# Round numbers: 0=First Four, 1=R1, 2=R2, 3=S16, 4=E8, 5=F4, 6=Champ
ROUND_LABEL = {0: "First Four", 1: "R1", 2: "R2", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}
ROUND_PTS   = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10}
MARGIN_BONUS = [(30, 3), (20, 2), (10, 1)]   # (threshold, bonus pts per win)
UPSET_BONUS  = 2                               # pts when lower seed (higher #) wins

# ---------------------------------------------------------------------------
# Simulation physics
# ---------------------------------------------------------------------------
MARGIN_STD   = 11.5    # historical NCAA tournament margin std dev (points)
EM_TO_MARGIN = 0.44    # adj_em difference → expected scoring margin

# ---------------------------------------------------------------------------
# Model features (must match training order exactly)
# ---------------------------------------------------------------------------
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
# Team name mapping: SEEDS_2026 names → KenPom 2026 names
# ---------------------------------------------------------------------------
TO_KENPOM = {
    "UConn":         "Connecticut",
    "Saint Mary's":  "St. Mary's",
    "Miami":         "Miami FL",
    "Miami OH":      "Miami OH",
    "Cal Baptist":   "Cal Baptist",
    "Prairie View":  "Prairie View A&M",
    "Iowa St.":      "Iowa St.",
    "Michigan St.":  "Michigan St.",
    "Ohio St.":      "Ohio St.",
    "Utah St.":      "Utah St.",
    "Texas A&M":     "Texas A&M",
    "North Carolina":"North Carolina",
    "North Dakota St.": "North Dakota St.",
    "Saint Louis":   "Saint Louis",
    "NC State":      "NC State",
    "Tennessee St.": "Tennessee St.",
    "Wright St.":    "Wright St.",
    "Kennesaw St.":  "Kennesaw St.",
    "Queens":        "Queens",
    "High Point":    "High Point",
    "Northern Iowa": "Northern Iowa",
}


def kp_name(name: str) -> str:
    return TO_KENPOM.get(name, name)


# ---------------------------------------------------------------------------
# 2026 Seeds
# ---------------------------------------------------------------------------
SEEDS = {
    'Duke': 1, 'Arizona': 1, 'Michigan': 1, 'Florida': 1,
    'Houston': 2, 'UConn': 2, 'Iowa St.': 2, 'Purdue': 2,
    'Michigan St.': 3, 'Illinois': 3, 'Gonzaga': 3, 'Virginia': 3,
    'Nebraska': 4, 'Alabama': 4, 'Kansas': 4, 'Arkansas': 4,
    'Vanderbilt': 5, "St. John's": 5, 'Texas Tech': 5, 'Wisconsin': 5,
    'Tennessee': 6, 'North Carolina': 6, 'Louisville': 6, 'BYU': 6,
    'Kentucky': 7, "Saint Mary's": 7, 'Miami': 7, 'UCLA': 7,
    'Clemson': 8, 'Villanova': 8, 'Ohio St.': 8, 'Georgia': 8,
    'Utah St.': 9, 'TCU': 9, 'Saint Louis': 9, 'Iowa': 9,
    'Santa Clara': 10, 'UCF': 10, 'Missouri': 10, 'Texas A&M': 10,
    'NC State': 11, 'Texas': 11, 'SMU': 11, 'Miami OH': 11,
    'VCU': 11, 'South Florida': 11,
    'McNeese': 12, 'Akron': 12, 'Northern Iowa': 12, 'High Point': 12,
    'Cal Baptist': 13, 'Hofstra': 13, 'Troy': 13, 'Hawaii': 13,
    'North Dakota St.': 14, 'Penn': 14, 'Wright St.': 14, 'Kennesaw St.': 14,
    'Tennessee St.': 15, 'Idaho': 15, 'Furman': 15, 'Queens': 15,
    'Siena': 16, 'LIU': 16, 'Howard': 16, 'UMBC': 16,
    'Lehigh': 16, 'Prairie View': 16,
}

# ---------------------------------------------------------------------------
# 2026 Bracket definition
# ---------------------------------------------------------------------------
# First Four games (round 0) — winners advance to the seed-11/16 slots in
# their assigned region.
# Format: (team_a, seed_a, team_b, seed_b, ff_slot_id)
FIRST_FOUR = [
    ("NC State",  11, "Texas",    11, "__FF11A__"),  # winner → East seed-11 slot
    ("SMU",       11, "Miami OH", 11, "__FF11B__"),  # winner → South seed-11 slot
    ("Siena",     16, "LIU",      16, "__FF16A__"),  # winner → East seed-16 slot
    ("Howard",    16, "UMBC",     16, "__FF16B__"),  # winner → South seed-16 slot
]

# 4 regions — each is a list of 16 (seed, team) in standard bracket order:
# 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
# Slots prefixed "__FF*__" are resolved from First Four winners at sim time.
REGIONS = {
    "East": [
        (1,  "Duke"),             (16, "__FF16A__"),
        (8,  "Clemson"),          (9,  "Utah St."),
        (5,  "Vanderbilt"),       (12, "McNeese"),
        (4,  "Nebraska"),         (13, "Cal Baptist"),
        (6,  "Tennessee"),        (11, "__FF11A__"),
        (3,  "Michigan St."),     (14, "North Dakota St."),
        (7,  "Kentucky"),         (10, "Santa Clara"),
        (2,  "Houston"),          (15, "Tennessee St."),
    ],
    "West": [
        (1,  "Arizona"),          (16, "Lehigh"),
        (8,  "Villanova"),        (9,  "TCU"),
        (5,  "St. John's"),       (12, "Akron"),
        (4,  "Alabama"),          (13, "Hofstra"),
        (6,  "North Carolina"),   (11, "VCU"),
        (3,  "Illinois"),         (14, "Penn"),
        (7,  "Saint Mary's"),     (10, "UCF"),
        (2,  "UConn"),            (15, "Idaho"),
    ],
    "South": [
        (1,  "Michigan"),         (16, "__FF16B__"),
        (8,  "Ohio St."),         (9,  "Saint Louis"),
        (5,  "Texas Tech"),       (12, "Northern Iowa"),
        (4,  "Kansas"),           (13, "Troy"),
        (6,  "Louisville"),       (11, "__FF11B__"),
        (3,  "Gonzaga"),          (14, "Wright St."),
        (7,  "Miami"),            (10, "Missouri"),
        (2,  "Iowa St."),         (15, "Furman"),
    ],
    "Midwest": [
        (1,  "Florida"),          (16, "Prairie View"),
        (8,  "Georgia"),          (9,  "Iowa"),
        (5,  "Wisconsin"),        (12, "High Point"),
        (4,  "Arkansas"),         (13, "Hawaii"),
        (6,  "BYU"),              (11, "South Florida"),
        (3,  "Virginia"),         (14, "Kennesaw St."),
        (7,  "UCLA"),             (10, "Texas A&M"),
        (2,  "Purdue"),           (15, "Queens"),
    ],
}

# Final Four cross: East vs West, South vs Midwest
FINAL_FOUR_MATCHUPS = [("East", "West"), ("South", "Midwest")]


# ---------------------------------------------------------------------------
# KenPom lookup with seed-based fallback
# ---------------------------------------------------------------------------

def build_kenpom_index(kp_df: pd.DataFrame) -> dict:
    """Build {kenpom_team_name: row} lookup."""
    return {row["team"]: row for _, row in kp_df.iterrows()}


def get_kp_row(team: str, seed: int, kp_index: dict) -> pd.Series:
    """Return KenPom row for team, falling back to a seed-based estimate."""
    lookup = kp_name(team)
    if lookup in kp_index:
        return kp_index[lookup]

    # Seed-based fallback (approximate power curve fit to historical adj_em by seed)
    adj_em   = max(-5, 35 - (seed - 1) * 2.3)
    adj_off  = 117 - (seed - 1) * 0.6
    adj_def  = 99  + (seed - 1) * 0.8
    adj_tmp  = 68.0
    eff_diff = adj_em
    return pd.Series({
        "team": team, "adj_em": adj_em, "adj_off_eff": adj_off,
        "adj_def_eff": adj_def, "adj_tempo": adj_tmp,
        "efficiency_differential": eff_diff,
    })


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(ra: pd.Series, rb: pd.Series, seed_a: int, seed_b: int) -> np.ndarray:
    off_adv = ra["adj_off_eff"] - rb["adj_def_eff"]
    def_adv = rb["adj_off_eff"] - ra["adj_def_eff"]
    return np.array([[
        off_adv,
        def_adv,
        off_adv - def_adv,
        ra["adj_tempo"] - rb["adj_tempo"],
        ra["adj_em"]    - rb["adj_em"],
        ra["efficiency_differential"] - rb["efficiency_differential"],
        seed_a - seed_b,
        ra["adj_off_eff"], ra["adj_def_eff"], ra["adj_em"],
        rb["adj_off_eff"], rb["adj_def_eff"], rb["adj_em"],
    ]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Pre-compute win probability cache
# ---------------------------------------------------------------------------

def precompute_probs(
    all_teams: list[tuple[str, int]],
    kp_index: dict,
    model,
) -> dict[tuple[str, str], float]:
    """Return {(team_a, team_b): P(team_a wins)} for every unique pair."""
    cache: dict[tuple[str, str], float] = {}
    teams = list(all_teams)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            ta, sa = teams[i]
            tb, sb = teams[j]
            ra = get_kp_row(ta, sa, kp_index)
            rb = get_kp_row(tb, sb, kp_index)
            feat = build_features(ra, rb, sa, sb)
            p = float(model.predict_proba(feat)[0, 1])
            cache[(ta, tb)] = p
            cache[(tb, ta)] = 1.0 - p
    return cache


# ---------------------------------------------------------------------------
# Single-game simulation
# ---------------------------------------------------------------------------

def sim_game(
    team_a: str, seed_a: int,
    team_b: str, seed_b: int,
    game_round: int,
    prob_cache: dict,
    kp_index: dict,
    rng: np.random.Generator,
) -> tuple[str, int, float]:
    """Simulate one game. Returns (winner, winner_seed, fantasy_pts_for_winner)."""
    p_a = prob_cache.get((team_a, team_b))
    if p_a is None:
        # Not pre-cached (shouldn't happen, but handle gracefully)
        ra = get_kp_row(team_a, seed_a, kp_index)
        rb = get_kp_row(team_b, seed_b, kp_index)
        feat = build_features(ra, rb, seed_a, seed_b)
        # model not available here — use seed-based prob as fallback
        seed_diff = seed_b - seed_a  # positive means a is favored by seeding
        p_a = 0.5 + seed_diff * 0.04  # crude linear seed fallback

    won_a = rng.random() < p_a

    if won_a:
        winner, winner_seed, loser_seed = team_a, seed_a, seed_b
        em_diff = (kp_index.get(kp_name(team_a), pd.Series({"adj_em": 0}))["adj_em"]
                   - kp_index.get(kp_name(team_b), pd.Series({"adj_em": 0}))["adj_em"])
    else:
        winner, winner_seed, loser_seed = team_b, seed_b, seed_a
        em_diff = (kp_index.get(kp_name(team_b), pd.Series({"adj_em": 0}))["adj_em"]
                   - kp_index.get(kp_name(team_a), pd.Series({"adj_em": 0}))["adj_em"])

    # Simulate scoring margin
    exp_margin = max(0.0, em_diff * EM_TO_MARGIN)
    margin     = max(1.0, float(rng.normal(exp_margin, MARGIN_STD)))

    # Calculate fantasy points for winner
    pts = ROUND_PTS[game_round]
    for threshold, bonus in MARGIN_BONUS:
        if margin >= threshold:
            pts += bonus
            break
    if winner_seed > loser_seed:   # upset: lower seed (higher number) won
        pts += UPSET_BONUS

    return winner, winner_seed, pts


# ---------------------------------------------------------------------------
# Region simulation (R1 → E8)
# ---------------------------------------------------------------------------

def sim_region(
    region_teams: list[tuple[int, str]],
    prob_cache: dict,
    kp_index: dict,
    rng: np.random.Generator,
) -> tuple[dict[str, float], str, int]:
    """
    Simulate all rounds of a single region.
    Returns (pts_dict, champion_name, champion_seed).
    """
    pts: dict[str, float] = defaultdict(float)
    current = list(region_teams)  # [(seed, team), ...] — 16 items

    for game_round in [1, 2, 3, 4]:   # R1=1, R2=2, S16=3, E8=4
        next_round = []
        for i in range(0, len(current), 2):
            seed_a, team_a = current[i]
            seed_b, team_b = current[i + 1]
            winner, winner_seed, game_pts = sim_game(
                team_a, seed_a, team_b, seed_b,
                game_round, prob_cache, kp_index, rng,
            )
            pts[winner] += game_pts
            next_round.append((winner_seed, winner))
        current = next_round

    champ_seed, champ = current[0]
    return dict(pts), champ, champ_seed


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def sim_tournament(
    ff_results: dict[str, tuple[str, int]],  # slot_id → (winner_team, winner_seed)
    prob_cache: dict,
    kp_index: dict,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Simulate one full tournament. Returns {team: total_fantasy_pts}."""
    total_pts: dict[str, float] = defaultdict(float)

    # ---- Regional rounds (R1 → E8) ----
    region_champs: dict[str, tuple[str, int]] = {}   # region → (team, seed)

    for region_name, slots in REGIONS.items():
        # Resolve First Four placeholders
        resolved = []
        for seed, team in slots:
            if team.startswith("__FF"):
                actual_team, actual_seed = ff_results[team]
                resolved.append((actual_seed, actual_team))
            else:
                resolved.append((seed, team))

        region_pts, champ, champ_seed = sim_region(
            resolved, prob_cache, kp_index, rng
        )
        for team, pts in region_pts.items():
            total_pts[team] += pts
        region_champs[region_name] = (champ, champ_seed)

    # ---- Final Four ----
    final_champs: list[tuple[str, int]] = []
    for reg_a, reg_b in FINAL_FOUR_MATCHUPS:
        team_a, seed_a = region_champs[reg_a]
        team_b, seed_b = region_champs[reg_b]
        winner, winner_seed, game_pts = sim_game(
            team_a, seed_a, team_b, seed_b,
            5, prob_cache, kp_index, rng,
        )
        total_pts[winner] += game_pts
        final_champs.append((winner, winner_seed))

    # ---- Championship ----
    (team_a, seed_a), (team_b, seed_b) = final_champs
    winner, _, game_pts = sim_game(
        team_a, seed_a, team_b, seed_b,
        6, prob_cache, kp_index, rng,
    )
    total_pts[winner] += game_pts

    return dict(total_pts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="2026 NCAA Auction Value Generator")
    p.add_argument("--budget", type=float, default=200.0, help="Total auction budget (default: 200)")
    p.add_argument("--sims",   type=int,   default=50_000, help="Monte Carlo iterations (default: 50000)")
    p.add_argument("--min-price", type=float, default=0.50, help="Minimum auction price (default: 0.50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR = OUTPUT_PATH.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]

    # ---- Load KenPom ----
    print(f"Loading KenPom: {KENPOM_PATH}")
    kp_df     = pd.read_csv(KENPOM_PATH)
    kp_index  = build_kenpom_index(kp_df)

    # ---- Collect all 68 teams ----
    all_teams: list[tuple[str, int]] = [(t, s) for t, s in SEEDS.items()]
    print(f"Teams in bracket: {len(all_teams)}")

    # ---- Pre-compute win probabilities for all pairs ----
    print(f"Pre-computing win probabilities for all team pairs...")
    prob_cache = precompute_probs(all_teams, kp_index, model)
    print(f"  Cached {len(prob_cache)} directed pair probabilities.")

    # ---- Pre-compute First Four expected outcomes ----
    # For Monte Carlo, we re-simulate FF each iteration;
    # but we also compute the FF winner probabilities for the display table.
    ff_probs: dict[str, dict[str, float]] = {}  # slot → {team: p_advance}
    for ta, sa, tb, sb, slot in FIRST_FOUR:
        p_a = prob_cache.get((ta, tb), 0.5)
        ff_probs[slot] = {ta: p_a, tb: 1 - p_a}

    # ---- Monte Carlo simulation ----
    print(f"\nRunning {args.sims:,} tournament simulations...")
    rng = np.random.default_rng(42)

    accumulated: dict[str, float] = defaultdict(float)
    rounds_won:  dict[str, list]  = defaultdict(list)  # track distribution

    for sim_i in range(args.sims):
        # Simulate First Four for this iteration
        ff_results: dict[str, tuple[str, int]] = {}
        for ta, sa, tb, sb, slot in FIRST_FOUR:
            p_a = prob_cache.get((ta, tb), 0.5)
            if rng.random() < p_a:
                ff_results[slot] = (ta, sa)
            else:
                ff_results[slot] = (tb, sb)

        # Add First Four points to accumulated totals
        for ta, sa, tb, sb, slot in FIRST_FOUR:
            winner, winner_seed = ff_results[slot]
            loser_seed          = sb if winner == ta else sa
            # simulate game margin for FF
            ra = get_kp_row(ta, sa, kp_index)
            rb = get_kp_row(tb, sb, kp_index)
            if winner == ta:
                em_diff = ra["adj_em"] - rb["adj_em"]
            else:
                em_diff = rb["adj_em"] - ra["adj_em"]
            exp_margin = max(0.0, em_diff * EM_TO_MARGIN)
            margin     = max(1.0, float(rng.normal(exp_margin, MARGIN_STD)))
            pts        = ROUND_PTS[0]
            for threshold, bonus in MARGIN_BONUS:
                if margin >= threshold:
                    pts += bonus
                    break
            if winner_seed > loser_seed:
                pts += UPSET_BONUS
            accumulated[winner] += pts

        # Simulate full tournament
        sim_pts = sim_tournament(ff_results, prob_cache, kp_index, rng)
        for team, pts in sim_pts.items():
            accumulated[team] += pts

        if (sim_i + 1) % 10_000 == 0:
            print(f"  {sim_i + 1:,} / {args.sims:,} sims complete")

    # ---- Compute expected points per team ----
    exp_pts: dict[str, float] = {
        team: accumulated[team] / args.sims for team in SEEDS
    }

    total_exp = sum(exp_pts.values())
    print(f"\nTotal expected fantasy points across all teams: {total_exp:.1f}")
    print(f"Budget: ${args.budget:.0f}")

    # ---- Scale to auction values ----
    min_price = args.min_price
    raw_values = {t: (ep / total_exp) * args.budget for t, ep in exp_pts.items()}

    # Apply minimum price floor and rescale
    floor_cost = sum(min_price for t, v in raw_values.items() if v < min_price)
    above_floor = {t: v for t, v in raw_values.items() if v >= min_price}
    scale = (args.budget - floor_cost) / sum(above_floor.values()) if above_floor else 1.0
    auction_values = {
        t: max(min_price, v * scale) for t, v in raw_values.items()
    }

    # ---- Build output table ----
    rows = []
    for team, seed in sorted(SEEDS.items(), key=lambda x: (-exp_pts.get(x[0], 0))):
        ep    = exp_pts.get(team, 0.0)
        val   = auction_values.get(team, min_price)

        # KenPom stats for context
        kp_row = get_kp_row(team, seed, kp_index)
        adj_em = kp_row["adj_em"]

        # Compute round-reach probabilities from probability cache
        # (approximated from seed-based historical rates weighted by adj_em)
        rows.append({
            "rank":       0,
            "team":       team,
            "seed":       seed,
            "adj_em":     round(adj_em, 1),
            "exp_pts":    round(ep, 2),
            "auction_$":  round(val, 2),
        })

    df = pd.DataFrame(rows).sort_values("exp_pts", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # ---- Print table ----
    print("\n" + "=" * 75)
    print(f"  2026 NCAA TOURNAMENT AUCTION VALUES  (budget=${args.budget:.0f}, sims={args.sims:,})")
    print("=" * 75)
    print(f"\n  {'Rank':<5} {'Team':<22} {'Seed':>5} {'AdjEM':>7} {'Exp Pts':>9} {'Auction $':>10}")
    print(f"  {'-' * 63}")

    for _, row in df.iterrows():
        tag = ""
        if row["seed"] <= 4:
            tag = "  ★★" if row["auction_$"] >= 20 else "  ★"
        print(
            f"  {int(row['rank']):<5} {row['team']:<22} #{int(row['seed']):<4} "
            f"{row['adj_em']:>7.1f} {row['exp_pts']:>9.2f} {row['auction_$']:>9.2f}{tag}"
        )
        # Insert visual separator between seed groups
        if int(row["rank"]) < len(df):
            next_seed = df.loc[df["rank"] == row["rank"] + 1, "seed"].values
            if len(next_seed) > 0 and next_seed[0] != row["seed"]:
                print(f"  {'·' * 63}")

    print(f"\n  Total auction values: ${df['auction_$'].sum():.2f}")
    print(f"  Highest value      : {df.iloc[0]['team']} (${df.iloc[0]['auction_$']:.2f})")
    print(f"  Lowest value       : {df.iloc[-1]['team']} (${df.iloc[-1]['auction_$']:.2f})")

    # ---- First Four breakdown ----
    print(f"\n  FIRST FOUR WIN PROBABILITIES:")
    print(f"  {'Matchup':<45} {'P(Team A wins)':>15}")
    print(f"  {'-' * 62}")
    for ta, sa, tb, sb, slot in FIRST_FOUR:
        p_a = prob_cache.get((ta, tb), 0.5)
        print(f"  {ta} (#{sa}) vs {tb} (#{sb})  →  {p_a:.1%} / {1-p_a:.1%}")

    # ---- Save CSV ----
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Full table saved → {OUTPUT_PATH}")

    # ---- Tiered summary ----
    print(f"\n{'=' * 75}")
    print(f"  AUCTION TIERS  (use these as bid guardrails)")
    print(f"{'=' * 75}")
    tiers = [
        ("Tier 1 — Championship contenders ($25+)",    lambda r: r["auction_$"] >= 25),
        ("Tier 2 — Sweet 16+ expected ($10–$24)",      lambda r: 10 <= r["auction_$"] < 25),
        ("Tier 3 — Value picks ($3–$9)",               lambda r: 3  <= r["auction_$"] < 10),
        ("Tier 4 — Dart throws ($1–$2)",               lambda r: 1  <= r["auction_$"] < 3),
        ("Tier 5 — Floor bids (<$1)",                  lambda r: r["auction_$"] < 1),
    ]
    for label, fn in tiers:
        subset = df[df.apply(fn, axis=1)]
        if subset.empty:
            continue
        print(f"\n  {label}")
        for _, row in subset.iterrows():
            print(f"    #{int(row['seed']):<3} {row['team']:<22}  ${row['auction_$']:.2f}  (exp {row['exp_pts']:.1f} pts)")

    print()


if __name__ == "__main__":
    main()
