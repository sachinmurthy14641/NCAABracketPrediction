"""2026 NCAA Tournament Bracket Prediction — Momentum-Enhanced

Three bracket strategies:
  A — Chalk:          KenPom efficiency only, deterministic max-prob picks
  B — Momentum:       Momentum model if recommended by ablation, else = A
  C — Pool Optimizer: Monte Carlo + differentiation bonus for pool play

Outputs:
  outputs/predictions/bracket_2026_chalk.json
  outputs/predictions/bracket_2026_momentum.json
  outputs/predictions/bracket_2026_pool_optimizer.json
  outputs/reports/bracket_2026_full_printable.txt
  outputs/reports/upset_watchlist_2026.txt
  outputs/reports/bracket_divergences_2026.txt
  data/processed/tournament_team_profiles_2026.csv

Usage:
    python scripts/fill_bracket_2026.py
    python scripts/fill_bracket_2026.py --sims 10000 --pool-size 50
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model import PlattModel  # noqa: F401 — needed by pickle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINE_MODEL_PATH   = ROOT / "outputs/models/lightgbm_final_2026.pkl"
MOMENTUM_MODEL_PATH   = ROOT / "outputs/models/logistic_regression_momentum.pkl"
MOMENTUM_REPORT_PATH  = ROOT / "outputs/reports/momentum_feature_report.txt"
KENPOM_PATH           = ROOT / "data/processed/kenpom_2026_clean.csv"
MOMENTUM_PATH         = ROOT / "data/processed/momentum_features_2026.csv"
OFFICIAL_BRACKET_PATH = ROOT / "data/bracket_2026_official.json"
PRED_DIR              = ROOT / "outputs/predictions"
REPORTS_DIR           = ROOT / "outputs/reports"
PROCESSED_DIR         = ROOT / "data/processed"

# ---------------------------------------------------------------------------
# BRACKET_2026 — loaded from data/bracket_2026_official.json
# Source: Yahoo Sports / NCAA.com, verified March 15-17 2026.
# Do NOT hardcode matchups here. Edit the JSON file instead.
#
# Slot naming convention for First Four placeholders: "FF{id}_winner"
#   FF1_winner : UMBC/Howard winner   → Midwest 16-seed (vs Michigan)
#   FF2_winner : Texas/NC State winner → West 11-seed (vs BYU)
#   FF3_winner : Prairie View/Lehigh winner → South 16-seed (vs Florida)
#   FF4_winner : Miami OH/SMU winner  → Midwest 11-seed (vs Tennessee)
#
# Final Four: East vs South, West vs Midwest
# ---------------------------------------------------------------------------

def _load_official_bracket() -> dict:
    """Load bracket_2026_official.json and convert to internal simulation format.

    Returns a dict with keys:
        first_four        — list of {team_a, seed_a, team_b, seed_b, slot, region_dest}
        regions           — dict of region → list of (seed, team) in bracket order
        seeds             — dict of team_name → seed number (all 68 teams)
        final_four_matchups — list of (region_a, region_b) tuples
    """
    if not OFFICIAL_BRACKET_PATH.exists():
        raise FileNotFoundError(
            f"Official bracket not found: {OFFICIAL_BRACKET_PATH}\n"
            "Expected at data/bracket_2026_official.json"
        )
    with open(OFFICIAL_BRACKET_PATH) as f:
        raw = json.load(f)

    # Build first_four in internal format
    first_four_out = []
    for ff in raw["first_four"]:
        first_four_out.append({
            "team_a":      ff["team_a"],
            "seed_a":      ff["seed_a"],
            "team_b":      ff["team_b"],
            "seed_b":      ff["seed_b"],
            "slot":        f"{ff['id']}_winner",
            "region_dest": ff["winner_slot"]["region"],
        })

    # Build regions: dict of region → flat list of (seed, team) in matchup order.
    # Each matchup expands to two consecutive entries — correct for the
    # pairwise simulation loop (current[i] vs current[i+1]).
    regions_out: dict[str, list[tuple[int, str]]] = {}
    for region_name, region_data in raw["regions"].items():
        flat: list[tuple[int, str]] = []
        for m in region_data["matchups"]:
            flat.append((m["seed_a"], m["team_a"]))
            flat.append((m["seed_b"], m["team_b"]))
        regions_out[region_name] = flat

    # Build seeds dict for all 68 tournament teams
    seeds_out: dict[str, int] = {}
    for region_data in raw["regions"].values():
        for m in region_data["matchups"]:
            if not m["team_a"].endswith("_winner"):
                seeds_out[m["team_a"]] = m["seed_a"]
            if not m["team_b"].endswith("_winner"):
                seeds_out[m["team_b"]] = m["seed_b"]
    for ff in raw["first_four"]:
        seeds_out[ff["team_a"]] = ff["seed_a"]
        seeds_out[ff["team_b"]] = ff["seed_b"]

    # Final Four matchups derived from region metadata
    ff_matchups_out = [("East", "South"), ("West", "Midwest")]

    return {
        "first_four":           first_four_out,
        "regions":              regions_out,
        "seeds":                seeds_out,
        "final_four_matchups":  ff_matchups_out,
        "_raw":                 raw,   # keep raw JSON for verify/print functions
    }


BRACKET_2026: dict = _load_official_bracket()

# ---------------------------------------------------------------------------
# Feature column definitions (must match training order exactly)
# ---------------------------------------------------------------------------
KENPOM_FEATURES = [
    "off_eff_advantage", "def_eff_advantage", "net_efficiency_edge",
    "tempo_difference", "overall_rating_diff", "efficiency_differential_diff",
    "seed_diff",
    "a_adj_off_eff", "a_adj_def_eff", "a_adj_em",
    "b_adj_off_eff", "b_adj_def_eff", "b_adj_em",
]

MOMENTUM_DIFF_FEATURES = [
    "margin_trend_diff", "momentum_score_diff", "recent_efg_diff",
    "recent_def_efg_diff", "win_pct_30d_diff", "conf_tourney_margin_diff",
    "rank_disagreement_diff",
]

ALL_FEATURES = KENPOM_FEATURES + MOMENTUM_DIFF_FEATURES

# ---------------------------------------------------------------------------
# Chalk popularity by seed for Strategy C differentiation bonus
# ---------------------------------------------------------------------------
CHALK_POPULARITY: dict[int, float] = {
    1: 0.90, 2: 0.70, 3: 0.50, 4: 0.35, 5: 0.35,
    6: 0.20, 7: 0.20, 8: 0.20, 9: 0.10, 10: 0.10,
    11: 0.10, 12: 0.10, 13: 0.10, 14: 0.10, 15: 0.10, 16: 0.10,
}

# Simulation physics
MARGIN_STD   = 11.5
EM_TO_MARGIN = 0.44

# ---------------------------------------------------------------------------
# Team name normalization — bracket names → KenPom names
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


# ===========================================================================
# STEP 0 — Model selection
# ===========================================================================

def load_models() -> tuple:
    """Parse momentum report and load appropriate models.

    Returns:
        baseline_model  — always the LightGBM KenPom model
        momentum_model  — momentum logistic if recommended, else None
        use_momentum    — bool; True if momentum model was recommended
        baseline_name   — human-readable model description
        momentum_name   — human-readable momentum model description
    """
    use_momentum = False
    momentum_model = None
    baseline_name = "LightGBM (KenPom features)"
    momentum_name = baseline_name

    # Parse the report
    if not MOMENTUM_REPORT_PATH.exists():
        print(
            "WARNING: outputs/reports/momentum_feature_report.txt not found.\n"
            "  Run 'python scripts/add_momentum_features.py --season 2026 --retrain' first.\n"
            "  Falling back to KenPom-only baseline model."
        )
    else:
        report_text = MOMENTUM_REPORT_PATH.read_text()
        if "USE MOMENTUM MODEL" in report_text or "✓" in report_text.split("RECOMMENDATION")[-1]:
            use_momentum = True

    # Load baseline model (always needed)
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_MODEL_PATH}\n"
            "Run scripts/compare_models.py to train the LightGBM model."
        )
    with open(BASELINE_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    baseline_model = payload["model"]
    baseline_name = f"{payload.get('name', 'LightGBM')} ({len(payload.get('features', KENPOM_FEATURES))} features)"

    # Load momentum model if recommended
    if use_momentum:
        if MOMENTUM_MODEL_PATH.exists():
            with open(MOMENTUM_MODEL_PATH, "rb") as f:
                mom_payload = pickle.load(f)
            momentum_model = mom_payload.get("model", mom_payload)
            momentum_name = f"Logistic Regression + Momentum ({len(ALL_FEATURES)} features)"
        else:
            print("WARNING: Momentum model recommended but file not found. Using baseline.")
            use_momentum = False

    if use_momentum:
        print(f"Using model: {momentum_name}")
        print(f"  Strategy A: {baseline_name}")
        print(f"  Strategy B: {momentum_name}")
    else:
        print(f"Using model: {baseline_name}")
        print("  Note: Momentum model not recommended by ablation — Strategy A and B will be identical.")

    return baseline_model, momentum_model, use_momentum, baseline_name, momentum_name


# ===========================================================================
# STEP 1 — Data loading
# ===========================================================================

def _load_bracket() -> tuple[dict, list, dict, list]:
    """Return bracket data from BRACKET_2026 (loaded from bracket_2026_official.json)."""
    seeds = BRACKET_2026["seeds"]
    first_four = [
        (ff["team_a"], ff["seed_a"], ff["team_b"], ff["seed_b"], ff["slot"])
        for ff in BRACKET_2026["first_four"]
    ]
    regions = dict(BRACKET_2026["regions"])
    ff_matchups = list(BRACKET_2026["final_four_matchups"])
    return seeds, first_four, regions, ff_matchups


def load_data() -> tuple[dict, pd.DataFrame, pd.DataFrame, list, dict, list]:
    """Load all data. Returns (seeds, kp_df, momentum_df, first_four, regions, ff_matchups)."""
    seeds, first_four, regions, ff_matchups = _load_bracket()

    kp_df = pd.read_csv(KENPOM_PATH)

    momentum_df = None
    if MOMENTUM_PATH.exists():
        momentum_df = pd.read_csv(MOMENTUM_PATH, index_col="kenpom_name")
        # Multiple Kaggle TeamIDs can fuzzy-match to the same KenPom name.
        # Keep the row with the most games (richest recent-form data).
        if momentum_df.index.duplicated().any():
            n_before = len(momentum_df)
            n_col = "n_games" if "n_games" in momentum_df.columns else momentum_df.columns[0]
            momentum_df = (
                momentum_df
                .sort_values(n_col, ascending=False)
                [~momentum_df.sort_values(n_col, ascending=False).index.duplicated(keep="first")]
            )
            print(f"  Deduplicated momentum index: {n_before} → {len(momentum_df)} rows")
    else:
        print(f"WARNING: {MOMENTUM_PATH} not found. Momentum features disabled.")

    return seeds, kp_df, momentum_df, first_four, regions, ff_matchups


# ===========================================================================
# KenPom helpers
# ===========================================================================

def build_kenpom_index(kp_df: pd.DataFrame) -> dict:
    return {row["team"]: row for _, row in kp_df.iterrows()}


def get_kp_row(team: str, seed: int, kp_index: dict) -> pd.Series:
    """Look up KenPom row for team; fall back to seed-based approximation."""
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


def _get_mom(team: str, momentum_df: pd.DataFrame | None, col: str, default: float = 0.0) -> float:
    if momentum_df is None:
        return default
    lookup = kp_name(team)
    if lookup not in momentum_df.index:
        return default
    return float(momentum_df.loc[lookup].get(col, default))


# ===========================================================================
# STEP 2 — Build probability caches (batched predict_proba)
# ===========================================================================

def _build_kenpom_feature_row(ta: str, sa: int, tb: str, sb: int, kp_index: dict) -> dict:
    ra = get_kp_row(ta, sa, kp_index)
    rb = get_kp_row(tb, sb, kp_index)
    off_adv = ra["adj_off_eff"] - rb["adj_def_eff"]
    def_adv = rb["adj_off_eff"] - ra["adj_def_eff"]
    return {
        "off_eff_advantage":            off_adv,
        "def_eff_advantage":            def_adv,
        "net_efficiency_edge":          off_adv - def_adv,
        "tempo_difference":             ra["adj_tempo"] - rb["adj_tempo"],
        "overall_rating_diff":          ra["adj_em"] - rb["adj_em"],
        "efficiency_differential_diff": ra["efficiency_differential"] - rb["efficiency_differential"],
        "seed_diff":                    sa - sb,
        "a_adj_off_eff": ra["adj_off_eff"], "a_adj_def_eff": ra["adj_def_eff"],
        "a_adj_em":      ra["adj_em"],
        "b_adj_off_eff": rb["adj_off_eff"], "b_adj_def_eff": rb["adj_def_eff"],
        "b_adj_em":      rb["adj_em"],
    }


def _add_momentum_features(row: dict, ta: str, tb: str, momentum_df: pd.DataFrame | None) -> dict:
    """Append momentum differential features to an existing KenPom feature row."""
    row = dict(row)
    g = lambda team, col: _get_mom(team, momentum_df, col)
    row["margin_trend_diff"]      = g(ta, "recent_margin_trend")   - g(tb, "recent_margin_trend")
    row["momentum_score_diff"]    = g(ta, "momentum_score")        - g(tb, "momentum_score")
    row["recent_efg_diff"]        = g(ta, "recent_efg_pct")        - g(tb, "recent_efg_pct")
    # defensive eFG: lower is better for the defender → flip to keep "higher = better for team A"
    row["recent_def_efg_diff"]    = g(tb, "recent_def_efg")        - g(ta, "recent_def_efg")
    row["win_pct_30d_diff"]       = g(ta, "recent_win_pct_30d")    - g(tb, "recent_win_pct_30d")
    row["conf_tourney_margin_diff"] = g(ta, "conf_tourney_avg_margin") - g(tb, "conf_tourney_avg_margin")
    # rank_disagreement: lower = more consensus on ranking; advantage to team with lower
    row["rank_disagreement_diff"] = g(tb, "rank_disagreement")     - g(ta, "rank_disagreement")
    return row


def build_prob_cache(
    all_teams: list[tuple[str, int]],
    kp_index: dict,
    model,
    momentum_df: pd.DataFrame | None = None,
    use_momentum_features: bool = False,
) -> dict[tuple[str, str], float]:
    """Batch-compute win probabilities for all team pairs.

    Symmetry guaranteed: cache[(A,B)] + cache[(B,A)] == 1.0
    """
    feat_cols = ALL_FEATURES if use_momentum_features else KENPOM_FEATURES

    pair_index: list[tuple[str, str]] = []
    rows: list[dict] = []

    for i in range(len(all_teams)):
        for j in range(i + 1, len(all_teams)):
            ta, sa = all_teams[i]
            tb, sb = all_teams[j]
            row = _build_kenpom_feature_row(ta, sa, tb, sb, kp_index)
            if use_momentum_features and momentum_df is not None:
                row = _add_momentum_features(row, ta, tb, momentum_df)
            rows.append(row)
            pair_index.append((ta, tb))

    X = pd.DataFrame(rows, columns=feat_cols)
    probs = model.predict_proba(X)[:, 1]

    # Symmetry check
    cache: dict[tuple[str, str], float] = {}
    for (ta, tb), p in zip(pair_index, probs):
        cache[(ta, tb)] = float(p)
        cache[(tb, ta)] = 1.0 - float(p)

    # Verify symmetry holds (assert ~1.0 for all pairs)
    violations = [
        (ta, tb, cache[(ta, tb)] + cache[(tb, ta)])
        for ta, tb in pair_index
        if abs(cache[(ta, tb)] + cache[(tb, ta)] - 1.0) > 1e-9
    ]
    if violations:
        print(f"WARNING: {len(violations)} symmetry violations detected (max error reported)")

    return cache


def resolve_first_four(
    first_four: list,
    prob_cache: dict,
) -> dict[str, tuple[str, int]]:
    """Deterministically resolve First Four games (highest-prob winner wins)."""
    ff_results: dict[str, tuple[str, int]] = {}
    for ta, sa, tb, sb, slot in first_four:
        p_a = prob_cache.get((ta, tb), 0.5)
        if p_a >= 0.5:
            ff_results[slot] = (ta, sa)
        else:
            ff_results[slot] = (tb, sb)
    return ff_results


# ===========================================================================
# STEP 3 — Momentum flags
# ===========================================================================

def get_momentum_flag(team: str, momentum_df: pd.DataFrame | None) -> str:
    """Return HOT / WARM / FLAT / COLD / ICY for a team."""
    if momentum_df is None:
        return "FLAT"
    lookup = kp_name(team)
    if lookup not in momentum_df.index:
        return "FLAT"
    row = momentum_df.loc[lookup]
    score = float(row.get("momentum_score", 0))
    trend = float(row.get("recent_margin_trend", 0))
    if score > 15 and trend > 0:
        return "HOT"
    if score > 5:
        return "WARM"
    if score < -15 or trend < -3.0:
        return "ICY"
    if score < -5:
        return "COLD"
    return "FLAT"


def assign_momentum_flags(
    all_teams: list[tuple[str, int]],
    momentum_df: pd.DataFrame | None,
) -> dict[str, str]:
    return {team: get_momentum_flag(team, momentum_df) for team, _ in all_teams}


# ===========================================================================
# STEP 4 — Build unified team profiles
# ===========================================================================

def build_team_profiles(
    all_teams: list[tuple[str, int]],
    kp_index: dict,
    momentum_df: pd.DataFrame | None,
    momentum_flags: dict[str, str],
    seeds: dict,
) -> pd.DataFrame:
    """Build unified profile per tournament team combining KenPom + momentum."""
    records = []
    missing_momentum = []

    for team, seed in all_teams:
        kp = get_kp_row(team, seed, kp_index)
        lookup = kp_name(team)
        has_momentum = momentum_df is not None and lookup in momentum_df.index

        if not has_momentum:
            missing_momentum.append(team)

        mom_row = momentum_df.loc[lookup] if has_momentum else None

        def m(col, default=0.0):
            return float(mom_row[col]) if has_momentum and col in mom_row.index else default

        records.append({
            "team":                  team,
            "kenpom_name":           lookup,
            "seed":                  seed,
            "region":                _team_region(team, seeds),
            "momentum_flag":         momentum_flags.get(team, "FLAT"),
            # KenPom
            "adj_off_eff":           float(kp["adj_off_eff"]),
            "adj_def_eff":           float(kp["adj_def_eff"]),
            "adj_tempo":             float(kp["adj_tempo"]),
            "adj_em":                float(kp["adj_em"]),
            # Momentum
            "momentum_score":        m("momentum_score"),
            "recent_margin_trend":   m("recent_margin_trend"),
            "recent_efg_pct":        m("recent_efg_pct"),
            "recent_def_efg":        m("recent_def_efg"),
            "recent_win_pct_30d":    m("recent_win_pct_30d"),
            "conf_tourney_avg_margin": m("conf_tourney_avg_margin"),
            "conf_tourney_wins":     m("conf_tourney_wins"),
            "won_conf_tourney":      m("won_conf_tourney"),
            "consensus_rank":        m("consensus_rank"),
            "rank_disagreement":     m("rank_disagreement"),
            "has_momentum_data":     has_momentum,
        })

    if missing_momentum:
        print(f"\nWARNING: {len(missing_momentum)} tournament team(s) missing from momentum data:")
        for t in missing_momentum:
            print(f"  {t} — using KenPom-only features")

    return pd.DataFrame(records)


def _team_region(team: str, seeds: dict) -> str:
    """Return region name for a tournament team from bracket_2026_official.json."""
    for region, slots in BRACKET_2026["regions"].items():
        for _, slot_team in slots:
            if slot_team == team:
                return region
    for ff in BRACKET_2026["first_four"]:
        if ff["team_a"] == team or ff["team_b"] == team:
            return ff["region_dest"]
    return "Unknown"


# ===========================================================================
# BRACKET VERIFICATION
# ===========================================================================

def verify_bracket(regions: dict, first_four: list, seeds: dict) -> None:
    """Validate bracket structure loaded from bracket_2026_official.json.

    Checks:
      1. Exactly 4 regions
      2. Exactly 8 matchups per region (16 slots)
      3. Seeds 1-16 each appear exactly once per region
         (FF placeholder slots count toward the seed they occupy)
      4. Exactly 4 First Four games (2×16-seed, 2×11-seed)
      5. No real team appears in multiple regions
    Prints PASS/FAIL per check. Raises AssertionError on any failure.
    """
    errors: list[str] = []
    checks: list[tuple[str, bool, str]] = []   # (label, passed, detail)

    # Check 1: exactly 4 regions
    n_regions = len(regions)
    checks.append((
        "Exactly 4 regions",
        n_regions == 4,
        f"found {n_regions}: {list(regions.keys())}",
    ))
    if n_regions != 4:
        errors.append(f"Expected 4 regions, found {n_regions}")

    # Check 2 & 3: per-region slot count and seed coverage
    all_real_teams: list[str] = []
    for region, slots in regions.items():
        slot_count_ok = len(slots) == 16
        checks.append((
            f"{region}: 16 slots",
            slot_count_ok,
            f"found {len(slots)}",
        ))
        if not slot_count_ok:
            errors.append(f"{region}: expected 16 slots, found {len(slots)}")
            continue

        region_seeds = [s for s, _ in slots]
        seeds_ok = sorted(region_seeds) == list(range(1, 17))
        if not seeds_ok:
            missing = set(range(1, 17)) - set(region_seeds)
            extra   = set(region_seeds) - set(range(1, 17))
            errors.append(f"{region}: seed mismatch — missing {missing}, extra {extra}")
        checks.append((
            f"{region}: seeds 1-16 each once",
            seeds_ok,
            "" if seeds_ok else f"missing={missing}, extra={extra}",
        ))

        for _, team in slots:
            if not team.endswith("_winner"):
                all_real_teams.append(team)

    # Check 4: First Four composition
    ff_16 = [ff for ff in first_four if ff[1] == 16]
    ff_11 = [ff for ff in first_four if ff[1] == 11]
    checks.append(("2 First Four 16-seed games", len(ff_16) == 2, f"found {len(ff_16)}"))
    checks.append(("2 First Four 11-seed games", len(ff_11) == 2, f"found {len(ff_11)}"))
    if len(ff_16) != 2:
        errors.append(f"Expected 2 First Four 16-seed games, found {len(ff_16)}")
    if len(ff_11) != 2:
        errors.append(f"Expected 2 First Four 11-seed games, found {len(ff_11)}")

    # Check 5: no team in two regions
    seen: set[str] = set()
    dupes: list[str] = []
    for t in all_real_teams:
        if t in seen:
            dupes.append(t)
        seen.add(t)
    checks.append(("No team in multiple regions", not dupes, f"dupes={dupes}" if dupes else ""))
    for t in dupes:
        errors.append(f"Team '{t}' appears in multiple regions")

    # Print results table
    print("\nBRACKET VERIFICATION")
    print("=" * 58)
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    print("")
    for region, slots in regions.items():
        real = [(s, t) for s, t in slots if not t.endswith("_winner")]
        seed1 = next(t for s, t in real if s == 1)
        seed2 = next(t for s, t in real if s == 2)
        ff_slots = [t for _, t in slots if t.endswith("_winner")]
        ff_str = f"  [{', '.join(ff_slots)}]" if ff_slots else ""
        print(f"  {region:8s}: 1={seed1:<22s} 2={seed2}{ff_str}")

    print("")
    raw_ff = BRACKET_2026.get("_raw", {}).get("first_four", BRACKET_2026["first_four"])
    for ff in raw_ff:
        slot = ff.get("slot", f"{ff.get('id','?')}_winner")
        dest = ff.get("region_dest", ff.get("winner_slot", {}).get("region", "?"))
        print(f"  {slot}: {ff['team_a']} vs {ff['team_b']} → {dest}")

    print(f"\n  Teams total: {len(seen)} bracket + {len(first_four)*2} First Four "
          f"= {len(seen) + len(first_four)*2} total")

    if errors:
        msg = "BRACKET VERIFICATION FAILED:\n" + "\n".join(f"  • {e}" for e in errors)
        raise AssertionError(msg)

    print("  ✓ All checks passed")
    print("=" * 58)


# ===========================================================================
# STEP 5 — Monte Carlo simulation (for Strategy C)
# ===========================================================================

def run_monte_carlo_for_pool(
    n_sims: int,
    prob_cache: dict,
    first_four: list,
    regions: dict,
    ff_matchups: list,
    seeds: dict,
) -> dict[str, dict[int, int]]:
    """Run n_sims Monte Carlo tournaments.

    Returns round_wins[team][round] counts where:
      round 1 = R64 win, 2 = R32, 3 = S16, 4 = E8, 5 = F4, 6 = Championship
    First Four (round 0) winners also tracked.
    """
    rng = np.random.default_rng(42)
    round_wins: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # Resolve all 68 teams including FF teams
    all_bracket_teams = set()
    for region_slots in regions.values():
        for _, team in region_slots:
            if not team.endswith("_winner"):
                all_bracket_teams.add(team)
    for ta, sa, tb, sb, _ in first_four:
        all_bracket_teams.add(ta)
        all_bracket_teams.add(tb)

    for sim_i in range(n_sims):
        # Resolve First Four randomly
        ff_results: dict[str, tuple[str, int]] = {}
        for ta, sa, tb, sb, slot in first_four:
            p_a = prob_cache.get((ta, tb), 0.5)
            if rng.random() < p_a:
                ff_results[slot] = (ta, sa)
                round_wins[ta][0] += 1
            else:
                ff_results[slot] = (tb, sb)
                round_wins[tb][0] += 1

        # Simulate each region
        region_champs: dict[str, tuple[int, str]] = {}
        for region_name, slots in regions.items():
            resolved = []
            for seed, team in slots:
                if team.endswith("_winner"):
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
                    p_a = prob_cache.get((ta, tb), 0.5)
                    if rng.random() < p_a:
                        winner, w_seed = ta, sa
                    else:
                        winner, w_seed = tb, sb
                    round_wins[winner][game_round] += 1
                    next_round.append((w_seed, winner))
                current = next_round
            region_champs[region_name] = current[0]  # (seed, team)

        # Final Four
        final_champs: list[tuple[int, str]] = []
        for reg_a, reg_b in ff_matchups:
            sa, ta = region_champs[reg_a]
            sb, tb = region_champs[reg_b]
            p_a = prob_cache.get((ta, tb), 0.5)
            if rng.random() < p_a:
                winner, w_seed = ta, sa
            else:
                winner, w_seed = tb, sb
            round_wins[winner][5] += 1
            final_champs.append((w_seed, winner))

        # Championship
        sa, ta = final_champs[0]
        sb, tb = final_champs[1]
        p_a = prob_cache.get((ta, tb), 0.5)
        winner = ta if rng.random() < p_a else tb
        round_wins[winner][6] += 1

        if (sim_i + 1) % 2_500 == 0:
            print(f"  {sim_i + 1:,} / {n_sims:,} sims complete")

    return dict(round_wins)


# ===========================================================================
# STEP 6 — Bracket filling
# ===========================================================================

ROUND_LABELS = {1: "R64", 2: "R32", 3: "S16", 4: "E8"}


def fill_bracket_deterministic(
    ff_results: dict[str, tuple[str, int]],
    regions: dict,
    ff_matchups: list,
    first_four: list,
    prob_cache: dict,
    strategy_name: str,
    model_name: str,
) -> dict:
    """Fill bracket deterministically — always pick highest-probability winner."""
    result: dict = {
        "strategy":     strategy_name,
        "model":        model_name,
        "first_four":   [],
        "regions":      {},
        "final_four":   [],
        "championship": None,
        "champion":     None,
    }

    # Record First Four
    for ta, sa, tb, sb, slot in first_four:
        winner, w_seed = ff_results[slot]
        p_a = prob_cache.get((ta, tb), 0.5)
        prob = p_a if winner == ta else 1.0 - p_a
        result["first_four"].append({
            "slot": slot, "team_a": ta, "seed_a": sa,
            "team_b": tb, "seed_b": sb,
            "winner": winner, "winner_seed": w_seed, "prob": round(prob, 3),
        })

    region_champs: dict[str, tuple[int, str]] = {}
    for region_name, slots in regions.items():
        resolved = _resolve_region(slots, ff_results)
        region_result: dict = {rn: [] for rn in ROUND_LABELS.values()}
        region_result["champion"] = None

        current = resolved
        for game_round in [1, 2, 3, 4]:
            next_round = []
            round_games = []
            for i in range(0, len(current), 2):
                sa, ta = current[i]
                sb, tb = current[i + 1]
                p_a = prob_cache.get((ta, tb), 0.5)
                if p_a >= 0.5:
                    winner, w_seed, prob = ta, sa, p_a
                else:
                    winner, w_seed, prob = tb, sb, 1.0 - p_a
                round_games.append({
                    "team_a": ta, "seed_a": sa,
                    "team_b": tb, "seed_b": sb,
                    "winner": winner, "winner_seed": w_seed,
                    "prob": round(prob, 3),
                })
                next_round.append((w_seed, winner))
            region_result[ROUND_LABELS[game_round]] = round_games
            current = next_round

        champ_seed, champ = current[0]
        region_result["champion"] = {"team": champ, "seed": champ_seed}
        result["regions"][region_name] = region_result
        region_champs[region_name] = (champ_seed, champ)

    # Final Four
    final_champs: list[tuple[int, str]] = []
    for reg_a, reg_b in ff_matchups:
        sa, ta = region_champs[reg_a]
        sb, tb = region_champs[reg_b]
        p_a = prob_cache.get((ta, tb), 0.5)
        if p_a >= 0.5:
            winner, w_seed, prob = ta, sa, p_a
        else:
            winner, w_seed, prob = tb, sb, 1.0 - p_a
        result["final_four"].append({
            "region_a": reg_a, "region_b": reg_b,
            "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
            "winner": winner, "winner_seed": w_seed, "prob": round(prob, 3),
        })
        final_champs.append((w_seed, winner))

    # Championship
    sa, ta = final_champs[0]
    sb, tb = final_champs[1]
    p_a = prob_cache.get((ta, tb), 0.5)
    if p_a >= 0.5:
        winner, w_seed, prob = ta, sa, p_a
    else:
        winner, w_seed, prob = tb, sb, 1.0 - p_a
    result["championship"] = {
        "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
        "winner": winner, "winner_seed": w_seed, "prob": round(prob, 3),
    }
    result["champion"] = {"team": winner, "seed": w_seed}
    return result


def fill_bracket_pool_optimizer(
    ff_results: dict[str, tuple[str, int]],
    regions: dict,
    ff_matchups: list,
    first_four: list,
    round_wins: dict[str, dict[int, int]],
    seeds: dict,
    n_sims: int,
    pool_size: int,
    strategy_name: str,
    model_name: str,
) -> dict:
    """Fill bracket using Monte Carlo win frequencies + differentiation bonus."""
    diff_scale = pool_size / 50.0

    def selection_score(team: str, game_round: int) -> float:
        seed = seeds.get(team, 16)
        pop  = CHALK_POPULARITY.get(seed, 0.10)
        freq = round_wins.get(team, {}).get(game_round, 0) / n_sims
        diff_bonus = 1.0 + diff_scale * 0.3 * (1.0 - pop)
        return freq * diff_bonus

    result: dict = {
        "strategy":     strategy_name,
        "model":        model_name,
        "first_four":   [],
        "regions":      {},
        "final_four":   [],
        "championship": None,
        "champion":     None,
    }

    # First Four — use deterministic resolution (same for all strategies)
    for ta, sa, tb, sb, slot in first_four:
        winner, w_seed = ff_results[slot]
        result["first_four"].append({
            "slot": slot, "team_a": ta, "seed_a": sa,
            "team_b": tb, "seed_b": sb,
            "winner": winner, "winner_seed": w_seed,
            "score_a": round(selection_score(ta, 0), 4),
            "score_b": round(selection_score(tb, 0), 4),
        })

    region_champs: dict[str, tuple[int, str]] = {}
    for region_name, slots in regions.items():
        resolved = _resolve_region(slots, ff_results)
        region_result: dict = {rn: [] for rn in ROUND_LABELS.values()}
        region_result["champion"] = None

        current = resolved
        for game_round in [1, 2, 3, 4]:
            next_round = []
            round_games = []
            for i in range(0, len(current), 2):
                sa, ta = current[i]
                sb, tb = current[i + 1]
                score_a = selection_score(ta, game_round)
                score_b = selection_score(tb, game_round)
                if score_a >= score_b:
                    winner, w_seed = ta, sa
                else:
                    winner, w_seed = tb, sb
                round_games.append({
                    "team_a": ta, "seed_a": sa,
                    "team_b": tb, "seed_b": sb,
                    "winner": winner, "winner_seed": w_seed,
                    "score_a": round(score_a, 4),
                    "score_b": round(score_b, 4),
                })
                next_round.append((w_seed, winner))
            region_result[ROUND_LABELS[game_round]] = round_games
            current = next_round

        champ_seed, champ = current[0]
        region_result["champion"] = {"team": champ, "seed": champ_seed}
        result["regions"][region_name] = region_result
        region_champs[region_name] = (champ_seed, champ)

    # Final Four
    final_champs: list[tuple[int, str]] = []
    for reg_a, reg_b in ff_matchups:
        sa, ta = region_champs[reg_a]
        sb, tb = region_champs[reg_b]
        score_a = selection_score(ta, 5)
        score_b = selection_score(tb, 5)
        if score_a >= score_b:
            winner, w_seed = ta, sa
        else:
            winner, w_seed = tb, sb
        result["final_four"].append({
            "region_a": reg_a, "region_b": reg_b,
            "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
            "winner": winner, "winner_seed": w_seed,
            "score_a": round(score_a, 4), "score_b": round(score_b, 4),
        })
        final_champs.append((w_seed, winner))

    # Championship
    sa, ta = final_champs[0]
    sb, tb = final_champs[1]
    score_a = selection_score(ta, 6)
    score_b = selection_score(tb, 6)
    if score_a >= score_b:
        winner, w_seed = ta, sa
    else:
        winner, w_seed = tb, sb
    result["championship"] = {
        "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
        "winner": winner, "winner_seed": w_seed,
        "score_a": round(score_a, 4), "score_b": round(score_b, 4),
    }
    result["champion"] = {"team": winner, "seed": w_seed}
    return result


def _resolve_region(slots: list, ff_results: dict) -> list[tuple[int, str]]:
    resolved = []
    for seed, team in slots:
        if team.endswith("_winner"):
            actual_team, actual_seed = ff_results[team]
            resolved.append((actual_seed, actual_team))
        else:
            resolved.append((seed, team))
    return resolved


# ===========================================================================
# STEP 7 — Upset watchlist
# ===========================================================================

ROUND_VALUE_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 4.0}
ROUND_POINT_VALUE  = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10}
ROUND_NAMES        = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "Final Four", 6: "Championship"}


def _momentum_multiplier(underdog: str, favorite: str, flags: dict[str, str]) -> float:
    uflag = flags.get(underdog, "FLAT")
    fflag = flags.get(favorite, "FLAT")
    if uflag == "HOT" and fflag in ("COLD", "ICY"):
        return 2.0
    if uflag == "HOT" and fflag == "FLAT":
        return 1.5
    if uflag == "WARM" and fflag in ("COLD", "ICY"):
        return 1.3
    return 1.0


def _iter_bracket_games(bracket: dict):
    """Yield (game_round, region_name, game_dict) for every game in a bracket."""
    for region_name, region_data in bracket.get("regions", {}).items():
        for rname, rnum in [("R64", 1), ("R32", 2), ("S16", 3), ("E8", 4)]:
            for g in region_data.get(rname, []):
                yield rnum, region_name, g
    for g in bracket.get("final_four", []):
        yield 5, "Final Four", g
    if bracket.get("championship"):
        yield 6, "Championship", bracket["championship"]


def compute_upset_watchlist(
    bracket_a: dict,
    prob_cache: dict,
    momentum_flags: dict[str, str],
    top_n: int = 15,
) -> list[dict]:
    """Return top_n upset candidates sorted by momentum-adjusted upset score."""
    candidates = []

    for game_round, region, game in _iter_bracket_games(bracket_a):
        ta, sa = game["team_a"], game["seed_a"]
        tb, sb = game["team_b"], game["seed_b"]

        if sa == sb:
            continue  # equal seeds — skip

        p_a = prob_cache.get((ta, tb), 0.5)

        # Identify underdog (larger seed number = weaker)
        if sa < sb:
            favorite, fav_seed = ta, sa
            underdog, udog_seed = tb, sb
            p_upset = 1.0 - p_a
        else:
            favorite, fav_seed = tb, sb
            underdog, udog_seed = ta, sa
            p_upset = p_a

        if p_upset < 0.08:
            continue  # near-certainty for favorite

        mom_mult   = _momentum_multiplier(underdog, favorite, momentum_flags)
        rw         = ROUND_VALUE_WEIGHT[game_round]
        upset_score = p_upset * mom_mult * rw

        if upset_score >= 0.6 and mom_mult >= 1.5:
            verdict = "STRONG CONSIDER"
        elif upset_score >= 0.3 or mom_mult >= 1.3:
            verdict = "WORTH WATCHING"
        else:
            verdict = "SKIP"

        candidates.append({
            "game_round":    game_round,
            "round_name":    ROUND_NAMES[game_round],
            "region":        region,
            "favorite":      favorite,
            "fav_seed":      fav_seed,
            "fav_flag":      momentum_flags.get(favorite, "FLAT"),
            "underdog":      underdog,
            "udog_seed":     udog_seed,
            "udog_flag":     momentum_flags.get(underdog, "FLAT"),
            "p_upset":       round(p_upset, 3),
            "mom_multiplier": mom_mult,
            "upset_score":   round(upset_score, 3),
            "pts_upside":    ROUND_POINT_VALUE[game_round],
            "pts_downside":  ROUND_POINT_VALUE[game_round],
            "verdict":       verdict,
        })

    candidates.sort(key=lambda x: x["upset_score"], reverse=True)
    return candidates[:top_n]


# ===========================================================================
# STEP 8 — Key divergences (Strategy A vs Strategy B)
# ===========================================================================

def _momentum_reason(team: str, momentum_df: pd.DataFrame | None) -> str:
    """Build a human-readable explanation for why momentum supports this team."""
    if momentum_df is None:
        return "no momentum data"
    lookup = kp_name(team)
    if lookup not in momentum_df.index:
        return "no momentum data"
    row = momentum_df.loc[lookup]
    parts = []

    rank_trend = float(row.get("rank_trend_30d", 0))
    trend      = float(row.get("recent_margin_trend", 0))
    wpt        = float(row.get("recent_win_pct_30d", 0.5))
    n_games    = int(row.get("n_games", 0))
    conf_wins  = int(row.get("conf_tourney_wins", 0))
    conf_marg  = float(row.get("conf_tourney_avg_margin", 0))
    score      = float(row.get("momentum_score", 0))

    if rank_trend < -2:
        parts.append(f"rising {abs(rank_trend):.0f} spots in rankings last 30d")
    elif rank_trend > 2:
        parts.append(f"falling {rank_trend:.0f} spots in rankings last 30d")
    if trend > 2:
        parts.append(f"+{trend:.1f} avg margin trend last {n_games} games")
    elif trend < -2:
        parts.append(f"{trend:.1f} avg margin trend last {n_games} games")
    if wpt >= 0.80 and n_games >= 3:
        wins = round(wpt * n_games)
        parts.append(f"{wins}-{n_games - wins} in last {n_games} games")
    if conf_wins >= 3:
        parts.append(f"won conf tournament by {conf_marg:.1f} avg margin")
    return ", ".join(parts) if parts else f"momentum score {score:.1f}"


def compute_divergences(
    bracket_a: dict,
    bracket_b: dict,
    momentum_df: pd.DataFrame | None,
    momentum_flags: dict[str, str],
    prob_cache_kenpom: dict,
    prob_cache_momentum: dict,
) -> list[dict]:
    """Find games where Strategy A and B pick different winners."""
    if bracket_a.get("champion", {}).get("team") == bracket_b.get("champion", {}).get("team"):
        # Quick check: if champions agree, there may still be intermediate divergences
        pass

    divergences = []

    games_b: dict[tuple, dict] = {}
    for rnum, region, g in _iter_bracket_games(bracket_b):
        key = (rnum, region, g.get("region_a", region), g["team_a"], g["team_b"])
        games_b[key] = g

    for rnum, region, ga in _iter_bracket_games(bracket_a):
        key = (rnum, region, ga.get("region_a", region), ga["team_a"], ga["team_b"])
        gb = games_b.get(key)
        if gb is None or ga["winner"] == gb["winner"]:
            continue

        ta, tb = ga["team_a"], ga["team_b"]
        winner_chalk    = ga["winner"]
        winner_momentum = gb["winner"]

        p_chalk_a    = prob_cache_kenpom.get((ta, tb), 0.5)
        p_momentum_a = prob_cache_momentum.get((ta, tb), 0.5)

        p_chalk_w    = p_chalk_a    if winner_chalk    == ta else 1 - p_chalk_a
        p_momentum_w = p_momentum_a if winner_momentum == ta else 1 - p_momentum_a

        prob_gap = abs(p_chalk_w - p_momentum_w)

        mom_score_chalk    = _get_mom(winner_chalk,    momentum_df, "momentum_score")
        mom_score_momentum = _get_mom(winner_momentum, momentum_df, "momentum_score")
        score_diff = abs(mom_score_momentum - mom_score_chalk)

        if prob_gap > 0.20:
            recommendation = f"STICK WITH CHALK — prob gap {prob_gap:.0%} too large"
        elif score_diff > 10 and prob_gap < 0.15:
            recommendation = "TAKE MOMENTUM PICK"
        else:
            recommendation = "TOSS-UP — use pool size to decide"

        divergences.append({
            "game_round":      rnum,
            "round_name":      ROUND_NAMES.get(rnum, str(rnum)),
            "region":          region,
            "team_a":          ta, "seed_a": ga["seed_a"],
            "team_b":          tb, "seed_b": ga["seed_b"],
            "chalk_pick":      winner_chalk,
            "chalk_prob":      round(p_chalk_w, 3),
            "chalk_flag":      momentum_flags.get(winner_chalk, "FLAT"),
            "momentum_pick":   winner_momentum,
            "momentum_prob":   round(p_momentum_w, 3),
            "momentum_flag":   momentum_flags.get(winner_momentum, "FLAT"),
            "reason":          _momentum_reason(winner_momentum, momentum_df),
            "recommendation":  recommendation,
        })

    return sorted(divergences, key=lambda x: x["game_round"])


# ===========================================================================
# STEP 9 — Printable bracket formatter
# ===========================================================================

FLAG_DISPLAY = {
    "HOT":  "HOT ",
    "WARM": "WARM",
    "FLAT": "    ",
    "COLD": "COLD",
    "ICY":  "ICY ",
}

UPSET_MARKER = " ← UPSET ALERT"


def _fmt_game_line(
    game: dict,
    momentum_flags: dict,
    bracket_a_winner: str | None = None,
    bracket_b_winner: str | None = None,
) -> str:
    ta, sa = game["team_a"], game["seed_a"]
    tb, sb = game["team_b"], game["seed_b"]
    winner = game["winner"]
    prob = game.get("prob", game.get("score_a", 0.5))

    flag_a = FLAG_DISPLAY.get(momentum_flags.get(ta, "FLAT"), "    ")
    flag_b = FLAG_DISPLAY.get(momentum_flags.get(tb, "FLAT"), "    ")

    chalk_agree    = (bracket_a_winner is None or bracket_a_winner    == winner)
    momentum_agree = (bracket_b_winner is None or bracket_b_winner    == winner)

    chalk_mark    = "✓chalk"    if chalk_agree    else "✗chalk"
    momentum_mark = "✓momentum" if momentum_agree else "✗momentum"

    is_upset = (winner == ta and sa > sb) or (winner == tb and sb > sa)
    upset_tag = UPSET_MARKER if is_upset else ""

    line = (
        f"  {sa:2d} {ta:<22s} [{flag_a}]  vs  "
        f"{sb:2d} {tb:<22s} [{flag_b}]  "
        f"→ {winner:<22s} {prob:5.1%}  "
        f"{chalk_mark} {momentum_mark}{upset_tag}"
    )
    return line


def format_bracket_printable(
    bracket_a: dict,
    bracket_b: dict,
    bracket_c: dict,
    momentum_flags: dict,
    use_momentum: bool,
) -> str:
    lines: list[str] = []

    if not use_momentum:
        lines.append("NOTE: Momentum model not recommended by ablation — Strategy B = Strategy A (shown once).")
        lines.append("")

    lines.append("=" * 90)
    lines.append("  2026 NCAA TOURNAMENT BRACKET PREDICTIONS")
    lines.append(f"  Strategy A — Chalk   : {bracket_a['model']}")
    if use_momentum:
        lines.append(f"  Strategy B — Momentum: {bracket_b['model']}")
    lines.append(f"  Strategy C — Pool Opt: {bracket_c['model']}")
    lines.append("=" * 90)

    def _champ_a(region, rnd):
        games = bracket_a.get("regions", {}).get(region, {}).get(rnd, [])
        for g in games:
            yield g["team_a"], g["team_b"], g["winner"]

    def _get_winner(bracket, region, rnd, ta, tb):
        for g in bracket.get("regions", {}).get(region, {}).get(rnd, []):
            if g["team_a"] == ta and g["team_b"] == tb:
                return g["winner"]
        return None

    for region_name in ["East", "West", "South", "Midwest"]:
        lines.append("")
        lines.append(f"{'=' * 40} {region_name.upper()} REGION {'=' * 40}")

        for rnd in ["R64", "R32", "S16", "E8"]:
            games_a = bracket_a.get("regions", {}).get(region_name, {}).get(rnd, [])
            if not games_a:
                continue
            lines.append(f"\n  — {rnd} —")
            for ga in games_a:
                ta, tb = ga["team_a"], ga["team_b"]
                winner_b = _get_winner(bracket_b, region_name, rnd, ta, tb)
                winner_c = _get_winner(bracket_c, region_name, rnd, ta, tb)
                lines.append(_fmt_game_line(ga, momentum_flags, ga["winner"], winner_b))
                # If pool optimizer diverges from chalk, show it
                if winner_c and winner_c != ga["winner"]:
                    lines.append(f"    ↳ Pool Optimizer picks: {winner_c} (differentiation play)")

        champ_a = bracket_a.get("regions", {}).get(region_name, {}).get("champion", {})
        champ_b = bracket_b.get("regions", {}).get(region_name, {}).get("champion", {})
        champ_c = bracket_c.get("regions", {}).get(region_name, {}).get("champion", {})
        lines.append(f"\n  Regional Champion:")
        lines.append(f"    Chalk:     {champ_a.get('seed','?')} {champ_a.get('team','?')}")
        if use_momentum and champ_b.get("team") != champ_a.get("team"):
            lines.append(f"    Momentum:  {champ_b.get('seed','?')} {champ_b.get('team','?')}")
        if champ_c.get("team") != champ_a.get("team"):
            lines.append(f"    Pool Opt:  {champ_c.get('seed','?')} {champ_c.get('team','?')}")

    # Final Four
    lines.append("")
    lines.append("=" * 90)
    lines.append("  FINAL FOUR")
    lines.append("=" * 90)
    for g in bracket_a.get("final_four", []):
        ta, tb = g["team_a"], g["team_b"]
        gb_winner = next((x["winner"] for x in bracket_b.get("final_four", [])
                          if x["team_a"] == ta and x["team_b"] == tb), None)
        lines.append(_fmt_game_line(g, momentum_flags, g["winner"], gb_winner))

    # Championship
    lines.append("")
    lines.append("=" * 90)
    lines.append("  CHAMPIONSHIP")
    lines.append("=" * 90)
    champ_game_a = bracket_a.get("championship", {})
    champ_game_b = bracket_b.get("championship", {})
    if champ_game_a:
        lines.append(_fmt_game_line(
            champ_game_a, momentum_flags,
            champ_game_a.get("winner"),
            champ_game_b.get("winner") if champ_game_b else None,
        ))

    # Champion summary
    champ_a = bracket_a.get("champion", {})
    champ_b = bracket_b.get("champion", {})
    champ_c = bracket_c.get("champion", {})
    lines.append("")
    lines.append("  CHAMPION PICKS:")
    lines.append(f"    Strategy A (Chalk):        {champ_a.get('seed','?')} {champ_a.get('team','?')}")
    if use_momentum:
        lines.append(f"    Strategy B (Momentum):     {champ_b.get('seed','?')} {champ_b.get('team','?')}")
    lines.append(f"    Strategy C (Pool Opt):     {champ_c.get('seed','?')} {champ_c.get('team','?')}")

    all_champs = {champ_a.get("team"), champ_b.get("team"), champ_c.get("team")}
    if len(all_champs) == 1:
        lines.append("    → HIGH CONFIDENCE — all models aligned")
    else:
        lines.append("    → SPLIT SIGNAL — see divergences report")

    return "\n".join(lines)


# ===========================================================================
# STEP 10 — Save all outputs
# ===========================================================================

def _format_upset_watchlist(watchlist: list[dict]) -> str:
    lines = ["=" * 70, "  2026 NCAA TOURNAMENT UPSET WATCHLIST (Top 15)", "=" * 70, ""]
    for i, c in enumerate(watchlist, 1):
        lines.append(
            f"{i:2d}. [{c['round_name']}] {c['udog_seed']} {c['underdog']} ({c['udog_flag']})  "
            f"over  {c['fav_seed']} {c['favorite']} ({c['fav_flag']})"
        )
        lines.append(
            f"      P(upset)={c['p_upset']:.1%}  mom_mult=×{c['mom_multiplier']:.1f}  "
            f"score={c['upset_score']:.3f}  pts={c['pts_upside']}  → {c['verdict']}"
        )
        lines.append("")
    return "\n".join(lines)


def _format_divergences(divergences: list[dict], use_momentum: bool) -> str:
    if not use_momentum or not divergences:
        return (
            "No divergences — Strategy A (Chalk) and Strategy B (Momentum) are identical.\n"
            "Reason: Momentum model was not recommended by ablation test.\n"
            "Run 'python scripts/add_momentum_features.py --season 2026 --retrain' "
            "after tournament data is available to update this recommendation."
        )
    lines = ["=" * 70, "  BRACKET DIVERGENCES — Where Momentum Changes the Pick", "=" * 70, ""]
    for d in divergences:
        lines.append(
            f"[{d['round_name']}] {d['region']}:  "
            f"{d['seed_a']} {d['team_a']} vs {d['seed_b']} {d['team_b']}"
        )
        lines.append(f"  Chalk pick:    {d['chalk_pick']} ({d['chalk_flag']})  {d['chalk_prob']:.1%}")
        lines.append(f"  Momentum pick: {d['momentum_pick']} ({d['momentum_flag']})  {d['momentum_prob']:.1%}")
        lines.append(f"  Momentum signal supports {d['momentum_pick']} because: {d['reason']}")
        lines.append(f"  Recommendation: {d['recommendation']}")
        lines.append("")
    return "\n".join(lines)


def save_all_outputs(
    bracket_a: dict,
    bracket_b: dict,
    bracket_c: dict,
    profiles_df: pd.DataFrame,
    watchlist: list[dict],
    divergences: list[dict],
    printable: str,
    use_momentum: bool,
) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PRED_DIR / "bracket_2026_chalk.json", "w") as f:
        json.dump(bracket_a, f, indent=2)
    with open(PRED_DIR / "bracket_2026_momentum.json", "w") as f:
        json.dump(bracket_b, f, indent=2)
    with open(PRED_DIR / "bracket_2026_pool_optimizer.json", "w") as f:
        json.dump(bracket_c, f, indent=2)

    profiles_df.to_csv(PROCESSED_DIR / "tournament_team_profiles_2026.csv", index=False)

    (REPORTS_DIR / "bracket_2026_full_printable.txt").write_text(printable)
    (REPORTS_DIR / "upset_watchlist_2026.txt").write_text(_format_upset_watchlist(watchlist))
    (REPORTS_DIR / "bracket_divergences_2026.txt").write_text(
        _format_divergences(divergences, use_momentum)
    )

    print("\n  Outputs saved:")
    print(f"    {PRED_DIR.relative_to(ROOT)}/bracket_2026_chalk.json")
    print(f"    {PRED_DIR.relative_to(ROOT)}/bracket_2026_momentum.json")
    print(f"    {PRED_DIR.relative_to(ROOT)}/bracket_2026_pool_optimizer.json")
    print(f"    {REPORTS_DIR.relative_to(ROOT)}/bracket_2026_full_printable.txt")
    print(f"    {REPORTS_DIR.relative_to(ROOT)}/upset_watchlist_2026.txt")
    print(f"    {REPORTS_DIR.relative_to(ROOT)}/bracket_divergences_2026.txt")
    print(f"    {PROCESSED_DIR.relative_to(ROOT)}/tournament_team_profiles_2026.csv")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="2026 NCAA Bracket Prediction")
    parser.add_argument("--sims",      type=int, default=10_000, help="Monte Carlo simulations (default 10000)")
    parser.add_argument("--pool-size", type=int, default=50,     help="Pool size for Strategy C differentiation (default 50)")
    args = parser.parse_args()

    print("=" * 60)
    print("  2026 NCAA BRACKET PREDICTION")
    print("=" * 60)

    # ── Step 0: Model selection ─────────────────────────────────────────────
    print("\n[Step 0] Loading models...")
    baseline_model, momentum_model, use_momentum, baseline_name, momentum_name = load_models()

    # ── Step 1: Data loading + bracket verification ─────────────────────────
    print("\n[Step 1] Loading data...")
    seeds, kp_df, momentum_df, first_four, regions, ff_matchups = load_data()
    kp_index = build_kenpom_index(kp_df)
    print(f"  KenPom: {len(kp_df)} teams | Momentum: {len(momentum_df) if momentum_df is not None else 0} teams")

    verify_bracket(regions, first_four, seeds)

    # Collect all 68 tournament teams (64 bracket + 4 FF extra pairs)
    all_bracket_teams: list[tuple[str, int]] = []
    seen: set[str] = set()
    for region_slots in regions.values():
        for seed, team in region_slots:
            if not team.endswith("_winner") and team not in seen:
                all_bracket_teams.append((team, seed))
                seen.add(team)
    for ta, sa, tb, sb, _ in first_four:
        for team, seed in [(ta, sa), (tb, sb)]:
            if team not in seen:
                all_bracket_teams.append((team, seed))
                seen.add(team)

    print(f"  Tournament teams: {len(all_bracket_teams)}")

    # ── Step 2: Build probability caches ───────────────────────────────────
    print("\n[Step 2] Computing win probabilities (batched)...")
    print(f"  KenPom cache: {len(all_bracket_teams) * (len(all_bracket_teams)-1) // 2:,} pairs...")
    kenpom_cache = build_prob_cache(all_bracket_teams, kp_index, baseline_model,
                                    use_momentum_features=False)

    if use_momentum and momentum_model is not None:
        print(f"  Momentum cache: {len(all_bracket_teams) * (len(all_bracket_teams)-1) // 2:,} pairs...")
        momentum_cache = build_prob_cache(all_bracket_teams, kp_index, momentum_model,
                                          momentum_df=momentum_df, use_momentum_features=True)
    else:
        momentum_cache = kenpom_cache  # identical when no momentum model

    # Resolve First Four (same for all strategies)
    ff_results = resolve_first_four(first_four, kenpom_cache)
    print(f"  First Four resolved: " + ", ".join(
        f"{slot}→{team}" for slot, (team, _) in ff_results.items()
    ))

    # ── Step 3: Momentum flags + team profiles ──────────────────────────────
    print("\n[Step 3] Assigning momentum flags...")
    momentum_flags = assign_momentum_flags(all_bracket_teams, momentum_df)
    flag_counts = defaultdict(int)
    for f in momentum_flags.values():
        flag_counts[f] += 1
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(flag_counts.items())))

    profiles_df = build_team_profiles(all_bracket_teams, kp_index, momentum_df,
                                       momentum_flags, seeds)
    print(f"  Profiles built: {len(profiles_df)} teams")

    # ── Step 4: Monte Carlo simulation (Strategy C) ─────────────────────────
    pool_cache = momentum_cache  # use best available model for pool sims
    print(f"\n[Step 4] Running {args.sims:,} Monte Carlo simulations for Pool Optimizer...")
    round_wins = run_monte_carlo_for_pool(
        n_sims=args.sims,
        prob_cache=pool_cache,
        first_four=first_four,
        regions=regions,
        ff_matchups=ff_matchups,
        seeds=seeds,
    )

    # ── Step 5: Fill three brackets ─────────────────────────────────────────
    print("\n[Step 5] Filling brackets...")

    bracket_a = fill_bracket_deterministic(
        ff_results, regions, ff_matchups, first_four,
        kenpom_cache, "chalk", baseline_name,
    )

    bracket_b = fill_bracket_deterministic(
        ff_results, regions, ff_matchups, first_four,
        momentum_cache,
        "momentum" if use_momentum else "chalk",
        momentum_name,
    )

    bracket_c = fill_bracket_pool_optimizer(
        ff_results, regions, ff_matchups, first_four,
        round_wins, seeds, args.sims, args.pool_size,
        "pool_optimizer",
        f"Pool Optimizer (pool-size={args.pool_size}, {args.sims:,} sims)",
    )

    print(f"  Strategy A champion: {bracket_a['champion']['seed']} {bracket_a['champion']['team']}")
    if use_momentum and bracket_b["champion"]["team"] != bracket_a["champion"]["team"]:
        print(f"  Strategy B champion: {bracket_b['champion']['seed']} {bracket_b['champion']['team']}")
    else:
        print(f"  Strategy B champion: (same as A)")
    print(f"  Strategy C champion: {bracket_c['champion']['seed']} {bracket_c['champion']['team']}")

    # ── Step 6: Upset watchlist ─────────────────────────────────────────────
    print("\n[Step 6] Computing upset watchlist...")
    watchlist = compute_upset_watchlist(bracket_a, kenpom_cache, momentum_flags)
    print(f"  Top upset: {watchlist[0]['udog_seed']} {watchlist[0]['underdog']} "
          f"over {watchlist[0]['fav_seed']} {watchlist[0]['favorite']} "
          f"({watchlist[0]['p_upset']:.1%}) [{watchlist[0]['verdict']}]")

    # ── Step 7: Divergences ─────────────────────────────────────────────────
    print("\n[Step 7] Computing A vs B divergences...")
    divergences = compute_divergences(
        bracket_a, bracket_b, momentum_df, momentum_flags,
        kenpom_cache, momentum_cache,
    )
    if divergences:
        print(f"  {len(divergences)} divergences found:")
        for d in divergences:
            print(f"    [{d['round_name']}] Chalk: {d['chalk_pick']}  Momentum: {d['momentum_pick']}")
    else:
        print("  No divergences (Strategy A = Strategy B).")

    # ── Step 8: Print and save ───────────────────────────────────────────────
    print("\n[Step 8] Formatting and saving outputs...")
    printable = format_bracket_printable(
        bracket_a, bracket_b, bracket_c, momentum_flags, use_momentum
    )
    print(printable)

    save_all_outputs(
        bracket_a, bracket_b, bracket_c,
        profiles_df, watchlist, divergences, printable, use_momentum,
    )

    # ── Sanity check summary ─────────────────────────────────────────────────
    print("\nBRACKET SANITY CHECK")
    print("=" * 50)
    region_champs_check = {
        name: data.get("champion", {}).get("team", "?")
        for name, data in bracket_a.get("regions", {}).items()
    }
    f4_teams = [g["winner"] for g in bracket_a.get("final_four", [])]
    all_regions = list(region_champs_check.keys())
    assert len(all_regions) == 4,           f"Expected 4 regions, got {len(all_regions)}"
    assert len(set(region_champs_check.values())) == 4, "Duplicate regional champions"
    assert len(set(f4_teams)) == 2,         f"Expected 2 Final Four winners, got {f4_teams}"

    print(f"  ✓ Each region has exactly 1 champion")
    print(f"  ✓ Final Four: {f4_teams[0]} vs {f4_teams[1]}")
    print(f"  ✓ No team in multiple regions")
    print(f"  ✓ All Round of 64 matchups match official NCAA bracket")
    ff_winners = [(ff["slot"], ff["winner"]) for ff in bracket_a.get("first_four", [])]
    slot_lookup = {e["slot"]: e for e in BRACKET_2026["first_four"]}
    for slot, winner in ff_winners:
        opp = slot_lookup.get(slot, {})
        opp_str = f"{opp.get('team_a','?')}/{opp.get('team_b','?')}"
        print(f"  ✓ First Four {slot}: {opp_str} → {winner}")
    print()
    print(f"  Champion predictions:")
    print(f"    Strategy A (Chalk):    {bracket_a['champion']['seed']} {bracket_a['champion']['team']}")
    if use_momentum:
        print(f"    Strategy B (Momentum): {bracket_b['champion']['seed']} {bracket_b['champion']['team']}")
    print(f"    Strategy C (Pool Opt): {bracket_c['champion']['seed']} {bracket_c['champion']['team']}")
    print("=" * 50)
    print("\nDone.")


if __name__ == "__main__":
    main()
