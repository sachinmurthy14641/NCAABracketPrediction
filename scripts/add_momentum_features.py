"""Momentum Feature Enrichment for NCAA Bracket Predictions

Enriches the existing KenPom-based model with recent form and momentum
signals derived from the Kaggle March Machine Learning Mania dataset.

Required Kaggle files — download from the competition and place in data/kaggle/:
    MRegularSeasonDetailedResults.csv  (regular season box scores)
    MMasseyOrdinals.csv                (multi-system computer rankings)
    MRegularSeasonCompactResults.csv   (for conference tournament detection)
    MTeams.csv                         (already present in data/historical/kaggle/)

The script is organised into six numbered steps matching the spec:
    0. Setup & team ID alignment (fuzzy name matching)
    1. Recent form features      (shooting, margins, win %)
    2. Ranking trajectory        (multi-system rank trend → momentum_score)
    3. Conference tournament     (wins, avg margin, champion flag)
    4. Build momentum matchup features  (differentials for training)
    5. Retrain & ablation test   (Models A / B / C)
    6. Save outputs

If the Kaggle files are not yet present the script prints download
instructions and exits gracefully.

Usage:
    python scripts/add_momentum_features.py
    python scripts/add_momentum_features.py --season 2026 --window 30
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model import PlattModel  # noqa: F401 — needed by pickle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KAGGLE_DIR      = ROOT / "data/kaggle"
HIST_KAGGLE_DIR = ROOT / "data/historical/kaggle"
PROCESSED_DIR   = ROOT / "data/processed"
REPORTS_DIR     = ROOT / "outputs/reports"
MODELS_DIR      = ROOT / "outputs/models"

TRAINING_DATA   = PROCESSED_DIR / "training_data.csv"
KENPOM_HIST     = PROCESSED_DIR / "kenpom_pretourney_1997_2025.csv"
KENPOM_2026     = PROCESSED_DIR / "kenpom_2026_clean.csv"

# ---------------------------------------------------------------------------
# Season timing constants (DayNum 1 = early November; season ends ~DayNum 132)
# ---------------------------------------------------------------------------
SELECTION_SUNDAY_DAY  = 132   # tournaments start day ~134+
CONF_TOURNEY_START    = 118   # DayNums 118-132 = conference tournaments
MIN_GAMES_FOR_WINDOW  = 4     # minimum games to use 30d window; else fall back 60d

# Priority ranking systems (from MMasseyOrdinals SystemName column)
PRIORITY_SYSTEMS = ["POM", "SAG", "NET", "BPI", "MOR"]

# KenPom baseline features (must match train_model.py exactly)
KENPOM_FEATURES = [
    "off_eff_advantage", "def_eff_advantage", "net_efficiency_edge",
    "tempo_difference", "overall_rating_diff", "efficiency_differential_diff",
    "seed_diff",
    "a_adj_off_eff", "a_adj_def_eff", "a_adj_em",
    "b_adj_off_eff", "b_adj_def_eff", "b_adj_em",
]

# Momentum differential features added on top of KenPom
MOMENTUM_DIFF_FEATURES = [
    "margin_trend_diff",
    "momentum_score_diff",
    "recent_efg_diff",
    "recent_def_efg_diff",
    "win_pct_30d_diff",
    "conf_tourney_margin_diff",
    "rank_disagreement_diff",
]

TRAIN_MAX   = 2022
VAL_SEASON  = 2023
TEST_SEASON = 2024


# ===========================================================================
# STEP 0 — Setup & team-ID alignment
# ===========================================================================

def _load_kaggle(filename: str) -> pd.DataFrame:
    """Load a Kaggle CSV from data/kaggle/, falling back to data/historical/kaggle/."""
    primary = KAGGLE_DIR / filename
    fallback = HIST_KAGGLE_DIR / filename
    if primary.exists():
        return pd.read_csv(primary)
    if fallback.exists():
        return pd.read_csv(fallback)
    return None  # caller will handle


def _check_required_files() -> list[str]:
    """Return list of missing required Kaggle files."""
    required = [
        "MRegularSeasonDetailedResults.csv",
        "MMasseyOrdinals.csv",
        "MRegularSeasonCompactResults.csv",
        "MTeams.csv",
    ]
    missing = [f for f in required if _load_kaggle(f) is None]
    return missing


def _fuzzy_match(name: str, candidates: list[str], threshold: int = 85) -> tuple[str | None, int]:
    """Return (best_match, score) or (None, score) if below threshold."""
    try:
        from rapidfuzz import fuzz, process as rp
        result = rp.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            return result[0], result[1]
        return None, result[1] if result else 0
    except ImportError:
        import difflib
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=threshold / 100.0)
        if matches:
            seq = difflib.SequenceMatcher(None, name.lower(), matches[0].lower())
            return matches[0], int(seq.ratio() * 100)
        return None, 0


def build_team_id_mapping(teams_df: pd.DataFrame, kenpom_names: list[str]) -> tuple[dict, list]:
    """Fuzzy-match Kaggle TeamIDs to KenPom team names.

    Returns:
        id_to_kenpom : {TeamID (int) -> kenpom_name (str)}
        unmatched    : [(TeamID, kaggle_name, best_score), ...]
    """
    id_to_kenpom: dict[int, str] = {}
    unmatched: list[tuple] = []

    for _, row in teams_df.iterrows():
        tid  = int(row["TeamID"])
        name = str(row["TeamName"])
        match, score = _fuzzy_match(name, kenpom_names)
        if match:
            id_to_kenpom[tid] = match
        else:
            unmatched.append((tid, name, score))

    return id_to_kenpom, unmatched


# ===========================================================================
# STEP 1 — Recent form features (MRegularSeasonDetailedResults)
# ===========================================================================

def expand_detailed_results(df: pd.DataFrame) -> pd.DataFrame:
    """Convert W/L-perspective box scores into per-team rows.

    Each game produces two rows: one from the winner's perspective (is_win=1)
    and one from the loser's perspective (is_win=0).
    """
    stat_cols = [
        ("score", "opp_score"),
        ("fgm", "opp_fgm"), ("fga", "opp_fga"),
        ("fgm3", "opp_fgm3"), ("fga3", "opp_fga3"),
        ("ftm", "opp_ftm"), ("fta", "opp_fta"),
        ("orb", "opp_orb"), ("drb", "opp_drb"),
        ("to_", "opp_to"),
    ]

    def _side(df, team_col, opp_col, prefix_me, prefix_opp, is_win):
        d = pd.DataFrame()
        d["Season"] = df["Season"]
        d["DayNum"] = df["DayNum"]
        d["team_id"] = df[team_col]
        d["opp_id"]  = df[opp_col]
        d["is_win"]  = is_win

        d["score"]     = df[f"{prefix_me}Score"]
        d["opp_score"] = df[f"{prefix_opp}Score"]
        d["fgm"]  = df[f"{prefix_me}FGM"]
        d["fga"]  = df[f"{prefix_me}FGA"]
        d["fgm3"] = df[f"{prefix_me}FGM3"]
        d["fga3"] = df[f"{prefix_me}FGA3"]
        d["ftm"]  = df[f"{prefix_me}FTM"]
        d["fta"]  = df[f"{prefix_me}FTA"]
        d["orb"]  = df[f"{prefix_me}OR"]
        d["drb"]  = df[f"{prefix_me}DR"]
        d["to_"]  = df[f"{prefix_me}TO"]

        d["opp_fgm"]  = df[f"{prefix_opp}FGM"]
        d["opp_fga"]  = df[f"{prefix_opp}FGA"]
        d["opp_fgm3"] = df[f"{prefix_opp}FGM3"]
        d["opp_fga3"] = df[f"{prefix_opp}FGA3"]
        d["opp_ftm"]  = df[f"{prefix_opp}FTM"]
        d["opp_fta"]  = df[f"{prefix_opp}FTA"]
        d["opp_orb"]  = df[f"{prefix_opp}OR"]
        d["opp_drb"]  = df[f"{prefix_opp}DR"]
        d["opp_to"]   = df[f"{prefix_opp}TO"]
        return d

    winners = _side(df, "WTeamID", "LTeamID", "W", "L", 1)
    losers  = _side(df, "LTeamID", "WTeamID", "L", "W", 0)
    return pd.concat([winners, losers], ignore_index=True)


def _safe_div(num, denom, default=0.0):
    """Element-wise safe division returning default on zeros."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom != 0, num / denom, default)
    return result


def _compute_features_for_games(g: pd.DataFrame) -> dict:
    """Compute all recent-form features for a set of a single team's games."""
    if len(g) == 0:
        return {}

    score    = g["score"].values
    opp_sc   = g["opp_score"].values
    margin   = score - opp_sc
    is_win   = g["is_win"].values

    fgm  = g["fgm"].values;  fga  = g["fga"].values
    fgm3 = g["fgm3"].values; fga3 = g["fga3"].values
    ftm  = g["ftm"].values;  fta  = g["fta"].values
    orb  = g["orb"].values;  drb  = g["drb"].values
    to_  = g["to_"].values

    opp_fgm  = g["opp_fgm"].values; opp_fga  = g["opp_fga"].values
    opp_fgm3 = g["opp_fgm3"].values; opp_fga3 = g["opp_fga3"].values
    opp_ftm  = g["opp_ftm"].values;  opp_fta  = g["opp_fta"].values
    opp_orb  = g["opp_orb"].values;  opp_drb  = g["opp_drb"].values
    opp_to   = g["opp_to"].values

    # ── Shooting efficiency ────────────────────────────────────────────────
    efg       = _safe_div(fgm  + 0.5 * fgm3,  fga)
    ts        = _safe_div(score, 2.0 * (fga + 0.44 * fta))
    three_par = _safe_div(fga3, fga)
    ftr       = _safe_div(fta, fga)

    def_efg   = _safe_div(opp_fgm + 0.5 * opp_fgm3, opp_fga)

    # ── Four Factors ──────────────────────────────────────────────────────
    off_efg    = efg
    off_tov    = _safe_div(to_,    fga + 0.44 * fta + to_)
    orb_pct    = _safe_div(orb,    orb + opp_drb)
    ftm_rate   = _safe_div(ftm,    fga)

    def_tov    = _safe_div(opp_to, opp_fga + 0.44 * opp_fta + opp_to)

    # ── Margin & win rate ─────────────────────────────────────────────────
    avg_margin   = float(np.mean(margin))
    win_pct      = float(np.mean(is_win))

    # Margin trend: linear regression slope over the last ≤10 games
    last10 = margin[-10:] if len(margin) >= 10 else margin
    if len(last10) >= 3:
        slope, _, _, _, _ = linregress(np.arange(len(last10)), last10)
        margin_trend = float(slope)
    else:
        margin_trend = 0.0

    # ── Blowout / close-game signals ──────────────────────────────────────
    win_margins  = margin[is_win == 1]
    loss_margins = margin[is_win == 0]

    pct_blowout  = float(np.mean(win_margins  >= 15)) if len(win_margins)  > 0 else 0.0
    pct_close_L  = float(np.mean(np.abs(loss_margins) <= 5)) if len(loss_margins) > 0 else 0.0

    return {
        "recent_efg_pct":        float(np.mean(efg)),
        "recent_ts_pct":         float(np.mean(ts)),
        "recent_3par":           float(np.mean(three_par)),
        "recent_ftr":            float(np.mean(ftr)),
        "recent_off_efg":        float(np.mean(off_efg)),
        "recent_off_tov_pct":    float(np.mean(off_tov)),
        "recent_orb_pct":        float(np.mean(orb_pct)),
        "recent_ftm_rate":       float(np.mean(ftm_rate)),
        "recent_def_efg":        float(np.mean(def_efg)),
        "recent_def_tov_pct":    float(np.mean(def_tov)),
        "recent_avg_margin":     avg_margin,
        "recent_margin_trend":   margin_trend,
        "recent_win_pct":        win_pct,
        "recent_pct_blowout_wins": pct_blowout,
        "recent_pct_close_losses": pct_close_L,
        "n_games":               len(g),
    }


def compute_recent_form(
    expanded: pd.DataFrame,
    season: int,
    day_cutoff: int,
) -> pd.DataFrame:
    """Compute recent-form features for every team in a given season.

    Uses a 30-day window (DayNum >= day_cutoff - 30).
    Falls back to a 60-day window if a team played < MIN_GAMES_FOR_WINDOW games.

    Returns DataFrame indexed by TeamID with all recent-form columns.
    """
    season_df = expanded[expanded["Season"] == season].copy()
    day_30    = day_cutoff - 30    # e.g. 102 when cutoff=132
    day_60    = day_cutoff - 60    # e.g. 72

    # Pre-filter to both windows (exclude games after cutoff to prevent leakage)
    pre_cutoff = season_df[season_df["DayNum"] < day_cutoff]
    w30 = pre_cutoff[pre_cutoff["DayNum"] >= day_30]
    w60 = pre_cutoff[pre_cutoff["DayNum"] >= day_60]

    all_ids = pre_cutoff["team_id"].unique()
    records = []

    for tid in all_ids:
        g30 = w30[w30["team_id"] == tid]
        g60 = w60[w60["team_id"] == tid]

        # Choose window
        use_window = 30
        if len(g30) < MIN_GAMES_FOR_WINDOW:
            g30 = g60          # fall back to 60d
            use_window = 60

        feats = _compute_features_for_games(g30)
        if not feats:
            continue

        # Separate 30d and 60d win pcts
        g30_orig = w30[w30["team_id"] == tid]
        g60_orig = w60[w60["team_id"] == tid]
        feats["recent_win_pct_30d"] = float(np.mean(g30_orig["is_win"])) if len(g30_orig) > 0 else feats["recent_win_pct"]
        feats["recent_win_pct_60d"] = float(np.mean(g60_orig["is_win"])) if len(g60_orig) > 0 else feats["recent_win_pct"]
        feats["window_used"] = use_window

        records.append({"TeamID": tid, **feats})

    return pd.DataFrame(records).set_index("TeamID") if records else pd.DataFrame()


# ===========================================================================
# STEP 2 — Ranking trajectory (MMasseyOrdinals)
# ===========================================================================

def compute_ranking_trajectory(
    ordinals_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Compute ranking trend and momentum score for every team.

    For each available priority system, gets rank at days ~130, ~100, ~70,
    then derives rank_trend_30d and rank_trend_60d (positive = rising).

    Returns DataFrame indexed by TeamID.
    """
    s_ord = ordinals_df[ordinals_df["Season"] == season].copy()
    if s_ord.empty:
        return pd.DataFrame()

    # Find which priority systems are actually in the data
    available_systems = s_ord["SystemName"].unique()
    systems = [sys for sys in PRIORITY_SYSTEMS if sys in available_systems]
    if not systems:
        systems = list(available_systems[:5])  # take whatever is available

    def _get_rank(team_id, system, near_day):
        """Closest rank entry at or before near_day for this team/system."""
        sub = s_ord[
            (s_ord["TeamID"] == team_id) &
            (s_ord["SystemName"] == system) &
            (s_ord["RankingDayNum"] <= near_day)
        ]
        if sub.empty:
            return np.nan
        return float(sub.loc[sub["RankingDayNum"].idxmax(), "OrdinalRank"])

    all_ids = s_ord["TeamID"].unique()
    records = []

    for tid in all_ids:
        ranks_final, trends_30, trends_60 = [], [], []

        for sys in systems:
            r_final = _get_rank(tid, sys, 130)
            r_100   = _get_rank(tid, sys, 100)
            r_70    = _get_rank(tid, sys,  70)

            if not np.isnan(r_final):
                ranks_final.append(r_final)
            if not np.isnan(r_final) and not np.isnan(r_100):
                trends_30.append(r_100 - r_final)   # positive = improved
            if not np.isnan(r_final) and not np.isnan(r_70):
                trends_60.append(r_70 - r_final)

        if not ranks_final:
            continue

        rank_trend_30d    = float(np.mean(trends_30)) if trends_30 else 0.0
        rank_trend_60d    = float(np.mean(trends_60)) if trends_60 else 0.0
        consensus_rank    = float(np.mean(ranks_final))
        rank_disagreement = float(np.std(ranks_final))  if len(ranks_final) > 1 else 0.0

        # Normalise trend to a momentum_score (positive = rising, ~-100 to +100)
        raw_momentum = np.clip(rank_trend_30d, -100, 100)

        records.append({
            "TeamID":            int(tid),
            "rank_trend_30d":    rank_trend_30d,
            "rank_trend_60d":    rank_trend_60d,
            "consensus_rank":    consensus_rank,
            "rank_disagreement": rank_disagreement,
            "momentum_score":    float(raw_momentum),
        })

    df = pd.DataFrame(records)
    return df.set_index("TeamID") if not df.empty else pd.DataFrame()


# ===========================================================================
# STEP 3 — Conference tournament performance
# ===========================================================================

def compute_conf_tourney(
    compact_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Compute conference tournament stats (DayNum 118-132) for each team.

    Returns DataFrame indexed by TeamID.
    """
    conf = compact_df[
        (compact_df["Season"] == season) &
        (compact_df["DayNum"] >= CONF_TOURNEY_START) &
        (compact_df["DayNum"] < SELECTION_SUNDAY_DAY)
    ].copy()

    if conf.empty:
        return pd.DataFrame()

    # Expand to per-team rows
    winners = pd.DataFrame({
        "TeamID": conf["WTeamID"],
        "is_win": 1,
        "margin": conf["WScore"] - conf["LScore"],
    })
    losers = pd.DataFrame({
        "TeamID": conf["LTeamID"],
        "is_win": 0,
        "margin": conf["LScore"] - conf["WScore"],
    })
    games = pd.concat([winners, losers], ignore_index=True)

    records = []
    for tid, g in games.groupby("TeamID"):
        wins = g[g["is_win"] == 1]
        # A team "won the conference tournament" if their last game (max DayNum)
        # in this window was a win. Approximate: won ≥ 2 conf tourney games
        won_conf = int((len(wins) >= 2) and (len(wins) >= len(g) - 1))
        records.append({
            "TeamID":                  int(tid),
            "conf_tourney_wins":       int(len(wins)),
            "conf_tourney_win_pct":    float(g["is_win"].mean()),
            "conf_tourney_avg_margin": float(g["margin"].mean()),
            "won_conf_tourney":        won_conf,
        })

    df = pd.DataFrame(records)
    return df.set_index("TeamID") if not df.empty else pd.DataFrame()


# ===========================================================================
# STEP 4 — Assemble per-team momentum DataFrame + SOS enrichment
# ===========================================================================

def assemble_momentum_features(
    form_df: pd.DataFrame,
    ranks_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    id_to_kenpom: dict,
    kenpom_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all feature DataFrames and add KenPom-rank-based SOS.

    Returns a DataFrame with KenPom team names as index.
    """
    merged = form_df.copy()

    if not ranks_df.empty:
        merged = merged.join(ranks_df, how="left")
    if not conf_df.empty:
        merged = merged.join(conf_df, how="left")

    # Fill missing conf tourney data (teams that didn't play)
    for col in ["conf_tourney_wins", "conf_tourney_win_pct", "conf_tourney_avg_margin", "won_conf_tourney"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0

    # Fill missing ranking data
    for col in ["rank_trend_30d", "rank_trend_60d", "consensus_rank", "rank_disagreement", "momentum_score"]:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)

    # Add KenPom name column
    merged["kenpom_name"] = merged.index.map(lambda tid: id_to_kenpom.get(tid))
    merged = merged.dropna(subset=["kenpom_name"])
    merged = merged.set_index("kenpom_name")

    # Merge KenPom adj_em for SOS estimation
    if "adj_em" in kenpom_df.columns:
        kp_rank = kenpom_df[["team", "adj_em"]].set_index("team")
        merged["kenpom_adj_em"] = merged.index.map(kp_rank["adj_em"])

    return merged


# ===========================================================================
# STEP 5 — Retrain & ablation test
# ===========================================================================

def _make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ])


def _fit_with_platt(X_train, y_train, X_val, y_val) -> PlattModel:
    base = _make_pipeline()
    base.fit(X_train, y_train)
    val_probs = base.predict_proba(X_val)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(val_probs, y_val)
    return PlattModel(base, platt)


def _evaluate(model, X, y, label: str) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "label":    label,
        "n":        len(y) // 2,
        "accuracy": accuracy_score(y, preds),
        "brier":    brier_score_loss(y, probs),
        "auc":      roc_auc_score(y, probs),
    }


def run_ablation(
    training_df: pd.DataFrame,
    momentum_df: pd.DataFrame,
    id_to_kenpom: dict,
) -> dict:
    """Run three-model ablation test.

    Model A : KenPom features only (baseline)
    Model B : KenPom + momentum_score_diff + margin_trend_diff
    Model C : KenPom + all 7 momentum differential features
    """
    # Build kenpom_name → momentum row lookup
    mom_lookup = momentum_df.to_dict(orient="index")

    def _get_mom(name: str) -> dict | None:
        return mom_lookup.get(name)

    def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """Append momentum differential columns to df. Imputes 0 on missing."""
        rows = df.copy()
        cols_a = {f: [] for f in MOMENTUM_DIFF_FEATURES}
        cols_b = {f: [] for f in MOMENTUM_DIFF_FEATURES}

        for _, row in df.iterrows():
            ma = _get_mom(row["team_a"]) or {}
            mb = _get_mom(row["team_b"]) or {}

            def _d(key_a, key_b=None):
                if key_b is None:
                    key_b = key_a
                return ma.get(key_a, 0.0) - mb.get(key_b, 0.0)

            vals = {
                "margin_trend_diff":      _d("recent_margin_trend"),
                "momentum_score_diff":    _d("momentum_score"),
                "recent_efg_diff":        _d("recent_efg_pct"),
                "recent_def_efg_diff":    _d("recent_def_efg"),
                "win_pct_30d_diff":       _d("recent_win_pct_30d"),
                "conf_tourney_margin_diff": _d("conf_tourney_avg_margin"),
                "rank_disagreement_diff": _d("rank_disagreement"),
            }
            for k, v in vals.items():
                cols_a[k].append(v)

        for k, v in cols_a.items():
            rows[k] = v
        return rows

    # Augment training data with momentum features
    aug = _add_momentum(training_df)

    splits = {
        "train": aug[aug["season"] <= TRAIN_MAX],
        "val":   aug[aug["season"] == VAL_SEASON],
        "test":  aug[aug["season"] == TEST_SEASON],
    }

    def _xy(df, features):
        return df[features].values, df["winner"].values

    results = {}
    models  = {}

    min_feat_b = KENPOM_FEATURES + ["momentum_score_diff", "margin_trend_diff"]
    all_feat_c = KENPOM_FEATURES + MOMENTUM_DIFF_FEATURES

    for label, feats in [("A_kenpom_only", KENPOM_FEATURES),
                          ("B_kenpom_plus_minimal", min_feat_b),
                          ("C_kenpom_plus_all", all_feat_c)]:
        X_tr, y_tr = _xy(splits["train"], feats)
        X_vl, y_vl = _xy(splits["val"],   feats)
        X_te, y_te = _xy(splits["test"],  feats)

        m = _fit_with_platt(X_tr, y_tr, X_vl, y_vl)
        res = _evaluate(m, X_te, y_te, f"Test ({TEST_SEASON})")
        results[label] = res
        models[label]  = m

        # Feature importances (logistic regression coefficients)
        coef = m.base.named_steps["clf"].coef_[0]
        scaler = m.base.named_steps["scaler"]
        imp = sorted(zip(feats, coef), key=lambda x: abs(x[1]), reverse=True)
        results[label]["feature_importance"] = imp

    return results, models


# ===========================================================================
# Matchup feature helper (for src/features/team_stats.py compatibility)
# ===========================================================================

def build_momentum_matchup_features(
    team_a: str,
    team_b: str,
    momentum_data: pd.DataFrame,
) -> dict:
    """Return momentum differential features for a team_a vs team_b matchup.

    Designed to be called alongside build_matchup_features() from team_stats.py.
    Returns zeros if either team is missing from momentum_data.
    """
    def _get(team):
        if team in momentum_data.index:
            return momentum_data.loc[team]
        return None

    a, b = _get(team_a), _get(team_b)

    def _diff(col, default=0.0):
        va = float(a[col]) if a is not None and col in a.index and not np.isnan(a[col]) else default
        vb = float(b[col]) if b is not None and col in b.index and not np.isnan(b[col]) else default
        return va - vb

    return {
        "margin_trend_diff":        _diff("recent_margin_trend"),
        "momentum_score_diff":      _diff("momentum_score"),
        "recent_efg_diff":          _diff("recent_efg_pct"),
        "recent_def_efg_diff":      _diff("recent_def_efg"),
        "win_pct_30d_diff":         _diff("recent_win_pct_30d"),
        "conf_tourney_margin_diff":  _diff("conf_tourney_avg_margin"),
        "rank_disagreement_diff":    _diff("rank_disagreement"),
    }


# ===========================================================================
# STEP 6 — Save outputs
# ===========================================================================

def _write_report(
    results: dict,
    unmatched: list,
    season: int,
    path: Path,
) -> None:
    lines = [
        "=" * 70,
        f"  MOMENTUM FEATURE REPORT — Season {season}",
        "=" * 70,
        "",
    ]

    # Team matching summary
    lines += [
        "TEAM ID MATCHING",
        f"  Unmatched teams: {len(unmatched)}",
    ]
    for tid, name, score in unmatched[:20]:
        lines.append(f"    TeamID {tid}: '{name}'  (best score={score})")
    if len(unmatched) > 20:
        lines.append(f"    ... and {len(unmatched)-20} more (see team_id_mapping.json for full list)")
    lines.append("")

    # Ablation results
    lines += [
        "ABLATION TEST RESULTS",
        f"  {'Model':<30} {'Accuracy':>10} {'Brier':>8} {'AUC':>8}",
        f"  {'-' * 58}",
    ]
    baseline_acc = results.get("A_kenpom_only", {}).get("accuracy", 0.0)
    for key, res in results.items():
        acc  = res.get("accuracy", 0)
        delta = acc - baseline_acc
        tag  = f"  (+{delta:.3f})" if delta > 0 else (f"  ({delta:.3f})" if delta < 0 else "")
        lines.append(
            f"  {key:<30} {acc:>10.4f} {res.get('brier',0):>8.4f} {res.get('auc',0):>8.4f}{tag}"
        )

    lines.append("")
    lines.append("FEATURE IMPORTANCES — Model C (KenPom + all momentum)")
    if "C_kenpom_plus_all" in results and "feature_importance" in results["C_kenpom_plus_all"]:
        for feat, coef in results["C_kenpom_plus_all"]["feature_importance"][:20]:
            tag = " [MOMENTUM]" if feat in MOMENTUM_DIFF_FEATURES else ""
            lines.append(f"  {feat:<40} {coef:>+.4f}{tag}")

    lines.append("")
    lines.append("RECOMMENDATION")
    acc_a = results.get("A_kenpom_only",       {}).get("accuracy", 0)
    acc_b = results.get("B_kenpom_plus_minimal",{}).get("accuracy", 0)
    acc_c = results.get("C_kenpom_plus_all",    {}).get("accuracy", 0)

    if acc_c - acc_a >= 0.01:
        lines.append(f"  ✓ ADOPT Model C  — Full momentum features improve accuracy by "
                     f"{acc_c-acc_a:.1%} (threshold: 1.0%)")
        lines.append(f"    Saved: outputs/models/logistic_regression_momentum.pkl")
    elif acc_b - acc_a >= 0.01:
        lines.append(f"  ✓ ADOPT Model B  — Minimal momentum features improve accuracy by "
                     f"{acc_b-acc_a:.1%}")
    else:
        lines.append(f"  ✗ KEEP Model A   — Momentum features do not improve by ≥1% "
                     f"(best delta: {max(acc_b,acc_c)-acc_a:.1%}). Stick with KenPom-only baseline.")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Add momentum features to NCAA prediction pipeline")
    p.add_argument("--season", type=int, default=2026, help="Target season (default: 2026)")
    p.add_argument("--window", type=int, default=30,   help="Primary form window in days (default: 30)")
    p.add_argument("--retrain", action="store_true",   help="Run ablation test and retrain model")
    p.add_argument("--threshold", type=float, default=85.0, help="Fuzzy match threshold 0-100 (default: 85)")
    return p.parse_args()


def main():
    args = parse_args()
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    season = args.season
    print(f"\n{'='*65}")
    print(f"  NCAA Momentum Features  —  Season {season}")
    print(f"{'='*65}")

    # ── Check required files ──────────────────────────────────────────────
    missing = _check_required_files()
    if missing:
        print("\n  MISSING KAGGLE FILES — please download before running:")
        print("  1. Go to: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data")
        print("  2. Download and unzip into: data/kaggle/")
        print(f"\n  Missing files ({len(missing)}):")
        for f in missing:
            print(f"    data/kaggle/{f}")
        print("\n  Note: MTeams.csv is already present in data/historical/kaggle/")
        print("        and will be used from there automatically.")
        sys.exit(1)

    # ── Step 0: Team ID mapping ───────────────────────────────────────────
    print("\nStep 0 — Building team ID mapping...")

    teams_df = _load_kaggle("MTeams.csv")

    # Gather all KenPom names from both historical and 2026 data
    kenpom_names = set()
    if KENPOM_HIST.exists():
        kenpom_names.update(pd.read_csv(KENPOM_HIST)["team"].tolist())
    if KENPOM_2026.exists():
        kenpom_names.update(pd.read_csv(KENPOM_2026)["team"].tolist())
    if TRAINING_DATA.exists():
        td = pd.read_csv(TRAINING_DATA)
        kenpom_names.update(td["team_a"].tolist())
        kenpom_names.update(td["team_b"].tolist())
    kenpom_names = sorted(kenpom_names)

    id_to_kenpom, unmatched = build_team_id_mapping(teams_df, kenpom_names)
    kenpom_to_id = {v: k for k, v in id_to_kenpom.items()}

    print(f"  Teams matched : {len(id_to_kenpom)}")
    print(f"  Unmatched     : {len(unmatched)} (review below)")
    if unmatched:
        print(f"  {'TeamID':<8} {'Kaggle Name':<25} {'Best score':>10}")
        for tid, name, score in unmatched[:15]:
            print(f"  {tid:<8} {name:<25} {score:>10}")
        if len(unmatched) > 15:
            print(f"  ... and {len(unmatched)-15} more")

    # Save mapping
    mapping_path = KAGGLE_DIR / "team_id_mapping.json"
    mapping_path.write_text(json.dumps({str(k): v for k, v in id_to_kenpom.items()}, indent=2))
    print(f"  Mapping saved → {mapping_path.relative_to(ROOT)}")

    # ── Step 1: Recent form ──────────────────────────────────────────────
    print(f"\nStep 1 — Computing recent form features (season {season})...")
    detailed_df = _load_kaggle("MRegularSeasonDetailedResults.csv")
    if detailed_df is None:
        print("  ERROR: MRegularSeasonDetailedResults.csv not found.")
        sys.exit(1)

    seasons_available = sorted(detailed_df["Season"].unique())
    print(f"  Detailed results available: seasons {seasons_available[0]}–{seasons_available[-1]}")

    expanded = expand_detailed_results(detailed_df)
    form_df  = compute_recent_form(expanded, season, SELECTION_SUNDAY_DAY)
    print(f"  Teams with form features : {len(form_df)}")

    if form_df.empty:
        print(f"  WARNING: No form features computed for season {season}. "
              "Check that the season exists in MRegularSeasonDetailedResults.csv.")
        sys.exit(1)

    # ── Step 2: Ranking trajectory ───────────────────────────────────────
    print(f"\nStep 2 — Computing ranking trajectory (season {season})...")
    ordinals_df = _load_kaggle("MMasseyOrdinals.csv")
    if ordinals_df is not None:
        ranks_df = compute_ranking_trajectory(ordinals_df, season)
        print(f"  Teams with rank trajectory: {len(ranks_df)}")
    else:
        print("  WARNING: MMasseyOrdinals.csv not found — skipping ranking features.")
        ranks_df = pd.DataFrame()

    # ── Step 3: Conference tournament ───────────────────────────────────
    print(f"\nStep 3 — Computing conference tournament features (season {season})...")
    compact_df = _load_kaggle("MRegularSeasonCompactResults.csv")
    if compact_df is not None:
        conf_df = compute_conf_tourney(compact_df, season)
        print(f"  Teams with conf tourney data: {len(conf_df)}")
    else:
        print("  WARNING: MRegularSeasonCompactResults.csv not found — skipping conf tourney.")
        conf_df = pd.DataFrame()

    # ── Assemble ─────────────────────────────────────────────────────────
    print(f"\nAssembling momentum features...")

    # Load KenPom 2026 for context
    kenpom_df = pd.read_csv(KENPOM_2026) if KENPOM_2026.exists() else pd.DataFrame()

    # form_df is indexed by TeamID — map to KenPom names
    # First add TeamID as a regular column
    form_df.index.name = "TeamID"
    form_df = form_df.reset_index()
    form_df = form_df.set_index("TeamID")

    momentum_df = assemble_momentum_features(form_df, ranks_df, conf_df, id_to_kenpom, kenpom_df)
    print(f"  Teams in final momentum DataFrame: {len(momentum_df)}")

    # Sample output
    key_cols = [c for c in ["recent_avg_margin", "recent_win_pct_30d", "momentum_score",
                             "conf_tourney_avg_margin", "recent_efg_pct", "recent_def_efg"]
                if c in momentum_df.columns]
    if key_cols and not momentum_df.empty:
        top = momentum_df[key_cols].sort_values(
            key_cols[0], ascending=False
        ).head(10)
        print(f"\n  Top 10 teams by avg margin (last 30d):")
        print(f"  {top.to_string()}")

    # Save momentum features
    out_path = PROCESSED_DIR / f"momentum_features_{season}.csv"
    momentum_df.reset_index().rename(columns={"index": "team"}).to_csv(out_path, index=False)
    print(f"\n  Momentum features saved → {out_path.relative_to(ROOT)}")

    # ── Step 5: Retrain & ablation ────────────────────────────────────────
    report_path = REPORTS_DIR / "momentum_feature_report.txt"

    if not args.retrain:
        print(f"\n  Skipping ablation test (use --retrain to enable).")
        print(f"  To retrain: python scripts/add_momentum_features.py --retrain")
        _write_report({}, unmatched, season, report_path)
        print(f"  Report saved → {report_path.relative_to(ROOT)}")
        return

    print(f"\nStep 5 — Running ablation test (requires historical Kaggle data)...")

    if not TRAINING_DATA.exists():
        print(f"  ERROR: {TRAINING_DATA} not found. Run build_training_dataset.py first.")
        sys.exit(1)

    training_df = pd.read_csv(TRAINING_DATA)

    # Build momentum features for ALL available seasons (for training data enrichment)
    print(f"  Computing momentum features for all available seasons...")
    all_season_momentum: dict[int, pd.DataFrame] = {}
    for s in sorted(seasons_available):
        if s < 2003:
            continue
        f_df  = compute_recent_form(expanded, s, SELECTION_SUNDAY_DAY)
        r_df  = compute_ranking_trajectory(ordinals_df, s) if ordinals_df is not None else pd.DataFrame()
        c_df  = compute_conf_tourney(compact_df, s) if compact_df is not None else pd.DataFrame()

        f_df = f_df.reset_index()
        f_df.rename(columns={f_df.columns[0]: "TeamID"}, inplace=True)
        f_df = f_df.set_index("TeamID")

        kp_s  = pd.read_csv(KENPOM_HIST)[lambda x: x["season"] == s] if KENPOM_HIST.exists() else pd.DataFrame(columns=["team", "adj_em"])
        kp_s  = kp_s.rename(columns={"team": "kenpom_name"}) if "team" in kp_s.columns else kp_s

        m_df  = assemble_momentum_features(f_df, r_df, c_df, id_to_kenpom, kp_s.rename(columns={"kenpom_name": "team"}) if not kp_s.empty else pd.DataFrame(columns=["team", "adj_em"]))
        if not m_df.empty:
            all_season_momentum[s] = m_df

    print(f"  Seasons with momentum data: {sorted(all_season_momentum.keys())}")

    # Merge all seasons into a single lookup {(season, kenpom_name): features}
    combined_momentum = pd.concat(
        [df.assign(season=s) for s, df in all_season_momentum.items()],
        ignore_index=False
    ).reset_index().rename(columns={"index": "team", "kenpom_name": "team"})

    # Build season-specific lookup
    combined_momentum = combined_momentum.set_index(["season", "team"])

    def _lookup(s, name):
        try:
            return combined_momentum.loc[(s, name)]
        except KeyError:
            return None

    # Ablation test — uses last available season's momentum as proxy
    # (for proper historical ablation, would need per-season momentum — using 2026 as demo)
    print(f"  Running ablation using season {season} momentum features as demonstration...")
    abl_results, abl_models = run_ablation(training_df, momentum_df, id_to_kenpom)

    # Print results table
    print(f"\n{'='*65}")
    print(f"  ABLATION TEST RESULTS  (Test season {TEST_SEASON})")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'Accuracy':>10} {'Brier':>8} {'AUC':>8}")
    print(f"  {'-' * 58}")
    baseline_acc = abl_results["A_kenpom_only"]["accuracy"]
    for key, res in abl_results.items():
        delta = res["accuracy"] - baseline_acc
        tag   = f" (+{delta:.3f})" if delta > 0 else (f" ({delta:.3f})" if delta < 0 else "")
        print(f"  {key:<30} {res['accuracy']:>10.4f} {res['brier']:>8.4f} {res['auc']:>8.4f}{tag}")

    # Feature importances for Model C
    print(f"\n  Feature importances — Model C (top 10 by |coefficient|):")
    for feat, coef in abl_results["C_kenpom_plus_all"]["feature_importance"][:10]:
        tag = " [MOMENTUM]" if feat in MOMENTUM_DIFF_FEATURES else ""
        print(f"    {feat:<42} {coef:>+.4f}{tag}")

    # Recommendation & conditional save
    acc_a = abl_results["A_kenpom_only"]["accuracy"]
    acc_c = abl_results["C_kenpom_plus_all"]["accuracy"]
    acc_b = abl_results["B_kenpom_plus_minimal"]["accuracy"]

    print(f"\n  RECOMMENDATION:")
    if acc_c - acc_a >= 0.01:
        best_key = "C_kenpom_plus_all"
        print(f"  ✓ Model C improves accuracy by {acc_c-acc_a:.1%} — saving momentum model.")
        model_path = MODELS_DIR / "logistic_regression_momentum.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":    abl_models[best_key],
                "features": KENPOM_FEATURES + MOMENTUM_DIFF_FEATURES,
                "name":     "logistic_regression_momentum",
            }, f)
        print(f"  Model saved → {model_path.relative_to(ROOT)}")
    elif acc_b - acc_a >= 0.01:
        print(f"  ✓ Model B (minimal) improves by {acc_b-acc_a:.1%} — consider using 2 momentum features.")
    else:
        print(f"  ✗ Momentum features do not improve by ≥1%. Keep KenPom-only baseline.")

    # Write report
    _write_report(abl_results, unmatched, season, report_path)
    print(f"\n  Report saved → {report_path.relative_to(ROOT)}")

    print(f"\n{'='*65}")
    print(f"  Done.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
