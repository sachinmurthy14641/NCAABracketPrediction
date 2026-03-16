"""NCAA trading strategy: load production model, score Kalshi markets, emit trade signals.

Usage::

    from src.kalshi.strategy import NCAATradingStrategy

    strategy = NCAATradingStrategy()
    strategy.update_seeds({'Duke': 1, 'American': 16, 'Auburn': 1, ...})

    # Score a raw Kalshi market dict
    signal = strategy.evaluate_market({
        'ticker': 'NCAAM-2026-DUKE-R1',
        'title': 'Will Duke beat American?',
        'yes_price': 88,
    })

    # Or score a known matchup directly
    signal = strategy.evaluate_matchup('Duke', 1, 'American', 16, market_yes_price=88)
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.team_stats import build_matchup_features
from src.utils.team_mapping import normalize_team_name, find_closest_kenpom_team

logger = logging.getLogger(__name__)

MODEL_PATH  = Path("outputs/models/lightgbm_final_2026.pkl")
KENPOM_PATH = Path("data/processed/kenpom_2026_clean.csv")

# Feature columns must match FEATURE_COLS in train_model.py exactly
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

MIN_EDGE       = 0.05   # 5% minimum edge to generate a signal
HIGH_EDGE      = 0.15   # 15% → high confidence
MEDIUM_EDGE    = 0.08   # 8%  → medium confidence


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchupSignal:
    ticker:              str
    team_a:              str
    team_b:              str
    team_a_seed:         int
    team_b_seed:         int
    model_prob:          float   # P(team_a wins)
    market_yes_price:    int     # cents
    market_implied_prob: float
    edge:                float   # model_prob - market_implied_prob
    side:                str     # "yes" | "no" | "pass"
    team:                str     # team to back
    confidence:          str     # "high" | "medium" | "low"
    expected_value:      float   # EV per $1 risked
    matchup:             str     = field(init=False)

    def __post_init__(self) -> None:
        self.matchup = f"{self.team_a} (#{self.team_a_seed}) vs {self.team_b} (#{self.team_b_seed})"

    def __str__(self) -> str:
        return (
            f"{self.matchup}\n"
            f"  Ticker       : {self.ticker}\n"
            f"  Model prob   : {self.model_prob:.1%}\n"
            f"  Market price : {self.market_yes_price}¢  (implied {self.market_implied_prob:.1%})\n"
            f"  Edge         : {self.edge:+.1%}\n"
            f"  EV           : {self.expected_value:+.3f} per $1 risked\n"
            f"  Signal       : {self.side.upper()}  [{self.confidence.upper()}]"
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class NCAATradingStrategy:
    """Load the production LightGBM model and evaluate NCAA Kalshi markets."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        kenpom_path: Path = KENPOM_PATH,
    ) -> None:
        self._model    = self._load_model(model_path)
        self._kenpom   = self._load_kenpom(kenpom_path)
        self._features = self._model["features"]
        self.seeds: dict[str, int] = {}   # populated after Selection Sunday

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        logger.info("Loaded model '%s' from %s", payload.get("name", path.stem), path)
        return payload

    @staticmethod
    def _load_kenpom(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"KenPom 2026 data not found: {path}")
        df = pd.read_csv(path)
        if "Team" in df.columns:
            df = df.rename(columns={"Team": "team"})
        df["team"] = df["team"].str.strip()
        logger.info("Loaded KenPom 2026: %d teams", len(df))
        return df

    def update_seeds(self, seed_mapping: dict[str, int]) -> None:
        """Update tournament seeds after the bracket is announced.

        Args:
            seed_mapping: {team_name: seed_number}
                Example: {'Duke': 1, 'American': 16, 'Auburn': 1, ...}
        """
        self.seeds = seed_mapping
        logger.info("Updated seeds for %d teams", len(seed_mapping))

    # ------------------------------------------------------------------
    # Team lookup with fuzzy fallback
    # ------------------------------------------------------------------

    def _find_team(self, name: str) -> Optional[str]:
        """Return canonical KenPom team name.

        Resolution order:
          1. Normalize via TEAM_NAME_MAPPING (strips mascots, maps variants)
          2. Exact match against KenPom roster
          3. Fuzzy match via difflib (cutoff=0.75)
        """
        teams      = self._kenpom["team"]
        teams_list = teams.tolist()
        normalized = normalize_team_name(name)

        # Exact match (case-insensitive) on normalized name
        exact = teams[teams.str.lower() == normalized.lower()]
        if not exact.empty:
            if normalized != name:
                logger.debug("Mapped '%s' → '%s'", name, exact.iloc[0])
            return exact.iloc[0]

        # Fuzzy match as last resort
        fuzzy = find_closest_kenpom_team(normalized, teams_list, cutoff=0.75)
        if fuzzy:
            logger.debug("Fuzzy matched '%s' (normalized: '%s') → '%s'", name, normalized, fuzzy)
            return fuzzy

        logger.warning("Team not found in KenPom 2026: '%s' (normalized: '%s')", name, normalized)
        return None

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _predict_win_probability(
        self,
        team_a: str,
        team_b: str,
        seed_a: Optional[int] = None,
        seed_b: Optional[int] = None,
    ) -> Optional[float]:
        """Return P(team_a beats team_b), or None if data is unavailable."""
        canon_a = self._find_team(team_a)
        canon_b = self._find_team(team_b)
        if canon_a is None or canon_b is None:
            return None

        try:
            kp_features = build_matchup_features(canon_a, canon_b, self._kenpom)
        except ValueError as exc:
            logger.warning("Feature build failed for %s vs %s: %s", team_a, team_b, exc)
            return None

        # Resolve seeds: argument → seeds dict → neutral placeholder
        sa = seed_a if seed_a is not None else self.seeds.get(canon_a, self.seeds.get(team_a, 8))
        sb = seed_b if seed_b is not None else self.seeds.get(canon_b, self.seeds.get(team_b, 8))

        # Raw KenPom stats for the individual-team features
        idx = self._kenpom.set_index("team")
        a_row = idx.loc[canon_a]
        b_row = idx.loc[canon_b]

        row = {
            **kp_features,
            "seed_diff":    sa - sb,
            "a_adj_off_eff": a_row["adj_off_eff"],
            "a_adj_def_eff": a_row["adj_def_eff"],
            "a_adj_em":      a_row["adj_em"],
            "b_adj_off_eff": b_row["adj_off_eff"],
            "b_adj_def_eff": b_row["adj_def_eff"],
            "b_adj_em":      b_row["adj_em"],
        }

        X    = pd.DataFrame([row])[self._features]
        prob = float(self._model["model"].predict_proba(X.values)[:, 1][0])

        logger.debug(
            "P(%s beats %s) = %.3f  [seeds %d vs %d]",
            canon_a, canon_b, prob, sa, sb,
        )
        return prob

    # ------------------------------------------------------------------
    # Market title parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_teams_from_title(title: str) -> Optional[tuple[str, str]]:
        """Extract (team_a, team_b) from a Kalshi market title.

        Handles patterns:
          - "Will Duke beat American?"
          - "Will Duke beat American on March 20?"
          - "Duke vs American winner"
          - "Duke vs. American"
        """
        # Pattern 1: "Will TEAM_A beat TEAM_B"
        m = re.search(r"Will (.+?) beat (.+?)(?:\s+on\b|\?|$)", title, re.IGNORECASE)
        if m:
            return normalize_team_name(m.group(1).strip()), normalize_team_name(m.group(2).strip())

        # Pattern 2: "TEAM_A vs[.] TEAM_B"
        m = re.search(r"(.+?)\s+vs\.?\s+(.+?)(?:\s+winner|\?|$)", title, re.IGNORECASE)
        if m:
            return normalize_team_name(m.group(1).strip()), normalize_team_name(m.group(2).strip())

        logger.debug("Could not parse teams from title: '%s'", title)
        return None

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_signal(
        prob: float,
        market_yes_price: int,
        ticker: str,
        team_a: str,
        team_b: str,
        seed_a: int,
        seed_b: int,
    ) -> MatchupSignal:
        market_prob = market_yes_price / 100.0
        edge        = prob - market_prob

        if edge > 0:
            side = "yes"
            team = team_a
            ev   = prob * (1 - market_prob) - (1 - prob) * market_prob
        else:
            side = "no"
            team = team_b
            ev   = (1 - prob) * market_prob - prob * (1 - market_prob)

        if abs(edge) < MIN_EDGE:
            side = "pass"
            team = ""

        if abs(edge) >= HIGH_EDGE:
            confidence = "high"
        elif abs(edge) >= MEDIUM_EDGE:
            confidence = "medium"
        else:
            confidence = "low"

        return MatchupSignal(
            ticker=ticker,
            team_a=team_a,
            team_b=team_b,
            team_a_seed=seed_a,
            team_b_seed=seed_b,
            model_prob=prob,
            market_yes_price=market_yes_price,
            market_implied_prob=market_prob,
            edge=edge,
            side=side,
            team=team,
            confidence=confidence,
            expected_value=ev,
        )

    def evaluate_matchup(
        self,
        team_a: str,
        seed_a: int,
        team_b: str,
        seed_b: int,
        market_yes_price: int,
        ticker: str = "",
    ) -> Optional[MatchupSignal]:
        """Score a known matchup against a Kalshi YES price (in cents).

        market_yes_price=88 means the market implies team_a wins with 88% probability.
        """
        prob = self._predict_win_probability(team_a, team_b, seed_a, seed_b)
        if prob is None:
            logger.warning("Skipping %s vs %s — could not compute probability", team_a, team_b)
            return None

        signal = self._compute_signal(prob, market_yes_price, ticker, team_a, team_b, seed_a, seed_b)
        logger.info(
            "[%s]  %s vs %s  →  model=%.1f%%  market=%d¢  edge=%+.1f%%  signal=%s",
            ticker or "—", team_a, team_b, prob * 100, market_yes_price,
            signal.edge * 100, signal.side.upper(),
        )
        return signal

    def evaluate_market(self, market: dict) -> Optional[MatchupSignal]:
        """Evaluate a raw Kalshi market dict and return a trade signal if edge exists.

        Expected market keys: ticker, title, yes_price (cents).
        Seeds are looked up from self.seeds (call update_seeds() first).

        Returns None if teams can't be parsed, data is missing, or edge < MIN_EDGE.
        """
        ticker    = market.get("ticker", "")
        title     = market.get("title", "")
        yes_price = market.get("yes_price", market.get("yes_bid", 50))

        teams = self._parse_teams_from_title(title)
        if teams is None:
            logger.warning("[%s] Could not parse teams from title: '%s'", ticker, title)
            return None

        team_a, team_b = teams
        seed_a = self.seeds.get(team_a, 8)
        seed_b = self.seeds.get(team_b, 8)

        if not self.seeds:
            logger.warning("Seeds not yet loaded — using neutral seed=8 for all teams. "
                           "Call update_seeds() after Selection Sunday.")

        return self.evaluate_matchup(team_a, seed_a, team_b, seed_b, yes_price, ticker)

    # ------------------------------------------------------------------
    # Batch scanning
    # ------------------------------------------------------------------

    def scan_markets(
        self,
        matchups: list[dict],
        verbose: bool = True,
    ) -> list[MatchupSignal]:
        """Score a list of matchups and return all signals (including PASS).

        Each dict: {team_a, seed_a, team_b, seed_b, market_yes_price, ticker (opt)}
        """
        signals: list[MatchupSignal] = []

        for m in matchups:
            sig = self.evaluate_matchup(
                team_a=m["team_a"], seed_a=m["seed_a"],
                team_b=m["team_b"], seed_b=m["seed_b"],
                market_yes_price=m["market_yes_price"],
                ticker=m.get("ticker", ""),
            )
            if sig is None:
                continue
            signals.append(sig)
            if verbose:
                print(f"\n{sig}")

        actionable = [s for s in signals if s.side != "pass"]
        if verbose:
            print(f"\n{'=' * 55}")
            print(f"  {len(actionable)} actionable / {len(signals)} total matchups scored")
            if actionable:
                print("\n  ACTIONABLE SIGNALS:")
                for s in actionable:
                    print(f"    {s.side.upper():>7}  {s.matchup:<50}  edge={s.edge:+.1%}  [{s.confidence}]")

        return signals
