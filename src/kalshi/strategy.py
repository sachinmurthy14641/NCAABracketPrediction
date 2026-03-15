"""Trading strategy: load production model, score matchups, find Kalshi edges.

Usage::

    from src.kalshi.strategy import NCAATradingStrategy

    strategy = NCAATradingStrategy()
    signal = strategy.evaluate_matchup("Duke", 1, "Alabama", 8, market_yes_price=65)
    print(signal)
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH   = Path("outputs/models/lightgbm_final_2026.pkl")
KENPOM_PATH  = Path("data/processed/kenpom_2026_clean.csv")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MatchupSignal:
    team_a: str
    team_b: str
    team_a_seed: int
    team_b_seed: int
    model_prob: float          # P(team_a wins) from our model
    market_yes_price: int      # Kalshi YES price in cents (= implied prob × 100)
    market_implied_prob: float # market_yes_price / 100
    edge: float                # model_prob - market_implied_prob
    recommendation: str        # "BUY YES", "BUY NO", "PASS"
    confidence: str            # "HIGH", "MEDIUM", "LOW"
    expected_value: float      # rough EV per $1 risked

    def __str__(self) -> str:
        return (
            f"{self.team_a} (#{self.team_a_seed}) vs {self.team_b} (#{self.team_b_seed})\n"
            f"  Model prob   : {self.model_prob:.1%}\n"
            f"  Market price : {self.market_yes_price}¢  (implied {self.market_implied_prob:.1%})\n"
            f"  Edge         : {self.edge:+.1%}\n"
            f"  EV           : {self.expected_value:+.3f} per $1 risked\n"
            f"  Signal       : {self.recommendation}  [{self.confidence}]"
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class NCAATradingStrategy:
    """Load the production LightGBM model and score NCAA tournament matchups."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        kenpom_path: Path = KENPOM_PATH,
        min_edge: float = 0.05,
        min_confidence: float = 0.60,
    ):
        self.min_edge       = min_edge
        self.min_confidence = min_confidence
        self._model         = self._load_model(model_path)
        self._kenpom        = self._load_kenpom(kenpom_path)
        self._features      = self._model["features"]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        logger.info("Loaded model: %s", payload.get("name", path.name))
        return payload

    @staticmethod
    def _load_kenpom(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"KenPom 2026 data not found: {path}")
        df = pd.read_csv(path)
        # Normalise team name column
        if "Team" in df.columns:
            df = df.rename(columns={"Team": "team"})
        df["team"] = df["team"].str.strip()
        logger.info("Loaded KenPom 2026: %d teams", len(df))
        return df

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _get_team_stats(self, team_name: str) -> Optional[pd.Series]:
        match = self._kenpom[self._kenpom["team"].str.lower() == team_name.lower()]
        if match.empty:
            # Fuzzy fallback: contains match
            match = self._kenpom[self._kenpom["team"].str.lower().str.contains(
                team_name.lower(), na=False
            )]
        if match.empty:
            logger.warning("Team not found in KenPom 2026: '%s'", team_name)
            return None
        return match.iloc[0]

    def _build_feature_row(
        self,
        stats_a: pd.Series,
        stats_b: pd.Series,
        seed_a: int,
        seed_b: int,
    ) -> pd.DataFrame:
        """Build the feature vector matching FEATURE_COLS from train_model.py."""
        a_off = stats_a.get("adj_off_eff", stats_a.get("AdjOE", float("nan")))
        a_def = stats_a.get("adj_def_eff", stats_a.get("AdjDE", float("nan")))
        a_em  = stats_a.get("adj_em",      stats_a.get("AdjEM", float("nan")))

        b_off = stats_b.get("adj_off_eff", stats_b.get("AdjOE", float("nan")))
        b_def = stats_b.get("adj_def_eff", stats_b.get("AdjDE", float("nan")))
        b_em  = stats_b.get("adj_em",      stats_b.get("AdjEM", float("nan")))

        # Reproduce exact formulas from build_training_dataset.py
        off_eff_advantage = a_off - b_def          # team_a offense vs team_b defense
        def_eff_advantage = b_off - a_def          # team_b offense vs team_a defense
        net_efficiency_edge = off_eff_advantage - def_eff_advantage

        a_eff_diff = stats_a.get("efficiency_differential", a_em)
        b_eff_diff = stats_b.get("efficiency_differential", b_em)

        row = {
            "off_eff_advantage":            off_eff_advantage,
            "def_eff_advantage":            def_eff_advantage,
            "net_efficiency_edge":          net_efficiency_edge,
            "tempo_difference":             stats_a.get("adj_tempo", 0) - stats_b.get("adj_tempo", 0),
            "overall_rating_diff":          a_em - b_em,
            "efficiency_differential_diff": a_eff_diff - b_eff_diff,
            "seed_diff":                    seed_a - seed_b,
            "a_adj_off_eff":                a_off,
            "a_adj_def_eff":                a_def,
            "a_adj_em":                     a_em,
            "b_adj_off_eff":                b_off,
            "b_adj_def_eff":                b_def,
            "b_adj_em":                     b_em,
        }
        return pd.DataFrame([row])[self._features]

    def predict_matchup(
        self,
        team_a: str,
        seed_a: int,
        team_b: str,
        seed_b: int,
    ) -> Optional[float]:
        """Return P(team_a wins). Returns None if team not found."""
        stats_a = self._get_team_stats(team_a)
        stats_b = self._get_team_stats(team_b)
        if stats_a is None or stats_b is None:
            return None

        X   = self._build_feature_row(stats_a, stats_b, seed_a, seed_b)
        prob = self._model["model"].predict_proba(X.values)[:, 1][0]
        return float(prob)

    # ------------------------------------------------------------------
    # Trading signal
    # ------------------------------------------------------------------

    def evaluate_matchup(
        self,
        team_a: str,
        seed_a: int,
        team_b: str,
        seed_b: int,
        market_yes_price: int,          # Kalshi YES price in cents
    ) -> Optional[MatchupSignal]:
        """
        Compare model probability to Kalshi market price.
        market_yes_price=65 means the market thinks team_a wins with 65% probability.
        """
        prob = self.predict_matchup(team_a, seed_a, team_b, seed_b)
        if prob is None:
            return None

        market_prob = market_yes_price / 100.0
        edge        = prob - market_prob

        # Expected value: if we BUY YES at price p, EV = prob*(1-p) - (1-prob)*p
        #                 if we BUY NO  at price p, EV = (1-prob)*p - prob*(1-p)
        if edge > 0:
            ev  = prob * (1 - market_prob) - (1 - prob) * market_prob
            rec = "BUY YES"
        else:
            ev  = (1 - prob) * market_prob - prob * (1 - market_prob)
            rec = "BUY NO"

        if abs(edge) < self.min_edge:
            rec = "PASS"

        if abs(edge) >= 0.10:
            conf = "HIGH"
        elif abs(edge) >= 0.05:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        return MatchupSignal(
            team_a=team_a,
            team_b=team_b,
            team_a_seed=seed_a,
            team_b_seed=seed_b,
            model_prob=prob,
            market_yes_price=market_yes_price,
            market_implied_prob=market_prob,
            edge=edge,
            recommendation=rec,
            confidence=conf,
            expected_value=ev,
        )

    def scan_markets(
        self,
        matchups: list[dict],
        verbose: bool = True,
    ) -> list[MatchupSignal]:
        """
        Score a list of matchups against Kalshi market prices.

        Each matchup dict:
            {team_a, seed_a, team_b, seed_b, market_yes_price, ticker (optional)}
        """
        signals = []
        for m in matchups:
            sig = self.evaluate_matchup(
                team_a=m["team_a"], seed_a=m["seed_a"],
                team_b=m["team_b"], seed_b=m["seed_b"],
                market_yes_price=m["market_yes_price"],
            )
            if sig is None:
                continue
            sig.ticker = m.get("ticker", "")  # type: ignore[attr-defined]
            signals.append(sig)
            if verbose:
                print(f"\n{sig}")

        actionable = [s for s in signals if s.recommendation != "PASS"]
        if verbose:
            print(f"\n{'='*55}")
            print(f"  {len(actionable)} actionable signals out of {len(signals)} matchups")
        return signals
