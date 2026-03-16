"""Continuous Kalshi market scanner — monitors NCAA markets and emits trade signals.

Usage::

    python scripts/scan_kalshi_markets.py
    python scripts/scan_kalshi_markets.py --interval 60
    python scripts/scan_kalshi_markets.py --min-edge 0.08
    python scripts/scan_kalshi_markets.py --env live
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# PlattModel needed for pickle deserialization
from scripts.train_model import PlattModel  # noqa: F401
from src.kalshi.client import KalshiClient
from src.kalshi.strategy import NCAATradingStrategy, MatchupSignal

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOGS_DIR = Path("outputs/logs")
NCAA_KEYWORDS = ["ncaa", "march madness", "tournament", "ncaam"]

# ===========================================================================
# SEEDS — 2026 tournament bracket
# ===========================================================================
SEEDS_2026 = {
    # No. 1 seeds
    'Duke': 1, 'Arizona': 1, 'Michigan': 1, 'Florida': 1,
    # No. 2 seeds
    'Houston': 2, 'UConn': 2, 'Iowa St.': 2, 'Purdue': 2,
    # No. 3 seeds
    'Michigan St.': 3, 'Illinois': 3, 'Gonzaga': 3, 'Virginia': 3,
    # No. 4 seeds
    'Nebraska': 4, 'Alabama': 4, 'Kansas': 4, 'Arkansas': 4,
    # No. 5 seeds
    'Vanderbilt': 5, "St. John's": 5, 'Texas Tech': 5, 'Wisconsin': 5,
    # No. 6 seeds
    'Tennessee': 6, 'North Carolina': 6, 'Louisville': 6, 'BYU': 6,
    # No. 7 seeds
    'Kentucky': 7, "Saint Mary's": 7, 'Miami': 7, 'UCLA': 7,
    # No. 8 seeds
    'Clemson': 8, 'Villanova': 8, 'Ohio St.': 8, 'Georgia': 8,
    # No. 9 seeds
    'Utah St.': 9, 'TCU': 9, 'Saint Louis': 9, 'Iowa': 9,
    # No. 10 seeds
    'Santa Clara': 10, 'UCF': 10, 'Missouri': 10, 'Texas A&M': 10,
    # No. 11 seeds (First Four)
    'NC State': 11, 'Texas': 11, 'SMU': 11, 'Miami OH': 11, 'VCU': 11, 'South Florida': 11,
    # No. 12 seeds
    'McNeese': 12, 'Akron': 12, 'Northern Iowa': 12, 'High Point': 12,
    # No. 13 seeds
    'Cal Baptist': 13, 'Hofstra': 13, 'Troy': 13, 'Hawaii': 13,
    # No. 14 seeds
    'North Dakota St.': 14, 'Penn': 14, 'Wright St.': 14, 'Kennesaw St.': 14,
    # No. 15 seeds
    'Tennessee St.': 15, 'Idaho': 15, 'Furman': 15, 'Queens': 15,
    # No. 16 seeds (First Four)
    'Siena': 16, 'LIU': 16, 'Howard': 16, 'UMBC': 16, 'Lehigh': 16, 'Prairie View': 16,
}
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalshi NCAA market scanner")
    parser.add_argument("--interval", type=int,   default=300,  help="Scan interval in seconds (default: 300)")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge to report (default: 0.05)")
    parser.add_argument("--env",      type=str,   default=None, help="Override KALSHI_ENV: 'demo' or 'live'")
    return parser.parse_args()


def is_ncaa_market(market: dict) -> bool:
    title = market.get("title", "").lower()
    return any(kw in title for kw in NCAA_KEYWORDS)


def save_opportunities(signals: list[MatchupSignal]) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"opportunities_{ts}.json"

    records = []
    for s in signals:
        records.append({
            "ticker":        s.ticker,
            "matchup":       s.matchup,
            "side":          s.side,
            "team":          s.team,
            "model_prob":    round(s.model_prob, 4),
            "market_prob":   round(s.market_implied_prob, 4),
            "edge":          round(s.edge, 4),
            "expected_value": round(s.expected_value, 4),
            "confidence":    s.confidence,
            "scanned_at":    datetime.now().isoformat(),
        })

    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    return path


def print_signals(signals: list[MatchupSignal], min_edge: float) -> None:
    actionable = [s for s in signals if s.side != "pass" and abs(s.edge) >= min_edge]
    if not actionable:
        print("  No opportunities meeting minimum edge criteria")
        return

    actionable.sort(key=lambda s: abs(s.edge), reverse=True)
    high   = [s for s in actionable if s.confidence == "high"]
    medium = [s for s in actionable if s.confidence == "medium"]
    low    = [s for s in actionable if s.confidence == "low"]

    print(f"\n{'='*70}")
    print(f"  OPPORTUNITIES FOUND: {len(actionable)}")
    print(f"{'='*70}")

    if high:
        print(f"\n  HIGH CONFIDENCE ({len(high)}):")
        for s in high:
            print(f"    {s.matchup}")
            print(f"      Signal : {s.side.upper()} {s.team}")
            print(f"      Model  : {s.model_prob:.1%}  |  Market: {s.market_implied_prob:.1%}  |  Edge: {s.edge:+.1%}")
            print(f"      EV     : {s.expected_value:+.3f} per $1 risked")

    if medium:
        print(f"\n  MEDIUM CONFIDENCE ({len(medium)}):")
        for s in medium:
            print(f"    {s.matchup}")
            print(f"      {s.side.upper()} {s.team}  |  edge={s.edge:+.1%}  |  EV={s.expected_value:+.3f}")

    if low:
        print(f"\n  LOW CONFIDENCE ({len(low)}):")
        for s in low:
            print(f"    {s.matchup}  |  {s.side.upper()}  edge={s.edge:+.1%}")

    print(f"{'='*70}")


def run_scan(
    client: KalshiClient,
    strategy: NCAATradingStrategy,
    scan_count: int,
    min_edge: float,
) -> list[MatchupSignal]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markets   = client.get_ncaa_markets()

    # Also scan any non-tagged markets that mention NCAA in title
    if not markets:
        print(f"[{timestamp}] Scan #{scan_count}: 0 NCAA markets found")
        print("  No NCAA tournament markets open yet (expected after Selection Sunday)")
        return []

    print(f"[{timestamp}] Scan #{scan_count}: {len(markets)} NCAA markets")

    # Convert market dicts to the format evaluate_market expects
    signals: list[MatchupSignal] = []
    for market in markets:
        yes_price = market.get("yes_bid") or market.get("yes_ask") or 50
        m = {
            "ticker":    market.get("ticker", ""),
            "title":     market.get("title", ""),
            "yes_price": yes_price,
        }
        sig = strategy.evaluate_market(m)
        if sig is not None:
            signals.append(sig)

    print_signals(signals, min_edge)

    actionable = [s for s in signals if s.side != "pass" and abs(s.edge) >= min_edge]
    if actionable:
        path = save_opportunities(actionable)
        print(f"\n  Saved {len(actionable)} opportunities -> {path}")

    return signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Override env if passed via CLI
    if args.env:
        os.environ["KALSHI_ENV"] = args.env

    env = os.getenv("KALSHI_ENV", "demo")

    print(f"Kalshi NCAA Market Scanner")
    print(f"  Environment  : {env.upper()}")
    print(f"  Scan interval: {args.interval}s")
    print(f"  Min edge     : {args.min_edge:.1%}")
    print(f"  Model        : outputs/models/lightgbm_final_2026.pkl")
    print()

    strategy = NCAATradingStrategy()
    strategy.update_seeds(SEEDS_2026)
    print(f"  Seeds loaded: {len(SEEDS_2026)} teams\n")

    scan_count = 0

    try:
        with KalshiClient(env=env) as client:
            while True:
                scan_count += 1
                try:
                    run_scan(client, strategy, scan_count, args.min_edge)
                except Exception as exc:
                    print(f"  Error during scan #{scan_count}: {exc}")
                    logger.exception("Scan error")

                print(f"  Next scan in {args.interval}s... (Ctrl+C to stop)")
                print("-" * 70)
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nScanner stopped.")
        print(f"Total scans completed: {scan_count}")


if __name__ == "__main__":
    main()
