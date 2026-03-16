"""One-time NCAA market scan for Selection Sunday night.

Run once after the bracket is announced and seeds are entered below.

Usage::

    python scripts/scan_sunday_night.py
    python scripts/scan_sunday_night.py --env live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import PlattModel  # noqa: F401 — needed for pickle
from src.kalshi.client import KalshiClient
from src.kalshi.strategy import NCAATradingStrategy, MatchupSignal

OUTPUT_PATH = Path("outputs/sunday_night_scan.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# SEEDS — fill in after Selection Sunday bracket is announced
# ===========================================================================
SEEDS_2026 = {
    # No. 1 seeds
    'Duke': 1,
    'Arizona': 1,
    'Michigan': 1,
    'Florida': 1,
    
    # No. 2 seeds
    'Houston': 2,
    'UConn': 2,
    'Iowa St.': 2,
    'Purdue': 2,
    
    # No. 3 seeds
    'Michigan St.': 3,
    'Illinois': 3,
    'Gonzaga': 3,
    'Virginia': 3,
    
    # No. 4 seeds
    'Nebraska': 4,
    'Alabama': 4,
    'Kansas': 4,
    'Arkansas': 4,
    
    # No. 5 seeds
    'Vanderbilt': 5,
    "St. John's": 5,
    'Texas Tech': 5,
    'Wisconsin': 5,
    
    # No. 6 seeds
    'Tennessee': 6,
    'North Carolina': 6,
    'Louisville': 6,
    'BYU': 6,
    
    # No. 7 seeds
    'Kentucky': 7,
    "Saint Mary's": 7,
    'Miami': 7,  # Miami (FL)
    'UCLA': 7,
    
    # No. 8 seeds
    'Clemson': 8,
    'Villanova': 8,
    'Ohio St.': 8,
    'Georgia': 8,
    
    # No. 9 seeds
    'Utah St.': 9,
    'TCU': 9,
    'Saint Louis': 9,
    'Iowa': 9,
    
    # No. 10 seeds
    'Santa Clara': 10,
    'UCF': 10,
    'Missouri': 10,
    'Texas A&M': 10,
    
    # No. 11 seeds (6 teams - First Four)
    'NC State': 11,
    'Texas': 11,
    'SMU': 11,
    'Miami OH': 11,
    'VCU': 11,
    'South Florida': 11,
    
    # No. 12 seeds
    'McNeese': 12,
    'Akron': 12,
    'Northern Iowa': 12,
    'High Point': 12,
    
    # No. 13 seeds
    'Cal Baptist': 13,
    'Hofstra': 13,
    'Troy': 13,
    'Hawaii': 13,
    
    # No. 14 seeds
    'North Dakota St.': 14,
    'Penn': 14,
    'Wright St.': 14,
    'Kennesaw St.': 14,
    
    # No. 15 seeds
    'Tennessee St.': 15,
    'Idaho': 15,
    'Furman': 15,
    'Queens': 15,  # Queens (NC)
    
    # No. 16 seeds (6 teams - First Four)
    'Siena': 16,
    'LIU': 16,
    'Howard': 16,
    'UMBC': 16,
    'Lehigh': 16,
    'Prairie View': 16,
}
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selection Sunday one-time scan")
    parser.add_argument("--env", type=str, default=None, help="'demo' or 'live' (overrides .env)")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge to display (default 0.05)")
    return parser.parse_args()


def display_signals(signals: list[MatchupSignal], min_edge: float) -> None:
    actionable = sorted(
        [s for s in signals if s.side != "pass" and abs(s.edge) >= min_edge],
        key=lambda s: abs(s.edge), reverse=True,
    )
    high   = [s for s in actionable if s.confidence == "high"]
    medium = [s for s in actionable if s.confidence == "medium"]
    low    = [s for s in actionable if s.confidence == "low"]

    print(f"\n{'='*70}")
    print(f"  SCAN RESULTS: {len(actionable)} actionable / {len(signals)} total signals")
    print(f"{'='*70}")

    if high:
        print(f"\n  HIGH CONFIDENCE ({len(high)}):\n")
        for i, s in enumerate(high, 1):
            print(f"  {i}. {s.matchup}")
            print(f"     Ticker    : {s.ticker or '(not yet listed)'}")
            print(f"     Signal    : {s.side.upper()} {s.team}")
            print(f"     Market    : {s.market_implied_prob:.1%}  |  Model: {s.model_prob:.1%}  |  Edge: {s.edge:+.1%}")
            print(f"     EV        : {s.expected_value:+.3f} per $1 risked")
            print()

    if medium:
        print(f"  MEDIUM CONFIDENCE ({len(medium)}):\n")
        for i, s in enumerate(medium, 1):
            print(f"  {i}. {s.matchup}")
            print(f"     {s.side.upper()} {s.team}  |  edge={s.edge:+.1%}  |  EV={s.expected_value:+.3f}")
        print()

    if low:
        print(f"  LOW CONFIDENCE ({len(low)}):")
        for s in low:
            print(f"    {s.matchup}  |  {s.side.upper()}  edge={s.edge:+.1%}")
        print()

    if not actionable:
        print("  No opportunities meeting minimum edge criteria.")
        print("  Either markets are efficiently priced or seeds haven't been loaded yet.")

    if high:
        print(f"\n{'='*70}")
        print("  RECOMMENDED TRADES (HIGH CONFIDENCE):")
        print(f"{'='*70}\n")
        for s in high[:5]:
            print(f"  {s.matchup}")
            print(f"    Action : BUY {s.side.upper()} on {s.team}")
            print(f"    Price  : ~{s.market_implied_prob:.0%}  (model says {s.model_prob:.0%})")
            print(f"    Edge   : {s.edge:+.1%}   EV: {s.expected_value:+.3f}")
            print(f"    Ticker : {s.ticker or '(look up on Kalshi)'}")
            print()


def save_results(signals: list[MatchupSignal]) -> None:
    records = []
    for s in signals:
        records.append({
            "ticker":         s.ticker,
            "matchup":        s.matchup,
            "team_a":         s.team_a,
            "team_b":         s.team_b,
            "team_a_seed":    s.team_a_seed,
            "team_b_seed":    s.team_b_seed,
            "side":           s.side,
            "team":           s.team,
            "model_prob":     round(s.model_prob, 4),
            "market_prob":    round(s.market_implied_prob, 4),
            "edge":           round(s.edge, 4),
            "expected_value": round(s.expected_value, 4),
            "confidence":     s.confidence,
            "scanned_at":     datetime.now().isoformat(),
        })
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Full results saved -> {OUTPUT_PATH}")


def main() -> None:
    args = parse_args()
    if args.env:
        os.environ["KALSHI_ENV"] = args.env

    env = os.getenv("KALSHI_ENV", "demo")

    print("NCAA Tournament Market Scan — Selection Sunday")
    print(f"  Environment : {env.upper()}")
    print(f"  Min edge    : {args.min_edge:.1%}")
    print(f"  Scanned at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    strategy = NCAATradingStrategy()

    if SEEDS_2026:
        strategy.update_seeds(SEEDS_2026)
        print(f"  Seeds loaded: {len(SEEDS_2026)} teams")
    else:
        print("  WARNING: SEEDS_2026 is empty — fill in the bracket at the top of this script.")
        print("           Using neutral seed=8 for all teams (predictions will be less accurate).")

    print()

    with KalshiClient(env=env) as client:
        markets = client.get_ncaa_markets()

    print(f"  NCAA markets found: {len(markets)}")

    if not markets:
        print("\n  No tournament markets available yet.")
        print("  Markets typically go live on Selection Sunday evening.")
        print("  Try again after ~6pm ET on Selection Sunday.")
        return

    print()
    signals: list[MatchupSignal] = []
    for market in markets:
        yes_price = market.get("yes_bid") or market.get("yes_ask") or 50
        sig = strategy.evaluate_market({
            "ticker":    market.get("ticker", ""),
            "title":     market.get("title", ""),
            "yes_price": yes_price,
        })
        if sig is not None:
            signals.append(sig)

    display_signals(signals, args.min_edge)
    print()
    save_results(signals)


if __name__ == "__main__":
    main()
