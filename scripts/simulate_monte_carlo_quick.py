"""Monte Carlo simulation of the 2026 NCAA Tournament.

Loads the bracket structure and the production model, then simulates the
full tournament N times, tracking how often each team reaches each round.

Usage:
    python scripts/simulate_monte_carlo_quick.py
    python scripts/simulate_monte_carlo_quick.py --simulations 5000
    python scripts/simulate_monte_carlo_quick.py --simulations 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model import PlattModel  # noqa: F401 — needed for pickle
from src.kalshi.strategy import NCAATradingStrategy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BRACKET_PATH = ROOT / "data" / "brackets" / "bracket_2026_structure.json"
OUTPUT_DIR = ROOT / "outputs" / "brackets"
OUTPUT_PATH = OUTPUT_DIR / "monte_carlo_results_2026.json"

# Flat seed map for the 2026 field (used to seed model lookups).
# First Four teams included with their shared seed.
SEEDS_2026: dict[str, int] = {
    # East
    "Duke": 1, "Ohio State": 8, "St. John's": 5, "Kansas": 4,
    "Louisville": 6, "Michigan State": 3, "UCLA": 7, "UConn": 2,
    "Siena": 16, "TCU": 9, "Northern Iowa": 12, "Cal Baptist": 13,
    "South Florida": 11, "North Dakota State": 14, "UCF": 10, "Furman": 15,
    # West
    "Arizona": 1, "Villanova": 8, "Wisconsin": 5, "Arkansas": 4,
    "BYU": 6, "Gonzaga": 3, "Miami (Fla.)": 7, "Purdue": 2,
    "Long Island University": 16, "Utah State": 9, "High Point": 12, "Hawaii": 13,
    "Kennesaw State": 14, "Missouri": 10, "Queens": 15,
    # Midwest
    "Michigan": 1, "Georgia": 8, "Texas Tech": 5, "Alabama": 4,
    "Tennessee": 6, "Virginia": 3, "Kentucky": 7, "Iowa State": 2,
    "Saint Louis": 9, "Akron": 12, "Hofstra": 13, "Wright State": 14,
    "Santa Clara": 10, "Tennessee State": 15,
    # South
    "Florida": 1, "Clemson": 8, "Vanderbilt": 5, "Nebraska": 4,
    "North Carolina": 6, "Illinois": 3, "Saint Mary's": 7, "Houston": 2,
    "Iowa": 9, "McNeese": 12, "Troy": 13, "Penn": 14,
    "VCU": 11, "Texas A&M": 10, "Idaho": 15,
    # First Four teams (share the seed of the winner slot)
    "UMBC": 16, "Howard": 16,                 # FF1 → Midwest 16
    "Texas": 11, "NC State": 11,              # FF2 → West 11
    "Prairie View A&M": 16, "Lehigh": 16,    # FF3 → South 16
    "Miami (Ohio)": 11, "SMU": 11,           # FF4 → Midwest 11
}

# Round labels used in tracking and output.
# "champion" is the single tournament winner; "championship" holds both finalists.
ROUNDS = ["round_of_32", "sweet_16", "elite_8", "final_four", "championship", "champion"]
ROUND_DISPLAY = {
    "round_of_32":  "Round of 32",
    "sweet_16":     "Sweet 16",
    "elite_8":      "Elite 8",
    "final_four":   "Final Four",
    "championship": "Championship Game",
    "champion":     "Champion",
}


# ---------------------------------------------------------------------------
# Bracket loading
# ---------------------------------------------------------------------------

def load_bracket(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single tournament simulation
# ---------------------------------------------------------------------------

class TournamentSimulator:
    """Simulates one full tournament run and returns per-round appearances."""

    def __init__(self, bracket: dict, model: NCAATradingStrategy) -> None:
        self.bracket = bracket
        self.model = model

        # Map FF placeholder label → FF game_id (e.g. "TBD_FF1" → 1)
        self._ff_game_id: dict[str, int] = {}
        for g in bracket["first_four"]:
            self._ff_game_id[f"TBD_{g['ff_id']}"] = g["game_id"]

    def simulate(self) -> dict[str, set[str]]:
        """Run one tournament. Returns {round_key: {team_name, ...}}.

        Tracking convention: each round's set contains the PARTICIPANTS of that
        round (i.e., teams that won enough games to play in it), not the winners.
        This gives the correct "probability of reaching X round" semantics:
          round_of_64  → all 64 bracket teams (not tracked — trivially 100%)
          round_of_32  → won their R64 game
          sweet_16     → won their R32 game
          elite_8      → won their S16 game
          final_four   → won their E8 game (region champion)
          championship → won their F4 game (finalist)
          champion     → won the championship game
        """
        # game_id → (team_name, seed)
        winners: dict[int, tuple[str, int]] = {}

        appearances: dict[str, set[str]] = {r: set() for r in ROUNDS}

        # --- First Four ---
        for g in self.bracket["first_four"]:
            winner = self._play(g["team_a"], g["seed_a"], g["team_b"], g["seed_b"])
            seed = g["seed_a"]  # both teams share the same seed
            winners[g["game_id"]] = (winner, seed)

        # --- Regional rounds ---
        for region_key, region in self.bracket["regions"].items():
            r64_games = region["round_of_64"]

            # Resolve TBD_FF# placeholders then play each R64 game.
            # Both participants are implicitly in the R64 (not tracked).
            # The winner advances to the R32.
            for g in r64_games:
                team_a, seed_a = g["team_a"], g["seed_a"]
                team_b, seed_b = g["team_b"], g["seed_b"]

                if team_a.startswith("TBD_"):
                    team_a, seed_a = winners[self._ff_game_id[team_a]]
                if team_b.startswith("TBD_"):
                    team_b, seed_b = winners[self._ff_game_id[team_b]]

                w = self._play(team_a, seed_a, team_b, seed_b)
                w_seed = seed_a if w == team_a else seed_b
                winners[g["game_id"]] = (w, w_seed)
                appearances["round_of_32"].add(w)   # winner reached the R32

            # R32, S16, E8: participants are the winners of the previous round.
            # We add both participants to the current round's set, then add the
            # winner to the *next* round's set.
            progression = [
                ("round_of_32",  "sweet_16"),
                ("sweet_16",     "elite_8"),
                ("elite_8",      "final_four"),
            ]
            for current_round, next_round in progression:
                for g in region[current_round]:
                    id_a, id_b = g["winner_of"]
                    team_a, seed_a = winners[id_a]
                    team_b, seed_b = winners[id_b]
                    # Both teams are participating in current_round
                    appearances[current_round].add(team_a)
                    appearances[current_round].add(team_b)
                    w = self._play(team_a, seed_a, team_b, seed_b)
                    w_seed = seed_a if w == team_a else seed_b
                    winners[g["game_id"]] = (w, w_seed)
                    appearances[next_round].add(w)  # winner advances

        # --- Final Four ---
        # Both participants are the E8 winners already added to final_four above.
        # The F4 winner advances to the championship game.
        for g in self.bracket["final_four"]["games"]:
            id_a, id_b = g["winner_of"]
            team_a, seed_a = winners[id_a]
            team_b, seed_b = winners[id_b]
            appearances["final_four"].add(team_a)
            appearances["final_four"].add(team_b)
            w = self._play(team_a, seed_a, team_b, seed_b)
            w_seed = seed_a if w == team_a else seed_b
            winners[g["game_id"]] = (w, w_seed)
            appearances["championship"].add(w)  # finalist

        # --- Championship ---
        champ_g = self.bracket["championship"]
        id_a, id_b = champ_g["winner_of"]
        team_a, seed_a = winners[id_a]
        team_b, seed_b = winners[id_b]
        appearances["championship"].add(team_a)
        appearances["championship"].add(team_b)
        champion = self._play(team_a, seed_a, team_b, seed_b)
        appearances["champion"] = {champion}  # single tournament winner

        return appearances

    def _play(self, team_a: str, seed_a: int, team_b: str, seed_b: int) -> str:
        """Stochastically determine a game winner using the model."""
        prob = self.model._predict_win_probability(team_a, team_b, seed_a, seed_b)
        if prob is None:
            # Fallback: use seed to estimate probability (lower seed = better)
            prob = max(0.1, min(0.9, 0.5 + (seed_b - seed_a) * 0.04))
        return team_a if random.random() < prob else team_b


# ---------------------------------------------------------------------------
# Run N simulations and aggregate
# ---------------------------------------------------------------------------

def run_simulations(
    bracket: dict,
    model: NCAATradingStrategy,
    n: int,
) -> dict[str, dict[str, int]]:
    """Return {round_key: {team: appearance_count}} across all simulations."""
    simulator = TournamentSimulator(bracket, model)
    counts: dict[str, dict[str, int]] = {r: defaultdict(int) for r in ROUNDS}

    for i in range(n):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Simulating... {i + 1}/{n}", end="\r", flush=True)
        appearances = simulator.simulate()
        for round_key in ROUNDS:
            for team in appearances.get(round_key, set()):
                counts[round_key][team] += 1

    print(f"  Simulating... {n}/{n}  done.        ")
    return counts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(counts: dict[str, dict[str, int]], n: int) -> None:
    # (round_key, top_n_to_display)
    display_plan = [
        ("champion",      15),
        ("championship",  15),
        ("final_four",    20),
        ("elite_8",       30),
        ("sweet_16",      None),
    ]

    print(f"\n{'=' * 65}")
    print(f"  2026 NCAA TOURNAMENT — MONTE CARLO RESULTS  ({n:,} simulations)")
    print(f"{'=' * 65}")

    for round_key, top_n in display_plan:
        label = ROUND_DISPLAY[round_key]
        data = counts[round_key]
        if not data:
            continue
        sorted_teams = sorted(data.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            sorted_teams = sorted_teams[:top_n]

        top_label = f"top {top_n}" if top_n else "all teams"
        print(f"\n  {label.upper()} PROBABILITY ({top_label}):\n")
        for rank, (team, cnt) in enumerate(sorted_teams, 1):
            prob = cnt / n
            bar = "█" * int(prob * 30)
            seed = SEEDS_2026.get(team, "?")
            print(f"  {rank:2d}. #{seed:<2} {team:<26}  {prob:5.1%}  {bar}")

    # Summary line
    print(f"\n{'=' * 65}")
    champ_data = sorted(counts["champion"].items(), key=lambda x: x[1], reverse=True)
    if champ_data:
        top_team, top_cnt = champ_data[0]
        print(f"  Most likely champion: #{SEEDS_2026.get(top_team, '?')} {top_team} ({top_cnt/n:.1%})")
    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def save_results(counts: dict[str, dict[str, int]], n: int, path: Path) -> None:
    output = {
        "_meta": {
            "year": 2026,
            "simulations": n,
            "generated_at": datetime.now().isoformat(),
            "generated_by": "scripts/simulate_monte_carlo_quick.py",
        },
        "probabilities": {},
    }

    for round_key in ROUNDS:
        data = counts.get(round_key, {})
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        output["probabilities"][round_key] = [
            {
                "team": team,
                "seed": SEEDS_2026.get(team),
                "appearances": cnt,
                "probability": round(cnt / n, 4),
            }
            for team, cnt in sorted_items
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo NCAA Tournament Simulation")
    parser.add_argument("--simulations", "-n", type=int, default=1000,
                        help="Number of tournament simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"  Random seed: {args.seed}")

    print(f"\n  Loading bracket: {BRACKET_PATH}")
    bracket = load_bracket(BRACKET_PATH)

    print("  Loading model...")
    model = NCAATradingStrategy()
    model.update_seeds(SEEDS_2026)
    print(f"  Seeds loaded: {len(SEEDS_2026)} teams")
    print(f"\n  Running {args.simulations:,} simulations...\n")

    counts = run_simulations(bracket, model, args.simulations)

    print_results(counts, args.simulations)
    save_results(counts, args.simulations, OUTPUT_PATH)


if __name__ == "__main__":
    main()
