"""Process Kaggle March Madness data into a unified tournament results dataset.

Loads game results, team names, and seeds from the Kaggle files and produces
a clean CSV with one row per game for seasons 1997-2025.

Output: data/historical/tournament_results_1997_2025.csv

Usage::

    python scripts/process_tournament_results.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR
from src.data.preprocessors import normalize_team_names

KAGGLE_DIR = Path(__file__).resolve().parent.parent / "data" / "historical" / "kaggle"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "historical" / "tournament_results_1997_2025.csv"

SEASON_MIN = 1997
SEASON_MAX = 2025

# Kaggle abbreviated names → KenPom full names
KAGGLE_TO_KENPOM: dict[str, str] = {
    "Abilene Chr":      "Abilene Christian",
    "Alabama St":       "Alabama St.",
    "American Univ":    "American",
    "Appalachian St":   "Appalachian St.",
    "Arizona St":       "Arizona St.",
    "Ark Little Rock":  "Arkansas Little Rock",
    "Ark Pine Bluff":   "Arkansas Pine Bluff",
    "Boise St":         "Boise St.",
    "Boston Univ":      "Boston University",
    "C Michigan":       "Central Michigan",
    "CS Bakersfield":   "Cal St. Bakersfield",
    "CS Fullerton":     "Cal St. Fullerton",
    "CS Northridge":    "Cal St. Northridge",
    "Central Conn":     "Central Connecticut",
    "Cleveland St":     "Cleveland St.",
    "Coastal Car":      "Coastal Carolina",
    "Col Charleston":   "College of Charleston",
    "Colorado St":      "Colorado St.",
    "Coppin St":        "Coppin St.",
    "Delaware St":      "Delaware St.",
    "E Kentucky":       "Eastern Kentucky",
    "E Washington":     "Eastern Washington",
    "ETSU":             "East Tennessee St.",
    "F Dickinson":      "Fairleigh Dickinson",
    "FL Atlantic":      "Florida Atlantic",
    "FL Gulf Coast":    "Florida Gulf Coast",
    "FGCU":             "Florida Gulf Coast",
    "Florida St":       "Florida St.",
    "Fresno St":        "Fresno St.",
    "G Washington":     "George Washington",
    "Georgia St":       "Georgia St.",
    "Grambling":        "Grambling St.",
    "McNeese St":       "McNeese St.",
    "NE Omaha":         "Nebraska Omaha",
    "St Francis PA":    "St. Francis PA",
    "IL Chicago":       "Illinois Chicago",
    "Indiana St":       "Indiana St.",
    "Iowa St":          "Iowa St.",
    "Jackson St":       "Jackson St.",
    "Jacksonville St":  "Jacksonville St.",
    "Kansas St":        "Kansas St.",
    "Kennesaw":         "Kennesaw St.",
    "Long Beach St":    "Long Beach St.",
    "Loyola-Chicago":   "Loyola Chicago",
    "MS Valley St":     "Mississippi Valley St.",
    "MTSU":             "Middle Tennessee",
    "Michigan St":      "Michigan St.",
    "Mississippi St":   "Mississippi St.",
    "Monmouth NJ":      "Monmouth",
    "Montana St":       "Montana St.",
    "Morehead St":      "Morehead St.",
    "Morgan St":        "Morgan St.",
    "Mt St Mary's":     "Mount St. Mary's",
    "Murray St":        "Murray St.",
    "N Colorado":       "Northern Colorado",
    "N Dakota St":      "North Dakota St.",
    "N Kentucky":       "Northern Kentucky",
    "NC A&T":           "North Carolina A&T",
    "NC Central":       "North Carolina Central",
    "New Mexico St":    "New Mexico St.",
    "Norfolk St":       "Norfolk St.",
    "Northwestern LA":  "Northwestern St.",
    "Ohio St":          "Ohio St.",
    "Oklahoma St":      "Oklahoma St.",
    "Oregon St":        "Oregon St.",
    "Penn St":          "Penn St.",
    "Portland St":      "Portland St.",
    "Prairie View":     "Prairie View A&M",
    "S Carolina St":    "South Carolina St.",
    "S Dakota St":      "South Dakota St.",
    "S Illinois":       "Southern Illinois",
    "SE Louisiana":     "Southeastern Louisiana",
    "SE Missouri St":   "Southeast Missouri St.",
    "SF Austin":        "Stephen F. Austin",
    "SUNY Albany":      "Albany",
    "Sam Houston St":   "Sam Houston St.",
    "San Diego St":     "San Diego St.",
    "Southern Univ":    "Southern",
    "St Bonaventure":   "St. Bonaventure",
    "St John's":        "St. John's",
    "St Joseph's PA":   "St. Joseph's",
    "St Louis":         "Saint Louis",
    "St Mary's CA":     "St. Mary's",
    "St Peter's":       "St. Peter's",
    "TAM C. Christi":   "Texas A&M Corpus Christi",
    "TX Southern":      "Texas Southern",
    "UT San Antonio":   "UTSA",
    "Utah St":          "Utah St.",
    "W Michigan":       "Western Michigan",
    "WI Green Bay":     "Wisconsin Green Bay",
    "WI Milwaukee":     "Wisconsin Milwaukee",
    "WKU":              "Western Kentucky",
    "Washington St":    "Washington St.",
    "Weber St":         "Weber St.",
    "Wichita St":       "Wichita St.",
    "Wright St":        "Wright St.",
}


def load_team_map() -> dict[int, str]:
    """Return {TeamID: TeamName} from MTeams.csv."""
    df = pd.read_csv(KAGGLE_DIR / "MTeams.csv")
    return dict(zip(df["TeamID"], df["TeamName"]))


def load_seeds() -> pd.DataFrame:
    """Return DataFrame with [Season, TeamID, seed] where seed is an integer."""
    df = pd.read_csv(KAGGLE_DIR / "MNCAATourneySeeds.csv")
    # Seed strings like 'W01', 'X16', 'Y12a' — extract first 1-2 digits after the letter
    df["seed"] = df["Seed"].str.extract(r"([0-9]+)")[0].astype(int)
    return df[["Season", "TeamID", "seed"]]


def load_results(team_map: dict[int, str], seeds: pd.DataFrame) -> pd.DataFrame:
    """Load game results and join team names + seeds."""
    df = pd.read_csv(KAGGLE_DIR / "MNCAATourneyDetailedResults.csv")

    # Filter to target seasons
    df = df[(df["Season"] >= SEASON_MIN) & (df["Season"] <= SEASON_MAX)].copy()

    # Map IDs to names
    df["team_a"] = df["WTeamID"].map(team_map)
    df["team_b"] = df["LTeamID"].map(team_map)

    unmapped = df["team_a"].isna().sum() + df["team_b"].isna().sum()
    if unmapped:
        print(f"  WARNING: {unmapped} team IDs could not be mapped to names.")

    df = df.dropna(subset=["team_a", "team_b"])

    # Join seeds for winners
    seed_map = seeds.set_index(["Season", "TeamID"])["seed"]
    df["team_a_seed"] = df.set_index(["Season", "WTeamID"]).index.map(seed_map.get)
    df["team_b_seed"] = df.set_index(["Season", "LTeamID"]).index.map(seed_map.get)

    # Fallback: merge-based seed lookup (more robust)
    seeds_w = seeds.rename(columns={"TeamID": "WTeamID", "seed": "team_a_seed"})
    seeds_l = seeds.rename(columns={"TeamID": "LTeamID", "seed": "team_b_seed"})
    df = df.drop(columns=["team_a_seed", "team_b_seed"], errors="ignore")
    df = df.merge(seeds_w, on=["Season", "WTeamID"], how="left")
    df = df.merge(seeds_l, on=["Season", "LTeamID"], how="left")

    df["winner"] = 1  # team_a (winner) always wins

    out = df[["Season", "team_a", "team_b", "team_a_seed", "team_b_seed",
              "WScore", "LScore", "winner"]].rename(columns={
        "Season": "season",
        "WScore": "score_a",
        "LScore": "score_b",
    })

    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Kaggle→KenPom mapping then standard name normalization to both team columns."""
    for col in ("team_a", "team_b"):
        df[col] = df[col].replace(KAGGLE_TO_KENPOM)
        tmp = df[[col]].rename(columns={col: "team"})
        tmp = normalize_team_names(tmp)
        df[col] = tmp["team"]
    return df


def main() -> None:
    print("Loading team map...")
    team_map = load_team_map()
    print(f"  {len(team_map):,} teams loaded.")

    print("Loading seeds...")
    seeds = load_seeds()
    print(f"  {len(seeds):,} seed records loaded.")

    print("Loading and processing game results...")
    results = load_results(team_map, seeds)

    print("Normalizing team names...")
    results = normalize(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total games     : {len(results):,}")
    print(f"Seasons covered : {results['season'].min()}–{results['season'].max()}")
    print(f"Output          : {OUTPUT_PATH}")

    print("\nGames per season:")
    print(results.groupby("season").size().to_string())

    print("\nFirst 5 games:")
    print(results.head().to_string(index=False))

    # Check for teams in results not in KenPom processed data
    kenpom_path = PROCESSED_DATA_DIR / "kenpom_pretourney_1997_2025.csv"
    if kenpom_path.exists():
        kenpom = pd.read_csv(kenpom_path)
        kenpom_teams = set(kenpom["team"].unique())
        result_teams = set(results["team_a"].unique()) | set(results["team_b"].unique())
        mismatches = result_teams - kenpom_teams
        if mismatches:
            print(f"\nTeam name mismatches vs KenPom ({len(mismatches)}):")
            for t in sorted(mismatches):
                print(f"  - {t}")
        else:
            print("\nNo team name mismatches vs KenPom data.")

    print("=" * 60)


if __name__ == "__main__":
    main()
