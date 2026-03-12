"""Matchup feature engineering from preprocessed KenPom season data."""

import pandas as pd


def build_matchup_features(
    team_a: str,
    team_b: str,
    season_data: pd.DataFrame,
) -> dict:
    """Compute matchup-specific features between two teams.

    Args:
        team_a: Name of the first team (perspective team — positive edge = team_a favored).
        team_b: Name of the second team.
        season_data: DataFrame with columns [team, adj_off_eff, adj_def_eff,
                     adj_tempo, adj_em, efficiency_differential].

    Returns:
        Dict with 6 matchup features:
            - off_eff_advantage: team_a's offense vs team_b's defense
            - def_eff_advantage: team_b's offense vs team_a's defense (lower = better for team_a)
            - net_efficiency_edge: off_eff_advantage - def_eff_advantage
            - tempo_difference: team_a tempo minus team_b tempo
            - overall_rating_diff: team_a AdjEM minus team_b AdjEM
            - efficiency_differential_diff: team_a eff_diff minus team_b eff_diff

    Raises:
        ValueError: If either team is not found in season_data.
    """
    index = season_data.set_index("team")

    missing = [t for t in (team_a, team_b) if t not in index.index]
    if missing:
        available = sorted(index.index.tolist())
        raise ValueError(
            f"Team(s) not found in season data: {missing}. "
            f"Available teams ({len(available)}): {available}"
        )

    a = index.loc[team_a]
    b = index.loc[team_b]

    off_eff_advantage = a["adj_off_eff"] - b["adj_def_eff"]
    def_eff_advantage = b["adj_off_eff"] - a["adj_def_eff"]

    return {
        "off_eff_advantage": off_eff_advantage,
        "def_eff_advantage": def_eff_advantage,
        "net_efficiency_edge": off_eff_advantage - def_eff_advantage,
        "tempo_difference": a["adj_tempo"] - b["adj_tempo"],
        "overall_rating_diff": a["adj_em"] - b["adj_em"],
        "efficiency_differential_diff": a["efficiency_differential"] - b["efficiency_differential"],
    }


def build_feature_matrix(
    matchups: list[tuple[str, str]],
    season_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build a feature matrix from a list of team matchups.

    Args:
        matchups: List of (team_a, team_b) tuples.
        season_data: DataFrame with preprocessed KenPom season stats.

    Returns:
        DataFrame with one row per matchup. Columns are the 6 matchup features
        plus 'team_a' and 'team_b' identifiers.
    """
    rows = []
    for team_a, team_b in matchups:
        features = build_matchup_features(team_a, team_b, season_data)
        rows.append({"team_a": team_a, "team_b": team_b, **features})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    data = pd.read_csv("data/processed/kenpom_2026_clean.csv")

    print("=== Duke vs North Carolina ===")
    features = build_matchup_features("Duke", "North Carolina", data)
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Feature matrix (3 matchups) ===")
    matrix = build_feature_matrix(
        [("Duke", "North Carolina"), ("Alabama", "Auburn"), ("Houston", "Tennessee")],
        data,
    )
    print(matrix.to_string(index=False))
