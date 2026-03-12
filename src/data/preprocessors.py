"""Preprocessing utilities for KenPom team stats DataFrames."""

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Common KenPom name → standard name mappings.
# Add entries here as mismatches surface against other data sources
# (e.g. bracket data, historical results).
# ---------------------------------------------------------------------------
_NAME_OVERRIDES: dict[str, str] = {
    "N.C. State": "NC State",
    "Texas A&M Corpus Chris": "Texas A&M Corpus Christi",
    "Saint Joseph's": "St. Joseph's",
    "Saint Mary's": "St. Mary's",
    "Saint Peter's": "St. Peter's",
    "William & Mary": "William & Mary",  # already fine, kept for clarity
}


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize team names in the ``team`` column.

    Steps applied (in order):
    1. Strip leading/trailing whitespace.
    2. Collapse internal runs of whitespace to a single space.
    3. Apply explicit overrides from ``_NAME_OVERRIDES``.

    Args:
        df: DataFrame containing a ``team`` column.

    Returns:
        New DataFrame with the ``team`` column cleaned. All other columns
        are left unchanged.
    """
    if "team" not in df.columns:
        raise ValueError("DataFrame must contain a 'team' column.")

    out = df.copy()
    out["team"] = (
        out["team"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    out["team"] = out["team"].replace(_NAME_OVERRIDES)
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features derived from KenPom efficiency columns.

    Features added:
    - ``efficiency_differential``: ``adj_off_eff - adj_def_eff``
      Measures how much better a team's offense is than its defense.
      Positive values indicate offence-dominant teams; negative values
      indicate defence-dominant teams. Strongly correlated with ``adj_em``
      but expressed in an interpretable unit (points per 100 possessions).

    Args:
        df: DataFrame with ``adj_off_eff`` and ``adj_def_eff`` columns.

    Returns:
        New DataFrame with the additional derived column appended.
    """
    required = {"adj_off_eff", "adj_def_eff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    out = df.copy()
    out["efficiency_differential"] = out["adj_off_eff"] - out["adj_def_eff"]
    return out
