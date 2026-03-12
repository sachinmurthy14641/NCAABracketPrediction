"""Process all historical KenPom pre-tournament CSV files into a single dataset.

Handles three distinct column schemas across the 1997-2025 data:

    Schema A (1997-1998): AdjOE, AdjDE, AdjEM                       — no seed
    Schema B (1999-2000): AdjOE, AdjDE, EM  (Pythag present but skipped) — no seed
    Schema C (2001-2002): ORtg,  DRtg,  NetRtg                      — has seed
    Schema D (2003-2025): AdjOE, AdjDE, AdjEM                       — has seed

Output: data/processed/kenpom_pretourney_1997_2025.csv

Usage::

    python scripts/process_historical_kenpom.py
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.preprocessors import add_derived_features, normalize_team_names

# ---------------------------------------------------------------------------
# File → year mapping (actual filenames on disk, no underscores)
# ---------------------------------------------------------------------------
FILE_YEAR_MAP: dict[str, int] = {
    "summary97.csv": 1997,
    "summary98.csv": 1998,
    "summary99.csv": 1999,
    "summary00.csv": 2000,
    **{f"summary{yy:02d}_pt.csv": 2000 + yy for yy in range(1, 26)},
}

# ---------------------------------------------------------------------------
# Column rename candidates, tried left-to-right (first match wins).
# ---------------------------------------------------------------------------
_OFF_CANDIDATES = ["AdjOE", "ORtg"]   # → adj_off_eff
_DEF_CANDIDATES = ["AdjDE", "DRtg"]   # → adj_def_eff
_EM_CANDIDATES  = ["AdjEM", "NetRtg", "EM"]  # → adj_em
_TEMPO_COL      = "AdjTempo"           # → adj_tempo
_TEAM_COL       = "TeamName"           # → team
_SEASON_COL     = "Season"             # case-insensitive match

OUTPUT_COLS = [
    "team", "season", "adj_off_eff", "adj_def_eff", "adj_tempo",
    "adj_em", "efficiency_differential", "seed",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in columns (case-insensitive)."""
    col_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    return None


def _process_file(path: Path, year: int) -> tuple[pd.DataFrame | None, list[str]]:
    """Load and normalise one KenPom CSV.

    Returns:
        (DataFrame, warnings_list) — DataFrame is None if the file is unusable.
    """
    warns: list[str] = []

    try:
        df = pd.read_csv(path, na_values=["NULL", ""])
    except Exception as exc:
        warns.append(f"{path.name}: failed to read — {exc}")
        return None, warns

    cols = df.columns.tolist()

    # -- Season column (may be 'Season' or 'season') -----------------------
    season_col = _find_col(cols, [_SEASON_COL, "season"])
    if season_col:
        # Sanity-check: season in file should match year derived from filename
        file_years = df[season_col].dropna().unique()
        if len(file_years) == 1 and int(file_years[0]) != year:
            warns.append(
                f"{path.name}: Season column says {int(file_years[0])} "
                f"but filename implies {year}. Using filename year."
            )
    else:
        warns.append(f"{path.name}: No Season column found; using filename year {year}.")

    # -- Required columns --------------------------------------------------
    team_col   = _find_col(cols, [_TEAM_COL])
    off_col    = _find_col(cols, _OFF_CANDIDATES)
    def_col    = _find_col(cols, _DEF_CANDIDATES)
    em_col     = _find_col(cols, _EM_CANDIDATES)
    tempo_col  = _find_col(cols, [_TEMPO_COL])

    missing_required = []
    if not team_col:
        missing_required.append("TeamName")
    if not off_col:
        missing_required.append(f"adj_off_eff (tried: {_OFF_CANDIDATES})")
    if not def_col:
        missing_required.append(f"adj_def_eff (tried: {_DEF_CANDIDATES})")
    if not tempo_col:
        missing_required.append(_TEMPO_COL)

    if missing_required:
        warns.append(f"{path.name}: SKIPPED — missing required columns: {missing_required}")
        return None, warns

    if not em_col:
        warns.append(f"{path.name}: adj_em not found (tried {_EM_CANDIDATES}); will be NaN.")

    # -- Seed (optional) ---------------------------------------------------
    seed_col = _find_col(cols, ["seed"])

    # -- Build standardised DataFrame -------------------------------------
    rename = {
        team_col:  "team",
        off_col:   "adj_off_eff",
        def_col:   "adj_def_eff",
        tempo_col: "adj_tempo",
    }
    if em_col:
        rename[em_col] = "adj_em"
    if seed_col:
        rename[seed_col] = "seed"

    keep = list(rename.keys())
    out = df[keep].rename(columns=rename).copy()
    out["season"] = year

    if "adj_em" not in out.columns:
        out["adj_em"] = float("nan")
    if "seed" not in out.columns:
        out["seed"] = float("nan")

    # Convert seed to nullable int where possible
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")

    # -- Drop rows missing essential values --------------------------------
    before = len(out)
    out = out.dropna(subset=["team", "adj_off_eff", "adj_def_eff", "adj_tempo"])
    dropped = before - len(out)
    if dropped:
        warns.append(f"{path.name}: dropped {dropped} rows with missing core values.")

    # -- Preprocessing pipeline -------------------------------------------
    out = normalize_team_names(out)
    out = add_derived_features(out)

    # -- Reorder to canonical column order --------------------------------
    for col in OUTPUT_COLS:
        if col not in out.columns:
            out[col] = float("nan")
    out = out[OUTPUT_COLS]

    return out, warns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    frames: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    processed_years: list[int] = []
    skipped_files: list[str] = []

    for filename, year in sorted(FILE_YEAR_MAP.items(), key=lambda x: x[1]):
        path = RAW_DATA_DIR / filename
        if not path.exists():
            all_warnings.append(f"{filename}: file not found — skipping.")
            skipped_files.append(filename)
            continue

        df, warns = _process_file(path, year)
        all_warnings.extend(warns)

        if df is not None and not df.empty:
            frames.append(df)
            processed_years.append(year)
        else:
            skipped_files.append(filename)

    # -- Combine & save ----------------------------------------------------
    if not frames:
        print("ERROR: No data collected. Check warnings above.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "kenpom_pretourney_1997_2025.csv"
    combined.to_csv(out_path, index=False)

    # -- Summary -----------------------------------------------------------
    print("=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Years processed : {len(processed_years)}  ({min(processed_years)}–{max(processed_years)})")
    print(f"Total team-seasons: {len(combined):,}")
    print(f"Output          : {out_path}")

    if skipped_files:
        print(f"\nSkipped files ({len(skipped_files)}):")
        for f in skipped_files:
            print(f"  - {f}")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  ! {w}")
    else:
        print("\nNo warnings.")

    print("=" * 60)

    # -- Spot-check: seasons and row counts --------------------------------
    print("\nRows per season:")
    season_counts = combined.groupby("season").size().reset_index(name="teams")
    print(season_counts.to_string(index=False))


if __name__ == "__main__":
    main()
