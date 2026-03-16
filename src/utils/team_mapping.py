"""Team name normalization: Kalshi → KenPom.

KenPom uses specific abbreviations and formats that may differ from
Kalshi market titles. This module provides a lookup table and fuzzy-
matching fallback so that team names resolve correctly regardless of
which source they come from.

Usage::

    from src.utils.team_mapping import normalize_team_name, find_closest_kenpom_team

    kp_name = normalize_team_name("UConn Huskies")   # -> "Connecticut"
    kp_name = normalize_team_name("Miami (FL)")       # -> "Miami FL"
"""

from __future__ import annotations

from difflib import get_close_matches

# ---------------------------------------------------------------------------
# Nickname suffixes to strip before lookup
# ---------------------------------------------------------------------------
_SUFFIXES_TO_STRIP = [
    " Huskies", " Blue Devils", " Tar Heels", " Wildcats", " Bulldogs",
    " Tigers", " Bears", " Bruins", " Trojans", " Ducks", " Cavaliers",
    " Volunteers", " Gators", " Longhorns", " Jayhawks", " Spartans",
    " Wolverines", " Hoosiers", " Buckeyes", " Hawkeyes", " Boilermakers",
    " Cornhuskers", " Cyclones", " Sooners", " Cowboys", " Cougars",
    " Utes", " Aggies", " Horned Frogs", " Mustangs", " Owls",
    " Panthers", " Eagles", " Cardinals", " Seminoles", " Crimson Tide",
    " War Eagles", " Rebels", " Razorbacks", " Mountaineers", " Yellow Jackets",
    " Demon Deacons", " Orange", " Pirates", " Gamecocks", " Terrapins",
    " Fighting Irish", " Zags", " Ramblers", " Flyers", " Musketeers",
    " Hoyas", " Friars", " Bluejays", " Monarchs", " Flames", " Bisons",
]

# ---------------------------------------------------------------------------
# Explicit Kalshi name → KenPom name mapping
# ---------------------------------------------------------------------------
TEAM_NAME_MAPPING: dict[str, str] = {
    # Connecticut variants
    "UConn":                "Connecticut",
    "UConn Huskies":        "Connecticut",
    "Uconn":                "Connecticut",

    # Miami variants
    "Miami":                "Miami FL",
    "Miami (FL)":           "Miami FL",
    "Miami FL":             "Miami FL",
    "Miami (Ohio)":         "Miami OH",
    "Miami Ohio":           "Miami OH",

    # Saint/St. variants
    "Saint Mary's":         "St. Mary's",
    "St Mary's":            "St. Mary's",
    "St. Marys":            "St. Mary's",
    "Saint Mary's (CA)":    "St. Mary's",
    "Saint Louis":          "Saint Louis",
    "St. Louis":            "Saint Louis",
    "Saint Joseph's":       "St. Joseph's",
    "St Joseph's":          "St. Joseph's",
    "Saint Peter's":        "St. Peter's",
    "St Peter's":           "St. Peter's",
    "Saint John's":         "St. John's",
    "St John's":            "St. John's",
    "Saint Bonaventure":    "St. Bonaventure",
    "St Bonaventure":       "St. Bonaventure",
    "Mount St. Mary's":     "Mount St. Mary's",
    "Mt. St. Mary's":       "Mount St. Mary's",
    "Mount Saint Mary's":   "Mount St. Mary's",

    # Abbreviated names that Kalshi might spell out
    "Southern California":  "USC",
    "Louisiana State":      "LSU",
    "Virginia Commonwealth": "VCU",
    "Nevada Las Vegas":     "UNLV",
    "Central Florida":      "UCF",
    "Southern Methodist":   "SMU",
    "Brigham Young":        "BYU",
    "Texas Christian":      "TCU",

    # Abbreviations Kalshi might use that need expanding
    "UNC":                  "North Carolina",
    "NC State":             "NC State",
    "NCSU":                 "NC State",
    "Pitt":                 "Pittsburgh",
    "Ole Miss":             "Mississippi",
    "Miss State":           "Mississippi St.",
    "Mississippi State":    "Mississippi St.",
    "Michigan State":       "Michigan St.",
    "Ohio State":           "Ohio St.",
    "Kansas State":         "Kansas St.",
    "Iowa State":           "Iowa St.",
    "Arizona State":        "Arizona St.",
    "Colorado State":       "Colorado St.",
    "Florida State":        "Florida St.",
    "Oregon State":         "Oregon St.",
    "Washington State":     "Washington St.",
    "Penn State":           "Penn St.",
    "Boise State":          "Boise St.",
    "Utah State":           "Utah St.",
    "New Mexico State":     "New Mexico St.",
    "Fresno State":         "Fresno St.",
    "San Diego State":      "San Diego St.",
    "North Dakota State":   "North Dakota St.",
    "South Dakota State":   "South Dakota St.",
    "Weber State":          "Weber St.",
    "Montana State":        "Montana St.",
    "Murray State":         "Murray St.",
    "Morehead State":       "Morehead St.",
    "Indiana State":        "Indiana St.",
    "Illinois State":       "Illinois St.",
    "Ball State":           "Ball St.",
    "Kent State":           "Kent St.",
    "Cleveland State":      "Cleveland St.",
    "Jacksonville State":   "Jacksonville St.",
    "Sam Houston State":    "Sam Houston St.",
    "Stephen F. Austin":    "Stephen F. Austin",
    "SFA":                  "Stephen F. Austin",

    # Other common variants
    "ETSU":                 "East Tennessee St.",
    "East Tennessee State": "East Tennessee St.",
    "SIUE":                 "SIU Edwardsville",
    "Southern Illinois Edwardsville": "SIU Edwardsville",
    "FIU":                  "FIU",
    "Florida International": "FIU",
    "UTSA":                 "UTSA",
    "UT San Antonio":       "UTSA",
    "UTEP":                 "UTEP",
    "UT El Paso":           "UTEP",
    "Texas El Paso":        "UTEP",
    "UAB":                  "UAB",
    "Alabama-Birmingham":   "UAB",
    "UMass":                "Massachusetts",
    "Massachusetts":        "Massachusetts",
    "UMBC":                 "UMBC",
    "Loyola Chicago":       "Loyola Chicago",
    "Loyola (IL)":          "Loyola Chicago",
    "Loyola-Chicago":       "Loyola Chicago",
    "Loyola MD":            "Loyola MD",
    "Loyola (MD)":          "Loyola MD",
    "George Washington":    "George Washington",
    "GWU":                  "George Washington",
    "American":             "American",
    "American University":  "American",
    "Norfolk State":        "Norfolk St.",
    "Alabama State":        "Alabama St.",
    "Grambling State":      "Grambling St.",
    "Prairie View":         "Prairie View A&M",
    "Prairie View A&M":     "Prairie View A&M",
    "Texas A&M Corpus Christi": "Texas A&M Corpus Christi",
    "TAMUCC":               "Texas A&M Corpus Christi",
    "Nebraska Omaha":       "Nebraska Omaha",
    "UNCG":                 "UNC Greensboro",
    "UNC Greensboro":       "UNC Greensboro",
    "VCU Rams":             "VCU",
    "Wichita State":        "Wichita St.",
    "Wichita St":           "Wichita St.",
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    """Convert a Kalshi team name to its KenPom equivalent.

    Steps:
    1. Strip common mascot suffixes (e.g. ' Huskies', ' Blue Devils')
    2. Look up in explicit TEAM_NAME_MAPPING
    3. Return cleaned name if no mapping found

    Args:
        name: Team name from a Kalshi market title or ticker.

    Returns:
        Canonical KenPom team name, or the cleaned input if no mapping exists.
    """
    name = name.strip()

    # Strip mascot suffixes
    for suffix in _SUFFIXES_TO_STRIP:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break

    # Explicit mapping
    if name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[name]

    return name


def find_closest_kenpom_team(
    name: str,
    kenpom_teams: list[str],
    cutoff: float = 0.75,
) -> str | None:
    """Fuzzy-match a team name against the KenPom roster.

    Used as a last-resort fallback when normalize_team_name() still doesn't
    find an exact match in the KenPom DataFrame.

    Args:
        name: Candidate team name (should be normalized first).
        kenpom_teams: List of canonical team names from KenPom data.
        cutoff: Minimum similarity score (0–1). Lower = more permissive.

    Returns:
        Best matching KenPom team name, or None if no match above cutoff.
    """
    normalized = normalize_team_name(name)
    matches = get_close_matches(normalized, kenpom_teams, n=1, cutoff=cutoff)
    return matches[0] if matches else None
