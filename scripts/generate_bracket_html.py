"""Generate a self-contained HTML bracket visualization for the 2026 NCAA Tournament.

Reads the official bracket and Monte Carlo results, then writes a single HTML file
that can be opened directly in any browser with no internet connection required.

Usage:
    python scripts/generate_bracket_html.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRACKET_PATH  = ROOT / "data" / "bracket_2026_official.json"
MC_PATH       = ROOT / "outputs" / "brackets" / "monte_carlo_results_2026.json"
OUTPUT_PATH   = ROOT / "outputs" / "brackets" / "bracket_2026_visualization.html"

# Final Four pairing: which regions face off in each semifinal
# East/South on the left half, West/Midwest on the right half
LEFT_REGIONS  = ["East",    "South"]
RIGHT_REGIONS = ["West",    "Midwest"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[dict, dict]:
    with open(BRACKET_PATH) as f:
        bracket = json.load(f)
    with open(MC_PATH) as f:
        mc = json.load(f)
    return bracket, mc


def build_prob_lookup(mc: dict) -> dict[str, dict[str, float]]:
    """Return {round_key: {team_name: probability}}."""
    lookup: dict[str, dict[str, float]] = {}
    for rnd, entries in mc["probabilities"].items():
        lookup[rnd] = {e["team"]: e["probability"] for e in entries}
    return lookup


def resolve_ff_name(team_name: str, bracket: dict) -> str:
    """Convert FF_winner placeholders to readable labels."""
    if not team_name.startswith("FF") or not team_name.endswith("_winner"):
        return team_name
    ff_id = team_name.replace("_winner", "")   # e.g. "FF2"
    for g in bracket["first_four"]:
        if g["id"] == ff_id:
            return f"{g['team_a']} / {g['team_b']}"
    return team_name


# ---------------------------------------------------------------------------
# Probability → color (white → gold for champions)
# ---------------------------------------------------------------------------

def champ_color(prob: float | None) -> str:
    """Map champion probability to a background hex color.

    0%   → #f8f9fa  (near-white)
    ~5%  → light yellow
    ~50% → gold
    95%+ → deep gold / amber
    """
    if prob is None or prob == 0:
        return "#f8f9fa"
    # Clamp and scale 0–1
    p = max(0.0, min(1.0, prob))
    # Interpolate: white(248,249,250) → amber(255,180,0)
    r = int(248 + (255 - 248) * p)
    g = int(249 + (180 - 249) * p)
    b = int(250 + (0   - 250) * p)
    return f"rgb({r},{g},{b})"


def ff_reach_color(prob: float | None) -> str:
    """Map Final Four probability to a green-tinted background."""
    if prob is None or prob == 0:
        return "#f8f9fa"
    p = max(0.0, min(1.0, prob))
    r = int(248 - 80  * p)
    g = int(249 - 40  * p)
    b = int(250 - 130 * p)
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# HTML component builders
# ---------------------------------------------------------------------------

def team_slot(seed: int, name: str, probs: dict[str, dict], note: str = "") -> str:
    """Render one team row inside a game card."""
    champ_p   = probs.get("champion",    {}).get(name, 0)
    ff_p      = probs.get("final_four",  {}).get(name, 0)
    bg        = champ_color(champ_p)
    ff_bg     = ff_reach_color(ff_p)

    champ_str = f"{champ_p*100:.1f}%" if champ_p >= 0.001 else "—"
    ff_str    = f"{ff_p*100:.0f}%"    if ff_p    >= 0.01  else "—"

    is_ff_team = "FF" in name and "/" in name  # first-four placeholder

    return f"""
      <div class="team-row" style="background:{bg}" title="Champion: {champ_str} | Final Four: {ff_str}">
        <span class="seed">{'??' if is_ff_team else seed}</span>
        <span class="tname {'ff-team' if is_ff_team else ''}">{name}</span>
        <span class="champ-pct">{champ_str}</span>
      </div>"""


def game_card(seed_a: int, team_a: str, seed_b: int, team_b: str, probs: dict) -> str:
    return f"""
    <div class="game-card">
      {team_slot(seed_a, team_a, probs)}
      {team_slot(seed_b, team_b, probs)}
    </div>"""


def region_r64_games(region_data: dict, bracket: dict, probs: dict) -> list[tuple]:
    """Return ordered list of (seed_a, team_a, seed_b, team_b) for a region's R64."""
    matchups = region_data["matchups"]
    SEED_ORDER = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    index = {(m["seed_a"], m["seed_b"]): m for m in matchups}
    result = []
    for sa, sb in SEED_ORDER:
        m = index[(sa, sb)]
        tb = resolve_ff_name(m["team_b"], bracket)
        result.append((m["seed_a"], m["team_a"], m["seed_b"], tb))
    return result


def predicted_path(region_data: dict, bracket: dict, probs: dict) -> list[list[tuple]]:
    """
    Simulate the chalk bracket path using champion-probability-ranked picks.
    Returns a list of rounds, each round a list of (seed, team) tuples (winners).
    """
    # Build the region games in order
    matchups_r64 = region_r64_games(region_data, bracket, probs)

    def winner(sa, ta, sb, tb):
        """Pick the team with higher champion probability; fall back to lower seed."""
        pa = probs.get("champion", {}).get(ta, 0) + probs.get("final_four", {}).get(ta, 0)
        pb = probs.get("champion", {}).get(tb, 0) + probs.get("final_four", {}).get(tb, 0)
        if pa >= pb:
            return sa, ta
        return sb, tb

    rounds: list[list[tuple]] = []

    # R32: pick winners of each R64 pair
    r32 = [winner(sa, ta, sb, tb) for (sa, ta, sb, tb) in matchups_r64]
    rounds.append(r32)

    # S16: pair adjacent R32 winners
    s16 = [winner(r32[i][0], r32[i][1], r32[i+1][0], r32[i+1][1]) for i in range(0, 8, 2)]
    rounds.append(s16)

    # E8
    e8 = [winner(s16[0][0], s16[0][1], s16[1][0], s16[1][1]),
          winner(s16[2][0], s16[2][1], s16[3][0], s16[3][1])]
    rounds.append(e8)

    # Region champ
    champ = [winner(e8[0][0], e8[0][1], e8[1][0], e8[1][1])]
    rounds.append(champ)

    return rounds


# ---------------------------------------------------------------------------
# HTML bracket column builders
# ---------------------------------------------------------------------------

def r64_column(games: list[tuple], probs: dict) -> str:
    cards = "".join(
        game_card(sa, ta, sb, tb, probs)
        for sa, ta, sb, tb in games
    )
    return f'<div class="round r64">{cards}</div>'


def later_round_column(entries: list[tuple], round_class: str, probs: dict) -> str:
    """Render a column of individual team slots (predicted winners of a round)."""
    slots = ""
    for seed, name in entries:
        bg = champ_color(probs.get("champion", {}).get(name, 0))
        ff_p = probs.get("final_four", {}).get(name, 0)
        champ_p = probs.get("champion", {}).get(name, 0)
        champ_str = f"{champ_p*100:.1f}%" if champ_p >= 0.001 else "—"
        slots += f"""
        <div class="later-slot" style="background:{bg}" title="Champion prob: {champ_str}">
          <span class="seed">{seed}</span>
          <span class="tname">{name}</span>
          <span class="champ-pct">{champ_str}</span>
        </div>"""
    return f'<div class="round {round_class}">{slots}</div>'


def region_bracket(region_name: str, region_data: dict, bracket: dict,
                   probs: dict, flip: bool = False) -> str:
    """Build the full bracket columns for one region."""
    games     = region_r64_games(region_data, bracket, probs)
    all_rounds = predicted_path(region_data, bracket, probs)

    r64_col  = r64_column(games, probs)
    r32_col  = later_round_column(all_rounds[0], "r32",  probs)
    s16_col  = later_round_column(all_rounds[1], "s16",  probs)
    e8_col   = later_round_column(all_rounds[2], "e8",   probs)
    champ_col = later_round_column(all_rounds[3], "rchamp", probs)

    round_labels = [
        ("R64",      "r64"),
        ("R32",      "r32"),
        ("S16",      "s16"),
        ("Elite 8",  "e8"),
        ("Region\nChamp", "rchamp"),
    ]
    header = '<div class="round-headers">'
    if flip:
        round_labels = list(reversed(round_labels))
    for label, _ in round_labels:
        header += f'<div class="rh">{label}</div>'
    header += '</div>'

    cols = [r64_col, r32_col, s16_col, e8_col, champ_col]
    if flip:
        cols = list(reversed(cols))

    inner = "".join(cols)
    direction = "row-reverse" if flip else "row"

    return f"""
    <div class="region-block">
      <div class="region-title">{region_name.upper()} REGION
        <span class="region-sub">@ {region_data.get('location','')}</span>
      </div>
      {header}
      <div class="region-rounds" style="flex-direction:{direction}">
        {inner}
      </div>
    </div>"""


def ff_and_champ_block(bracket: dict, probs: dict,
                       left_champs: list[tuple], right_champs: list[tuple]) -> str:
    """Build the center Final Four + Championship block."""

    def pick_winner(teams: list[tuple]) -> tuple:
        best = max(teams, key=lambda t: probs.get("champion", {}).get(t[1], 0))
        return best

    # Left FF: East champ vs South champ
    lc1, lc2 = left_champs   # (seed, name) for each left-half region champ
    ff_left_winner = pick_winner([lc1, lc2])

    # Right FF: West champ vs Midwest champ
    rc1, rc2 = right_champs
    ff_right_winner = pick_winner([rc1, rc2])

    # Championship
    champion = pick_winner([ff_left_winner, ff_right_winner])

    def slot(seed, name, extra_class=""):
        bg = champ_color(probs.get("champion", {}).get(name, 0))
        cp = probs.get("champion", {}).get(name, 0)
        cs = f"{cp*100:.1f}%" if cp >= 0.001 else "—"
        return f"""<div class="ff-slot {extra_class}" style="background:{bg}" title="Champion prob: {cs}">
          <span class="seed">{seed}</span> <span class="tname">{name}</span>
          <span class="champ-pct">{cs}</span>
        </div>"""

    loc = bracket.get("final_four", {}).get("location", "Indianapolis")
    dates = bracket.get("final_four", {}).get("dates", "April 4 & 6")

    return f"""
    <div class="ff-block">
      <div class="ff-title">FINAL FOUR</div>
      <div class="ff-sub">{loc} · {dates}</div>

      <div class="ff-semis">
        <div class="ff-semi">
          <div class="ff-label">East vs South</div>
          {slot(lc1[0], lc1[1])}
          {slot(lc2[0], lc2[1])}
          <div class="ff-winner-label">→ Finalist</div>
          {slot(ff_left_winner[0], ff_left_winner[1], "ff-winner")}
        </div>

        <div class="champ-block">
          <div class="champ-title">🏆 CHAMPIONSHIP</div>
          {slot(ff_left_winner[0],  ff_left_winner[1],  "champ-team")}
          {slot(ff_right_winner[0], ff_right_winner[1], "champ-team")}
          <div class="champ-winner-label">Predicted Champion</div>
          {slot(champion[0], champion[1], "champ-winner")}
        </div>

        <div class="ff-semi">
          <div class="ff-label">West vs Midwest</div>
          {slot(rc1[0], rc1[1])}
          {slot(rc2[0], rc2[1])}
          <div class="ff-winner-label">→ Finalist</div>
          {slot(ff_right_winner[0], ff_right_winner[1], "ff-winner")}
        </div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Full page
# ---------------------------------------------------------------------------

def build_html(bracket: dict, mc: dict) -> str:
    probs = build_prob_lookup(mc)
    sims  = mc["_meta"]["simulations"]

    regions = bracket["regions"]

    # Build chalk predicted region champions for FF block
    def region_champ(rname):
        rd = predicted_path(regions[rname], bracket, probs)
        return rd[-1][0]  # (seed, name)

    left_champs  = [region_champ("East"),    region_champ("South")]
    right_champs = [region_champ("West"),    region_champ("Midwest")]

    east_html    = region_bracket("East",    regions["East"],    bracket, probs, flip=False)
    south_html   = region_bracket("South",   regions["South"],   bracket, probs, flip=False)
    west_html    = region_bracket("West",    regions["West"],    bracket, probs, flip=True)
    midwest_html = region_bracket("Midwest", regions["Midwest"], bracket, probs, flip=True)
    ff_html      = ff_and_champ_block(bracket, probs, left_champs, right_champs)

    # First Four summary
    ff_rows = ""
    for g in bracket["first_four"]:
        slot = g["winner_slot"]
        ff_rows += f"""
        <tr>
          <td>{g['id']}</td>
          <td>{g.get('date','')}</td>
          <td>#{g['seed_a']} {g['team_a']} vs #{g['seed_b']} {g['team_b']}</td>
          <td>{slot['region']} — plays #{slot['seed']} {slot['opponent']}</td>
        </tr>"""

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; }
    body { background: #1a1a2e; color: #eee; padding: 16px; }
    h1 { text-align: center; color: #ffd700; font-size: 1.4rem; margin-bottom: 4px; }
    .subtitle { text-align: center; color: #aaa; font-size: 0.8rem; margin-bottom: 16px; }

    /* Legend */
    .legend { display:flex; justify-content:center; gap:16px; margin-bottom:16px; flex-wrap:wrap; }
    .legend-item { display:flex; align-items:center; gap:6px; font-size:0.75rem; color:#ccc; }
    .legend-swatch { width:18px; height:14px; border-radius:3px; border:1px solid #555; }

    /* First Four table */
    .ff-table-wrap { margin: 0 auto 20px auto; max-width: 700px; }
    .ff-table-wrap h3 { color: #ffd700; font-size:0.85rem; margin-bottom:6px; text-align:center; }
    table { width:100%; border-collapse:collapse; font-size:0.72rem; }
    th, td { padding:5px 8px; border:1px solid #333; }
    th { background:#2a2a4a; color:#ffd700; }
    td { background:#16213e; color:#ddd; }

    /* Main bracket layout */
    .bracket-outer { display:flex; gap:10px; justify-content:center; align-items:flex-start; flex-wrap:nowrap; }
    .left-half, .right-half { display:flex; flex-direction:column; gap:10px; }

    /* Region */
    .region-block { background:#16213e; border-radius:8px; padding:10px; min-width:520px; }
    .region-title { font-size:0.85rem; font-weight:700; color:#ffd700; margin-bottom:4px; }
    .region-sub { font-size:0.7rem; color:#aaa; font-weight:400; margin-left:8px; }
    .round-headers { display:flex; gap:4px; margin-bottom:4px; }
    .rh { flex:1; text-align:center; font-size:0.62rem; color:#888; font-weight:600;
           text-transform:uppercase; letter-spacing:.5px; white-space:pre-line; }
    .region-rounds { display:flex; gap:4px; align-items:stretch; }

    /* Round columns */
    .round { display:flex; flex-direction:column; flex:1; gap:4px; }
    .r64  { flex:2; }
    .r32  { flex:1.6; justify-content:space-around; }
    .s16  { flex:1.3; justify-content:space-around; }
    .e8   { flex:1.1; justify-content:center; }
    .rchamp { flex:1; justify-content:center; }

    /* Game card */
    .game-card { border:1px solid #2a2a4a; border-radius:5px; overflow:hidden; }

    /* Team row (R64 games) */
    .team-row { display:flex; align-items:center; padding:3px 5px; border-bottom:1px solid #2a2a4a;
                cursor:default; transition:filter .15s; }
    .team-row:last-child { border-bottom:none; }
    .team-row:hover { filter:brightness(1.1); }
    .seed { font-size:0.65rem; color:#888; width:18px; flex-shrink:0; font-weight:700; }
    .tname { flex:1; font-size:0.68rem; color:#ddd; white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis; }
    .ff-team { font-style:italic; color:#aaa; font-size:0.62rem; }
    .champ-pct { font-size:0.62rem; color:#666; width:30px; text-align:right; flex-shrink:0; }

    /* Later round slots */
    .later-slot { display:flex; align-items:center; padding:4px 6px; border-radius:4px;
                  border:1px solid #2a2a4a; margin:1px 0; cursor:default; }
    .later-slot:hover { filter:brightness(1.1); }

    /* Final Four & Championship center block */
    .ff-block { background:#0f3460; border-radius:10px; padding:14px; min-width:340px;
                border:2px solid #ffd700; align-self:center; }
    .ff-title { text-align:center; color:#ffd700; font-size:1rem; font-weight:700; margin-bottom:2px; }
    .ff-sub { text-align:center; color:#aaa; font-size:0.72rem; margin-bottom:12px; }
    .ff-semis { display:flex; gap:12px; justify-content:center; align-items:flex-start; }
    .ff-semi { display:flex; flex-direction:column; gap:4px; min-width:120px; }
    .ff-label { font-size:0.68rem; color:#aaa; text-align:center; margin-bottom:2px; }
    .ff-winner-label, .champ-winner-label { font-size:0.62rem; color:#888; text-align:center; margin:2px 0; }
    .ff-slot { display:flex; align-items:center; padding:4px 6px; border-radius:4px;
               border:1px solid #2a2a4a; }
    .ff-slot .tname { font-size:0.68rem; }
    .ff-winner { border:1px solid #4caf50 !important; }
    .champ-block { display:flex; flex-direction:column; gap:4px; min-width:130px; align-items:stretch; }
    .champ-title { text-align:center; font-size:0.78rem; font-weight:700; color:#ffd700;
                   margin-bottom:4px; }
    .champ-team { border:1px solid #555 !important; }
    .champ-winner { border:2px solid #ffd700 !important; font-weight:700; }
    .champ-winner .tname { font-size:0.75rem; color:#333; font-weight:700; }
    .champ-winner .seed { color:#555; }
    .champ-winner .champ-pct { color:#555; font-weight:700; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>2026 NCAA Tournament Bracket</title>
  <style>{css}</style>
</head>
<body>
  <h1>2026 NCAA Tournament Bracket — Model Predictions</h1>
  <p class="subtitle">
    Chalk picks from LightGBM model · Champion % from {sims:,}-simulation Monte Carlo
    · Color: white = low · gold = high champion probability
  </p>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:#f8f9fa"></div> &lt;1% champion prob
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgb(252,224,125)"></div> ~5% champ prob
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgb(255,200,62)"></div> ~50% champ prob
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:rgb(255,180,0)"></div> 90%+ champ prob
    </div>
  </div>

  <div class="ff-table-wrap">
    <h3>First Four (already played / playing now)</h3>
    <table>
      <thead><tr><th>Game</th><th>Date</th><th>Matchup</th><th>Winner plays</th></tr></thead>
      <tbody>{ff_rows}</tbody>
    </table>
  </div>

  <div class="bracket-outer">
    <div class="left-half">
      {east_html}
      {south_html}
    </div>

    {ff_html}

    <div class="right-half">
      {west_html}
      {midwest_html}
    </div>
  </div>
</body>
</html>"""


def main() -> None:
    print("Loading data...")
    bracket, mc = load_data()
    print("Building HTML...")
    html = build_html(bracket, mc)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved → {OUTPUT_PATH}")
    print(f"Open in browser: file://{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
