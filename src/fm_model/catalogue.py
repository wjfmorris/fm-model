"""FM26 import guidance shown by the Streamlit app."""

from .data import ATTRIBUTES, PLAYER_METRICS
from .drivers import ATTACK_CANDIDATES, DEFENCE_CANDIDATES


DATA_CATALOGUE = {
    "league_table": {
        "required": ["league", "season", "position", "matches", "goals_for", "goals_against"],
        "recommended": ["points", "points_adjustment", "team_id"],
        "notes": "One completed league phase per league-season; include final positions 1..N.",
    },
    "team_metrics": {
        "required": ["league", "season", "matches", "goals_for", "goals_against"],
        "attacking": list(ATTACK_CANDIDATES),
        "defensive": list(DEFENCE_CANDIDATES),
        "notes": "Counts are converted to per-match rates. Keep xG, shots, chances, set pieces, pressing and goalkeeping fields where available.",
    },
    "historical_player_export": {
        "recommended_identity": ["player_name", "player_id", "league", "season", "team_id", "role_group", "minutes", "age"],
        "performance": list(PLAYER_METRICS),
        "attributes": list(ATTRIBUTES),
        "notes": "Use every club and every position across multiple completed seasons. The final season is held out when enough history exists so attribute -> outcome relationships can be validated chronologically.",
    },
    "player_export": {
        "recommended_identity": ["player_name", "player_id", "league", "team_id", "role_group", "minutes", "age"],
        "performance": list(PLAYER_METRICS),
        "attributes": list(ATTRIBUTES),
        "market": ["market_value", "asking_price", "transfer_fee", "wage", "annual_wage", "contract_months", "owned"],
        "notes": "Current squad plus the broadest realistic transfer market. FMST26 labels such as Guide Value, Minutes Played, Non-Penalty xG, Tackles Completed, Saves per 90 and xG Prevented are canonicalised automatically. Supply league metadata when the export omits it; do not mix raw totals and per-90 columns without labels.",
    },
}
