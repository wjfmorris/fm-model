"""Exact data-collection guidance for the FM26/FΜST26 workflow."""

from __future__ import annotations

from .data import ATTRIBUTES
from .drivers import ATTACK_CANDIDATES, DEFENCE_CANDIDATES


IDENTITY_FIELDS = [
    "player_name", "player_id", "team_id", "league", "season", "role_group",
    "minutes", "age",
]

MARKET_FIELDS = [
    "market_value", "asking_price", "transfer_fee", "wage", "annual_wage",
    "contract_months", "owned",
]

ATTACK_PLAYER_STATS = [
    "goals", "goals_p90", "non_penalty_xg", "non_penalty_xg_p90", "xg", "xg_p90",
    "shots", "shots_p90", "shots_on_target", "shots_on_target_p90",
    "big_chances", "big_chances_p90", "xa", "xa_p90", "assists", "assists_p90",
    "key_passes", "key_passes_p90", "chances_created", "chances_created_p90",
    "touches_in_box", "touches_in_box_p90", "penalty_area_entries", "penalty_area_entries_p90",
    "progressive_passes", "progressive_passes_p90", "successful_dribbles",
    "successful_dribbles_p90", "crosses_completed", "crosses_completed_p90",
    "headers_won", "headers_won_p90",
]

DEFENCE_PLAYER_STATS = [
    "tackles_won", "tackles_won_p90", "interceptions", "interceptions_p90",
    "blocks", "blocks_p90", "clearances", "clearances_p90",
    "headers_won", "headers_won_p90", "aerial_duels_won", "aerial_duels_won_p90",
    "pressures", "pressures_p90", "successful_pressures", "successful_pressures_p90",
    "possession_won", "possession_won_p90", "possession_lost", "possession_lost_p90",
    "errors_leading_to_shot", "errors_leading_to_goal",
]

GK_PLAYER_STATS = [
    "saves", "saves_p90", "save_pct", "clean_sheets",
    "xg_prevented", "xg_prevented_p90", "goals_prevented", "goals_prevented_p90",
]

POSITION_SCOPE = {
    "ST": "all strikers / centre-forwards",
    "AM_W": "all attacking midfielders and wide attackers",
    "CM_DM": "all central and defensive midfielders",
    "FB_WB": "all full-backs and wing-backs",
    "CB": "all centre-backs",
    "GK": "all goalkeepers",
}


def collection_plan(*, historical_seasons=3, include_goalkeepers=True):
    """Return the exact files and scope recommended before using the app."""

    positions = dict(POSITION_SCOPE)
    if not include_goalkeepers:
        positions.pop("GK", None)

    return {
        "principle": (
            "Collect broadly first, then let the model decide what matters. Do not pre-filter "
            "to attributes or statistics you personally expect to be important."
        ),
        "league_tables": {
            "seasons": f"At least {historical_seasons}; 4-5 is better.",
            "clubs": "Every club in each target league, every completed season.",
            "positions": "Not applicable.",
            "fields": [
                "league", "season", "team_id", "position", "matches",
                "goals_for", "goals_against", "points", "points_adjustment",
            ],
            "purpose": "Learn the GF/GA combinations associated with the requested finishing position.",
        },
        "team_history": {
            "seasons": f"At least {historical_seasons}; use the same seasons as the league tables.",
            "clubs": "Every club in the target league(s), not only your club or the strongest teams.",
            "positions": "Team-level file.",
            "required_fields": ["league", "season", "team_id", "matches", "goals_for", "goals_against"],
            "attacking_fields": list(ATTACK_CANDIDATES),
            "defensive_fields": list(DEFENCE_CANDIDATES),
            "purpose": "Learn which team processes predict scoring and conceding.",
        },
        "historical_players": {
            "seasons": (
                f"At least {historical_seasons} completed seasons if available. "
                "One row per player-season; keep low-minute rows but expect the model to down-weight/flag them."
            ),
            "clubs": "Every player from every club in the historical league sample.",
            "positions": positions,
            "identity_fields": list(IDENTITY_FIELDS),
            "performance_fields": list(dict.fromkeys(
                ATTACK_PLAYER_STATS + DEFENCE_PLAYER_STATS + (GK_PLAYER_STATS if include_goalkeepers else [])
            )),
            "attribute_fields": list(ATTRIBUTES),
            "market_fields": list(MARKET_FIELDS),
            "purpose": (
                "Learn attribute -> player-outcome relationships with a held-out final season, "
                "and create position-specific recruitment profiles."
            ),
        },
        "current_market": {
            "seasons": "Current save only.",
            "clubs": (
                "Your full squad plus the broadest realistic recruitment pool. Include every club/leagues "
                "you would genuinely buy from rather than exporting only players you already like."
            ),
            "positions": positions,
            "identity_fields": list(IDENTITY_FIELDS),
            "performance_fields": list(dict.fromkeys(
                ATTACK_PLAYER_STATS + DEFENCE_PLAYER_STATS + (GK_PLAYER_STATS if include_goalkeepers else [])
            )),
            "attribute_fields": list(ATTRIBUTES),
            "market_fields": list(MARKET_FIELDS),
            "purpose": (
                "Forecast replacement scenarios, identify expensive low-contribution squad members, "
                "and search for cheaper players who close the GF/GA gap."
            ),
        },
        "exclude_from_performance_models": [
            "Current Ability (CA)",
            "Potential Ability (PA)",
            "market_value",
            "asking_price",
            "transfer_fee",
            "wage",
            "reputation",
        ],
        "why_excluded": (
            "Price and hidden FM ability ratings are used only for market comparison/validation. "
            "They must not tell the football-performance model who is good."
        ),
    }
