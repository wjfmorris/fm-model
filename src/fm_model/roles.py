"""Canonical position groups and formation presets used by the squad planner."""

from __future__ import annotations

import re


ROLE_LABELS = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB_WB": "Full-back / wing-back",
    "CM_DM": "Central / defensive midfield",
    "AM_W": "Attacking midfield / wing",
    "ST": "Striker",
    "OTHER": "Other",
}

FORMATION_PRESETS = {
    "4-2-3-1": {"GK": 1, "CB": 2, "FB_WB": 2, "CM_DM": 2, "AM_W": 3, "ST": 1},
    "4-3-3": {"GK": 1, "CB": 2, "FB_WB": 2, "CM_DM": 3, "AM_W": 2, "ST": 1},
    "4-4-2": {"GK": 1, "CB": 2, "FB_WB": 2, "CM_DM": 2, "AM_W": 2, "ST": 2},
    "3-4-2-1": {"GK": 1, "CB": 3, "FB_WB": 2, "CM_DM": 2, "AM_W": 2, "ST": 1},
}


def canonical_role(value) -> str:
    """Map common FM/FΜST position labels into stable planning groups."""

    text = str(value or "").upper().strip()
    compact = re.sub(r"[^A-Z0-9]+", " ", text)
    tokens = set(compact.split())

    if "GK" in tokens or "GOALKEEPER" in compact:
        return "GK"
    if tokens.intersection({"ST", "CF"}) or "STRIKER" in compact or "FORWARD" in compact:
        return "ST"

    # FM commonly displays positions as D (C), WB (L), M (C), AM (R), ST (C),
    # sometimes with several positions in one string. Normalisation above turns
    # these into token pairs such as {"D", "C"} or {"AM", "R"}.
    if "AM" in tokens or tokens.intersection({"AML", "AMR", "AMC", "LW", "RW"}) or "ATTACKING MID" in compact or "WINGER" in compact:
        return "AM_W"
    if "WB" in tokens or tokens.intersection({"WBL", "WBR", "DL", "DR", "LB", "RB", "FB"}) or "WING BACK" in compact or "FULL BACK" in compact:
        return "FB_WB"
    if tokens.intersection({"DC", "CB", "SW"}) or ("D" in tokens and "C" in tokens) or "CENTRE BACK" in compact or "CENTER BACK" in compact:
        return "CB"
    if "DM" in tokens or tokens.intersection({"MC", "CM", "DMC"}) or ("M" in tokens and "C" in tokens) or "CENTRAL MID" in compact or "DEFENSIVE MID" in compact:
        return "CM_DM"
    if "D" in tokens and tokens.intersection({"L", "R"}):
        return "FB_WB"
    return "OTHER"


def formation_slots(name: str) -> dict[str, int]:
    if name not in FORMATION_PRESETS:
        raise ValueError(f"Unknown formation {name!r}.")
    return dict(FORMATION_PRESETS[name])
