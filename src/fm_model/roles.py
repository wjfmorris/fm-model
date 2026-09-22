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
    # Parenthesised side combinations belong to their own position token.
    # Expand D (RC), M (LC), etc. without confusing them with unrelated roles.
    text = re.sub(r"\b(D|M|WB|AM)\s*\(([RLC]+)\)",
                  lambda m: m.group(1) + " " + " ".join(m.group(2)), text)
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
    if "M" in tokens and tokens.intersection({"L", "R"}):
        return "AM_W"
    if "D" in tokens and tokens.intersection({"L", "R"}):
        return "FB_WB"
    return "OTHER"


def formation_slots(name: str) -> dict[str, int]:
    if name not in FORMATION_PRESETS:
        raise ValueError(f"Unknown formation {name!r}.")
    return dict(FORMATION_PRESETS[name])


# Exact tactical positions shown to the user. These are intentionally more
# specific than the modelling groups above; the statistical model still pools
# positions where sample sizes would otherwise be too small.
INTENDED_POSITION_TO_ROLE = {
    "GK": "GK",
    "DL": "FB_WB",
    "DR": "FB_WB",
    "WBL": "FB_WB",
    "WBR": "FB_WB",
    "DC": "CB",
    "DM": "CM_DM",
    "MC": "CM_DM",
    "ML": "AM_W",
    "MR": "AM_W",
    "AML": "AM_W",
    "AMC": "AM_W",
    "AMR": "AM_W",
    "ST": "ST",
}
INTENDED_POSITION_OPTIONS = tuple(INTENDED_POSITION_TO_ROLE)


def intended_position_role(value: str) -> str:
    """Map an exact intended position back to the broad modelling group."""

    key = str(value or "").upper().strip()
    if key not in INTENDED_POSITION_TO_ROLE:
        raise ValueError(f"Unknown intended position {value!r}.")
    return INTENDED_POSITION_TO_ROLE[key]


def exact_positions(value) -> list[str]:
    """Extract exact FM positions while preserving FM's displayed order."""

    text = str(value or "").upper().strip()
    if not text:
        return []

    result: list[str] = []

    def add(position: str):
        if position in INTENDED_POSITION_TO_ROLE and position not in result:
            result.append(position)

    aliases = {
        "GK": "GK",
        "DC": "DC", "CB": "DC",
        "DL": "DL", "LB": "DL",
        "DR": "DR", "RB": "DR",
        "WBL": "WBL", "WBR": "WBR",
        "DM": "DM", "DMC": "DM",
        "MC": "MC", "CM": "MC",
        "ML": "ML", "MR": "MR",
        "AML": "AML", "AMC": "AMC", "AMR": "AMR",
        "ST": "ST", "CF": "ST",
    }

    # Work left-to-right so "DM, M (C)" suggests DM before MC and a
    # multifunctional FM label retains the order the game presents.
    for segment in re.split(r"\s*,\s*", text):
        matched_ranges = []
        for match in re.finditer(r"((?:D|WB|M|AM)(?:/(?:D|WB|M|AM))*)\s*\(([RLC]+)\)", segment):
            matched_ranges.append(match.span())
            families = match.group(1).split("/")
            sides = match.group(2)
            for family in families:
                for side in sides:
                    if family == "D":
                        add("DC" if side == "C" else f"D{side}")
                    elif family == "WB" and side in {"L", "R"}:
                        add(f"WB{side}")
                    elif family == "M":
                        add("MC" if side == "C" else f"M{side}")
                    elif family == "AM":
                        add(f"AM{side}")

        # Remove the parenthesised expressions already expanded, then parse
        # any bare canonical labels in the same segment.
        remainder = segment
        for left, right in reversed(matched_ranges):
            remainder = remainder[:left] + " " + remainder[right:]
        compact = re.sub(r"[^A-Z0-9]+", " ", remainder)
        for token in compact.split():
            if token in aliases:
                add(aliases[token])

        # ST (C) is not part of the D/M/AM family parser.
        if re.search(r"(^|[^A-Z])ST(?:\s*\(C\))?([^A-Z]|$)", segment):
            add("ST")

    return result


def suggest_intended_position(raw_position, role_group: str | None = None) -> str:
    """Choose an editable exact-position default for a player."""

    options = exact_positions(raw_position)
    if role_group:
        same_group = [p for p in options if INTENDED_POSITION_TO_ROLE[p] == role_group]
        if same_group:
            return same_group[0]
    if options:
        return options[0]
    fallback = {
        "GK": "GK",
        "CB": "DC",
        "FB_WB": "DL",
        "CM_DM": "MC",
        "AM_W": "AMC",
        "ST": "ST",
    }
    return fallback.get(str(role_group or ""), "MC")
