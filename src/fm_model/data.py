"""Explicit units, conservative parsing and league-table validation."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import DataError
from .roles import canonical_role


ALIASES = {
    "pos": "position", "rank": "position", "position": "position",
    "gf": "goals_for", "for": "goals_for", "ga": "goals_against", "against": "goals_against",
    "pts": "points", "pld": "matches", "played": "matches", "mp": "matches",
    "division": "league", "competition": "league", "club": "team_id", "team": "team_id",
    "uid": "player_id", "unique_id": "player_id", "name": "player_name", "player": "player_name",
    "player_name": "player_name", "age": "age", "club_name": "team_id", "team_id": "team_id",
    "league_name": "league", "league": "league", "season": "season", "year": "season",
    "value": "market_value", "guide_value": "market_value", "market_value": "market_value",
    "transfer_value": "transfer_value", "asking_price": "asking_price",
    "fee": "transfer_fee", "transfer_fee": "transfer_fee", "wage": "wage", "weekly_wage": "weekly_wage",
    "annual_wage": "annual_wage", "contract_months": "contract_months", "reputation": "reputation",
    "mins": "minutes", "minutes_played": "minutes", "best_pos": "role_group",
    "best_position": "role_group", "role": "role_group",
    "xg": "xg", "xa": "xa", "expected_goals": "xg", "expected_assists": "xa", "gls": "goals", "ast": "assists",
    "non_penalty_xg": "non_penalty_xg", "np_xg": "non_penalty_xg",
    "tackles_completed": "tackles_won", "saves_per_90": "saves_p90", "xg_prevented": "xg_prevented",
    "sh/90": "shots_p90", "shots/90": "shots_p90", "sot/90": "shots_on_target_p90", "xg/90": "xg_p90", "xa/90": "xa_p90",
    "gls/90": "goals_p90", "asts/90": "assists_p90", "key/90": "key_passes_p90",
    "pr_passes/90": "progressive_passes_p90", "drb/90": "dribbles_p90",
    "tck/90": "tackles_won_p90", "int/90": "interceptions_p90",
    "hdrs_w/90": "headers_won_p90", "pres_a/90": "pressures_p90",
    "acc": "acceleration", "pac": "pace", "jum": "jumping_reach", "str": "strength",
    "fin": "finishing", "cmp": "composure", "otb": "off_the_ball", "pas": "passing",
    "vis": "vision", "dec": "decisions", "tec": "technique", "cro": "crossing",
    "tck": "tackling", "mar": "marking", "pos_attr": "positioning", "ant": "anticipation",
    "ref": "reflexes", "han": "handling", "one": "one_on_ones", "aer": "aerial_reach",
}

# Exact FMST26 display headings. Percentages stay on the displayed 0..100 scale.
# Keep ambiguous event labels distinct (CCC, tackles and pressures) rather than
# silently treating them as a different statistic.
FMST_STAT_COLUMNS = {
    "goals_per_90": "goals_p90", "xg_per_90": "xg_p90",
    "xa_per_90": "xa_p90", "np_xg_per_90": "non_penalty_xg_p90",
    "xg_overperformance": "xg_overperformance", "shots_per_90": "shots_p90",
    "shots_on_target_per_90": "shots_on_target_p90", "shot_accuracy_%": "shot_accuracy_pct",
    "ccc_per_90": "clear_cut_chances_p90", "shots_outside_box_per_90": "shots_outside_box_p90",
    "pass_completion_%": "pass_completion_pct", "passes_completed_per_90": "passes_completed_p90",
    "passes_attempted_per_90": "passes_attempted_p90", "key_passes_per_90": "key_passes_p90",
    "progressive_passes_per_90": "progressive_passes_p90", "crosses_attempted": "crosses_attempted",
    "crosses_completed": "crosses_completed", "cross_completion_%": "cross_completion_pct",
    "open_play_crosses_att": "open_play_crosses_attempted",
    "open_play_crosses_comp": "open_play_crosses_completed", "open_play_cross_%": "open_play_cross_pct",
    "open_play_key_passes_p90": "open_play_key_passes_p90", "tackles_per_90": "tackles_p90",
    "tackles_attempted": "tackles_attempted", "tackle_success_%": "tackle_success_pct",
    "interceptions_per_90": "interceptions_p90", "clearances_per_90": "clearances_p90",
    "blocks_per_90": "blocks_p90", "key_tackles_per_90": "key_tackles_p90",
    "pressures_per_90": "pressures_p90", "pressures_attempted_per_90": "pressures_attempted_p90",
    "pressure_success_%": "pressure_success_pct", "fouls_made": "fouls_made",
    "possession_won_per_90": "possession_won_p90", "possession_lost_per_90": "possession_lost_p90",
    "shots_blocked_per_90": "shots_blocked_p90", "mistakes_leading_to_goals": "errors_leading_to_goal",
    "dribbles_per_90": "dribbles_p90", "headers_attempted": "headers_attempted",
    "headers_won": "headers_won", "header_win_%": "header_win_pct",
    "save_percentage": "save_pct", "saves_per_90": "saves_p90", "clean_sheets": "clean_sheets",
    "goals_conceded_per_90": "goals_conceded_p90", "xg_prevented": "xg_prevented",
    "penalty_save_%": "penalty_save_pct",
}
ALIASES.update(FMST_STAT_COLUMNS)
FMST_NUMERIC_COLUMNS = frozenset(FMST_STAT_COLUMNS.values())

ATTRIBUTES = (
    "corners", "crossing", "dribbling", "finishing", "first_touch", "free_kick_taking",
    "heading", "long_shots", "long_throws", "marking", "passing", "penalty_taking",
    "tackling", "technique", "aggression", "anticipation", "bravery", "composure",
    "concentration", "decisions", "determination", "flair", "leadership", "off_the_ball",
    "positioning", "teamwork", "vision", "work_rate", "acceleration", "agility", "balance",
    "jumping_reach", "natural_fitness", "pace", "stamina", "strength", "aerial_reach",
    "command_of_area", "communication", "eccentricity", "handling", "kicking",
    "one_on_ones", "reflexes", "rushing_out", "punching_tendency", "throwing",
)

PLAYER_METRICS = (
    "shots_p90", "xg_p90", "non_penalty_xg_p90", "goals_p90", "xa_p90", "assists_p90",
    "key_passes_p90",
    "progressive_passes_p90", "dribbles_p90", "crosses_completed_p90",
    "tackles_won_p90", "interceptions_p90", "headers_won_p90", "pressures_p90",
    "possession_won_p90", "possession_lost_p90", "saves_p90", "goals_prevented_p90", "xg_prevented_p90",
)


def require(frame: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataError("Missing columns: " + ", ".join(missing))


def numeric(frame: pd.DataFrame, columns, *, nonnegative=True, nullable=False) -> None:
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        bad = ~np.isfinite(values)
        if nullable:
            bad &= frame[col].notna()
        if bad.any() or (nonnegative and (values.dropna() < 0).any()):
            raise DataError(f"{col} must contain {'nonnegative ' if nonnegative else ''}finite numbers.")
        frame[col] = values


def season_number(value) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return int(value)
    m = re.fullmatch(r"\s*(\d{4})(?:[/\-]\d{2,4})?\s*", str(value))
    if not m:
        raise DataError(f"Season {value!r} must be a year or a label such as 2025/26.")
    return int(m.group(1))


def parse_number(value, *, decimal=".", ranges="error") -> float:
    """Parse display numbers. Never silently average masked attributes or prices."""
    if decimal not in {".", ","} or ranges not in {"error", "midpoint", "lower", "upper"}:
        raise DataError("Invalid decimal or ranges setting.")
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = (str(value).strip().replace("\u00a0", "").replace(" ", "")
         .replace("−", "-").replace("'", "").replace("’", ""))
    # FMST uses '-£1' as a display placeholder for an unavailable guide value,
    # rather than a meaningful negative price. Treat it as missing evidence.
    if s.lower() in {"-£1", "£-1", "-€1", "€-1", "-$1", "$-1"}:
        return float("nan")
    if s.lower() in {"", "-", "—", "n/a", "unknown", "none"}:
        return float("nan")
    s = re.sub(r"[£€$]", "", s)
    s = re.sub(r"(?i)(p/w|/wk|/week|perweek)$", "", s)
    parts = re.split(r"[–—]|(?<=[0-9kKmMbB%])-(?=[+\\-]?[0-9.])", s, maxsplit=1)
    if len(parts) == 2:
        if ranges == "error":
            raise DataError(f"Range {value!r}: choose lower/upper/midpoint explicitly or retain separate bounds.")
        lo, hi = (parse_number(p, decimal=decimal) for p in parts)
        return {"lower": lo, "upper": hi, "midpoint": (lo + hi) / 2}[ranges]
    s = s.replace("%", "")  # Percentages remain 0..100, never fractions implicitly.
    s = s.replace(".", "").replace(",", ".") if decimal == "," else s.replace(",", "")
    m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))([kKmMbB]?)", s)
    if not m:
        raise DataError(f"Cannot parse {value!r}; map units and column meaning explicitly.")
    return float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6, "b": 1e9}[m.group(2).lower()]


def read_table(source, *, column_map=None, numeric_columns=(), decimal=".", ranges="error",
               table_index=0, format=None) -> pd.DataFrame:
    """CSV/semicolon CSV/TSV, local HTML or XLSX; accepts file-like Streamlit uploads.

    Canonical snake_case columns pass through. Unknown columns are retained. No URL reads.
    Ambiguous labels (e.g. Pos for player position) must be mapped by the caller.
    """
    name = str(getattr(source, "name", source)).lower()
    if name.startswith(("https:", "http:")):
        raise DataError("Upload a file; remote URLs are not read by the importer.")
    kind = format or Path(name).suffix.lstrip(".")
    if kind in {"xlsx", "xlsm"}:
        frame = pd.read_excel(source)
    else:
        raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                raw = raw.decode("utf-16")
        if kind in {"html", "htm"} or "<table" in raw[:4000].lower():
            tables = pd.read_html(io.StringIO(raw), keep_default_na=True)
            if not 0 <= table_index < len(tables):
                raise DataError("HTML table_index is out of range.")
            frame = tables[table_index]
        else:
            frame = pd.read_csv(io.StringIO(raw), sep=None, engine="python", dtype=str)
    mapping = column_map or {}
    columns = []
    for col in frame.columns:
        label = str(col).strip()
        normal = re.sub(r"[\s\-]+", "_", label.lower())
        columns.append(mapping.get(label, ALIASES.get(normal, normal)))
    if len(columns) != len(set(columns)):
        raise DataError("Column mapping produces duplicate names; supply an explicit column_map.")
    frame.columns = columns
    for col in numeric_columns:
        require(frame, [col])
        if col == "season":
            frame[col] = frame[col].map(season_number)
        else:
            frame[col] = frame[col].map(lambda v: parse_number(v, decimal=decimal, ranges=ranges))
    return frame


def prepare_player_export(frame: pd.DataFrame, *, league=None, season=None, owned=None) -> pd.DataFrame:
    """Add save metadata to a canonical FMST/FM player export.

    FMST's statistics table normally contains player, club and performance fields,
    but not the competition or save season. Those are context supplied by the
    user. A market export can leave ``owned`` unset; a squad export can pass
    ``owned=True``. Existing columns are preserved and only missing metadata is
    filled.
    """
    f = frame.copy()
    require(f, ["player_name", "minutes"])
    if "league" not in f:
        if league is None:
            raise DataError("Player export has no league column; supply league=... for this save/export.")
        f["league"] = league
    elif league is not None:
        f["league"] = f["league"].fillna(league)
    if season is not None and "season" not in f:
        f["season"] = season
    if owned is not None and "owned" not in f:
        f["owned"] = owned
    if "role_group" not in f and "position" in f:
        f["role_group"] = f["position"].map(canonical_role)
    return f


def read_player_export(source, *, league=None, season=None, owned=None, column_map=None,
                       numeric_columns=(), decimal=".", ranges="error", table_index=0,
                       format=None) -> pd.DataFrame:
    """Read an FMST/FM player export and attach context absent from the table.

    This is a convenience wrapper around :func:`read_table`. It recognises FMST
    labels such as ``Guide Value`` and ``Saves per 90`` through ``ALIASES`` and
    accepts CSV, HTML and XLSX sources supported by ``read_table``.
    """
    frame = read_table(source, column_map=column_map, numeric_columns=numeric_columns,
                       decimal=decimal, ranges=ranges, table_index=table_index, format=format)
    return prepare_player_export(frame, league=league, season=season, owned=owned)


def validate_league_table(frame: pd.DataFrame) -> pd.DataFrame:
    require(frame, ["league", "season", "position", "matches", "goals_for", "goals_against"])
    f = frame.copy()
    if f.empty or f.league.isna().any():
        raise DataError("Provide complete league tables with nonempty league identifiers.")
    f["league"] = f.league.astype(str)
    f["season"] = f.season.map(season_number)
    cols = ["position", "matches", "goals_for", "goals_against"]
    for col in cols:
        f[col] = f[col].map(lambda value: parse_number(value, ranges="error"))
    numeric(f, cols)
    if not np.allclose(f[cols], np.round(f[cols])):
        raise DataError("League positions, matches and goals must be integers.")
    if (f.matches <= 0).any():
        raise DataError("Completed tables need matches > 0; do not mix partial seasons.")
    if "points" in f:
        f["points"] = f["points"].map(lambda value: parse_number(value, ranges="error"))
        numeric(f, ["points"], nonnegative=False, nullable=True)
        if (f.points > 3 * f.matches).any():
            raise DataError("Points exceed the supported three-points-for-a-win maximum.")
    if "points_adjustment" in f:
        f["points_adjustment"] = f["points_adjustment"].map(lambda value: parse_number(value, ranges="error"))
        numeric(f, ["points_adjustment"], nonnegative=False)
    for key, g in f.groupby(["league", "season"]):
        n = len(g)
        if n < 4 or sorted(g.position.astype(int)) != list(range(1, n + 1)):
            raise DataError(f"{key}: supply each final position 1..N exactly once (at least four clubs).")
        if g.matches.nunique() != 1:
            raise DataError(f"{key}: unequal matches; use one completed, comparable league phase.")
        if int(g.goals_for.sum()) != int(g.goals_against.sum()):
            raise DataError(f"{key}: total goals for ({int(g.goals_for.sum())}) must equal "
                            f"total goals against ({int(g.goals_against.sum())}). Check the source table.")
        if g.goals_for.sum() <= 0:
            raise DataError(f"{key}: no goals; cannot learn a scoring environment.")
        if "team_id" in g and g.team_id.duplicated().any():
            raise DataError(f"{key}: duplicate team_id.")
        if "points" in g and g.points.notna().all():
            if (g.sort_values("position").points.diff().dropna() > 0).any():
                raise DataError(f"{key}: points do not match finishing order; check split tables/deductions.")
    f["n_teams"] = f.groupby(["league", "season"]).position.transform("size")
    f["goal_rate"] = f.groupby(["league", "season"]).goals_for.transform("sum") / f.groupby(
        ["league", "season"]).matches.transform("sum")
    return f.sort_values(["season", "league", "position"]).reset_index(drop=True)


def add_per90(frame: pd.DataFrame, columns) -> pd.DataFrame:
    require(frame, ["minutes", *columns])
    f = frame.copy()
    numeric(f, ["minutes", *columns])
    for col in columns:
        f[col + "_p90"] = f[col] * 90 / f.minutes.replace(0, np.nan)
    return f
