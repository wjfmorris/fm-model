"""Helpers used by the Streamlit front end.

The UI stays deliberately thin: parsing, ownership inference, model construction
and example loading live here so they can be unit tested without a browser.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pandas as pd

from .data import (
    ATTRIBUTES, FMST_NUMERIC_COLUMNS, PLAYER_METRICS,
    season_number, parse_number, prepare_player_export, read_table,
)
from .errors import DataError
from .pipeline import MoneyballModel
from .players import PRICE_COLUMNS, RAW_COUNT_METRICS, WAGE_COLUMNS


PLAYER_NUMERIC_COLUMNS = (
    set(ATTRIBUTES)
    | set(FMST_NUMERIC_COLUMNS)
    | set(PLAYER_METRICS)
    | set(RAW_COUNT_METRICS)
    | set(PRICE_COLUMNS)
    | set(WAGE_COLUMNS)
    | {"minutes", "age", "contract_months", "reputation", "expected_minutes", "fmst_guide_value"}
)

OWNERSHIP_MODES = {
    "auto": "Use owned column if present, otherwise infer from club",
    "club": "Infer ownership from club",
    "none": "Treat everyone as a market player",
    "all": "Treat everyone as my squad",
    "column": "Require and use the owned column",
}


def clone_upload(source):
    """Clone a Streamlit UploadedFile without depending on its current cursor."""

    if hasattr(source, "getvalue"):
        raw = source.getvalue()
        clone = io.BytesIO(raw)
        clone.name = getattr(source, "name", "upload.csv")
        return clone
    return source


def infer_season_from_name(name):
    """Infer a season start year from names such as players_2025_26.csv."""

    text = str(name or "")
    match = re.search(r"(?<!\d)(20\d{2})(?:[/_-](?:20)?\d{2})(?!\d)", text)
    return int(match.group(1)) if match else None


def merge_table_uploads(sources):
    """Merge table uploads, inferring a missing season from each filename."""

    if not sources:
        return None
    sources = list(sources) if isinstance(sources, (list, tuple)) else [sources]
    frames = []
    for source in sources:
        frame = read_table(clone_upload(source))
        if "season" not in frame:
            season = infer_season_from_name(getattr(source, "name", ""))
            if season is None:
                raise DataError(
                    f"{getattr(source, 'name', 'upload')}: no season column. "
                    "Add one or rename the file like team_2025_26.csv."
                )
            frame["season"] = season
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def read_player_for_app(
    source,
    *,
    league=None,
    season=None,
    ownership_mode="auto",
    owned_club=None,
    range_policy="error",
):
    """Read and normalise a player export for the UI.

    Numeric display ranges are resolved only for recognised numeric columns. This
    keeps identity fields untouched and makes the user's range policy explicit.
    """

    if ownership_mode not in OWNERSHIP_MODES:
        raise DataError(f"Unknown ownership mode {ownership_mode!r}.")
    if range_policy not in {"error", "lower", "midpoint", "upper"}:
        raise DataError("range_policy must be error, lower, midpoint or upper.")

    frame = read_table(clone_upload(source))
    frame = prepare_player_export(frame, league=league, season=season)

    for col in PLAYER_NUMERIC_COLUMNS.intersection(frame.columns):
        frame[col] = frame[col].map(lambda value: parse_number(value, ranges=range_policy))

    if ownership_mode == "column":
        if "owned" not in frame:
            raise DataError("Ownership mode requires an owned column, but the export does not contain one.")
    elif ownership_mode == "all":
        frame["owned"] = True
    elif ownership_mode == "none":
        frame["owned"] = False
    elif ownership_mode == "club":
        if not owned_club:
            raise DataError("Enter your club name when ownership is inferred from club.")
        if "team_id" not in frame:
            raise DataError("The player export has no club/team column to infer ownership from.")
        wanted = str(owned_club).strip().casefold()
        frame["owned"] = frame["team_id"].astype(str).str.strip().str.casefold().eq(wanted)
    elif "owned" not in frame:
        if owned_club and "team_id" in frame:
            wanted = str(owned_club).strip().casefold()
            frame["owned"] = frame["team_id"].astype(str).str.strip().str.casefold().eq(wanted)
        else:
            frame["owned"] = False

    return frame


def read_player_history_for_app(sources, *, league=None, season=None, range_policy="error"):
    """Read the latest player season by default; explicit years remain authoritative."""

    if not sources:
        return None
    sources = list(sources) if isinstance(sources, (list, tuple)) else [sources]
    frames = []
    for source in sources:
        frame = read_table(clone_upload(source))
        file_season = None
        if "season" not in frame:
            file_season = infer_season_from_name(getattr(source, "name", ""))
            if file_season is None:
                file_season = season
            if file_season is None:
                raise DataError(
                    f"{getattr(source, 'name', 'player history')}: no season column. "
                    "Select the player season, add season, or name the file players_2025_26.csv."
                )
        frame = prepare_player_export(frame, league=league, season=file_season)
        frame["season"] = frame["season"].map(season_number)
        for col in PLAYER_NUMERIC_COLUMNS.intersection(frame.columns):
            frame[col] = frame[col].map(lambda value: parse_number(value, ranges=range_policy))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)


def build_model(*, league_data=None, team_data=None, historical_player_data=None, player_data=None, include_all_available=False, allow_partial=False):
    """Fit every supplied model layer in dependency order."""

    if league_data is None and team_data is None and historical_player_data is None and player_data is None:
        raise DataError("Provide at least one data source before building the model.")

    model = MoneyballModel()
    model.upload_issues = []
    if league_data is not None:
        try:
            model.fit_league(league_data)
        except DataError as exc:
            if not allow_partial:
                raise
            model.league_model = None
            model.upload_issues.append("League targets unavailable: " + str(exc))
    if team_data is not None:
        model.fit_drivers(team_data, include_all_available=include_all_available)
    if historical_player_data is not None:
        model.fit_player_outcomes(historical_player_data)
    if player_data is None and historical_player_data is not None:
        player_data = historical_player_data
    if player_data is not None:
        model.fit_players(player_data)
    return model


def repository_root():
    return Path(__file__).resolve().parents[2]


def load_example_frames():
    """Load the synthetic repository examples used by the app demo and tests."""

    root = repository_root() / "data" / "templates"
    return {
        "league": read_table(root / "league_table.csv"),
        "team": read_table(root / "team_metrics.csv"),
        "player_history": read_table(root / "player_history.csv"),
        "players": read_table(root / "player_export.csv"),
    }


def serialise_model(model):
    return json.dumps(model.to_dict(), default=float, indent=2).encode("utf-8")


def frame_to_csv_bytes(frame: pd.DataFrame):
    return frame.to_csv(index=False).encode("utf-8")


def player_upload_report(frame):
    """Display import coverage and suspicious source values without changing them."""
    warnings = []
    if not any(c in frame for c in ATTRIBUTES):
        warnings.append("No player attributes were exported. Attribute learning is unavailable; statistics remain usable.")
    if "season" in frame and frame.season.nunique() == 1:
        warnings.append("One player season supplied. Comparisons describe this season; future-season validation is unavailable.")
    for completed, attempted in (("crosses_completed", "crosses_attempted"),
                                 ("open_play_crosses_completed", "open_play_crosses_attempted"),
                                 ("headers_won", "headers_attempted")):
        if completed in frame and attempted in frame:
            n = int((frame[completed] > frame[attempted]).sum())
            if n:
                warnings.append(f"{n} rows have {completed} greater than {attempted}; check the export. Values were retained.")
    if "fmst_guide_value" in frame:
        warnings.append("FMST Guide Value is retained for reference only and excluded from financial valuation.")
        counts = frame.fmst_guide_value.dropna().value_counts()
        if len(counts) and counts.iloc[0] >= max(5, len(frame) * 0.03):
            warnings.append(f"{int(counts.iloc[0])} players share guide value {counts.index[0]:,.0f}. Check repeated/capped values before acting on prices.")
    if not any(c in frame for c in WAGE_COLUMNS):
        warnings.append("No wages supplied. Price comparisons exclude wage and contract costs.")
    identity = {"player_name", "player_id", "position", "role_group", "team_id", "league", "season", "owned"}
    retained = [c for c in frame if c not in PLAYER_NUMERIC_COLUMNS and c not in identity]
    return {"rows": len(frame), "numeric_columns": sorted(set(frame) & PLAYER_NUMERIC_COLUMNS),
            "other_columns": retained, "warnings": warnings}
