"""Helpers used by the Streamlit front end.

The UI stays deliberately thin: parsing, ownership inference, model construction
and example loading live here so they can be unit tested without a browser.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from .data import ATTRIBUTES, PLAYER_METRICS, parse_number, prepare_player_export, read_table
from .errors import DataError
from .pipeline import MoneyballModel
from .players import PRICE_COLUMNS, RAW_COUNT_METRICS, WAGE_COLUMNS


PLAYER_NUMERIC_COLUMNS = (
    set(ATTRIBUTES)
    | set(PLAYER_METRICS)
    | set(RAW_COUNT_METRICS)
    | set(PRICE_COLUMNS)
    | set(WAGE_COLUMNS)
    | {"minutes", "age", "contract_months", "reputation", "expected_minutes"}
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


def build_model(*, league_data=None, team_data=None, player_data=None, include_all_available=False):
    """Fit every supplied model layer in dependency order."""

    if league_data is None and team_data is None and player_data is None:
        raise DataError("Provide at least one data source before building the model.")

    model = MoneyballModel()
    if league_data is not None:
        model.fit_league(league_data)
    if team_data is not None:
        model.fit_drivers(team_data, include_all_available=include_all_available)
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
        "players": read_table(root / "player_export.csv"),
    }


def serialise_model(model):
    return json.dumps(model.to_dict(), default=float, indent=2).encode("utf-8")


def frame_to_csv_bytes(frame: pd.DataFrame):
    return frame.to_csv(index=False).encode("utf-8")
