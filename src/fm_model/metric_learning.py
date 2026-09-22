"""Learn recruitment metrics from the uploaded FMST player pool.

No position is assigned pre-selected statistics. Every usable independent
numeric performance metric is evaluated only against football-relevant team
outcomes for that position, then validated out of sample across clubs.
"""
from __future__ import annotations

import unicodedata
import numpy as np
import pandas as pd

from .data import parse_number, require, season_number
from .errors import DataError
from .roles import ROLE_LABELS

NON_PERFORMANCE_COLUMNS = {
    "player_key", "player_id", "player_name", "team_id", "league", "season",
    "position", "role_group", "owned", "age", "minutes", "appearances", "index",
    "market_value", "fmst_guide_value", "transfer_value", "asking_price",
    "transfer_fee", "fee", "wage", "weekly_wage", "annual_wage",
    "contract_months", "contract_years", "reputation", "currency",
}

LEAKAGE_OR_COMPOSITE_COLUMNS = {
    "goals", "goals_p90", "goals_outside_box", "goals_conceded_p90",
    "clean_sheets", "clean_sheets_p90", "goal_contributions_p90",
    "non_penalty_contributions_p90", "defensive_actions_p90",
    "attacking_actions_p90", "creative_actions_p90",
    "goalkeeping_actions_p90", "xg_overperformance",
}

RATE_REPLACEMENTS = {
    "xg_prevented": "xg_prevented_p90",
    "crosses_attempted": "crosses_attempted_p90",
    "open_play_crosses_attempted": "open_play_crosses_attempted_p90",
    "crosses_completed": "crosses_completed_p90",
    "open_play_crosses_completed": "open_play_crosses_completed_p90",
    "headers_won": "headers_won_p90",
    "headers_attempted": "headers_attempted_p90",
    "tackles_attempted": "tackles_attempted_p90",
    "fouls_made": "fouls_made_p90",
}


# Do not spend statistical power testing implausible role/outcome combinations.
# Hybrid positions remain two-way because their normal job can materially affect
# both attacking and defensive phases.
ROLE_OUTCOME_TARGETS = {
    "GK": ("preventing goals",),
    "CB": ("preventing goals",),
    "FB_WB": ("scoring", "preventing goals"),
    "CM_DM": ("scoring", "preventing goals"),
    "AM_W": ("scoring",),
    "ST": ("scoring",),
}


def role_outcomes(role: str) -> tuple[str, ...]:
    """Return the team outcomes the metric learner is allowed to test."""

    return ROLE_OUTCOME_TARGETS.get(str(role), ())

def _normal(value) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()

def _numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    values = []
    for value in series:
        try:
            values.append(parse_number(value))
        except (DataError, TypeError, ValueError):
            values.append(np.nan)
    return pd.Series(values, index=series.index, dtype=float)

def add_rate_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exposure-adjusted forms for count fields present in FMST exports."""
    f = frame.copy()
    if "minutes" not in f:
        return f
    minutes = pd.to_numeric(f["minutes"], errors="coerce").replace(0, np.nan)
    raw_rates = {
        "crosses_attempted": "crosses_attempted_p90",
        "open_play_crosses_attempted": "open_play_crosses_attempted_p90",
        "crosses_completed": "crosses_completed_p90",
        "open_play_crosses_completed": "open_play_crosses_completed_p90",
        "headers_won": "headers_won_p90",
        "headers_attempted": "headers_attempted_p90",
        "tackles_attempted": "tackles_attempted_p90",
        "fouls_made": "fouls_made_p90",
    }
    for raw, rate in raw_rates.items():
        if raw in f and rate not in f:
            values = _numeric_series(f[raw])
            if values.notna().any():
                f[rate] = values * 90.0 / minutes
    for attempts, pct, rate in (
        ("crosses_attempted", "cross_completion_pct", "crosses_completed_p90"),
        ("open_play_crosses_attempted", "open_play_cross_pct", "open_play_crosses_completed_p90"),
    ):
        if rate in f or attempts not in f or pct not in f:
            continue
        a, p = _numeric_series(f[attempts]), _numeric_series(f[pct])
        if a.notna().any() and p.notna().any():
            f[rate] = a * (p / 100.0) * 90.0 / minutes
    if "xg_prevented" in f and "xg_prevented_p90" not in f:
        xgp = _numeric_series(f["xg_prevented"])
        if xgp.notna().any():
            f["xg_prevented_p90"] = xgp * 90.0 / minutes
    return f

def metric_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe every imported column and whether it can enter learning."""
    f = add_rate_derivatives(frame)
    rows = []
    for col in f.columns:
        values = _numeric_series(f[col])
        numeric_count = int(values.notna().sum())
        unique = int(values.nunique(dropna=True))
        category, eligible = "performance", True
        reason = "Eligible independent numeric performance metric"
        if col in NON_PERFORMANCE_COLUMNS:
            category, eligible, reason = "context", False, "Identifier, exposure, context or financial field"
        elif col in LEAKAGE_OR_COMPOSITE_COLUMNS:
            category, eligible, reason = "outcome/composite", False, "Outcome leakage or undocumented composite"
        elif col in RATE_REPLACEMENTS and RATE_REPLACEMENTS[col] in f:
            category, eligible, reason = "raw total", False, "Use exposure-adjusted " + RATE_REPLACEMENTS[col]
        elif numeric_count == 0:
            category, eligible, reason = "non-numeric", False, "No numeric observations"
        elif unique < 2:
            category, eligible, reason = "constant", False, "No variation to learn from"
        rows.append({
            "column": col, "category": category, "eligible": bool(eligible),
            "numeric_rows": numeric_count, "unique_values": unique, "reason": reason,
        })
    return pd.DataFrame(rows)

def available_metrics(frame: pd.DataFrame) -> list[str]:
    inventory = metric_inventory(frame)
    return sorted(inventory.loc[inventory["eligible"], "column"].tolist())

def _player_derived_team_outcomes(pool: pd.DataFrame) -> pd.DataFrame:
    require(pool, ["team_id", "minutes"])
    f = pool.copy()
    f["minutes"] = _numeric_series(f["minutes"])
    rows = []
    for team, group in f.groupby("team_id"):
        max_minutes = float(group["minutes"].max()) if group["minutes"].notna().any() else np.nan
        scoring = np.nan
        if "goals" in group and max_minutes > 0:
            goals = _numeric_series(group["goals"])
            if goals.notna().any():
                scoring = float(goals.fillna(0).sum() * 90.0 / max_minutes)
        conceding = np.nan
        if "goals_conceded_p90" in group:
            gk = group[group["role_group"].eq("GK")].copy()
            if not gk.empty:
                rates, weights = _numeric_series(gk["goals_conceded_p90"]), _numeric_series(gk["minutes"])
                ok = rates.notna() & weights.gt(0)
                if ok.any():
                    conceding = float(np.average(rates[ok], weights=weights[ok]))
        rows.append({
            "team_id": team, "team_key": _normal(team),
            "goals_for_rate": scoring, "goals_against_rate": conceding,
            "outcome_source": "player export (same statistics scope)",
        })
    return pd.DataFrame(rows)

def _official_team_outcomes(pool: pd.DataFrame, league_table: pd.DataFrame | None) -> pd.DataFrame:
    if league_table is None or league_table.empty:
        return pd.DataFrame()
    required = {"league", "season", "team_id", "matches", "goals_for", "goals_against"}
    if not required.issubset(league_table.columns) or not {"league", "season"}.issubset(pool.columns):
        return pd.DataFrame()
    leagues, seasons = pool["league"].dropna().astype(str).unique(), pool["season"].dropna().unique()
    if len(leagues) != 1 or len(seasons) != 1:
        return pd.DataFrame()
    try:
        season = season_number(seasons[0])
    except DataError:
        return pd.DataFrame()
    table = league_table.copy()
    table = table[table["league"].map(_normal).eq(_normal(leagues[0]))]
    try:
        years = table["season"].map(season_number)
    except DataError:
        return pd.DataFrame()
    table = table[years.eq(season)].copy()
    if table.empty:
        return pd.DataFrame()
    matches, gf, ga = _numeric_series(table["matches"]), _numeric_series(table["goals_for"]), _numeric_series(table["goals_against"])
    valid = matches.gt(0) & gf.notna() & ga.notna()
    table = table.loc[valid, ["team_id"]].copy()
    table["team_key"] = table["team_id"].map(_normal)
    table["goals_for_rate"] = (gf[valid] / matches[valid]).to_numpy()
    table["goals_against_rate"] = (ga[valid] / matches[valid]).to_numpy()
    table["outcome_source"] = "same-season completed league table"
    return table

def team_outcomes(pool: pd.DataFrame, league_table: pd.DataFrame | None = None, *, scope=None) -> pd.DataFrame:
    official = _official_team_outcomes(pool, league_table)
    if str(scope or "").strip().casefold() == "league only" and len(official) >= 8:
        return official
    derived = _player_derived_team_outcomes(pool)
    if derived["goals_for_rate"].notna().sum() >= 8:
        return derived
    if len(official) >= 8:
        official = official.copy()
        official["outcome_source"] = "same-season league table (scope may differ from player statistics)"
        return official
    raise DataError(
        "Metric learning needs outcomes for at least eight clubs. Export the full league player pool "
        "with Goals and goalkeeper Goals Conceded/90, or provide a matching same-season league table."
    )

def _weighted_team_role_metric(pool: pd.DataFrame, role: str, metric: str, minimum_role_minutes: float) -> pd.DataFrame:
    f = pool[pool["role_group"].eq(role)].copy()
    if f.empty or metric not in f:
        return pd.DataFrame(columns=["team_key", "metric_value", "role_minutes"])
    f["minutes"], f[metric] = _numeric_series(f["minutes"]), _numeric_series(f[metric])
    f = f[f["minutes"].gt(0) & f[metric].notna()]
    if f.empty:
        return pd.DataFrame(columns=["team_key", "metric_value", "role_minutes"])
    f["_weighted"] = f[metric] * f["minutes"]
    agg = f.groupby("team_id", as_index=False).agg(weighted=("_weighted", "sum"), role_minutes=("minutes", "sum"))
    agg["metric_value"] = agg["weighted"] / agg["role_minutes"]
    agg = agg[agg["role_minutes"].ge(float(minimum_role_minutes))].copy()
    agg["team_key"] = agg["team_id"].map(_normal)
    return agg[["team_key", "metric_value", "role_minutes"]]

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr, yr = pd.Series(x).rank(method="average").to_numpy(float), pd.Series(y).rank(method="average").to_numpy(float)
    if np.nanstd(xr) <= 1e-12 or np.nanstd(yr) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])

def _loocv_univariate(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 8 or np.unique(x).size < 3 or np.unique(y).size < 3:
        return {"clubs": n, "validation_gain": np.nan, "spearman": np.nan, "slope": np.nan}
    model_errors, baseline_errors = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        xt, yt = x[mask], y[mask]
        if np.std(xt) <= 1e-12:
            continue
        slope, intercept = np.polyfit(xt, yt, 1)
        model_errors.append(abs(y[i] - (intercept + slope * x[i])))
        baseline_errors.append(abs(y[i] - float(np.mean(yt))))
    if len(model_errors) < 6:
        return {"clubs": n, "validation_gain": np.nan, "spearman": np.nan, "slope": np.nan}
    baseline_mae, model_mae = float(np.mean(baseline_errors)), float(np.mean(model_errors))
    gain = 0.0 if baseline_mae <= 1e-12 else 1.0 - model_mae / baseline_mae
    slope, _ = np.polyfit(x, y, 1)
    return {
        "clubs": n, "validation_gain": float(gain), "spearman": _spearman(x, y),
        "slope": float(slope), "model_mae": model_mae, "baseline_mae": baseline_mae,
    }

def learn_metric_importance(pool: pd.DataFrame, league_table: pd.DataFrame | None = None, *, scope=None, minimum_role_minutes=450, min_clubs=8) -> pd.DataFrame:
    """Test every eligible metric only against role-relevant team outcomes."""
    require(pool, ["team_id", "role_group", "minutes"])
    f = add_rate_derivatives(pool)
    metrics = available_metrics(f)
    outcomes = team_outcomes(f, league_table, scope=scope)
    rows = []
    for role in [r for r in ROLE_LABELS if r != "OTHER"]:
        for metric in metrics:
            agg = _weighted_team_role_metric(f, role, metric, minimum_role_minutes)
            joined = agg.merge(outcomes[["team_key", "goals_for_rate", "goals_against_rate", "outcome_source"]], on="team_key", how="inner", validate="one_to_one")
            if len(joined) < min_clubs:
                continue
            x = joined["metric_value"].to_numpy(float)
            allowed = role_outcomes(role)
            choices = []
            if "scoring" in allowed:
                choices.append((
                    "scoring",
                    _loocv_univariate(x, joined["goals_for_rate"].to_numpy(float)),
                ))
            if "preventing goals" in allowed:
                choices.append((
                    "preventing goals",
                    _loocv_univariate(x, -joined["goals_against_rate"].to_numpy(float)),
                ))
            if not choices:
                continue
            outcome, evidence = max(
                choices,
                key=lambda item: -np.inf if not np.isfinite(item[1].get("validation_gain", np.nan)) else item[1]["validation_gain"],
            )
            gain, corr, slope = evidence.get("validation_gain", np.nan), evidence.get("spearman", np.nan), evidence.get("slope", np.nan)
            if not np.isfinite(gain) or not np.isfinite(corr) or not np.isfinite(slope):
                continue
            coverage = min(1.0, len(joined) / max(float(len(outcomes)), 1.0))
            rows.append({
                "role_group": role, "position": ROLE_LABELS[role], "metric": metric,
                "tested_outcomes": " / ".join(role_outcomes(role)),
                "learned_outcome": outcome, "direction": "higher" if slope >= 0 else "lower",
                "validation_gain": float(gain), "spearman": float(corr),
                "importance_score": max(float(gain), 0.0) * abs(float(corr)) * coverage,
                "clubs": int(len(joined)), "coverage": float(coverage),
                "outcome_source": str(joined["outcome_source"].iloc[0]),
                "evidence": "Leave-one-club-out predictive association; not causal proof",
            })
    columns = ["role_group","position","metric","tested_outcomes","learned_outcome","direction","validation_gain","spearman","importance_score","clubs","coverage","outcome_source","evidence"]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        raise DataError("No metric had enough club-level variation and coverage to learn recruitment importance.")
    return result.sort_values(["role_group","importance_score","validation_gain","coverage"], ascending=[True,False,False,False]).reset_index(drop=True)

def select_learned_metrics(pool: pd.DataFrame, ranking: pd.DataFrame, *, max_metrics=4, minimum_validation_gain=0.0, redundancy_threshold=0.85) -> dict[str, list[str]]:
    """Choose strongest validated, non-redundant metrics per position."""
    if max_metrics < 1:
        raise DataError("max_metrics must be at least one.")
    f = add_rate_derivatives(pool)
    selected = {}
    for role in [r for r in ROLE_LABELS if r != "OTHER"]:
        ranked = ranking[
            ranking["role_group"].eq(role)
            & ranking["validation_gain"].gt(float(minimum_validation_gain))
            & ranking["importance_score"].gt(0)
        ].copy()
        chosen = []
        role_rows = f[f["role_group"].eq(role)]
        for metric in ranked["metric"].tolist():
            redundant = False
            for existing in chosen:
                pair = role_rows[[metric, existing]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) >= 8:
                    corr = pair.corr(method="spearman").iloc[0, 1]
                    if np.isfinite(corr) and abs(float(corr)) >= redundancy_threshold:
                        redundant = True
                        break
            if not redundant:
                chosen.append(metric)
            if len(chosen) >= int(max_metrics):
                break
        selected[role] = chosen
    return selected
