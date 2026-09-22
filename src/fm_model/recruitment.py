"""FMST-first recruitment: fixed peer benchmarks, explicit evidence and real costs.

These are descriptive screening profiles, not learned goal-impact weights. League
GF/GA targets remain a separately fitted model until compatible team data exists.
"""
from __future__ import annotations

import hashlib
import unicodedata

import numpy as np
import pandas as pd

from .data import FMST_NUMERIC_COLUMNS, parse_number, require, season_number
from .errors import DataError
from .roles import FORMATION_PRESETS, ROLE_LABELS, canonical_role

# Purpose-specific starting menus, exposed as editable choices in the UI.
FOCUSES = {
    "Shot stopping": ("save_pct", "xg_prevented_p90"),
    "Scoring opportunities": ("non_penalty_xg_p90", "shots_p90", "shots_on_target_p90"),
    "Chance creation": ("xa_p90", "open_play_key_passes_p90", "progressive_passes_p90"),
    "Ball progression": ("progressive_passes_p90", "dribbles_p90", "pass_completion_pct"),
    "Ball winning (context dependent)": ("interceptions_p90", "possession_won_p90", "tackle_success_pct"),
    "Aerial and defensive activity (context dependent)": ("header_win_pct", "interceptions_p90", "key_tackles_p90"),
}
DEFAULT_FOCUS = {
    "GK": "Shot stopping", "CB": "Aerial and defensive activity (context dependent)",
    "FB_WB": "Chance creation", "CM_DM": "Ball progression",
    "AM_W": "Chance creation", "ST": "Scoring opportunities",
}
# Composite actions have undocumented formulas; goals-derived fields are not
# allowed to masquerade as independent goal drivers.
EXCLUDED_METRICS = {
    "goals_outside_box", "goal_contributions_p90", "non_penalty_contributions_p90",
    "defensive_actions_p90", "attacking_actions_p90", "creative_actions_p90",
    "goalkeeping_actions_p90", "xg_overperformance",
}
COST_FIELDS = ("weekly_wage", "purchase_fee", "sale_proceeds", "contract_years", "additional_fees")
COST_METADATA = ("currency", "cost_note")

# Validate the fields this module consumes even if the importer has not parsed
# them. A long-running app can retain an older imported schema during a source
# update; direct library callers can also pass canonical columns as strings.
ANALYSIS_NUMERIC_COLUMNS = set().union(*map(set, FOCUSES.values())) | EXCLUDED_METRICS | {
    "appearances", "goals", "xg_p90", "goals_p90", "key_passes_p90", "tackles_p90",
    "pressures_attempted_p90", "pressure_success_pct", "clearances_p90", "blocks_p90",
    "sprints_p90", "distance_km_p90", "crosses_completed", "crosses_attempted",
    "open_play_crosses_completed", "open_play_crosses_attempted", "headers_won",
    "headers_attempted", "xg_prevented", "xg_prevented_p90",
}


def normal(value):
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def identify(frame):
    """Stable name+club keys across exports; ambiguous identities never auto-join."""
    f = frame.copy().reset_index(drop=True)
    require(f, ["player_name", "team_id", "minutes", "role_group"])
    for c in ("player_name", "team_id"):
        if f[c].isna().any() or f[c].astype(str).str.strip().eq("").any():
            raise DataError(f"Every player needs a nonempty {c} for safe matching.")
    f["player_key"] = [hashlib.sha256((normal(n) + "\x1f" + normal(c)).encode()).hexdigest()[:24]
                       for n, c in zip(f.player_name, f.team_id)]
    f = f.drop_duplicates()
    if f.player_key.duplicated().any():
        raise DataError("Ambiguous duplicate name/club rows. Export one season and resolve duplicate players first.")
    f["role_group"] = f.role_group.map(lambda v: v if v in ROLE_LABELS else canonical_role(v))
    f["minutes"] = f.minutes.map(parse_number)
    if (f.minutes < 0).any() or (~np.isfinite(f.minutes) & f.minutes.notna()).any():
        raise DataError("Minutes must be finite and nonnegative, or blank when unknown.")
    if "season" in f:
        f["season"] = f.season.map(season_number)
        if f.season.nunique() != 1:
            raise DataError("Use one statistics season per player upload. League tables may cover many seasons.")
    return f.reset_index(drop=True)


def clean_statistics(frame, *, league_matches=34):
    """Return a separate analysis copy; uploaded observations stay untouched."""
    league_matches = parse_number(league_matches)
    if not np.isfinite(league_matches) or league_matches <= 0:
        raise DataError("League match count must be a positive finite number.")
    f = identify(frame)
    issues = []
    def issue(field, count, message):
        issues.append({"field": field, "rows": int(count), "message": message})
    if f.minutes.isna().any():
        issue("minutes", f.minutes.isna().sum(), "Minutes missing; player retained with insufficient evidence and excluded from benchmarks.")
    for col in (set(FMST_NUMERIC_COLUMNS) | ANALYSIS_NUMERIC_COLUMNS).intersection(f):
        try:
            f[col] = f[col].map(parse_number).astype(float)
        except (DataError, TypeError, ValueError) as exc:
            raise DataError(f"Invalid numeric values in {col}: {exc}") from exc
        invalid = ~np.isfinite(f[col]) & f[col].notna()
        if col not in {"xg_prevented", "xg_prevented_p90", "xg_overperformance"}:
            invalid |= f[col].lt(0)
        if col.endswith("_pct"):
            invalid |= f[col].gt(100)
        if invalid.any():
            issue(col, invalid.sum(), "Impossible value excluded from analysis; source retained.")
            f.loc[invalid, col] = np.nan
    if "appearances" in f:
        count = f.appearances.gt(league_matches).sum()
        if count:
            issue("competition_scope", count,
                  "Appearances exceed league fixtures. Do not aggregate these statistics into league GF/GA.")
    if {"goals", "goals_outside_box"}.issubset(f):
        invalid = f.goals_outside_box.gt(f.goals)
        if invalid.any():
            issue("goals_outside_box", invalid.sum(), "Outside-box goals exceed total goals; column excluded.")
    for made, attempted in (("crosses_completed", "crosses_attempted"),
                            ("open_play_crosses_completed", "open_play_crosses_attempted"),
                            ("headers_won", "headers_attempted")):
        if made in f and attempted in f:
            invalid = f[made].gt(f[attempted])
            if invalid.any():
                issue(made, invalid.sum(), "Completed count exceeds attempts; completed value excluded.")
                f.loc[invalid, made] = np.nan
    if "xg_prevented" in f and "xg_prevented_p90" not in f:
        f["xg_prevented_p90"] = f.xg_prevented * 90 / f.minutes.replace(0, np.nan)
    for metric in EXCLUDED_METRICS.intersection(f):
        f[metric] = np.nan
    return f, pd.DataFrame(issues, columns=["field", "rows", "message"])


def prepare_inputs(pool, squad, *, league_matches=34):
    pool, p_issues = clean_statistics(pool, league_matches=league_matches)
    squad, s_issues = clean_statistics(squad, league_matches=league_matches)
    if squad.team_id.nunique() != 1:
        raise DataError("My squad must contain one club. Remove other clubs from the squad upload.")
    if pool.empty or squad.empty:
        raise DataError("Both the league statistics and squad exports need player rows.")
    if "season" in pool and "season" in squad and pool.season.iloc[0] != squad.season.iloc[0]:
        raise DataError("The league player pool and your squad must cover the same statistics season.")
    if "league" in pool and pool.league.nunique() != 1:
        raise DataError("The reference pool must contain one league; upload other leagues as targets later.")
    if "league" in pool and "league" in squad and set(squad.league.map(normal)) != set(pool.league.map(normal)):
        raise DataError("The reference pool and squad must belong to the same league.")
    # The pool is the fixed reference. Squad rows are never appended a second time.
    pool["owned"] = pool.player_key.isin(squad.player_key)
    squad["owned"] = True
    issues = pd.concat([p_issues.assign(source="League statistics"),
                        s_issues.assign(source="My squad")], ignore_index=True)
    return pool, squad, issues


def default_assignments(squad, formation="4-2-3-1"):
    result = squad[["player_key", "player_name", "position", "role_group", "minutes"]].copy() if "position" in squad else squad[["player_key", "player_name", "role_group", "minutes"]].copy()
    result["starter"] = False
    for role, slots in FORMATION_PRESETS[formation].items():
        ids = result[result.role_group.eq(role)].sort_values("minutes", ascending=False).head(slots).index
        result.loc[ids, "starter"] = True
    return result


def available_metrics(pool):
    candidates = set().union(*map(set, FOCUSES.values())) | {
        "xg_p90", "key_passes_p90", "goals_p90", "tackles_p90", "pressures_attempted_p90",
        "pressure_success_pct", "clearances_p90", "blocks_p90", "sprints_p90", "distance_km_p90",
    }
    return sorted(c for c in candidates if c in pool and pool[c].notna().any())


def profile_metrics(focus, pool):
    cols = list(FOCUSES[focus])
    if "non_penalty_xg_p90" in cols and ("non_penalty_xg_p90" not in pool or pool.non_penalty_xg_p90.notna().sum() < 5):
        cols[cols.index("non_penalty_xg_p90")] = "xg_p90"
    if "open_play_key_passes_p90" in cols and "open_play_key_passes_p90" not in pool:
        cols[cols.index("open_play_key_passes_p90")] = "key_passes_p90"
    return [c for c in cols if c in available_metrics(pool)]


def assess_squad(pool, squad, assignments, *, formation="4-2-3-1", minimum_minutes=900,
                 target_percentile=60, focuses=None, metrics=None):
    """Generate empirical screening ranges and starter shortfalls, never goal gains."""
    if not 0 < target_percentile < 100:
        raise DataError("Screening percentile must lie between 0 and 100.")
    if minimum_minutes <= 0:
        raise DataError("Minimum comparison minutes must be positive.")
    require(assignments, ["player_key", "role_group", "starter"])
    if assignments.player_key.duplicated().any() or set(assignments.player_key) != set(squad.player_key):
        raise DataError("Squad assignments must contain each squad player exactly once.")
    if not assignments.role_group.isin(ROLE_LABELS).all():
        raise DataError("Choose a valid position group for every player.")
    if not assignments.starter.isin([True, False]).all():
        raise DataError("Starter assignments must be true or false.")
    team = squad.drop(columns=["role_group"], errors="ignore").merge(
        assignments[["player_key", "role_group", "starter"]], on="player_key", validate="one_to_one")
    # Use confirmed squad positions in the reference too, without duplicating rows.
    reference = pool.copy()
    mapping = assignments.set_index("player_key").role_group
    reference["role_group"] = reference.player_key.map(mapping).fillna(reference.role_group)
    focuses = focuses or DEFAULT_FOCUS
    profiles, priorities, players = [], [], []
    for role, slots in FORMATION_PRESETS[formation].items():
        starters = team[team.role_group.eq(role) & team.starter]
        if len(starters) > slots:
            raise DataError(f"{ROLE_LABELS[role]}: select at most {slots} starters for {formation}.")
        peers = reference[reference.role_group.eq(role) & reference.minutes.ge(minimum_minutes)]
        focus = focuses.get(role, DEFAULT_FOCUS[role])
        selected = metrics.get(role, []) if metrics is not None else profile_metrics(focus, reference)
        group_profiles = []
        for metric in selected:
            if metric not in available_metrics(reference):
                continue
            values = peers[metric].dropna()
            if len(values) < 5 or values.nunique() < 2:
                continue
            target = float(values.quantile(target_percentile / 100))
            current = starters.loc[starters.minutes.ge(minimum_minutes), metric].dropna() if metric in starters else pd.Series(dtype=float)
            current_median = float(current.median()) if len(current) else np.nan
            current_pct = float(100 * ((values < current_median).mean() + .5 * (values == current_median).mean())) if len(current) else np.nan
            row = {"role_group": role, "focus": focus, "metric": metric,
                   "league_median": float(values.median()), "screening_target": target,
                   "peer_upper_quartile": float(values.quantile(.75)), "current_starter_median": current_median,
                   "current_percentile": current_pct, "percentile_gap": max(0, target_percentile - current_pct) if len(current) else np.nan,
                   "peer_count": len(values), "target_percentile": target_percentile,
                   "evidence": "Descriptive league benchmark; not learned goal impact"}
            group_profiles.append(row)
            profiles.append(row)
        for _, player in team[team.role_group.eq(role)].iterrows():
            observed = [(p, player.get(p["metric"], np.nan)) for p in group_profiles]
            valid = [(p, v) for p, v in observed if pd.notna(v)]
            count = sum(v >= p["screening_target"] for p, v in valid)
            enough = player.minutes >= minimum_minutes
            review = ("Insufficient minutes" if not enough else "Insufficient benchmark data" if not valid else
                      "Review profile fit" if count < len(group_profiles) / 2 else "Fits several screening metrics")
            players.append({"player_key": player.player_key, "player_name": player.player_name,
                            "role_group": role, "starter": bool(player.starter), "minutes": player.minutes,
                            "metrics_met": count, "metrics_observed": len(valid), "metrics_required": len(group_profiles),
                            "review": review})
        gaps = [p["percentile_gap"] for p in group_profiles if pd.notna(p["percentile_gap"])]
        missing = max(0, slots - len(starters))
        low = int((~starters.minutes.ge(minimum_minutes)).sum())
        review_names = [r["player_name"] for r in players if r["role_group"] == role and r["starter"] and r["review"] == "Review profile fit"]
        priorities.append({"role_group": role, "position": ROLE_LABELS[role], "focus": focus,
                           "unfilled_starter_slots": missing, "low_minutes_starters": low,
                           "average_percentile_shortfall": float(np.mean(gaps)) if gaps else np.nan,
                           "review_players": ", ".join(review_names), "supported_metrics": len(group_profiles),
                           "next_step": "Assign a starter or scout cover" if missing else
                           "Gather more playing evidence" if low or not gaps else
                           "Compare potential upgrades" if review_names else "Lower recruitment priority"})
    priorities = pd.DataFrame(priorities).sort_values(
        ["unfilled_starter_slots", "average_percentile_shortfall"], ascending=[False, False], na_position="last")
    return priorities.reset_index(drop=True), pd.DataFrame(profiles), pd.DataFrame(players)


def compare_targets(targets, squad, profile, role, *, minimum_minutes=900, league=None):
    """Apply unchanged league thresholds. Uploading targets cannot move benchmarks."""
    if profile.empty:
        return pd.DataFrame(), pd.DataFrame()
    profile = profile[profile.role_group.eq(role)]
    if profile.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, details = [], []
    combined = pd.concat([squad.assign(source="My squad"),
                          targets[~targets.player_key.isin(squad.player_key)].assign(source="Target")], ignore_index=True)
    for _, player in combined[combined.role_group.eq(role)].iterrows():
        seen, passed = 0, 0
        for p in profile.to_dict("records"):
            value = player.get(p["metric"], np.nan)
            met = bool(value >= p["screening_target"]) if pd.notna(value) else None
            seen += int(pd.notna(value))
            passed += int(met is True)
            details.append({"player_key": player.player_key, "player_name": player.player_name,
                            "metric": p["metric"], "value": value,
                            "screening_target": p["screening_target"], "meets_target": met})
        other_league = league is not None and normal(player.get("league", "")) != normal(league)
        rows.append({"player_key": player.player_key, "player_name": player.player_name,
                     "club": player.team_id, "source": player.source, "minutes": player.minutes,
                     "metrics_met": passed, "metrics_observed": seen, "metrics_required": len(profile),
                     "profile_match_pct": 100 * passed / len(profile),
                     "sufficient_minutes": bool(pd.notna(player.minutes) and player.minutes >= minimum_minutes),
                     "evidence": "Insufficient minutes" if pd.isna(player.minutes) or player.minutes < minimum_minutes else
                     "Unadjusted cross-league comparison" if other_league else
                     "Missing profile statistics" if seen < len(profile) else "Same-league descriptive comparison"})
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["sufficient_minutes", "profile_match_pct", "minutes"], ascending=False).reset_index(drop=True)
    return result, pd.DataFrame(details)


def empty_costs(players, currency="GBP"):
    f = players[["player_key", "player_name", "team_id", "owned"]].drop_duplicates("player_key").copy()
    for c in COST_FIELDS:
        f[c] = np.nan
    f["currency"] = currency
    f["cost_note"] = ""
    return f.reset_index(drop=True)


def merge_costs(players, saved=None, currency="GBP"):
    """Match edits by stable identity, never display order; don't overwrite ownership."""
    out = empty_costs(players, currency)
    if saved is None or saved.empty:
        return out
    require(saved, ["player_key"])
    if saved.player_key.duplicated().any():
        raise DataError("Cost file contains duplicate player keys.")
    saved = saved.set_index("player_key")
    for c in (*COST_FIELDS, *COST_METADATA):
        if c in saved:
            values = out.player_key.map(saved[c])
            out[c] = values.combine_first(out[c])
    for c in COST_FIELDS:
        out[c] = out[c].map(parse_number)
        if out[c].dropna().lt(0).any() or (~np.isfinite(out[c]) & out[c].notna()).any():
            raise DataError(f"{c} must be a nonnegative finite number or blank.")
    if out.contract_years.dropna().le(0).any():
        raise DataError("Contract years must be positive or blank.")
    return out


def cost_comparison(costs, *, horizon=3, currency="GBP"):
    if not 0 < horizon <= 10:
        raise DataError("Choose a comparison horizon between 0 and 10 years.")
    out = merge_costs(costs, costs, currency)
    wages_known = out.weekly_wage.notna()
    covered = out.contract_years.ge(horizon)
    same_currency = out.currency.astype(str).str.upper().eq(currency.upper())
    target = ~out.owned.astype(bool)
    complete = wages_known & covered & same_currency & out.additional_fees.notna() & (~target | out.purchase_fee.notna())
    out["horizon_years"] = horizon
    out["horizon_wages"] = out.weekly_wage * 52 * horizon
    # Current acquisition fees are sunk. Additional fees are future commitments.
    out["total_cost"] = (out.horizon_wages + out.additional_fees + out.purchase_fee.where(target, 0)).where(complete)
    out["cost_status"] = ["Complete" if ok else "Currency differs" if not curr else
                          "Contract shorter than horizon or unknown" if not cover else "Missing costs"
                          for ok, curr, cover in zip(complete, same_currency, covered)]
    return out


def replacement_cost(costed, outgoing_key, incoming_key):
    lookup = costed.set_index("player_key")
    out, incoming = lookup.loc[outgoing_key], lookup.loc[incoming_key]
    if not out.owned or incoming.owned:
        raise DataError("Choose an owned outgoing player and an external target.")
    if pd.isna(out.total_cost) or pd.isna(incoming.total_cost) or pd.isna(out.sale_proceeds):
        return None
    return float(incoming.total_cost - out.sale_proceeds - out.total_cost)
