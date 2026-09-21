"""Player contribution, replacement and market-value comparisons.

Player scores are deliberately built from the learned team-driver models when a
link exists. A player's metric is compared with a replacement-level reference in
the same league and role; the result is a contribution index, not an FM star
rating. Market decisions require a price or wage column and always expose missing
price/playing-time evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .data import ATTRIBUTES, FMST_NUMERIC_COLUMNS, PLAYER_METRICS, numeric, parse_number, require
from .drivers import GoalDriverModel
from .errors import DataError, NotFittedError


PLAYER_ALIASES = {
    "xg": ("xg_p90", "xg", "non_penalty_xg_p90", "non_penalty_xg", "expected_goals_p90", "expected_goals"),
    "xa": ("xa_p90", "xa", "expected_assists_p90", "expected_assists"),
    "assists": ("assists_p90", "assists"),
    "shots": ("shots_p90", "shots"),
    "shots_on_target": ("shots_on_target_p90", "shots_on_target"),
    "big_chances": ("big_chances_p90", "big_chances"),
    "chances_created": ("chances_created_p90", "key_passes_p90", "chances_created", "key_passes"),
    "key_passes": ("key_passes_p90", "key_passes"),
    "final_third_entries": ("final_third_entries_p90", "final_third_entries"),
    "penalty_area_entries": ("penalty_area_entries_p90", "penalty_area_entries"),
    "touches_in_box": ("touches_in_box_p90", "touches_in_box"),
    "progressive_passes": ("progressive_passes_p90", "progressive_passes"),
    "successful_dribbles": ("dribbles_p90", "successful_dribbles_p90", "successful_dribbles", "dribbles"),
    "crosses_completed": ("crosses_completed_p90", "crosses_completed"),
    "set_piece_goals": ("set_piece_goals_p90", "set_piece_goals"),
    "corner_goals": ("corner_goals_p90", "corner_goals"),
    "free_kick_goals": ("free_kick_goals_p90", "free_kick_goals"),
    "goals": ("goals_p90", "goals"),
    "xg_against": ("xg_against_p90", "xg_against"),
    "shots_against": ("shots_against_p90", "shots_against"),
    "shots_on_target_against": ("shots_on_target_against_p90", "shots_on_target_against"),
    "big_chances_against": ("big_chances_against_p90", "big_chances_against"),
    "chances_conceded": ("chances_conceded_p90", "chances_conceded"),
    "pressures": ("pressures_p90", "pressures"),
    "successful_pressures": ("successful_pressures_p90", "successful_pressures"),
    "tackles_won": ("tackles_won_p90", "tackles_won"),
    "interceptions": ("interceptions_p90", "interceptions"),
    "blocks": ("blocks_p90", "blocks"),
    "clearances": ("clearances_p90", "clearances"),
    "headers_won": ("headers_won_p90", "headers_won"),
    "aerial_duels_won": ("aerial_duels_won_p90", "aerial_duels_won"),
    "possession_won": ("possession_won_p90", "possession_won"),
    "goals_prevented": ("goals_prevented_p90", "goals_prevented", "xg_prevented_p90", "xg_prevented"),
    "saves": ("saves_p90", "saves_per_90", "saves"),
    "save_pct": ("save_pct",),
    "clean_sheets": ("clean_sheets",),
    "possession_lost": ("possession_lost_p90", "possession_lost"),
    "possession_lost_in_own_half": ("possession_lost_in_own_half_p90", "possession_lost_in_own_half"),
    "errors_leading_to_shot": ("errors_leading_to_shot",),
    "errors_leading_to_goal": ("errors_leading_to_goal",),
}

RAW_COUNT_METRICS = set(
    x for choices in PLAYER_ALIASES.values() for x in choices
    if not x.endswith("_p90") and not x.endswith("_pct") and not x.endswith("_per_90")
)
PRICE_COLUMNS = ("asking_price", "transfer_fee", "market_value", "value", "transfer_value")
WAGE_COLUMNS = ("annual_wage", "wage", "weekly_wage", "salary")
ATTRIBUTE_PROXY = {
    "attack": ("finishing", "off_the_ball", "composure", "technique", "passing", "vision", "crossing", "pace"),
    "defence": ("marking", "tackling", "positioning", "anticipation", "decisions", "concentration",
                "pace", "strength", "jumping_reach", "reflexes", "handling"),
}


def _safe_name(value):
    return str(value).strip() if pd.notna(value) else "Unknown"


def _as_bool(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    truthy = {"1", "true", "yes", "y", "owned", "ours", "current", "squad"}
    falsy = {"0", "false", "no", "n", "available", "market", "target"}
    output = []
    for value in values:
        if pd.isna(value) or str(value).strip().lower() in falsy:
            output.append(False)
        elif str(value).strip().lower() in truthy:
            output.append(True)
        else:
            raise DataError(f"Cannot interpret ownership value {value!r}; use true/false or yes/no.")
    return pd.Series(output, index=values.index, dtype=bool)


def _prepare_players(frame: pd.DataFrame, *, role_column="role_group", minimum_minutes=450):
    f = frame.copy()
    require(f, ["league", "minutes"])
    # FM tables may display 2,800 minutes or £1.5m values as text. Convert only
    # fields known to be numeric; identity/role columns stay untouched.
    numeric_labels = set(FMST_NUMERIC_COLUMNS) | set(PLAYER_METRICS) | set(ATTRIBUTES) | set(RAW_COUNT_METRICS) | {
        "minutes", "age", *PRICE_COLUMNS, *WAGE_COLUMNS, "contract_months", "reputation", "expected_minutes"
    }
    for col in numeric_labels.intersection(f.columns):
        f[col] = f[col].map(lambda value: parse_number(value, ranges="error"))
    # FMST includes players with no recorded appearances. Preserve those rows so
    # they can still be screened by value/attributes, but do not treat them as
    # observed performance evidence.
    numeric(f, ["minutes"], nullable=True)
    if (f.minutes < 0).any():
        raise DataError("minutes cannot be negative.")
    f["league"] = f.league.map(_safe_name)
    if role_column not in f:
        for candidate in ("position", "best_pos", "role", "position_group"):
            if candidate in f:
                role_column = candidate
                break
    if role_column not in f:
        f["role_group"] = "all"
        role_column = "role_group"
    f["role_group"] = f[role_column].map(_safe_name)
    # Derive p90 values only for recognised count metrics. Attributes such as Pace
    # must never be divided by minutes.
    for col in list(f.columns):
        if col.endswith("_p90") or col.endswith("_pct") or col not in RAW_COUNT_METRICS:
            continue
        if col + "_p90" not in f.columns:
            f[col + "_p90"] = pd.to_numeric(f[col], errors="coerce") * 90 / f.minutes.replace(0, np.nan)
    f["minutes_evidence"] = np.select(
        [f.minutes.isna(), f.minutes >= minimum_minutes],
        ["unavailable", "usable"],
        default="small_sample",
    )
    return f, role_column


def _find_player_column(team_feature: str, f: pd.DataFrame, mapping: Mapping[str, str] | None = None):
    if mapping and team_feature in mapping:
        col = mapping[team_feature]
        if col not in f:
            raise DataError(f"Player metric link {team_feature!r}->{col!r} is missing from input.")
        return col
    for candidate in PLAYER_ALIASES.get(team_feature, (team_feature + "_p90", team_feature)):
        if candidate in f:
            return candidate
    # Generic convention: FM team columns such as progressive_passes_p90 may already match.
    for candidate in (team_feature + "_p90", team_feature):
        if candidate in f:
            return candidate
    return None


def _scale(value, median, scale):
    return (value - median) / max(scale, 1e-8)


class PlayerValuationModel:
    """Score players against role-level replacement and compare with market cost.

    ``fit`` learns references and driver links. It does not pretend that a single
    player's raw output equals team output: ``score`` reports both contribution
    units and a cautious team-goal sensitivity when a team baseline is supplied.
    """

    def __init__(self, drivers: GoalDriverModel | None = None, *, minimum_minutes=450,
                 replacement_quantile=0.25, seed=26):
        self.drivers = drivers
        self.minimum_minutes = minimum_minutes
        self.replacement_quantile = replacement_quantile
        self.seed = seed

    def fit(self, frame: pd.DataFrame, *, player_metric_map=None, role_column="role_group"):
        f, self.role_column = _prepare_players(frame, role_column=role_column,
                                               minimum_minutes=self.minimum_minutes)
        if f.empty:
            raise DataError("Player input is empty.")
        if not 0 < self.replacement_quantile < 0.5:
            raise DataError("replacement_quantile must be between 0 and 0.5.")
        self.metric_map = {"attack": {}, "defence": {}}
        self.weights = {"attack": {}, "defence": {}}
        self.attribute_map = {direction: [c for c in cols if c in f.columns]
                              for direction, cols in ATTRIBUTE_PROXY.items()}
        if self.drivers is not None:
            for direction, features in (("attack", self.drivers.attack_features),
                                        ("defence", self.drivers.defensive_features)):
                effects = self.drivers.driver_report(direction)["drivers"]
                for effect in effects:
                    team_feature = effect["feature"]
                    col = _find_player_column(team_feature, f, player_metric_map)
                    if col is None:
                        continue
                    raw_weight = abs(float(effect.get("benefit", 0.0)))
                    # If a model effect is unavailable/flat, retain the link with a
                    # small weight so the app can show the missing-evidence reason.
                    self.metric_map[direction][team_feature] = col
                    self.weights[direction][team_feature] = max(raw_weight, 1e-6)
        # Add a well-defined fallback set when the driver model is not trained or the
        # team and player exports use different labels. These are labelled proxies.
        if not self.metric_map["attack"]:
            for feature in ("xg", "shots", "chances_created", "goals", "set_piece_goals"):
                col = _find_player_column(feature, f, player_metric_map)
                if col:
                    self.metric_map["attack"][feature] = col
                    self.weights["attack"][feature] = 1.0
        if not self.metric_map["defence"]:
            for feature in ("tackles_won", "interceptions", "headers_won", "pressures", "goals_prevented", "saves"):
                col = _find_player_column(feature, f, player_metric_map)
                if col:
                    self.metric_map["defence"][feature] = col
                    self.weights["defence"][feature] = 1.0
        if not self.metric_map["attack"] and not self.attribute_map["attack"]:
            raise DataError("No recognisable attacking performance or attributes. Include FM stats or pass player_metric_map.")
        if not self.metric_map["defence"] and not self.attribute_map["defence"]:
            raise DataError("No recognisable defensive performance or attributes. Include FM stats or pass player_metric_map.")
        if not self.metric_map["attack"] and not self.metric_map["defence"] and not (
                self.attribute_map["attack"] or self.attribute_map["defence"]):
            raise DataError("No recognisable player performance metrics. Include FM stats or pass player_metric_map.")

        self.references = {}
        for direction in ("attack", "defence"):
            for feature, col in self.metric_map[direction].items():
                self.references[feature] = self._make_references(f, col)
            for attribute in self.attribute_map[direction]:
                self.references["attribute::" + attribute] = self._make_references(f, attribute)
            total = sum(self.weights[direction].values())
            if total:
                self.weights[direction] = {k: v / total for k, v in self.weights[direction].items()}
        # Normalise market costs only if present; values remain in the file's currency.
        self.price_column = next((c for c in PRICE_COLUMNS if c in f), None)
        self.wage_column = next((c for c in WAGE_COLUMNS if c in f), None)
        if self.price_column:
            numeric(f, [self.price_column], nonnegative=True, nullable=True)
        if self.wage_column:
            numeric(f, [self.wage_column], nonnegative=True, nullable=True)
        self.training = f
        self.report = {
            "rows": len(f), "leagues": sorted(f.league.unique().tolist()),
            "metric_map": self.metric_map, "weights": self.weights,
            "attribute_map": self.attribute_map,
            "price_column": self.price_column, "wage_column": self.wage_column,
            "replacement_quantile": self.replacement_quantile,
            "evidence": "Driver-linked scores use learned team associations; fallback scores are proxies until linked data exists.",
        }
        return self

    def _make_references(self, f, col):
        output = {}
        eligible = f[f.minutes >= self.minimum_minutes]
        for league, g in eligible.groupby("league"):
            for role, h in g.groupby("role_group"):
                values = pd.to_numeric(h[col], errors="coerce").dropna().to_numpy(float)
                if len(values) >= 5:
                    output[(str(league), str(role))] = self._summary(values)
            values = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
            if len(values) >= 5:
                output[(str(league), "__all__")] = self._summary(values)
        values = pd.to_numeric(eligible[col], errors="coerce").dropna().to_numpy(float)
        if len(values) >= 5:
            output[("__all__", "__all__")] = self._summary(values)
        return output

    @staticmethod
    def _summary(values):
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)) * 1.4826)
        scale = max(mad, float(np.std(values)), 1e-8)
        return {"median": median, "scale": scale, "replacement": float(np.quantile(values, 0.25)), "n": len(values)}

    def _reference(self, feature, league, role):
        refs = self.references[feature]
        for key in ((str(league), str(role)), (str(league), "__all__"), ("__all__", "__all__")):
            if key in refs:
                return refs[key], key
        return None, None

    def _score_direction(self, f, direction):
        score = np.zeros(len(f), float)
        components = []
        for feature, col in self.metric_map[direction].items():
            ref_keys = []
            z = np.full(len(f), np.nan)
            for i, (_, row) in enumerate(f.iterrows()):
                ref, key = self._reference(feature, row.league, row.role_group)
                ref_keys.append(key)
                value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                if ref and np.isfinite(value):
                    z[i] = np.clip(_scale(value, ref["median"], ref["scale"]), -5, 5)
            # Direction is learned from the driver model's benefit where possible.
            sign = 1.0
            if self.drivers is not None:
                effect_rows = self.drivers.driver_report(direction)["drivers"]
                effect = next((x for x in effect_rows if x["feature"] == feature), None)
                if effect and effect.get("benefit", 0) < 0:
                    sign = -1.0
            oriented = np.nan_to_num(z * sign, nan=0.0)
            contribution = oriented * self.weights[direction].get(feature, 0)
            score += contribution
            components.append((feature, col, contribution, z, sign))
        return score, components

    def _score_attributes(self, f, direction):
        score = np.zeros(len(f), float)
        components = []
        attrs = self.attribute_map.get(direction, [])
        if not attrs:
            return score, components
        weight = 1.0 / len(attrs)
        for attribute in attrs:
            z = np.full(len(f), np.nan)
            for i, (_, row) in enumerate(f.iterrows()):
                ref, _ = self._reference("attribute::" + attribute, row.league, row.role_group)
                value = pd.to_numeric(pd.Series([row.get(attribute)]), errors="coerce").iloc[0]
                if ref and np.isfinite(value):
                    z[i] = np.clip(_scale(value, ref["median"], ref["scale"]), -5, 5)
            contribution = np.nan_to_num(z, nan=0.0) * weight
            score += contribution
            components.append(("attribute::" + attribute, attribute, contribution, z, 1.0))
        return score, components

    def _baseline_team(self, league):
        if self.drivers is None:
            return None
        f = getattr(self.drivers, "raw_frame", self.drivers.frame)
        subset = f[f.league.astype(str) == str(league)]
        if subset.empty:
            subset = f
        # Numeric medians preserve all categories; choose the most frequent league.
        row = subset.iloc[[0]].copy()
        for col in subset.columns:
            if pd.api.types.is_numeric_dtype(subset[col]):
                row[col] = float(subset[col].median())
        row["league"] = str(league)
        return row

    def _goal_sensitivity(self, row, *, attack_score, defence_score, league, matches):
        if self.drivers is None:
            return np.nan, np.nan, "no_goal_driver_model"
        baseline = self._baseline_team(league)
        if baseline is None:
            return np.nan, np.nan, "no_team_baseline"
        # A score unit is a weighted, role-standardised player contribution. We use
        # local driver-model sensitivities to translate it to approximate seasonal goals.
        # This is intentionally labelled sensitivity, not a guaranteed player's causal effect.
        baseline_prediction = self.drivers.predict(baseline)
        a_rep = baseline_prediction.predicted_goals_for_per_match.iloc[0]
        d_rep = baseline_prediction.predicted_goals_against_per_match.iloc[0]
        a_step = max(abs(a_rep) * 0.05, 0.01)
        d_step = max(abs(d_rep) * 0.05, 0.01)
        return float(attack_score * a_step * matches), float(-defence_score * d_step * matches), "local_model_sensitivity"

    def score(self, frame: pd.DataFrame, *, expected_minutes=None, matches=None, team_baseline=None):
        if not hasattr(self, "training"):
            raise NotFittedError("Fit player observations first.")
        f, _ = _prepare_players(frame, role_column=self.role_column, minimum_minutes=self.minimum_minutes)
        if expected_minutes is not None:
            if np.isscalar(expected_minutes):
                f["expected_minutes"] = float(expected_minutes)
            else:
                values = np.asarray(expected_minutes, dtype=float)
                if len(values) != len(f):
                    raise DataError("expected_minutes must be scalar or match the player rows.")
                f["expected_minutes"] = values
        elif "expected_minutes" not in f:
            # Current minutes are evidence, not a promise. A capped default avoids
            # assigning an entire season to a player with a tiny sample. Missing
            # minutes remain missing rather than being silently promoted to 450.
            f["expected_minutes"] = np.where(
                f.minutes.notna(), np.clip(f.minutes, self.minimum_minutes, 3_600), np.nan
            )
        numeric(f, ["expected_minutes"], nullable=True)
        if (f.expected_minutes < 0).any():
            raise DataError("expected_minutes cannot be negative.")
        attack_score, attack_components = self._score_direction(f, "attack")
        defence_score, defence_components = self._score_direction(f, "defence")
        attack_attributes, attack_attribute_components = self._score_attributes(f, "attack")
        defence_attributes, defence_attribute_components = self._score_attributes(f, "defence")
        metric_columns = list(dict.fromkeys(
            list(self.metric_map["attack"].values()) + list(self.metric_map["defence"].values())
        ))
        metric_evidence = (f[metric_columns].notna().any(axis=1).to_numpy(bool)
                           if metric_columns else np.zeros(len(f), dtype=bool))
        attribute_columns = list(dict.fromkeys(self.attribute_map["attack"] + self.attribute_map["defence"]))
        attribute_evidence = (f[attribute_columns].notna().any(axis=1).to_numpy(bool)
                              if attribute_columns else np.zeros(len(f), dtype=bool))
        # Attributes are the usable fallback for current-save targets whose seasonal
        # statistics are unavailable. When both are present they remain separate so
        # the app does not silently double-count the same skill.
        if not self.metric_map["attack"]:
            attack_score, attack_components = attack_attributes, attack_attribute_components
        if not self.metric_map["defence"]:
            defence_score, defence_components = defence_attributes, defence_attribute_components
        out = f.copy()
        out["attack_contribution_units"] = attack_score
        out["defensive_contribution_units"] = defence_score
        out["football_contribution_units"] = attack_score + defence_score
        out["attribute_attack_units"] = attack_attributes
        out["attribute_defence_units"] = defence_attributes
        out["profile_evidence"] = np.select(
            [metric_evidence, attribute_evidence, f.minutes.notna().to_numpy(bool)],
            ["performance", "attributes", "minutes_only"],
            default="none",
        )
        out["attribute_evidence"] = np.where(
            (f["minutes"].isna() | ~f["minutes"].ge(self.minimum_minutes)) &
            ((attack_attributes != 0) | (defence_attributes != 0)),
            "attribute_profile_available", "performance_profile_preferred")
        out["data_quality"] = np.select(
            [f.minutes.isna(), f.minutes >= self.minimum_minutes],
            ["no_minutes", "adequate_minutes"],
            default="small_sample",
        )
        out["evidence_note"] = "Role-relative standardised output; inspect component columns and model report."
        out["approx_goals_for_impact"] = np.nan
        out["approx_goals_against_reduction"] = np.nan
        out["team_impact_evidence"] = "not_calculated"
        for i, (_, row) in enumerate(out.iterrows()):
            if pd.isna(row.expected_minutes) and matches is None:
                a, d, evidence = np.nan, np.nan, "no_minutes_for_impact"
            else:
                row_matches = matches
                if row_matches is None:
                    row_matches = max(1, float(row.expected_minutes) / 90)
                a, d, evidence = self._goal_sensitivity(
                    row, attack_score=attack_score[i], defence_score=defence_score[i], league=row.league,
                    matches=int(row_matches))
            out.loc[row.name, "approx_goals_for_impact"] = a
            out.loc[row.name, "approx_goals_against_reduction"] = d
            out.loc[row.name, "team_impact_evidence"] = evidence
        # Add transparent component contributions, useful for Streamlit explainability.
        for direction, components in (("attack", attack_components), ("defence", defence_components)):
            for feature, col, contribution, z, sign in components:
                out[f"{direction}_{feature}_component"] = contribution
                out[f"{direction}_{feature}_percentile_signal"] = z * sign
        for direction, components in (("attack", attack_attribute_components), ("defence", defence_attribute_components)):
            for feature, col, contribution, z, sign in components:
                label = feature.replace("attribute::", "")
                out[f"{direction}_attribute_{label}_component"] = contribution
                out[f"{direction}_attribute_{label}_signal"] = np.nan_to_num(z, nan=0.0)
        out["market_price"] = self._market_price(out)
        out["annual_wage"] = self._annual_wage(out)
        return out

    def _market_price(self, f):
        if self.price_column and self.price_column in f:
            return pd.to_numeric(f[self.price_column], errors="coerce")
        return pd.Series(np.nan, index=f.index)

    def _annual_wage(self, f):
        if not self.wage_column or self.wage_column not in f:
            return pd.Series(np.nan, index=f.index)
        wage = pd.to_numeric(f[self.wage_column], errors="coerce")
        if self.wage_column in {"wage", "weekly_wage", "salary"}:
            return wage * 52
        return wage

    def decisions(self, frame: pd.DataFrame, *, owned_column="owned", status_column=None,
                  expected_minutes=None, matches=None, comparable_k=12, price_threshold=1.15):
        scored = self.score(frame, expected_minutes=expected_minutes, matches=matches)
        if not 1.0 < price_threshold <= 3:
            raise DataError("price_threshold must be > 1 and <= 3.")
        if owned_column not in scored:
            if status_column and status_column in scored:
                scored[owned_column] = scored[status_column].astype(str).str.lower().isin(
                    {"owned", "current", "squad", "ours"})
            else:
                scored[owned_column] = False
        scored[owned_column] = _as_bool(scored[owned_column])
        scores = scored.football_contribution_units.to_numpy(float)
        prices = scored.market_price.to_numpy(float)
        expected = np.full(len(scored), np.nan)
        ratio = np.full(len(scored), np.nan)
        # Empirical local cost frontier: the median price of the closest comparable
        # players in league/role, excluding the player itself. It avoids an arbitrary
        # currency-per-goal conversion and adapts to each save's market.
        for i, row in scored.reset_index(drop=True).iterrows():
            candidates = scored.reset_index(drop=True)
            candidates = candidates[(candidates.league == row.league) & (candidates.role_group == row.role_group)]
            candidates = candidates[candidates.index != i]
            candidates = candidates[np.isfinite(candidates.market_price) & np.isfinite(candidates.football_contribution_units)]
            if len(candidates) < max(3, comparable_k // 2):
                candidates = scored.reset_index(drop=True)
                candidates = candidates[(candidates.index != i) & np.isfinite(candidates.market_price) &
                                        np.isfinite(candidates.football_contribution_units)]
            if len(candidates):
                distances = np.abs(candidates.football_contribution_units - row.football_contribution_units)
                near = candidates.loc[distances.nsmallest(min(comparable_k, len(candidates))).index]
                expected[i] = float(np.median(near.market_price))
                if np.isfinite(prices[i]) and expected[i] > 0:
                    ratio[i] = prices[i] / expected[i]
        scored["expected_market_price_for_contribution"] = expected
        scored["price_vs_comparables"] = ratio
        scored["above_role_replacement"] = self._above_replacement(scored)
        decisions, reasons = [], []
        for i, row in scored.reset_index(drop=True).iterrows():
            price = row.market_price
            premium = row.price_vs_comparables
            above = row.above_role_replacement
            if row.profile_evidence == "none":
                decisions.append("INSUFFICIENT_PERFORMANCE_DATA")
                reasons.append("No minutes, performance statistics or attributes were exported for this player.")
            elif row.profile_evidence == "minutes_only":
                decisions.append("INSUFFICIENT_PERFORMANCE_DATA")
                reasons.append("Minutes are present but no recognisable performance or attribute measures were exported.")
            elif not np.isfinite(price):
                decisions.append("INSUFFICIENT_PRICE_DATA")
                reasons.append("Provide asking/market value or transfer fee to compare cost.")
            elif row[owned_column] and np.isfinite(premium) and premium > price_threshold and above <= 0:
                decisions.append("SELL")
                reasons.append("Market price is above comparable contribution while player is at/below replacement.")
            elif (not row[owned_column]) and np.isfinite(premium) and premium < 1 / price_threshold and above > 0:
                decisions.append("BUY")
                reasons.append("Contribution is above role replacement at a below-comparable price.")
            elif row[owned_column]:
                decisions.append("KEEP_REVIEW")
                reasons.append("No clear sell edge under the current replacement and market evidence.")
            else:
                decisions.append("WATCH")
                reasons.append("No clear buy edge; retain for tactical and contract review.")
        scored["decision"] = decisions
        scored["decision_reason"] = reasons
        scored["decision_confidence"] = np.where(
            (scored.minutes >= self.minimum_minutes) & np.isfinite(scored.market_price) & np.isfinite(expected), "higher", "lower")
        if self.drivers is None:
            scored["decision_confidence"] = "descriptive_proxy"
        return scored

    def _above_replacement(self, scored):
        output = []
        for _, row in scored.iterrows():
            peers = scored[(scored.league == row.league) & (scored.role_group == row.role_group)]
            peers = peers[np.isfinite(peers.football_contribution_units)]
            if len(peers) < 5:
                peers = scored[np.isfinite(scored.football_contribution_units)]
            if len(peers) == 0:
                output.append(np.nan)
            else:
                output.append(float(row.football_contribution_units - peers.football_contribution_units.quantile(self.replacement_quantile)))
        return output

    def attribute_values(self):
        if not hasattr(self, "training"):
            raise NotFittedError("Fit player observations first.")
        rows = []
        for direction in ("attack", "defence"):
            for feature, col in self.metric_map[direction].items():
                rows.append({"direction": direction, "team_driver": feature, "player_metric": col,
                             "weight": self.weights[direction].get(feature, 0),
                             "reference_groups": len(self.references.get(feature, {})),
                             "interpretation": "One role-relative standard deviation, oriented by learned driver benefit."})
            for attribute in self.attribute_map.get(direction, []):
                rows.append({"direction": direction, "team_driver": None, "player_metric": attribute,
                             "weight": 1 / max(len(self.attribute_map[direction]), 1),
                             "reference_groups": len(self.references.get("attribute::" + attribute, {})),
                             "interpretation": "Role-relative attribute proxy; not a learned causal team effect."})
        return pd.DataFrame(rows).sort_values(["direction", "weight"], ascending=[True, False]).reset_index(drop=True)

    def to_dict(self):
        return {"minimum_minutes": self.minimum_minutes, "replacement_quantile": self.replacement_quantile,
                "seed": self.seed, "metric_map": self.metric_map, "weights": self.weights,
                "attribute_map": self.attribute_map,
                "references": {k: {"||".join(key): value for key, value in v.items()} for k, v in self.references.items()},
                "role_column": self.role_column, "price_column": self.price_column, "wage_column": self.wage_column,
                "report": self.report, "training": self.training.to_json(orient="records"),
                "drivers": None if self.drivers is None else self.drivers.to_dict()}

    @classmethod
    def from_dict(cls, state):
        import io
        drivers = None if state["drivers"] is None else GoalDriverModel.from_dict(state["drivers"])
        obj = cls(drivers, minimum_minutes=state["minimum_minutes"], replacement_quantile=state["replacement_quantile"], seed=state["seed"])
        obj.metric_map, obj.weights, obj.role_column = state["metric_map"], state["weights"], state["role_column"]
        obj.attribute_map = state.get("attribute_map", {"attack": [], "defence": []})
        obj.references = {k: {tuple(key.split("||")): value for key, value in v.items()} for k, v in state["references"].items()}
        obj.price_column, obj.wage_column, obj.report = state["price_column"], state["wage_column"], state["report"]
        obj.training = pd.read_json(io.StringIO(state["training"]))
        return obj
