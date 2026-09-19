"""Learn what team processes predict goals for and goals against.

The model deliberately treats these as predictive associations. Team-level FM data
cannot, by itself, prove that changing one statistic causes a specific goal change.
The public outputs expose backtests, feature coverage and extrapolation flags so a
Streamlit user can see how much evidence supports a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import numeric, parse_number, require, season_number
from .errors import DataError, NotFittedError
from .learning import LearnedRegressor


# Names cover common FM Data Hub/export views and are intentionally suggestions.
# Explicit feature lists remain available because custom FM views may use different labels.
ATTACK_CANDIDATES = (
    "xg", "shots", "shots_on_target", "shots_inside_box", "big_chances", "clear_cut_chances",
    "key_passes", "chances_created", "final_third_entries", "penalty_area_entries", "touches_in_box",
    "progressive_passes", "successful_dribbles", "crosses_completed", "set_piece_goals",
    "corner_goals", "free_kick_goals", "penalty_goals", "goals_from_corners", "possession_pct",
)
DEFENCE_CANDIDATES = (
    "xg_against", "shots_against", "shots_on_target_against", "shots_inside_box_against",
    "big_chances_against", "chances_conceded", "penalty_area_entries_against", "touches_in_box_against",
    "final_third_entries_against", "pressures", "successful_pressures", "tackles_won",
    "interceptions", "blocks", "clearances", "headers_won", "aerial_duels_won", "possession_won",
    "goals_prevented", "set_piece_goals_against", "corner_goals_against", "penalty_goals_against",
    "errors_leading_to_shot", "errors_leading_to_goal", "possession_lost", "possession_lost_in_own_half",
    "saves", "save_pct", "clean_sheets",
)


def _rate_columns(frame: pd.DataFrame, features: Iterable[str], matches: str = "matches") -> pd.DataFrame:
    """Return raw features with per-match versions for count columns.

    FM exports use a mixture of season totals, per-90 values and percentages. A
    feature ending in ``_p90`` or ``_pct`` is retained; ordinary counts are converted
    to per-match values, keeping the exposure comparison meaningful across leagues.
    """
    f = frame.copy()
    require(f, [matches])
    f[matches] = f[matches].map(lambda value: parse_number(value, ranges="error"))
    numeric(f, [matches])
    if (f[matches] <= 0).any():
        raise DataError("matches must be positive for team driver modelling.")
    for col in features:
        if col not in f:
            continue
        f[col] = f[col].map(lambda value: parse_number(value, ranges="error"))
        numeric(f, [col], nonnegative=False, nullable=True)
        if not (col.endswith("_p90") or col.endswith("_pct") or col.endswith("_rate") or col.endswith("_ratio")):
            f[col] = f[col] / f[matches]
    return f


def _feature_effects(model: LearnedRegressor, frame: pd.DataFrame, features: list[str], *, direction: str):
    base = model.predict(frame, coverage=0.8).prediction.iloc[0]
    rows = []
    for col in features:
        if col not in frame:
            continue
        row = frame.copy()
        step = max(model.design.scales.get(col, 1.0), 1e-8)
        row[col] = row[col].iloc[0] + step
        changed = model.predict(row, coverage=0.8).prediction.iloc[0] - base
        # For defence, a negative predicted goals-against change is beneficial.
        benefit = -changed if direction == "defence" else changed
        rows.append({"feature": col, "one_training_sd_change": float(step),
                     "predicted_rate_change": float(changed), "benefit": float(benefit)})
    return sorted(rows, key=lambda x: -x["benefit"])


@dataclass
class DriverRoute:
    direction: str
    target_goals: float
    matches: int
    feasible: bool
    predicted_goals: float
    changes: list[dict]
    objective: float
    extrapolation: bool

    def to_dict(self):
        return vars(self)


class GoalDriverModel:
    """Two learned models: attack -> goals for and defence -> goals against."""

    def __init__(self, *, alpha=10.0, min_minutes=None, seed=26):
        self.alpha, self.min_minutes, self.seed = alpha, min_minutes, seed

    def fit(self, frame: pd.DataFrame, *, attacking_features=None, defensive_features=None,
            include_all_available=False):
        require(frame, ["league", "season", "matches", "goals_for", "goals_against"])
        f = frame.copy()
        # Team metric exports commonly use labels such as 2024/25 and formatted
        # display numbers. Canonicalise those fields here instead of requiring a
        # Streamlit caller to pre-clean them.
        f["season"] = f["season"].map(season_number)
        for col in ("matches", "goals_for", "goals_against"):
            f[col] = f[col].map(lambda value: parse_number(value, ranges="error"))
        # validate_league_table checks season/league consistency when team identifiers exist;
        # for driver tables we permit repeated or partial observations but retain canonical checks.
        numeric(f, ["matches", "goals_for", "goals_against"])
        if (f.matches <= 0).any():
            raise DataError("matches must be positive.")
        f["goals_for_rate"] = f.goals_for / f.matches
        f["goals_against_rate"] = f.goals_against / f.matches
        attack = list(attacking_features) if attacking_features is not None else [
            c for c in ATTACK_CANDIDATES if c in f.columns
        ]
        defence = list(defensive_features) if defensive_features is not None else [
            c for c in DEFENCE_CANDIDATES if c in f.columns
        ]
        if include_all_available:
            protected = {"league", "season", "matches", "goals_for", "goals_against",
                         "goals_for_rate", "goals_against_rate", "points", "position"}
            available = [c for c in f.columns if c not in protected and pd.api.types.is_numeric_dtype(f[c])]
            attack = list(dict.fromkeys(attack + available))
            defence = list(dict.fromkeys(defence + available))
        attack = [c for c in attack if c not in {"goals_for", "goals_against", "points", "position"}]
        defence = [c for c in defence if c not in {"goals_for", "goals_against", "points", "position"}]
        if not attack:
            raise DataError("No attacking features found. Export xG/shots/chance/set-piece fields or pass attacking_features.")
        if not defence:
            raise DataError("No defensive features found. Export xG/shots-against/pressing/goalkeeping fields or pass defensive_features.")
        raw_frame = f.copy()
        f = _rate_columns(f, list(dict.fromkeys(attack + defence)))
        # Team rows can have several observations per season. LearnedRegressor enforces a
        # chronological final-season holdout and applies recency weights.
        self.attack_features, self.defensive_features = attack, defence
        self.attack = LearnedRegressor(attack, ["league"], log_target=True, nonnegative=True,
                                       interactions=self._candidate_interactions(attack), seed=self.seed)
        self.defence = LearnedRegressor(defence, ["league"], log_target=True, nonnegative=True,
                                        interactions=self._candidate_interactions(defence), seed=self.seed)
        self.attack.fit(f, "goals_for_rate")
        self.defence.fit(f, "goals_against_rate")
        self.frame = f
        self.raw_frame = raw_frame
        self.report = {
            "attack": self.attack.report, "defence": self.defence.report,
            "attack_features": attack, "defensive_features": defence,
            "observations": len(f), "leagues": sorted(f.league.astype(str).unique()),
            "data_requirements": {
                "required": ["league", "season", "matches", "goals_for", "goals_against"],
                "attacking_examples": list(ATTACK_CANDIDATES),
                "defensive_examples": list(DEFENCE_CANDIDATES),
            },
            "interpretation": "Feature importance is predictive association and may reflect tactics, teammates or selection.",
        }
        return self

    @staticmethod
    def _candidate_interactions(features):
        # Keep the model compact and interpretable. Add only common volume x quality pairs.
        pairs = []
        if "shots" in features and "xg" in features:
            pairs.append(("shots", "xg"))
        if "chances_created" in features and "xg" in features:
            pairs.append(("chances_created", "xg"))
        return pairs

    def _check(self):
        if not hasattr(self, "attack"):
            raise NotFittedError("Fit team observations before using the goal-driver model.")

    def predict(self, frame: pd.DataFrame, *, matches=None, coverage=0.8):
        self._check()
        f = _rate_columns(frame, list(dict.fromkeys(self.attack_features + self.defensive_features)))
        a = self.attack.predict(f, coverage=coverage)
        d = self.defence.predict(f, coverage=coverage)
        n_matches = f["matches"] if matches is None else matches
        n_matches = np.asarray(n_matches, dtype=float)
        if np.any(n_matches <= 0):
            raise DataError("matches must be positive.")
        out = pd.DataFrame(index=f.index)
        out["predicted_goals_for_per_match"] = a.prediction
        out["goals_for_lower"] = a.lower
        out["goals_for_upper"] = a.upper
        out["predicted_goals_against_per_match"] = d.prediction
        out["goals_against_lower"] = d.lower
        out["goals_against_upper"] = d.upper
        out["predicted_goals_for"] = a.prediction.to_numpy() * n_matches
        out["predicted_goals_against"] = d.prediction.to_numpy() * n_matches
        out["attack_out_of_range_features"] = a.out_of_range_features
        out["defence_out_of_range_features"] = d.out_of_range_features
        return out

    def driver_report(self, direction):
        self._check()
        if direction not in {"attack", "defence"}:
            raise DataError("direction must be 'attack' or 'defence'.")
        model = self.attack if direction == "attack" else self.defence
        features = self.attack_features if direction == "attack" else self.defensive_features
        median = self.frame.loc[self.frame.league.astype(str) == str(self.frame.league.iloc[0])].copy()
        # Use a representative row for marginal feature effects.
        representative = self.frame.iloc[[0]].copy()
        effects = _feature_effects(model, representative, features, direction=direction)
        importances = {x["feature"]: x for x in model.report.get("importance", [])}
        rows = []
        for effect in effects:
            row = dict(effect)
            row["validation_mae_increase"] = importances.get(effect["feature"], {}).get("mae_increase")
            rows.append(row)
        return {"direction": direction, "model_report": model.report, "drivers": rows,
                "features_used": features, "note": "A driver can proxy for another correlated driver."}

    def route(self, baseline: pd.DataFrame, *, target_goals_for=None, target_goals_against=None,
              matches=None, direction=None):
        """Find low-distance metric routes to a target with bounds learned from history.

        The optimizer is a scenario tool, not a causal claim. It searches within observed
        feature ranges; a route that leaves that range is flagged as extrapolation.
        """
        self._check()
        if len(baseline) != 1:
            raise DataError("baseline must contain exactly one team row.")
        f = _rate_columns(baseline, list(dict.fromkeys(self.attack_features + self.defensive_features)))
        n_matches = int(matches if matches is not None else f.matches.iloc[0])
        if n_matches <= 0:
            raise DataError("matches must be positive.")
        directions = [direction] if direction else ["attack", "defence"]
        result = {}
        for d in directions:
            if d == "attack":
                model, features, target = self.attack, self.attack_features, target_goals_for
            elif d == "defence":
                model, features, target = self.defence, self.defensive_features, target_goals_against
            else:
                raise DataError("direction must be attack, defence or omitted.")
            if target is None:
                continue
            if target < 0 or not np.isfinite(target):
                raise DataError("Goal targets must be finite and nonnegative.")
            target_rate = target / n_matches
            base = f.iloc[0].copy()
            lows = np.array([model.design.limits[c][0] for c in features], float)
            highs = np.array([model.design.limits[c][1] for c in features], float)
            start = np.array([float(base[c]) for c in features], float)
            scales = np.array([max(model.design.scales.get(c, 1.0), 1e-8) for c in features])

            def make_row(x):
                row = f.copy()
                for col, value in zip(features, x):
                    row[col] = value
                return row

            def prediction(x):
                return float(model.predict(make_row(x), coverage=0.8).prediction.iloc[0])

            def objective(x):
                shortfall = max(target_rate - prediction(x), 0.0) if d == "attack" else max(prediction(x) - target_rate, 0.0)
                return float(np.sum(((x - start) / scales) ** 2) + 10_000 * shortfall**2)

            constraint = ({"type": "ineq", "fun": lambda x: prediction(x) - target_rate}
                          if d == "attack" else {"type": "ineq", "fun": lambda x: target_rate - prediction(x)})
            optimum = minimize(objective, np.clip(start, lows, highs), bounds=list(zip(lows, highs)),
                               constraints=constraint, method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
            # When the requested target lies outside the learned feature envelope,
            # SLSQP correctly reports infeasibility. Still return the best in-range
            # effort, so the user sees which levers move the forecast rather than a
            # misleading empty route.
            if not optimum.success:
                starts = [start, lows, highs, (lows + highs) / 2]
                candidates = []
                for seed in starts:
                    trial = minimize(objective, np.clip(seed, lows, highs), bounds=list(zip(lows, highs)),
                                     method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12})
                    candidates.append(trial)
                optimum = min(candidates, key=lambda trial: objective(trial.x))
            x = np.clip(optimum.x, lows, highs)
            pred = prediction(x) * n_matches
            changes = []
            for col, before, after, scale, lo, hi in zip(features, start, x, scales, lows, highs):
                changes.append({"feature": col, "baseline": float(before), "target_value": float(after),
                                "change": float(after - before), "standardised_change": float((after-before)/scale),
                                "training_min": float(lo), "training_max": float(hi),
                                "extrapolation": bool(after < lo - 1e-8 or after > hi + 1e-8)})
            changes.sort(key=lambda r: -abs(r["standardised_change"]))
            result[d] = DriverRoute(d, float(target), n_matches,
                                    bool((pred >= target - 1e-5 if d == "attack" else pred <= target + 1e-5)),
                                    float(pred), changes, float(optimum.fun),
                                    bool(any(c["extrapolation"] for c in changes))).to_dict()
        return result

    def to_dict(self):
        return {"alpha": self.alpha, "min_minutes": self.min_minutes, "seed": self.seed,
                "attack_features": self.attack_features, "defensive_features": self.defensive_features,
                "attack": self.attack.to_dict(), "defence": self.defence.to_dict(),
                "frame": self.frame.to_json(orient="records"), "raw_frame": self.raw_frame.to_json(orient="records"),
                "report": self.report}

    @classmethod
    def from_dict(cls, state):
        import io
        obj = cls(alpha=state["alpha"], min_minutes=state["min_minutes"], seed=state["seed"])
        obj.attack_features, obj.defensive_features = state["attack_features"], state["defensive_features"]
        obj.attack, obj.defence = LearnedRegressor.from_dict(state["attack"]), LearnedRegressor.from_dict(state["defence"])
        obj.frame = pd.read_json(io.StringIO(state["frame"]))
        obj.raw_frame = pd.read_json(io.StringIO(state.get("raw_frame", state["frame"])))
        obj.report = state["report"]
        return obj
