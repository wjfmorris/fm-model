"""Learn which FM attributes predict observable player outcomes.

This layer is optional. It requires historical player-season rows so the last
season can remain untouched for validation. When history is unavailable the rest
of the app still works, but attribute thresholds are labelled descriptive rather
than learned.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .data import ATTRIBUTES
from .errors import DataError, NotFittedError
from .learning import LearnedRegressor
from .players import _prepare_players
from .roles import canonical_role


OUTCOME_LABELS = {
    "chance_generation": "Chance generation (xG/90)",
    "chance_creation": "Chance creation (xA/key passes per 90)",
    "finishing": "Finishing above/below xG per 90",
    "defensive_output": "Defensive output proxy",
    "goalkeeping": "Goalkeeping xG prevented/90",
}


# Role-specific fits are only trained where the outcome is football-relevant.
# Every usable attribute is still considered by the global model first.
ROLE_OUTCOMES = {
    "ST": {"chance_generation", "chance_creation", "finishing"},
    "AM_W": {"chance_generation", "chance_creation", "finishing"},
    "CM_DM": {"chance_generation", "chance_creation", "defensive_output"},
    "FB_WB": {"chance_creation", "defensive_output"},
    "CB": {"defensive_output"},
    "GK": {"goalkeeping"},
}


class PlayerOutcomeModel:
    """Chronologically validated attribute -> player-outcome models."""

    def __init__(self, *, seed=26):
        self.seed = seed
        self.models: dict[str, dict[str, LearnedRegressor]] = {}
        self.report: dict = {}

    @staticmethod
    def _usable_attributes(frame: pd.DataFrame) -> list[str]:
        result = []
        for col in ATTRIBUTES:
            if col not in frame:
                continue
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.notna().mean() >= 0.6 and values.nunique(dropna=True) >= 2:
                result.append(col)
        return result

    @staticmethod
    def _defensive_proxy(frame: pd.DataFrame) -> pd.Series | None:
        cols = [
            c for c in (
                "tackles_won_p90", "interceptions_p90", "blocks_p90",
                "headers_won_p90", "pressures_p90", "possession_won_p90",
            ) if c in frame
        ]
        if len(cols) < 2:
            return None
        pieces = []
        groups = [frame["league"].astype(str), frame["season"], frame["canonical_position"]]
        for col in cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            mean = values.groupby(groups).transform("mean")
            std = values.groupby(groups).transform("std").replace(0, np.nan)
            pieces.append((values - mean) / std)
        return pd.concat(pieces, axis=1).mean(axis=1, skipna=True)

    def _targets(self, frame: pd.DataFrame) -> dict[str, tuple[str, bool]]:
        targets: dict[str, tuple[str, bool]] = {}
        chance_col = next(
            (c for c in ("non_penalty_xg_p90", "xg_p90") if c in frame and frame[c].notna().sum() >= 24),
            None,
        )
        if chance_col:
            targets["chance_generation"] = (chance_col, True)

        creation_col = next(
            (c for c in ("xa_p90", "key_passes_p90", "chances_created_p90")
             if c in frame and frame[c].notna().sum() >= 24),
            None,
        )
        if creation_col:
            targets["chance_creation"] = (creation_col, True)

        goals_col = "goals_p90" if "goals_p90" in frame else None
        if goals_col and chance_col:
            frame["finishing_over_xg_p90"] = (
                pd.to_numeric(frame[goals_col], errors="coerce")
                - pd.to_numeric(frame[chance_col], errors="coerce")
            )
            targets["finishing"] = ("finishing_over_xg_p90", False)

        defensive = self._defensive_proxy(frame)
        if defensive is not None and defensive.notna().sum() >= 24:
            frame["defensive_output_index"] = defensive
            targets["defensive_output"] = ("defensive_output_index", False)

        gk_col = next(
            (c for c in ("xg_prevented_p90", "goals_prevented_p90")
             if c in frame and frame[c].notna().sum() >= 24),
            None,
        )
        if gk_col:
            targets["goalkeeping"] = (gk_col, False)
        return targets

    @staticmethod
    def _enough(frame: pd.DataFrame, attributes: list[str], *, role_specific=False) -> bool:
        if "season" not in frame or frame["season"].nunique() < 2:
            return False
        required = max(36 if role_specific else 48, len(attributes) + (8 if role_specific else 12))
        if len(frame) < required:
            return False
        years = sorted(frame["season"].dropna().unique())
        if len(years) < 2:
            return False
        return len(frame[frame["season"] < years[-1]]) >= 12 and len(frame[frame["season"] == years[-1]]) >= 4

    def fit(self, frame: pd.DataFrame):
        prepared, role_col = _prepare_players(frame)
        if "season" not in prepared:
            self.report = {
                "available": False,
                "reason": "Historical player data needs a season column.",
                "models": {},
            }
            return self
        prepared["canonical_position"] = prepared[role_col].map(canonical_role)
        attributes = self._usable_attributes(prepared)
        if len(attributes) < 4:
            self.report = {
                "available": False,
                "reason": "At least four varying player attributes are needed.",
                "attributes": attributes,
                "models": {},
            }
            return self

        targets = self._targets(prepared)
        self.models = {}
        reports = {}
        skipped = {}
        for outcome, (target, nonnegative) in targets.items():
            self.models[outcome] = {}
            outcome_reports = {}

            global_rows = prepared[prepared[target].notna()].copy()
            if outcome == "goalkeeping":
                global_rows = global_rows[global_rows["canonical_position"].eq("GK")]
            else:
                global_rows = global_rows[~global_rows["canonical_position"].eq("GK")]
            if self._enough(global_rows, attributes):
                try:
                    model = LearnedRegressor(
                        attributes,
                        ["league", "canonical_position"],
                        log_target=False,
                        nonnegative=nonnegative,
                        seed=self.seed,
                    ).fit(global_rows, target)
                    self.models[outcome]["__all__"] = model
                    outcome_reports["__all__"] = model.report
                except DataError as exc:
                    skipped[f"{outcome}:global"] = str(exc)

            for role, subset in global_rows.groupby("canonical_position"):
                if outcome not in ROLE_OUTCOMES.get(str(role), set()):
                    continue
                if not self._enough(subset, attributes, role_specific=True):
                    continue
                try:
                    role_model = LearnedRegressor(
                        attributes,
                        ["league"],
                        log_target=False,
                        nonnegative=nonnegative,
                        seed=self.seed,
                    ).fit(subset.copy(), target)
                    self.models[outcome][str(role)] = role_model
                    outcome_reports[str(role)] = role_model.report
                except DataError as exc:
                    skipped[f"{outcome}:{role}"] = str(exc)

            if outcome_reports:
                reports[outcome] = outcome_reports

        self.frame = prepared
        self.attributes = attributes
        self.targets = {name: target for name, (target, _) in targets.items()}
        self.report = {
            "available": any(self.models.get(k) for k in self.models),
            "rows": len(prepared),
            "seasons": int(prepared["season"].nunique()),
            "attributes": attributes,
            "models": reports,
            "skipped": skipped,
            "interpretation": (
                "Attribute coefficients are predictive associations with player outcomes. "
                "They do not prove that changing an attribute causes the observed outcome."
            ),
        }
        if not self.report["available"]:
            self.report["reason"] = (
                "No attribute-outcome model had enough historical player-season evidence "
                "for chronological validation."
            )
        return self

    @property
    def available(self) -> bool:
        return bool(self.report.get("available"))

    def _model_for(self, outcome: str, role: str | None):
        if outcome not in self.models:
            return None
        role = canonical_role(role) if role else None
        return self.models[outcome].get(role) or self.models[outcome].get("__all__")

    def attribute_importance(self, outcome: str, *, role: str | None = None) -> pd.DataFrame:
        model = self._model_for(outcome, role)
        if model is None:
            return pd.DataFrame(
                columns=["attribute", "coefficient", "abs_coefficient", "validation_mae_increase", "direction"]
            )
        coefs = model.coefficients()
        importance = {
            row["feature"]: row.get("mae_increase")
            for row in model.report.get("importance", [])
        }
        rows = []
        for attribute in self.attributes:
            coefficient = float(coefs.get(attribute, 0.0))
            rows.append({
                "attribute": attribute,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
                "validation_mae_increase": importance.get(attribute),
                "direction": "higher_associated_with_more" if coefficient > 0 else (
                    "higher_associated_with_less" if coefficient < 0 else "flat"
                ),
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["_validation"] = pd.to_numeric(out["validation_mae_increase"], errors="coerce").fillna(0)
        return out.sort_values(["_validation", "abs_coefficient"], ascending=False).drop(columns="_validation").reset_index(drop=True)

    def predict_outcomes(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.available:
            raise NotFittedError("No validated player attribute-outcome models are available.")
        prepared, role_col = _prepare_players(frame)
        prepared["canonical_position"] = prepared[role_col].map(canonical_role)
        output = pd.DataFrame(index=prepared.index)

        for outcome in self.models:
            values = pd.Series(np.nan, index=prepared.index, dtype=float)
            evidence = pd.Series("unavailable", index=prepared.index, dtype=object)
            for role, idx in prepared.groupby("canonical_position").groups.items():
                model = self._model_for(outcome, role)
                if model is None:
                    continue
                rows = prepared.loc[idx].copy()
                if "canonical_position" in model.categorical_columns:
                    rows["canonical_position"] = role
                try:
                    prediction = model.predict(rows, coverage=0.8)
                except DataError:
                    continue
                values.loc[idx] = prediction["prediction"].to_numpy()
                evidence.loc[idx] = "role_model" if role in self.models[outcome] else "global_model"
            output[f"predicted_{outcome}"] = values
            output[f"{outcome}_evidence"] = evidence
        return output

    def to_dict(self):
        return {
            "seed": self.seed,
            "models": {
                outcome: {role: model.to_dict() for role, model in role_models.items()}
                for outcome, role_models in self.models.items()
            },
            "report": self.report,
            "attributes": getattr(self, "attributes", []),
            "targets": getattr(self, "targets", {}),
            "frame": getattr(self, "frame", pd.DataFrame()).to_json(orient="records"),
        }

    @classmethod
    def from_dict(cls, state):
        obj = cls(seed=state.get("seed", 26))
        obj.models = {
            outcome: {role: LearnedRegressor.from_dict(model_state) for role, model_state in role_models.items()}
            for outcome, role_models in state.get("models", {}).items()
        }
        obj.report = state.get("report", {})
        obj.attributes = state.get("attributes", [])
        obj.targets = state.get("targets", {})
        raw = state.get("frame", "[]")
        obj.frame = pd.read_json(io.StringIO(raw))
        return obj
