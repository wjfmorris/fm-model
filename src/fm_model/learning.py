"""Small inspectable regressions with chronological selection and an untouched last-season test."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .data import numeric, require, season_number
from .errors import DataError, NotFittedError


def ridge_solve(x, y, alpha=10.0, weights=None, positive=()):
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.mean(w)
    penalty = np.eye(x.shape[1]) * np.sqrt(alpha)
    penalty[0, 0] = 0  # Intercept is never penalised.
    a = np.vstack([x * np.sqrt(w[:, None]), penalty])
    b = np.r_[y * np.sqrt(w), np.zeros(x.shape[1])]
    if positive:
        lo = np.full(x.shape[1], -np.inf)
        lo[list(positive)] = 0
        return lsq_linear(a, b, bounds=(lo, np.inf)).x
    return np.linalg.lstsq(a, b, rcond=None)[0]


def time_weights(seasons, half_life=4.0):
    s = np.asarray(seasons, dtype=float)
    return np.power(0.5, (s.max() - s) / half_life)


@dataclass
class Design:
    numeric_columns: list[str]
    categorical_columns: list[str]
    interactions: list[tuple[str, str]]

    def fit(self, f):
        require(f, self.numeric_columns + self.categorical_columns)
        self.medians, self.scales, self.limits, self.categories = {}, {}, {}, {}
        for c in self.numeric_columns:
            a = pd.to_numeric(f[c], errors="raise").to_numpy(float)
            a = a[np.isfinite(a)]
            if not len(a):
                raise DataError(f"Training feature {c} has no finite observations.")
            self.medians[c] = float(np.median(a))
            self.scales[c] = max(float(np.std(a)), 1e-8)
            self.limits[c] = [float(a.min()), float(a.max())]
        for c in self.categorical_columns:
            if f[c].isna().any():
                raise DataError(f"Categorical feature {c} has missing values.")
            self.categories[c] = sorted(f[c].astype(str).unique().tolist())
        return self

    def transform(self, f, *, allow_unknown=False):
        require(f, self.numeric_columns + self.categorical_columns)
        columns, names, z = [np.ones(len(f))], ["intercept"], {}
        for c in self.numeric_columns:
            a = pd.to_numeric(f[c], errors="raise").to_numpy(float)
            missing = ~np.isfinite(a)
            z[c] = (np.where(missing, self.medians[c], a) - self.medians[c]) / self.scales[c]
            columns.extend([z[c], missing.astype(float)])
            names.extend([c, c + "__missing"])
        for c in self.categorical_columns:
            values = f[c].astype(str)
            unknown = sorted(set(values) - set(self.categories[c]))
            if unknown and not allow_unknown:
                raise DataError(f"Untrained {c}: {unknown}. Supply training data for this context.")
            for level in self.categories[c][1:]:
                columns.append((values == level).to_numpy(float))
                names.append(c + "=" + level)
        for left, right in self.interactions:
            columns.append(z[left] * z[right])
            names.append(left + "*" + right)
        return np.column_stack(columns), names

    def out_of_range(self, f):
        result = np.zeros(len(f), dtype=int)
        for c, (lo, hi) in self.limits.items():
            a = pd.to_numeric(f[c], errors="raise").to_numpy(float)
            result += ((a < lo) | (a > hi) | ~np.isfinite(a)).astype(int)
        return result

    def to_dict(self):
        return vars(self)

    @classmethod
    def from_dict(cls, state):
        obj = cls(state["numeric_columns"], state["categorical_columns"], state["interactions"])
        for key in ["medians", "scales", "limits", "categories"]:
            setattr(obj, key, state[key])
        return obj


class LearnedRegressor:
    """Coefficients are learned, not an FM attribute weight table.

    Last season is reserved for reporting; earlier seasons choose regularisation and
    optional interactions. Reported errors are never in-sample training errors.
    The deployment model then refits all supplied seasons. Intervals are empirical
    residual intervals, not causal estimates or guaranteed coverage.
    """

    def __init__(self, numeric_columns, categorical_columns=(), *, log_target=False,
                 nonnegative=True, interactions=(), seed=26):
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.log_target = log_target
        self.nonnegative = nonnegative
        self.interactions = [tuple(p) for p in interactions]
        self.seed = seed

    def _fit_once(self, f, y, alpha, interactions):
        design = Design(self.numeric_columns, self.categorical_columns, list(interactions)).fit(f)
        x, _ = design.transform(f)
        target = np.log1p(y) if self.log_target else y
        coef = ridge_solve(x, target, alpha, time_weights(f.season))
        # Smearing correction estimates the conditional arithmetic mean after log transform.
        smear = float(np.mean(np.exp(target - x @ coef))) if self.log_target else 1.0
        return design, coef, smear

    def _predict_once(self, fit, f, *, allow_unknown=False):
        design, coef, smear = fit
        x, _ = design.transform(f, allow_unknown=allow_unknown)
        y = x @ coef
        if self.log_target:
            y = np.exp(np.clip(y, -30, 20)) * smear - 1
        return np.maximum(y, 0) if self.nonnegative else y

    def fit(self, frame, target):
        require(frame, ["season", target, *self.numeric_columns, *self.categorical_columns])
        if target in self.numeric_columns or target in self.categorical_columns:
            raise DataError("Target cannot also be an input feature.")
        f = frame.copy()
        f["season"] = f.season.map(season_number)
        numeric(f, [target], nonnegative=self.nonnegative, nullable=True)
        f = f[f[target].notna()].reset_index(drop=True)
        years = sorted(f.season.unique())
        if len(f) < 24 or len(years) < 2:
            raise DataError(f"{target}: at least 24 observations across two seasons are required.")
        train = f[f.season < years[-1]]
        test = f[f.season == years[-1]]
        if len(train) < 12 or len(test) < 4:
            raise DataError(f"{target}: insufficient chronological training/test observations.")
        options = [(a, ()) for a in (1.0, 10.0, 100.0)]
        if self.interactions:
            options += [(a, tuple(self.interactions)) for a in (10.0, 100.0)]
        inner_years = years[1:-1]
        scores = []
        for alpha, interactions in options:
            errors = []
            for year in inner_years:
                tr, va = train[train.season < year], train[train.season == year]
                if len(tr) < 12 or len(va) < 4:
                    continue
                fitted = self._fit_once(tr, tr[target].to_numpy(), alpha, interactions)
                # Unknown contexts in validation are pooled; final prediction refuses them.
                p = self._predict_once(fitted, va, allow_unknown=True)
                errors.extend(np.abs(va[target].to_numpy() - p))
            scores.append(float(np.mean(errors)) if errors else np.inf)
        choice = int(np.argmin(scores)) if np.isfinite(scores).any() else 1
        self.alpha, selected_interactions = options[choice]
        evaluation_fit = self._fit_once(train, train[target].to_numpy(), self.alpha, selected_interactions)
        predictions = self._predict_once(evaluation_fit, test, allow_unknown=True)
        truth = test[target].to_numpy()
        self.residuals = (truth - predictions).tolist()
        baseline = np.full(len(test), train[target].mean())
        self.report = {
            "target": target, "training_rows": len(f), "test_season": int(years[-1]),
            "test_rows": len(test), "mae": float(np.mean(np.abs(truth - predictions))),
            "rmse": float(np.sqrt(np.mean((truth - predictions) ** 2))),
            "baseline_mae": float(np.mean(np.abs(truth - baseline))),
            "selected_alpha": self.alpha, "inner_validation_seasons": len(inner_years),
            "selected_interactions": [list(p) for p in selected_interactions],
            "interpretation": "Predictive association; not an intervention or causal effect.",
        }
        self.report["beats_baseline"] = self.report["mae"] < self.report["baseline_mae"]
        rng = np.random.default_rng(self.seed)
        importance = []
        base_mae = self.report["mae"]
        for col in self.numeric_columns:
            differences = []
            for _ in range(4):
                shuffled = test.copy()
                # Shuffle within league/role where possible, preserving broad context.
                groups = list(shuffled.groupby(self.categorical_columns).groups.values()) if (
                    self.categorical_columns) else [shuffled.index]
                for idx in groups:
                    shuffled.loc[idx, col] = rng.permutation(shuffled.loc[idx, col].to_numpy())
                pred = self._predict_once(evaluation_fit, shuffled, allow_unknown=True)
                differences.append(float(np.mean(np.abs(truth - pred)) - base_mae))
            importance.append({"feature": col, "mae_increase": float(np.mean(differences))})
        self.report["importance"] = sorted(importance, key=lambda r: -r["mae_increase"])
        self.design, self.coef, self.smear = self._fit_once(
            f, f[target].to_numpy(), self.alpha, selected_interactions)
        self.target = target
        self.last_season = int(years[-1])
        return self

    def predict(self, frame, *, coverage=0.8):
        if not hasattr(self, "coef"):
            raise NotFittedError("Fit the model first.")
        if not 0 < coverage < 1:
            raise DataError("coverage must be between zero and one.")
        mean = self._predict_once((self.design, self.coef, self.smear), frame)
        q = np.quantile(self.residuals, [(1 - coverage) / 2, (1 + coverage) / 2])
        lower, upper = mean + min(q[0], 0), mean + max(q[1], 0)
        if self.nonnegative:
            lower = np.maximum(lower, 0)
        return pd.DataFrame({"prediction": mean, "lower": lower, "upper": upper,
                             "out_of_range_features": self.design.out_of_range(frame)}, index=frame.index)

    def coefficients(self):
        _, names = self.design.transform(pd.DataFrame({
            **{c: [v] for c, v in self.design.medians.items()},
            **{c: [v[0]] for c, v in self.design.categories.items()},
        }))
        return dict(zip(names, self.coef.tolist()))

    def to_dict(self):
        return {"numeric_columns": self.numeric_columns, "categorical_columns": self.categorical_columns,
                "log_target": self.log_target, "nonnegative": self.nonnegative,
                "interactions": self.interactions, "seed": self.seed, "design": self.design.to_dict(),
                "coef": self.coef.tolist(), "smear": self.smear, "residuals": self.residuals,
                "report": self.report, "target": self.target, "last_season": self.last_season,
                "alpha": self.alpha}

    @classmethod
    def from_dict(cls, state):
        obj = cls(**{k: state[k] for k in ["numeric_columns", "categorical_columns", "log_target",
                                         "nonnegative", "interactions", "seed"]})
        obj.design = Design.from_dict(state["design"])
        obj.coef = np.array(state["coef"])
        for key in ["smear", "residuals", "report", "target", "last_season", "alpha"]:
            setattr(obj, key, state[key])
        return obj
