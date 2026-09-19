"""Goals for/against -> finishing position and points, with league-specific scoring trends."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_league_table
from .errors import DataError, NotFittedError
from .learning import ridge_solve, time_weights


def forecast_series(years, values, next_year=None):
    """Choose a recency-weighted mean or damped trend using expanding-window errors."""
    years, values = np.asarray(years, float), np.asarray(values, float)
    order = np.argsort(years)
    years, values = years[order], values[order]
    future = years[-1] + 1 if next_year is None else next_year

    def estimate(t, y, at, trend):
        w = time_weights(t)
        centre = np.average(t, weights=w)
        level = np.average(y, weights=w)
        slope = (np.sum(w * (t - centre) * (y - level)) /
                 (np.sum(w * (t - centre) ** 2) + 2.0)) if trend else 0.0
        return float(level + slope * (at - centre))

    errors = {False: [], True: []}
    for i in range(2, len(years)):
        for trend in errors:
            errors[trend].append(values[i] - estimate(years[:i], values[:i], years[i], trend))
    trend = bool(len(years) >= 4 and np.mean(np.square(errors[True])) <
                 0.95 * np.mean(np.square(errors[False])))
    prediction = estimate(years, values, future, trend)
    residuals = errors[trend]
    return {"prediction": prediction, "method": "damped_trend" if trend else "weighted_mean",
            "seasons": len(years), "backtest_errors": list(map(float, residuals)),
            "backtest_mae": float(np.mean(np.abs(residuals))) if residuals else None}


class LeagueModel:
    """First model works from complete tables alone; points are optional, never invented.

    Fit a monotone regression on log goal rates relative to the league-season pace.
    Rank is modelled on a 0..1 scale to accommodate different league sizes. Future
    scoring pace is forecast using ALL seasons with decaying, nonzero weights.
    Goal targets are a Pareto frontier, not two independently fitted quantiles.
    """

    def __init__(self, alpha=0.1, seed=26):
        self.alpha, self.seed = alpha, seed

    @staticmethod
    def _features(gf, ga, matches, pace):
        return np.column_stack([np.ones(np.size(gf)),
                                np.log1p(np.asarray(gf) / np.asarray(matches) / np.asarray(pace)),
                                -np.log1p(np.asarray(ga) / np.asarray(matches) / np.asarray(pace))])

    def _fit_coefficients(self, f, target):
        x = self._features(f.goals_for, f.goals_against, f.matches, f.goal_rate)
        return ridge_solve(x, target, self.alpha, time_weights(f.season), positive=(1, 2))

    def fit(self, frame):
        f = validate_league_table(frame)
        self.table = f
        self.leagues = sorted(f.league.unique())
        y = (f.n_teams - f.position) / (f.n_teams - 1)
        self.rank_coef = self._fit_coefficients(f, y)
        self.points_coef = None
        if "points" in f and f.points.notna().any():
            observed = f[f.points.notna()]
            adjustment = observed.get("points_adjustment", 0)
            self.points_coef = self._fit_coefficients(observed, (observed.points - adjustment) / observed.matches)
        residuals, point_errors, folds = [], [], []
        for year in sorted(f.season.unique())[1:]:
            train, test = f[f.season < year], f[f.season == year]
            coef = self._fit_coefficients(train, (train.n_teams - train.position) / (train.n_teams - 1))
            # Validation conditions on the observed goals and pace; it is NOT a preseason forecast test.
            pred = self._features(test.goals_for, test.goals_against, test.matches, test.goal_rate) @ coef
            truth = (test.n_teams - test.position) / (test.n_teams - 1)
            for league, error in zip(test.league, truth - pred):
                residuals.append({"league": league, "error": float(error)})
            folds.append({"season": int(year), "rank_mae": float(np.mean(
                np.abs(truth.to_numpy() - np.clip(pred, 0, 1)) * (test.n_teams.to_numpy() - 1)))})
            if "points" in train and train.points.notna().sum() >= 4:
                tr, te = train[train.points.notna()], test[test.points.notna()]
                c = self._fit_coefficients(tr, (tr.points - tr.get("points_adjustment", 0)) / tr.matches)
                p = self._features(te.goals_for, te.goals_against, te.matches, te.goal_rate) @ c
                point_errors.extend(((te.points - te.get("points_adjustment", 0)) / te.matches - p).tolist())
        self.residuals = residuals
        self.point_errors = point_errors
        self.report = {"seasons": int(f.season.nunique()), "tables": int(f.groupby(["league", "season"]).ngroups),
                       "rows": len(f), "validation": "expanding-season conditional goals-to-rank backtest",
                       "folds": folds, "rank_mae": float(np.mean([v["rank_mae"] for v in folds])) if folds else None,
                       "has_points": self.points_coef is not None,
                       "uncertainty_status": "empirical_backtest_residuals" if residuals else "unvalidated"}
        return self

    def context(self, league, *, matches=None, n_teams=None, next_season=None):
        if not hasattr(self, "table"):
            raise NotFittedError("Fit complete league tables first.")
        f = self.table[self.table.league == str(league)]
        if f.empty:
            raise DataError(f"No historical tables for league {league!r}.")
        latest = f[f.season == f.season.max()]
        year = int(f.season.max() + 1) if next_season is None else int(next_season)
        if year <= f.season.max():
            raise DataError("next_season must be later than the training history.")
        series = f.groupby("season").goal_rate.first()
        forecast = forecast_series(series.index, series.values, year)
        games = int(latest.matches.iloc[0]) if matches is None else matches
        size = len(latest) if n_teams is None else n_teams
        if games <= 0 or int(games) != games or size < 4 or int(size) != size:
            raise DataError("matches must be a positive integer and n_teams an integer >= 4.")
        # Bound extreme trend extrapolation; documented guard, not a football law.
        pace = float(np.clip(forecast["prediction"], series.min() * 0.5, series.max() * 1.5))
        return {"league": str(league), "season": year, "matches": int(games), "n_teams": int(size),
                "goals_per_team_match": pace, "trend": forecast}

    def evaluate(self, league, goals_for, goals_against, *, target_position=None, matches=None,
                 n_teams=None, next_season=None):
        ctx = self.context(league, matches=matches, n_teams=n_teams, next_season=next_season)
        gf, ga = np.broadcast_arrays(np.atleast_1d(goals_for).astype(float), np.atleast_1d(goals_against).astype(float))
        if (~np.isfinite(gf)).any() or (~np.isfinite(ga)).any() or (gf < 0).any() or (ga < 0).any():
            raise DataError("Goal totals must be finite and nonnegative.")
        x = self._features(gf, ga, ctx["matches"], ctx["goals_per_team_match"])
        rank_score = x @ self.rank_coef
        out = pd.DataFrame({"goals_for": gf, "goals_against": ga,
                            "predicted_position": ctx["n_teams"] - np.clip(rank_score, 0, 1) * (ctx["n_teams"] - 1)})
        if self.points_coef is not None:
            out["predicted_points"] = np.clip(x @ self.points_coef, 0, 3) * ctx["matches"]
        if target_position is not None:
            if int(target_position) != target_position or not 1 <= target_position <= ctx["n_teams"]:
                raise DataError("Target position is outside this league.")
            # Half-place threshold: continuous ranks rounded to integer finishing positions.
            threshold = (ctx["n_teams"] - target_position - 0.5) / (ctx["n_teams"] - 1)
            residuals = [r["error"] for r in self.residuals if r["league"] == str(league)]
            if len(residuals) < 12:
                residuals = [r["error"] for r in self.residuals]
            if residuals:
                # Integrate historical scoring-pace forecast errors without claiming exact coverage.
                errors = ctx["trend"]["backtest_errors"] or [0.0]
                pace_draws = np.maximum(ctx["goals_per_team_match"] + np.asarray(errors), 0.1)
                prob = np.zeros(len(out))
                for pace in pace_draws:
                    score = self._features(gf, ga, ctx["matches"], pace) @ self.rank_coef
                    # Laplace smoothing avoids spurious probabilities exactly 0 or 1.
                    prob += (np.sum(score[:, None] + np.asarray(residuals) >= threshold, axis=1) + 1) / (len(residuals) + 2)
                out["estimated_target_probability"] = prob / len(pace_draws)
            else:
                out["estimated_target_probability"] = np.nan
            out["mean_meets_target"] = rank_score >= threshold
        observed = self.table[self.table.league == str(league)]
        relative_for = gf / ctx["matches"] / ctx["goals_per_team_match"]
        relative_against = ga / ctx["matches"] / ctx["goals_per_team_match"]
        for_range = observed.goals_for / observed.matches / observed.goal_rate
        against_range = observed.goals_against / observed.matches / observed.goal_rate
        out["extrapolation"] = ((relative_for < for_range.min()) | (relative_for > for_range.max()) |
                                (relative_against < against_range.min()) | (relative_against > against_range.max()))
        return out

    def targets(self, league, position, *, probability=0.7, matches=None, n_teams=None,
                next_season=None, max_goals=None):
        if not 0 < probability < 1:
            raise DataError("probability must be between zero and one.")
        ctx = self.context(league, matches=matches, n_teams=n_teams, next_season=next_season)
        cap = int(np.ceil(ctx["goals_per_team_match"] * ctx["matches"] * 3)) if max_goals is None else max_goals
        if int(cap) != cap or not 1 <= cap <= 1000:
            raise DataError("max_goals must be an integer between 1 and 1000.")
        grid = np.arange(int(cap) + 1)
        # Build frontier with binary search: scoring more and conceding less is monotone.
        low, high = np.full(len(grid), -1), np.full(len(grid), int(cap) + 1)
        kwargs = {"target_position": position, "matches": ctx["matches"], "n_teams": ctx["n_teams"],
                  "next_season": ctx["season"]}
        if not self.residuals:
            raise DataError("At least two historical seasons are needed for probability-based targets.")
        while np.any(high - low > 1):
            mid = (low + high) // 2
            result = self.evaluate(league, grid, np.maximum(mid, 0), **kwargs)
            meets = result.estimated_target_probability.to_numpy() >= probability
            active = high - low > 1
            low = np.where(active & meets, mid, low)
            high = np.where(active & ~meets, mid, high)
        valid = low >= 0
        frontier = self.evaluate(league, grid[valid], low[valid], **kwargs)
        # Remove dominated pairs: same maximum GA with a higher required GF.
        frontier = frontier.loc[frontier.goals_against.diff().fillna(1) > 0].reset_index(drop=True)
        if frontier.empty:
            raise DataError("No goal pair reaches the requested probability in this search range.")
        f = self.table[self.table.league == str(league)]
        med_for = np.median(f.goals_for / f.matches / f.goal_rate) * ctx["goals_per_team_match"] * ctx["matches"]
        med_against = np.median(f.goals_against / f.matches / f.goal_rate) * ctx["goals_per_team_match"] * ctx["matches"]
        scale_for = max(np.std(f.goals_for / f.matches / f.goal_rate) * ctx["goals_per_team_match"] * ctx["matches"], 1)
        scale_against = max(np.std(f.goals_against / f.matches / f.goal_rate) * ctx["goals_per_team_match"] * ctx["matches"], 1)
        frontier["balanced_distance"] = ((frontier.goals_for - med_for) / scale_for) ** 2 + (
            (frontier.goals_against - med_against) / scale_against) ** 2
        supported = frontier[~frontier.extrapolation]
        candidates = supported if not supported.empty else frontier
        best = candidates.loc[candidates.balanced_distance.idxmin()].to_dict()
        return {"context": ctx, "target_position": int(position), "requested_probability": probability,
                "recommended": best, "frontier": frontier.to_dict("records"), "validation": self.report,
                "notes": ["Score at least GF AND concede at most GA for the same selected pair.",
                          "Probabilities are empirical model estimates, not guaranteed finishing positions.",
                          "The balanced pair minimises standardised distance from the league median, not transfer cost."]}

    def to_dict(self):
        return {"alpha": self.alpha, "seed": self.seed, "table": self.table.to_json(orient="records"),
                "rank_coef": self.rank_coef.tolist(),
                "points_coef": None if self.points_coef is None else self.points_coef.tolist(),
                "residuals": self.residuals, "point_errors": self.point_errors, "report": self.report}

    @classmethod
    def from_dict(cls, state):
        import io
        obj = cls(state["alpha"], state["seed"])
        obj.table = pd.read_json(io.StringIO(state["table"]))
        obj.table["league"] = obj.table.league.astype(str)
        obj.leagues = sorted(obj.table.league.unique())
        obj.rank_coef = np.array(state["rank_coef"])
        obj.points_coef = None if state["points_coef"] is None else np.array(state["points_coef"])
        obj.residuals, obj.point_errors, obj.report = state["residuals"], state["point_errors"], state["report"]
        return obj
