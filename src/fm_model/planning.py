"""End-to-end squad planning from league target to transfer recommendations."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd

from .data import season_number
from .errors import DataError, NotFittedError
from .players import ATTRIBUTE_PROXY
from .roles import FORMATION_PRESETS, ROLE_LABELS, canonical_role, formation_slots


class SquadPlanner:
    """Connect league targets, team forecasts and player replacement scenarios.

    The current forecast assumes the latest observed team-process profile repeats
    next season. Transfer effects are simulated by replacing the outgoing player's
    mapped per-90 contribution in that profile with the incoming player's output,
    then re-running the learned goal-driver model. This is a scenario model, not a
    claim that the transfer mechanically causes the exact goal change.
    """

    def __init__(self, league_model, driver_model, player_model, outcome_model=None, *, lineup_share=0.80):
        self.league_model = league_model
        self.driver_model = driver_model
        self.player_model = player_model
        self.outcome_model = outcome_model
        self.lineup_share = float(lineup_share)
        if not 0.4 <= self.lineup_share <= 1.0:
            raise ValueError("lineup_share must be between 0.4 and 1.0.")

    def _check(self):
        if self.league_model is None or self.driver_model is None or self.player_model is None:
            raise NotFittedError(
                "Squad planning requires league targets, team goal drivers and the player valuation model."
            )

    @staticmethod
    def _normal(value):
        return str(value).strip().casefold()

    def available_clubs(self, league=None):
        raw = getattr(self.driver_model, "raw_frame", None)
        if raw is None or "team_id" not in raw:
            return []
        f = raw
        if league is not None and "league" in f:
            same = f["league"].astype(str).eq(str(league))
            if same.any():
                f = f[same]
        return sorted(f["team_id"].dropna().astype(str).unique().tolist())

    def _baseline_team(self, league, club):
        raw = getattr(self.driver_model, "raw_frame", None)
        if raw is None or raw.empty:
            raise DataError("Team performance history is required for a current-team forecast.")

        f = raw.copy()
        if "league" in f:
            same_league = f["league"].astype(str).eq(str(league))
            if same_league.any():
                f = f[same_league]
        if f.empty:
            raise DataError(f"No team-performance rows are available for {league!r}.")

        evidence = "club_latest_team_profile"
        if club and "team_id" in f:
            same = f["team_id"].map(self._normal).eq(self._normal(club))
            if same.any():
                f = f[same]
            else:
                raise DataError(
                    f"No team-performance row matches club {club!r}. Select the club name exactly as it appears in the team export."
                )
        elif "team_id" in f and f["team_id"].nunique() > 1:
            raise DataError("Select your club so the app can forecast the current team rather than the league median.")

        if "season" in f:
            try:
                season_values = f["season"].map(season_number)
            except DataError:
                season_values = pd.to_numeric(f["season"], errors="coerce")
            if season_values.notna().any():
                latest = season_values.max()
                f = f[season_values.eq(latest)]

        if len(f) > 1:
            row = f.iloc[[0]].copy()
            for col in f.columns:
                values = pd.to_numeric(f[col], errors="coerce")
                if values.notna().any():
                    row[col] = float(values.median())
            evidence = "club_latest_team_profile_median"
        else:
            row = f.iloc[[0]].copy()
        return row, evidence

    def _forecast(self, baseline, matches):
        prediction = self.driver_model.predict(baseline, matches=int(matches))
        return {
            "goals_for": float(prediction["predicted_goals_for"].iloc[0]),
            "goals_against": float(prediction["predicted_goals_against"].iloc[0]),
            "attack_out_of_range_features": int(prediction["attack_out_of_range_features"].iloc[0]),
            "defence_out_of_range_features": int(prediction["defence_out_of_range_features"].iloc[0]),
        }


    def _historical_current_adjustment(self, baseline, scored, starter_ids, slots, club, matches):
        """Move the latest club profile from its historical XI to the current owned XI."""

        outcome = self.outcome_model
        if outcome is None or not getattr(outcome, "available", False):
            return baseline, {"goals_for": 0.0, "goals_against_reduction": 0.0}, {
                "applied": False,
                "reason": "No validated historical player-outcome layer is available.",
                "replacements": 0,
            }
        history = getattr(outcome, "frame", None)
        if history is None or history.empty or "team_id" not in history or "season" not in history:
            return baseline, {"goals_for": 0.0, "goals_against_reduction": 0.0}, {
                "applied": False,
                "reason": "Historical player rows do not contain club and season identifiers.",
                "replacements": 0,
            }

        hist = history[history["team_id"].map(self._normal).eq(self._normal(club))].copy()
        if hist.empty:
            return baseline, {"goals_for": 0.0, "goals_against_reduction": 0.0}, {
                "applied": False,
                "reason": "No historical player-season rows match the selected club.",
                "replacements": 0,
            }
        try:
            years = hist["season"].map(season_number)
            hist = hist[years.eq(years.max())].copy()
        except DataError:
            pass
        if "canonical_position" not in hist:
            hist["canonical_position"] = hist["role_group"].map(canonical_role)

        try:
            predictions = outcome.predict_outcomes(hist)
            for col in predictions:
                hist[col] = predictions[col].to_numpy()
        except (DataError, NotFittedError):
            pass

        current = scored[scored["_row_id"].isin(starter_ids)].copy()
        adjusted = baseline.copy()
        finishing_adjustment = 0.0
        goalkeeping_reduction = 0.0
        replacements = 0

        for position, count in slots.items():
            old_group = hist[hist["canonical_position"].eq(position)].copy()
            new_group = current[current["canonical_position"].eq(position)].copy()
            if old_group.empty or new_group.empty:
                continue
            old_group = old_group.sort_values("minutes", ascending=False, na_position="last").head(int(count))
            new_group = new_group.sort_values(
                ["football_contribution_units", "minutes"],
                ascending=[False, False],
                na_position="last",
            ).head(int(count))
            for (_, outgoing), (_, incoming) in zip(old_group.iterrows(), new_group.iterrows()):
                adjusted, used = self._apply_replacement(adjusted, outgoing, incoming)
                if used:
                    replacements += 1
                if position == "GK":
                    inc = self._finite(incoming.get("predicted_goalkeeping"))
                    out = self._finite(outgoing.get("predicted_goalkeeping")) or 0.0
                    if inc is not None:
                        goalkeeping_reduction += (inc - out) * self.lineup_share * int(matches)
                else:
                    inc = self._finite(incoming.get("predicted_finishing"))
                    out = self._finite(outgoing.get("predicted_finishing")) or 0.0
                    if inc is not None:
                        finishing_adjustment += (inc - out) * self.lineup_share * int(matches)

        return adjusted, {
            "goals_for": float(finishing_adjustment),
            "goals_against_reduction": float(goalkeeping_reduction),
        }, {
            "applied": bool(replacements),
            "reason": (
                "Latest historical starters were replaced position-by-position with the current owned XI."
                if replacements else
                "Historical/current position coverage was insufficient to adjust the club profile."
            ),
            "replacements": int(replacements),
        }

    def _forecast_with_adjustment(self, baseline, matches, adjustment=None):
        forecast = self._forecast(baseline, matches)
        adjustment = adjustment or {}
        forecast["goals_for"] = max(
            forecast["goals_for"] + float(adjustment.get("goals_for", 0.0)), 0.0
        )
        forecast["goals_against"] = max(
            forecast["goals_against"] - float(adjustment.get("goals_against_reduction", 0.0)), 0.0
        )
        forecast["attribute_finishing_adjustment"] = float(adjustment.get("goals_for", 0.0))
        forecast["attribute_goalkeeping_reduction"] = float(
            adjustment.get("goals_against_reduction", 0.0)
        )
        return forecast

    def _score_players(self, players, matches):
        scored = self.player_model.decisions(players, matches=int(matches)).reset_index(drop=True)
        scored["_row_id"] = np.arange(len(scored))
        scored["canonical_position"] = scored["role_group"].map(canonical_role)
        if self.outcome_model is not None and getattr(self.outcome_model, "available", False):
            try:
                predictions = self.outcome_model.predict_outcomes(scored)
                for col in predictions:
                    scored[col] = predictions[col].to_numpy()
            except (DataError, NotFittedError):
                pass
        return scored

    @staticmethod
    def _owned_mask(scored):
        if "owned" not in scored:
            return pd.Series(False, index=scored.index)
        values = scored["owned"]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False)
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "owned", "ours", "current", "squad"})

    def _select_lineup(self, scored, slots, *, owned_only):
        pool = scored[self._owned_mask(scored)].copy() if owned_only else scored.copy()
        selected = []
        missing = {}
        for position, count in slots.items():
            group = pool[pool["canonical_position"].eq(position)].copy()
            group = group.sort_values(
                ["football_contribution_units", "minutes"],
                ascending=[False, False],
                na_position="last",
            )
            take = group.head(int(count))
            selected.extend(take["_row_id"].astype(int).tolist())
            if len(take) < count:
                missing[position] = int(count - len(take))
        return selected, missing

    def _select_starters(self, scored, slots):
        return self._select_lineup(scored, slots, owned_only=True)

    def _feature_links(self):
        links = {}
        for direction in ("attack", "defence"):
            for team_feature, player_col in self.player_model.metric_map.get(direction, {}).items():
                links.setdefault(team_feature, player_col)
        return links

    def _score_historical_club(self, league, club, matches):
        if self.outcome_model is None or not hasattr(self.outcome_model, "frame"):
            return None
        history = self.outcome_model.frame.copy()
        if history.empty or "team_id" not in history:
            return None
        same_club = history["team_id"].astype(str).map(self._normal).eq(self._normal(club))
        history = history[same_club]
        if "league" in history:
            same_league = history["league"].astype(str).eq(str(league))
            if same_league.any():
                history = history[same_league]
        if history.empty:
            return None
        if "season" in history:
            years = pd.to_numeric(history["season"], errors="coerce")
            if years.notna().any():
                history = history[years.eq(years.max())]
        if history.empty:
            return None
        try:
            scored = self.player_model.score(history, matches=int(matches)).reset_index(drop=True)
        except (DataError, NotFittedError):
            return None
        scored["_row_id"] = np.arange(len(scored))
        scored["canonical_position"] = scored["role_group"].map(canonical_role)
        if getattr(self.outcome_model, "available", False):
            try:
                predictions = self.outcome_model.predict_outcomes(scored)
                for col in predictions:
                    scored[col] = predictions[col].to_numpy()
            except (DataError, NotFittedError):
                pass
        return scored

    def _lineup_feature_total(self, lineup, player_col, team_feature):
        values = []
        for _, row in lineup.iterrows():
            value = self._metric_value(row, player_col, team_feature)
            if value is not None:
                values.append(value)
        if not values:
            return None
        if team_feature.endswith(("_pct", "_rate", "_ratio")):
            return float(np.mean(values))
        return float(np.sum(values))

    def _adjust_baseline_for_current_squad(
        self, baseline, current_scored, starter_ids, slots, *, league, club, matches
    ):
        """Move the last team profile from last season's XI to the current XI.

        This makes the initial next-season forecast squad-aware when historical
        player seasons are available. Only player metrics already linked to learned
        team drivers are allowed to alter the baseline.
        """

        if not starter_ids:
            return baseline, {
                "status": "team_profile_only",
                "features_adjusted": 0,
                "reason": "No complete current XI could be identified.",
            }
        current_lineup = current_scored[current_scored["_row_id"].isin(starter_ids)].copy()
        if len(current_lineup) < sum(slots.values()):
            return baseline, {
                "status": "team_profile_only",
                "features_adjusted": 0,
                "reason": "The current export does not fill every formation slot.",
            }

        historical_scored = self._score_historical_club(league, club, matches)
        if historical_scored is None:
            return baseline, {
                "status": "team_profile_only",
                "features_adjusted": 0,
                "reason": "No usable previous player-season squad was available for this club.",
            }
        historical_ids, historical_missing = self._select_lineup(
            historical_scored, slots, owned_only=False
        )
        if historical_missing:
            return baseline, {
                "status": "team_profile_only",
                "features_adjusted": 0,
                "reason": "The historical player export does not fill the selected formation.",
            }
        historical_lineup = historical_scored[
            historical_scored["_row_id"].isin(historical_ids)
        ].copy()

        modified = baseline.copy()
        features = []
        for team_feature, player_col in self._feature_links().items():
            if team_feature not in modified:
                continue
            previous = self._lineup_feature_total(
                historical_lineup, player_col, team_feature
            )
            current = self._lineup_feature_total(
                current_lineup, player_col, team_feature
            )
            if previous is None or current is None:
                continue
            delta = (current - previous) * self.lineup_share
            if not team_feature.endswith(("_p90", "_pct", "_rate", "_ratio")):
                delta *= int(matches)
            base = self._finite(modified[team_feature].iloc[0])
            if base is None:
                continue
            updated = max(base + delta, 0.0)
            if team_feature.endswith("_pct"):
                updated = min(updated, 100.0)
            modified.loc[modified.index[0], team_feature] = updated
            features.append({
                "team_feature": team_feature,
                "player_metric": player_col,
                "previous_xi": previous,
                "current_xi": current,
                "baseline_change": float(delta),
            })

        return modified, {
            "status": "current_squad_adjusted" if features else "team_profile_only",
            "features_adjusted": len(features),
            "reason": (
                "Latest team profile adjusted by the mapped performance difference "
                "between last season's modelled XI and the current modelled XI."
                if features else
                "No player metrics overlapped with learned team drivers strongly enough to adjust the baseline."
            ),
            "features": features,
        }

    @staticmethod
    def _finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def _metric_value(self, row, player_col, team_feature):
        observed = self._finite(row.get(player_col))
        if team_feature in {"xg", "non_penalty_xg"}:
            predicted = self._finite(row.get("predicted_chance_generation"))
            if predicted is not None:
                if observed is None:
                    return predicted
                minutes = self._finite(row.get("minutes")) or 0.0
                # Attribute-predicted chance generation acts as a prior. Observed
                # performance receives more weight as minutes accumulate.
                observed_weight = minutes / (minutes + 900.0)
                return observed_weight * observed + (1.0 - observed_weight) * predicted
        return observed

    def _apply_replacement(self, baseline, outgoing, incoming):
        modified = baseline.copy()
        if len(modified) != 1:
            raise DataError("Team baseline must contain exactly one row.")
        base_matches = float(pd.to_numeric(modified["matches"], errors="coerce").iloc[0])
        if not np.isfinite(base_matches) or base_matches <= 0:
            raise DataError("Baseline team matches must be positive.")

        used = []
        for team_feature, player_col in self._feature_links().items():
            if team_feature not in modified:
                continue
            before = self._metric_value(outgoing, player_col, team_feature)
            after = self._metric_value(incoming, player_col, team_feature)
            if before is None or after is None:
                continue
            delta = (after - before) * self.lineup_share
            if team_feature.endswith(("_p90", "_pct", "_rate", "_ratio")):
                team_delta = delta
            else:
                team_delta = delta * base_matches
            current = self._finite(modified[team_feature].iloc[0])
            if current is None:
                continue
            modified.loc[modified.index[0], team_feature] = max(current + team_delta, 0.0)
            used.append({
                "team_feature": team_feature,
                "player_metric": player_col,
                "outgoing": before,
                "incoming": after,
                "team_feature_delta": float(team_delta),
            })
        return modified, used

    def _placeholder(self, pool, position):
        subset = pool[pool["canonical_position"].eq(position)]
        row = {}
        for col in pool.columns:
            values = pd.to_numeric(subset[col], errors="coerce") if col in subset else pd.Series(dtype=float)
            if values.notna().any():
                row[col] = float(values.quantile(0.25))
            elif col in subset and len(subset):
                row[col] = subset[col].iloc[0]
        row.update({
            "_row_id": -1,
            "player_name": "Replacement-level placeholder",
            "canonical_position": position,
            "role_group": ROLE_LABELS.get(position, position),
            "market_price": 0.0,
            "expected_market_price_for_contribution": np.nan,
            "price_vs_comparables": np.nan,
            "football_contribution_units": 0.0,
            "above_role_replacement": 0.0,
        })
        return pd.Series(row)

    def _simulate_candidates(self, baseline, current_forecast, outgoing, candidates, matches, base_adjustment=None):
        if candidates.empty:
            return pd.DataFrame()
        modified_rows = []
        metadata = []
        for _, candidate in candidates.iterrows():
            changed, used = self._apply_replacement(baseline, outgoing, candidate)
            modified_rows.append(changed.iloc[0])
            metadata.append((int(candidate["_row_id"]), used))
        batch = pd.DataFrame(modified_rows).reset_index(drop=True)
        prediction = self.driver_model.predict(batch, matches=int(matches))
        result = candidates.reset_index(drop=True).copy()
        base_adjustment = base_adjustment or {}
        projected_for = (
            prediction["predicted_goals_for"].to_numpy(dtype=float)
            + float(base_adjustment.get("goals_for", 0.0))
        )
        projected_against = (
            prediction["predicted_goals_against"].to_numpy(dtype=float)
            - float(base_adjustment.get("goals_against_reduction", 0.0))
        )

        # Historical attribute models add two same-unit corrections that team
        # process statistics cannot capture directly: finishing above/below xG,
        # and goalkeeper xG prevention. These remain predictions, not causal effects.
        attr_for = np.zeros(len(result), dtype=float)
        attr_against = np.zeros(len(result), dtype=float)
        outgoing_finish = self._finite(outgoing.get("predicted_finishing")) or 0.0
        outgoing_gk = self._finite(outgoing.get("predicted_goalkeeping")) or 0.0
        for j, (_, candidate) in enumerate(result.iterrows()):
            incoming_finish = self._finite(candidate.get("predicted_finishing"))
            if incoming_finish is not None and candidate.get("canonical_position") != "GK":
                attr_for[j] = (incoming_finish - outgoing_finish) * self.lineup_share * int(matches)
            incoming_gk = self._finite(candidate.get("predicted_goalkeeping"))
            if incoming_gk is not None and candidate.get("canonical_position") == "GK":
                attr_against[j] = (incoming_gk - outgoing_gk) * self.lineup_share * int(matches)

        result["attribute_finishing_adjustment"] = attr_for
        result["attribute_goalkeeping_reduction"] = attr_against
        result["projected_goals_for"] = np.maximum(projected_for + attr_for, 0.0)
        result["projected_goals_against"] = np.maximum(projected_against - attr_against, 0.0)
        result["gf_gain"] = result["projected_goals_for"] - current_forecast["goals_for"]
        result["ga_reduction"] = current_forecast["goals_against"] - result["projected_goals_against"]
        result["outgoing_row_id"] = int(outgoing.get("_row_id", -1))
        result["outgoing_player"] = str(outgoing.get("player_name", "Replacement-level placeholder"))
        result["mapped_features_used"] = [len(item[1]) for item in metadata]
        return result

    @staticmethod
    def _impact_score(frame, attack_gap, defence_gap):
        attack = np.maximum(pd.to_numeric(frame["gf_gain"], errors="coerce").fillna(0).to_numpy(), 0)
        defence = np.maximum(pd.to_numeric(frame["ga_reduction"], errors="coerce").fillna(0).to_numpy(), 0)
        score = np.zeros(len(frame), dtype=float)
        if attack_gap > 0:
            score += np.minimum(attack / max(attack_gap, 1e-8), 1.0)
        if defence_gap > 0:
            score += np.minimum(defence / max(defence_gap, 1e-8), 1.0)
        return score

    def _position_candidates(self, baseline, current_forecast, scored, starters, position, slots, matches,
                             attack_gap, defence_gap, base_adjustment=None):
        market = scored[
            (~self._owned_mask(scored))
            & scored["canonical_position"].eq(position)
            & ~scored["profile_evidence"].isin(["none", "minutes_only"])
        ].copy()
        if market.empty:
            return pd.DataFrame(), []

        outgoing_rows = scored[scored["_row_id"].isin(starters) & scored["canonical_position"].eq(position)]
        if outgoing_rows.empty:
            outgoing_rows = pd.DataFrame([self._placeholder(scored, position)])

        simulations = []
        for _, outgoing in outgoing_rows.iterrows():
            sim = self._simulate_candidates(
                baseline, current_forecast, outgoing, market, matches, base_adjustment=base_adjustment
            )
            if not sim.empty:
                simulations.append(sim)
        if not simulations:
            return pd.DataFrame(), []

        all_sim = pd.concat(simulations, ignore_index=True)
        all_sim["impact_score"] = self._impact_score(all_sim, attack_gap, defence_gap)
        all_sim["positive_goal_swing"] = (
            np.maximum(all_sim["gf_gain"], 0) + np.maximum(all_sim["ga_reduction"], 0)
        )
        all_sim = all_sim.sort_values(
            ["_row_id", "impact_score", "positive_goal_swing"],
            ascending=[True, False, False],
        ).drop_duplicates("_row_id", keep="first")

        price = pd.to_numeric(all_sim["market_price"], errors="coerce")
        expected = pd.to_numeric(all_sim["expected_market_price_for_contribution"], errors="coerce")
        all_sim["estimated_value_edge"] = expected - price
        all_sim["estimated_value_edge_pct"] = expected / price.replace(0, np.nan) - 1

        max_impact = float(all_sim["impact_score"].max()) if len(all_sim) else 0.0
        viable = all_sim[all_sim["impact_score"] >= max(max_impact * 0.50, 0.01)].copy()
        finite_price = np.isfinite(pd.to_numeric(viable["market_price"], errors="coerce"))
        priced = viable[finite_price].copy()
        ranking_pool = priced if not priced.empty else viable
        if ranking_pool.empty:
            return all_sim, []

        price_m = np.maximum(pd.to_numeric(ranking_pool["market_price"], errors="coerce").fillna(0).to_numpy() / 1e6, 0.25)
        value_bonus = np.maximum(pd.to_numeric(ranking_pool["estimated_value_edge_pct"], errors="coerce").fillna(0).to_numpy(), 0)
        ranking_pool["planner_value_score"] = (
            ranking_pool["impact_score"].to_numpy() * (1 + 0.25 * np.minimum(value_bonus, 2.0))
            / np.sqrt(price_m)
        )
        ranking_pool = ranking_pool.sort_values(
            ["planner_value_score", "impact_score", "positive_goal_swing"],
            ascending=False,
        )
        best = ranking_pool.iloc[0]

        profile = self._minimum_profile(
            scored,
            all_sim,
            position,
            best,
            attack_gap=attack_gap,
            defence_gap=defence_gap,
        )
        shortlist_columns = [
            c for c in (
                "player_name", "team_id", "market_price", "expected_market_price_for_contribution",
                "estimated_value_edge", "gf_gain", "ga_reduction", "impact_score",
                "planner_value_score", "football_contribution_units", "above_role_replacement",
                "decision_confidence", "mapped_features_used",
            ) if c in ranking_pool.columns
        ]
        shortlist = ranking_pool.head(8)[shortlist_columns].replace({np.nan: None}).to_dict("records")

        # Keep a compact set of viable alternatives for the package optimiser.
        # This means "cheapest route" can choose a cheaper second/third option in
        # a position when that player is still sufficient to reach the target.
        candidate_options = []
        for _, option in ranking_pool.head(4).iterrows():
            candidate_options.append({
                "position": position,
                "position_label": ROLE_LABELS.get(position, position),
                "slots": int(slots),
                "outgoing_row_id": int(option["outgoing_row_id"]),
                "outgoing_player": str(option["outgoing_player"]),
                "recommended_player_row_id": int(option["_row_id"]),
                "recommended_player": str(option.get("player_name", "")),
                "recommended_club": str(option.get("team_id", "")),
                "market_price": self._finite(option.get("market_price")),
                "expected_comparable_price": self._finite(
                    option.get("expected_market_price_for_contribution")
                ),
                "estimated_value_edge": self._finite(option.get("estimated_value_edge")),
                "gf_gain": float(option["gf_gain"]),
                "ga_reduction": float(option["ga_reduction"]),
                "impact_score": float(option["impact_score"]),
                "planner_value_score": float(option.get("planner_value_score", np.nan)),
                "mapped_features_used": int(option.get("mapped_features_used", 0)),
            })

        return all_sim, [{
            "position": position,
            "position_label": ROLE_LABELS.get(position, position),
            "slots": int(slots),
            "outgoing_row_id": int(best["outgoing_row_id"]),
            "outgoing_player": str(best["outgoing_player"]),
            "recommended_player_row_id": int(best["_row_id"]),
            "recommended_player": str(best.get("player_name", "")),
            "recommended_club": str(best.get("team_id", "")),
            "market_price": self._finite(best.get("market_price")),
            "expected_comparable_price": self._finite(best.get("expected_market_price_for_contribution")),
            "estimated_value_edge": self._finite(best.get("estimated_value_edge")),
            "gf_gain": float(best["gf_gain"]),
            "ga_reduction": float(best["ga_reduction"]),
            "impact_score": float(best["impact_score"]),
            "planner_value_score": float(best.get("planner_value_score", np.nan)),
            "mapped_features_used": int(best.get("mapped_features_used", 0)),
            "minimum_profile": profile,
            "candidate_shortlist": shortlist,
            "candidate_options": candidate_options,
        }]

    def _minimum_profile(self, scored, simulations, position, best, *, attack_gap, defence_gap):
        best_impact = float(best["impact_score"])
        cohort = simulations[
            (simulations["canonical_position"].eq(position))
            & (simulations["impact_score"] >= max(best_impact * 0.50, 0.01))
        ].copy()
        if len(cohort) < 3:
            cohort = simulations[simulations["canonical_position"].eq(position)].nlargest(
                min(5, len(simulations)), "impact_score"
            )
        ids = cohort["_row_id"].astype(int).tolist()
        players = scored[scored["_row_id"].isin(ids)].copy()
        if players.empty:
            return []

        rows = []
        relevant_metric_cols = []
        if attack_gap > 0:
            relevant_metric_cols.extend(self.player_model.metric_map.get("attack", {}).values())
        if defence_gap > 0:
            relevant_metric_cols.extend(self.player_model.metric_map.get("defence", {}).values())
        for col in dict.fromkeys(relevant_metric_cols):
            if col not in players:
                continue
            values = pd.to_numeric(players[col], errors="coerce").dropna()
            if len(values) < 3 or values.nunique() < 2:
                continue
            rows.append({
                "type": "stat",
                "feature": col,
                "minimum_typical_value": float(values.quantile(0.25)),
                "median_successful_candidate": float(values.median()),
                "sample": int(len(values)),
                "evidence": "driver-linked performance among candidates producing at least half the best available modelled impact",
            })

        attributes = []
        learned = False
        if self.outcome_model is not None and getattr(self.outcome_model, "available", False):
            relevant_outcomes = []
            if attack_gap > 0:
                relevant_outcomes += ["chance_generation", "chance_creation", "finishing"]
            if defence_gap > 0:
                relevant_outcomes += ["goalkeeping" if position == "GK" else "defensive_output"]
            seen = set()
            for outcome in relevant_outcomes:
                importance = self.outcome_model.attribute_importance(outcome, role=position)
                if importance.empty:
                    continue
                positive = importance[importance["coefficient"] > 0].head(8)
                for attribute in positive["attribute"]:
                    if attribute not in seen:
                        attributes.append(attribute)
                        seen.add(attribute)
            learned = bool(attributes)

        if not attributes:
            if attack_gap > 0:
                attributes.extend(ATTRIBUTE_PROXY["attack"])
            if defence_gap > 0:
                attributes.extend(ATTRIBUTE_PROXY["defence"])
            attributes = list(dict.fromkeys(attributes))

        for attribute in attributes[:10]:
            if attribute not in players:
                continue
            values = pd.to_numeric(players[attribute], errors="coerce").dropna()
            if len(values) < 3 or values.nunique() < 2:
                continue
            rows.append({
                "type": "attribute",
                "feature": attribute,
                "minimum_typical_value": float(values.quantile(0.25)),
                "median_successful_candidate": float(values.median()),
                "sample": int(len(values)),
                "evidence": (
                    "historically learned positive attribute-outcome association plus successful-candidate profile"
                    if learned else
                    "descriptive successful-candidate profile; attribute effect is not yet learned from historical seasons"
                ),
            })

        stats = [r for r in rows if r["type"] == "stat"][:8]
        attrs = [r for r in rows if r["type"] == "attribute"][:8]
        return stats + attrs

    def _sale_candidates(self, scored, starter_ids):
        owned = scored[self._owned_mask(scored)].copy()
        if owned.empty:
            return []
        ratio = pd.to_numeric(owned["price_vs_comparables"], errors="coerce")
        replacement = pd.to_numeric(owned["above_role_replacement"], errors="coerce")
        owned["_starter"] = owned["_row_id"].isin(starter_ids)
        owned["_sale_score"] = (
            np.maximum(ratio.fillna(1).to_numpy() - 1, 0) * 2
            + np.maximum(-replacement.fillna(0).to_numpy(), 0)
            + np.where(owned["_starter"], 0.0, 0.35)
        )
        candidates = owned[
            (owned["_sale_score"] > 0.10)
            & np.isfinite(pd.to_numeric(owned["market_price"], errors="coerce"))
        ].sort_values("_sale_score", ascending=False)

        result = []
        for _, row in candidates.head(8).iterrows():
            reasons = []
            premium = self._finite(row.get("price_vs_comparables"))
            above = self._finite(row.get("above_role_replacement"))
            if premium is not None and premium > 1:
                reasons.append(f"price is {premium:.2f}x comparable contribution")
            if above is not None and above < 0:
                reasons.append("below estimated role-replacement level")
            if not bool(row["_starter"]):
                reasons.append("outside the modelled starting XI")
            result.append({
                "player_name": str(row.get("player_name", "")),
                "position": str(row.get("canonical_position", "")),
                "market_price": self._finite(row.get("market_price")),
                "expected_comparable_price": self._finite(row.get("expected_market_price_for_contribution")),
                "price_vs_comparables": premium,
                "above_role_replacement": above,
                "starter": bool(row["_starter"]),
                "sale_score": float(row["_sale_score"]),
                "reason": "; ".join(reasons) if reasons else "review market value versus squad role",
            })
        return result

    def _simulate_combined_plan(self, baseline, scored, recommendations, matches, base_adjustment=None):
        modified = baseline.copy()
        transfers = []
        used_outgoing = set()
        for rec in recommendations:
            incoming_rows = scored[scored["_row_id"].eq(rec["recommended_player_row_id"])]
            if incoming_rows.empty:
                continue
            incoming = incoming_rows.iloc[0]
            outgoing_id = rec["outgoing_row_id"]
            if outgoing_id in used_outgoing and outgoing_id >= 0:
                continue
            outgoing_rows = scored[scored["_row_id"].eq(outgoing_id)]
            outgoing = outgoing_rows.iloc[0] if not outgoing_rows.empty else self._placeholder(
                scored, rec["position"]
            )
            modified, used = self._apply_replacement(modified, outgoing, incoming)
            used_outgoing.add(outgoing_id)
            transfers.append({
                **rec,
                "features_changed": len(used),
                "outgoing_market_price": self._finite(outgoing.get("market_price")),
            })

        prediction = self._forecast_with_adjustment(modified, matches, base_adjustment)

        # Apply the same finishing/GK corrections used in individual replacement
        # scenarios to the combined transfer package.
        finishing_adjustment = 0.0
        goalkeeping_reduction = 0.0
        for transfer in transfers:
            incoming_rows = scored[scored["_row_id"].eq(transfer["recommended_player_row_id"])]
            outgoing_rows = scored[scored["_row_id"].eq(transfer["outgoing_row_id"])]
            if incoming_rows.empty:
                continue
            incoming = incoming_rows.iloc[0]
            outgoing = outgoing_rows.iloc[0] if not outgoing_rows.empty else self._placeholder(
                scored, transfer["position"]
            )
            if transfer["position"] == "GK":
                inc = self._finite(incoming.get("predicted_goalkeeping"))
                out = self._finite(outgoing.get("predicted_goalkeeping")) or 0.0
                if inc is not None:
                    goalkeeping_reduction += (inc - out) * self.lineup_share * int(matches)
            else:
                inc = self._finite(incoming.get("predicted_finishing"))
                out = self._finite(outgoing.get("predicted_finishing")) or 0.0
                if inc is not None:
                    finishing_adjustment += (inc - out) * self.lineup_share * int(matches)
        prediction["goals_for"] = max(prediction["goals_for"] + finishing_adjustment, 0.0)
        prediction["goals_against"] = max(prediction["goals_against"] - goalkeeping_reduction, 0.0)
        prediction["attribute_finishing_adjustment"] = float(finishing_adjustment)
        prediction["attribute_goalkeeping_reduction"] = float(goalkeeping_reduction)

        incoming_cost = sum(x["market_price"] or 0 for x in transfers)
        outgoing_value = sum(x["outgoing_market_price"] or 0 for x in transfers)
        return prediction, transfers, float(incoming_cost - outgoing_value)

    def _select_transfer_package(
        self,
        baseline,
        scored,
        opportunities,
        *,
        league,
        target_position,
        requested_probability,
        context,
        max_recruits,
        base_adjustment=None,
    ):
        """Choose the cheapest recommendation subset that reaches the requested target.

        The search is deliberately small and inspectable: there is at most one
        value-selected recommendation per canonical position, so every subset up
        to the maximum recruit count can be evaluated by the actual combined model.
        """

        if not opportunities:
            forecast, transfers, net_spend = self._simulate_combined_plan(
                baseline, scored, [], context["matches"], base_adjustment=base_adjustment
            )
            evaluation = self.league_model.evaluate(
                league,
                forecast["goals_for"],
                forecast["goals_against"],
                target_position=int(target_position),
                matches=context["matches"],
                n_teams=context["n_teams"],
                next_season=context["season"],
            ).iloc[0]
            probability = self._finite(evaluation.get("estimated_target_probability"))
            return forecast, transfers, net_spend, {
                "packages_considered": 1,
                "target_reached": bool(
                    probability is not None and probability >= requested_probability
                ),
                "selection_reason": "No measurable positive recruitment opportunity was available.",
                "selected_probability": probability,
            }

        candidates = []
        limit = min(int(max_recruits), len(opportunities))
        for size in range(1, limit + 1):
            for position_subset in combinations(opportunities, size):
                option_groups = [
                    opportunity.get("candidate_options") or [opportunity]
                    for opportunity in position_subset
                ]
                for chosen_players in product(*option_groups):
                    forecast, transfers, net_spend = self._simulate_combined_plan(
                        baseline,
                        scored,
                        list(chosen_players),
                        context["matches"],
                        base_adjustment=base_adjustment,
                    )
                    if not transfers:
                        continue
                    evaluation = self.league_model.evaluate(
                        league,
                        forecast["goals_for"],
                        forecast["goals_against"],
                        target_position=int(target_position),
                        matches=context["matches"],
                        n_teams=context["n_teams"],
                        next_season=context["season"],
                    ).iloc[0]
                    probability = self._finite(evaluation.get("estimated_target_probability"))
                    candidates.append({
                        "forecast": forecast,
                        "transfers": transfers,
                        "net_spend": float(net_spend),
                        "probability": probability,
                        "predicted_position": float(evaluation["predicted_position"]),
                        "target_reached": bool(
                            probability is not None and probability >= float(requested_probability)
                        ),
                    })

        if not candidates:
            forecast, transfers, net_spend = self._simulate_combined_plan(
                baseline, scored, [], context["matches"], base_adjustment=base_adjustment
            )
            return forecast, transfers, net_spend, {
                "packages_considered": 0,
                "target_reached": False,
                "selection_reason": "No valid transfer package could be simulated from the current export.",
                "selected_probability": None,
            }

        feasible = [row for row in candidates if row["target_reached"]]
        if feasible:
            best = min(
                feasible,
                key=lambda row: (
                    row["net_spend"],
                    len(row["transfers"]),
                    -(row["probability"] if row["probability"] is not None else -1),
                ),
            )
            reason = (
                "Lowest modelled net-spend package among the tested position recommendations "
                "that reaches the requested target probability."
            )
        else:
            best = min(
                candidates,
                key=lambda row: (
                    -(row["probability"] if row["probability"] is not None else -1),
                    row["predicted_position"],
                    row["net_spend"],
                    len(row["transfers"]),
                ),
            )
            reason = (
                "No tested package reaches the requested probability within the signing limit; "
                "this package gives the strongest modelled progress, with net spend used as a tie-break."
            )

        return best["forecast"], best["transfers"], best["net_spend"], {
            "packages_considered": len(candidates),
            "target_reached": best["target_reached"],
            "selection_reason": reason,
            "selected_probability": best["probability"],
        }


    def plan(
        self,
        league,
        position,
        players,
        *,
        club,
        probability=0.70,
        formation="4-2-3-1",
        max_recruits=3,
        matches=None,
        n_teams=None,
        next_season=None,
    ):
        self._check()
        if formation not in FORMATION_PRESETS:
            raise DataError(f"Unknown formation {formation!r}.")
        if not 1 <= int(max_recruits) <= 8:
            raise DataError("max_recruits must be between 1 and 8.")

        scenario = self.league_model.targets(
            league,
            int(position),
            probability=float(probability),
            matches=matches,
            n_teams=n_teams,
            next_season=next_season,
        )
        context = scenario["context"]
        baseline, baseline_evidence = self._baseline_team(league, club)

        scored = self._score_players(players, context["matches"])
        slots = formation_slots(formation)
        starter_ids, missing = self._select_starters(scored, slots)
        current_baseline, current_adjustment, squad_adjustment_evidence = self._historical_current_adjustment(
            baseline,
            scored,
            starter_ids,
            slots,
            club,
            context["matches"],
        )
        current = self._forecast_with_adjustment(
            current_baseline, context["matches"], current_adjustment
        )
        target = scenario["recommended"]
        attack_gap = max(float(target["goals_for"]) - current["goals_for"], 0.0)
        defence_gap = max(current["goals_against"] - float(target["goals_against"]), 0.0)

        opportunities = []
        simulations_by_position = {}
        if attack_gap > 0 or defence_gap > 0:
            for role, slot_count in slots.items():
                simulations, recommendation = self._position_candidates(
                    current_baseline,
                    current,
                    scored,
                    starter_ids,
                    role,
                    slot_count,
                    context["matches"],
                    attack_gap,
                    defence_gap,
                    base_adjustment=current_adjustment,
                )
                simulations_by_position[role] = simulations
                opportunities.extend(recommendation)

        opportunities.sort(
            key=lambda row: (row["impact_score"], row.get("planner_value_score", -np.inf)),
            reverse=True,
        )
        after, transfer_plan, net_spend, package_selection = self._select_transfer_package(
            current_baseline,
            scored,
            opportunities,
            league=league,
            target_position=int(position),
            requested_probability=float(probability),
            context=context,
            max_recruits=int(max_recruits),
            base_adjustment=current_adjustment,
        )

        current_eval = self.league_model.evaluate(
            league,
            current["goals_for"],
            current["goals_against"],
            target_position=int(position),
            matches=context["matches"],
            n_teams=context["n_teams"],
            next_season=context["season"],
        ).iloc[0]
        after_eval = self.league_model.evaluate(
            league,
            after["goals_for"],
            after["goals_against"],
            target_position=int(position),
            matches=context["matches"],
            n_teams=context["n_teams"],
            next_season=context["season"],
        ).iloc[0]

        lineup = scored[scored["_row_id"].isin(starter_ids)][
            [
                c for c in (
                    "_row_id", "player_name", "team_id", "canonical_position", "role_group",
                    "football_contribution_units", "attack_contribution_units",
                    "defensive_contribution_units", "market_price",
                ) if c in scored
            ]
        ].to_dict("records")

        warnings = []
        if missing:
            detail = ", ".join(f"{ROLE_LABELS.get(k, k)}: {v}" for k, v in missing.items())
            warnings.append(
                "The owned-player export does not fill every formation slot. "
                f"Missing modelled starters: {detail}. Recruitment profiles for those groups use a replacement-level placeholder."
            )
        if squad_baseline.get("status") != "current_squad_adjusted":
            warnings.append(
                "The starting GF/GA forecast could not be adjusted from last season's XI to the current XI: "
                + squad_baseline.get("reason", "historical squad evidence was unavailable")
            )
        if current["attack_out_of_range_features"] or current["defence_out_of_range_features"]:
            warnings.append(
                "The current team profile contains driver values outside part of the historical training range."
            )
        if not (self.outcome_model is not None and getattr(self.outcome_model, "available", False)):
            warnings.append(
                "Historical attribute-outcome models are not available, so attribute cut-offs are descriptive profiles "
                "of successful candidates rather than learned attribute effects."
            )

        return {
            "league": str(league),
            "club": str(club),
            "formation": formation,
            "target": scenario,
            "current_forecast": {
                **current,
                "predicted_position": float(current_eval["predicted_position"]),
                "estimated_target_probability": self._finite(current_eval.get("estimated_target_probability")),
                "evidence": (
                    "Goal-driver prediction if the latest observed team-process profile repeats next season."
                ),
                "baseline_evidence": baseline_evidence,
                "squad_adjustment": squad_baseline,
                "squad_adjustment": squad_adjustment_evidence,
            },
            "gap": {
                "additional_goals_for": float(attack_gap),
                "fewer_goals_against": float(defence_gap),
            },
            "starting_xi": lineup,
            "missing_formation_slots": missing,
            "sale_candidates": self._sale_candidates(scored, starter_ids),
            "position_opportunities": opportunities,
            "recommended_transfers": transfer_plan,
            "after_transfer_forecast": {
                **after,
                "predicted_position": float(after_eval["predicted_position"]),
                "estimated_target_probability": self._finite(after_eval.get("estimated_target_probability")),
                "net_spend": net_spend,
                "target_reached": package_selection["target_reached"],
                "packages_considered": package_selection["packages_considered"],
                "selection_reason": package_selection["selection_reason"],
                "interpretation": (
                    "Combined scenario after replacing mapped player per-90 inputs and re-running the goal-driver model."
                ),
            },
            "model_evidence": {
                "attribute_outcomes": bool(
                    self.outcome_model is not None and getattr(self.outcome_model, "available", False)
                ),
                "lineup_share": self.lineup_share,
                "warnings": warnings,
                "current_squad_adjustment": squad_adjustment_evidence,
            },
        }
