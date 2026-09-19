"""End-to-end orchestration used by the future Streamlit front end."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .drivers import GoalDriverModel
from .errors import NotFittedError
from .league import LeagueModel
from .players import PlayerValuationModel


class MoneyballModel:
    """The complete target -> drivers -> player decisions pipeline.

    Every stage can be fitted and inspected independently. This lets Streamlit
    show which file populated a result and which stages are unavailable because
    a required data layer was not uploaded.
    """

    def __init__(self, *, seed=26):
        self.seed = seed
        self.league_model = None
        self.driver_model = None
        self.player_model = None

    def fit_league(self, league_table: pd.DataFrame):
        self.league_model = LeagueModel(seed=self.seed).fit(league_table)
        return self

    def fit_drivers(self, team_data: pd.DataFrame, *, attacking_features=None, defensive_features=None,
                    include_all_available=False):
        self.driver_model = GoalDriverModel(seed=self.seed).fit(
            team_data, attacking_features=attacking_features, defensive_features=defensive_features,
            include_all_available=include_all_available)
        return self

    def fit_players(self, player_data: pd.DataFrame, *, player_metric_map=None, role_column="role_group"):
        self.player_model = PlayerValuationModel(self.driver_model, seed=self.seed).fit(
            player_data, player_metric_map=player_metric_map, role_column=role_column)
        return self

    def readiness(self):
        return {
            "league_targets": self.league_model is not None,
            "goal_drivers": self.driver_model is not None,
            "player_values": self.player_model is not None,
            "market_decisions": self.player_model is not None,
            "next_step": ("Upload completed league tables" if self.league_model is None else
                           "Upload team metrics" if self.driver_model is None else
                           "Upload player export" if self.player_model is None else "Run a target scenario"),
        }

    def target_scenario(self, league, position, *, probability=0.7, matches=None, n_teams=None, next_season=None):
        if self.league_model is None:
            raise NotFittedError("Fit league tables before requesting targets.")
        return self.league_model.targets(league, position, probability=probability, matches=matches,
                                         n_teams=n_teams, next_season=next_season)

    def explain_scenario(self, league, position, *, baseline_team=None, probability=0.7,
                         matches=None, n_teams=None, next_season=None, players=None, owned_column="owned"):
        scenario = self.target_scenario(league, position, probability=probability, matches=matches,
                                        n_teams=n_teams, next_season=next_season)
        output = {"scenario": scenario, "drivers": None, "player_decisions": None}
        if self.driver_model is not None:
            output["drivers"] = {
                "attack": self.driver_model.driver_report("attack"),
                "defence": self.driver_model.driver_report("defence"),
            }
            if baseline_team is not None:
                rec = scenario["recommended"]
                output["drivers"]["route"] = self.driver_model.route(
                    baseline_team, target_goals_for=rec["goals_for"], target_goals_against=rec["goals_against"],
                    matches=scenario["context"]["matches"])
        if self.player_model is not None and players is not None:
            output["player_decisions"] = self.player_model.decisions(players, owned_column=owned_column,
                                                                      matches=matches)
        return output

    def player_decisions(self, players: pd.DataFrame, *, owned_column="owned", expected_minutes=None,
                         matches=None):
        if self.player_model is None:
            raise NotFittedError("Fit player observations before requesting player decisions.")
        return self.player_model.decisions(players, owned_column=owned_column,
                                           expected_minutes=expected_minutes, matches=matches)

    def to_dict(self):
        return {"seed": self.seed,
                "league_model": None if self.league_model is None else self.league_model.to_dict(),
                "driver_model": None if self.driver_model is None else self.driver_model.to_dict(),
                "player_model": None if self.player_model is None else self.player_model.to_dict()}

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), default=float, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, state):
        obj = cls(seed=state.get("seed", 26))
        if state.get("league_model") is not None:
            obj.league_model = LeagueModel.from_dict(state["league_model"])
        if state.get("driver_model") is not None:
            obj.driver_model = GoalDriverModel.from_dict(state["driver_model"])
        if state.get("player_model") is not None:
            obj.player_model = PlayerValuationModel.from_dict(state["player_model"])
        return obj

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
