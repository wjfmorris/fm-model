from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fm_model import DataError, GoalDriverModel, MoneyballModel, PlayerValuationModel
from fm_model.data import parse_number, read_player_export, read_table


def make_data(seed=7, seasons=5, teams=12):
    rng = np.random.default_rng(seed)
    league_rows, team_rows = [], []
    for league in ("Alpha", "Beta"):
        for season in range(2021, 2021 + seasons):
            strength = np.arange(teams)[::-1].astype(float) + rng.normal(0, 0.4, teams)
            gf = np.maximum(10, 34 + strength * 2 + rng.normal(0, 2, teams)).round().astype(int)
            ga = np.maximum(10, 48 - strength * 1.6 + rng.normal(0, 2, teams)).round().astype(int)
            ga[-1] += int(gf.sum() - ga.sum())
            for i in range(teams):
                row = {"league": league, "season": season, "position": i + 1,
                       "team_id": f"{league}-{season}-{i}", "matches": 22,
                       "goals_for": int(gf[i]), "goals_against": int(ga[i]),
                       "points": int(54 - i * 3)}
                league_rows.append(row)
                team_rows.append({**row,
                    "xg": max(1, 32 + strength[i] * 2 + rng.normal(0, 1)),
                    "shots": max(1, 230 + strength[i] * 10 + rng.normal(0, 8)),
                    "chances_created": max(1, 90 + strength[i] * 4 + rng.normal(0, 4)),
                    "set_piece_goals": max(0, 3 + strength[i] * 0.25 + rng.normal(0, 1)),
                    "xg_against": max(1, 45 - strength[i] * 1.5 + rng.normal(0, 2)),
                    "shots_against": max(1, 260 - strength[i] * 8 + rng.normal(0, 8)),
                    "pressures": max(1, 230 + strength[i] * 7 + rng.normal(0, 8)),
                    "interceptions": max(1, 180 + strength[i] * 5 + rng.normal(0, 7)),
                    "saves": max(1, 90 - strength[i] * 2 + rng.normal(0, 5)),
                })
    return pd.DataFrame(league_rows), pd.DataFrame(team_rows)


class ModelTests(unittest.TestCase):
    def test_input_parser_is_explicit_about_ranges(self):
        self.assertEqual(parse_number("£1.5m"), 1_500_000)
        self.assertTrue(np.isnan(parse_number("'-£1")))
        with self.assertRaises(DataError):
            parse_number("10-15")
        self.assertEqual(parse_number("10-15", ranges="midpoint"), 12.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.csv"
            path.write_text("League;Season;Pos;MP;GF;GA\nAlpha;2025/26;1;22;50;20\n", encoding="utf-8")
            frame = read_table(path, numeric_columns=["season", "position", "matches", "goals_for", "goals_against"])
            self.assertEqual(frame.iloc[0].position, 1)

    def test_fmst26_export_headers_and_context_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "league_statistics.csv"
            path.write_text(
                "Name,Position,Club,Age,Guide Value,Minutes Played,Goals,Assists,xA,Non-Penalty xG,"
                "Shots,Shots on Target,Key Passes,Progressive Passes,Tackles Completed,Interceptions,"
                "Pressures,Headers Won,Saves per 90,xG Prevented\n"
                "Player One,ST,Club A,24,£1.5M,1000,8,3,2.4,5.5,40,18,12,20,3,4,5,6,,\n"
                "Player Two,GK,Club A,29,'-£1,,0,0,0,0,0,0,0,0,,0,0,,2.4,2.2\n",
                encoding="utf-8",
            )
            frame = read_player_export(path, league="Example League", season=2026, owned=False)
        self.assertEqual(frame.loc[0, "player_name"], "Player One")
        self.assertEqual(frame.loc[0, "role_group"], "ST")
        self.assertEqual(frame.loc[0, "team_id"], "Club A")
        self.assertEqual(frame.loc[0, "tackles_won"], "3")
        self.assertEqual(frame.loc[0, "league"], "Example League")
        self.assertEqual(frame.loc[0, "season"], 2026)
        self.assertFalse(frame.loc[0, "owned"])

        scored = PlayerValuationModel().fit(frame).score(frame)
        self.assertTrue(np.isnan(scored.loc[0, "market_price"]))  # Guide Value is not a verified price.
        self.assertEqual(scored.loc[1, "data_quality"], "no_minutes")
        self.assertEqual(scored.loc[1, "team_impact_evidence"], "no_minutes_for_impact")
        decisions = PlayerValuationModel().fit(frame).decisions(frame)
        self.assertEqual(decisions.loc[1, "decision"], "INSUFFICIENT_PRICE_DATA")

    def test_league_model_uses_all_seasons_and_returns_frontier(self):
        league, _ = make_data()
        model = MoneyballModel().fit_league(league)
        result = model.target_scenario("Alpha", 4, probability=0.6)
        self.assertIn("frontier", result)
        self.assertGreaterEqual(result["recommended"]["goals_for"], 0)
        self.assertGreaterEqual(result["recommended"]["goals_against"], 0)
        self.assertEqual(result["validation"]["seasons"], 5)
        first = model.target_scenario("Alpha", 1, probability=0.6)["recommended"]
        fourth = result["recommended"]
        self.assertTrue(first["goals_for"] >= fourth["goals_for"] or first["goals_against"] <= fourth["goals_against"])

    def test_driver_model_reports_holdout_and_best_effort_route(self):
        _, team = make_data()
        model = GoalDriverModel().fit(team)
        self.assertIn("mae", model.report["attack"])
        self.assertIn("importance", model.report["attack"])
        self.assertEqual(model.report["attack"]["test_season"], 2025)
        baseline = team[team.league == "Alpha"].iloc[[0]]
        route = model.route(baseline, target_goals_for=150, target_goals_against=1, matches=22)
        self.assertIn("changes", route["attack"])
        self.assertIn("changes", route["defence"])
        self.assertTrue(any(abs(x["change"]) > 0 for x in route["attack"]["changes"]))

    def test_player_values_have_replacement_comparison_and_round_trip(self):
        league, team = make_data()
        driver = GoalDriverModel().fit(team)
        rng = np.random.default_rng(9)
        rows = []
        for league_name in ("Alpha", "Beta"):
            for i in range(24):
                role = ("ST", "MC", "DC")[i % 3]
                rows.append({"player_name": f"{league_name}-{i}", "league": league_name,
                    "role_group": role, "minutes": int(900 + rng.integers(0, 2_000)),
                    "xg_p90": max(0, rng.normal(0.2 if role == "ST" else 0.08, 0.04)),
                    "shots_p90": max(0, rng.normal(2.5 if role == "ST" else 1.0, 0.3)),
                    "chances_created_p90": max(0, rng.normal(1.7, 0.2)),
                    "set_piece_goals_p90": max(0, rng.normal(0.02, 0.01)),
                    "interceptions_p90": max(0, rng.normal(1.3, 0.2)),
                    "pressures_p90": max(0, rng.normal(8, 0.5)),
                    "market_value": float(max(1, rng.normal(10, 3))), "owned": i < 3})
        players = pd.DataFrame(rows)
        model = PlayerValuationModel(driver).fit(players)
        decisions = model.decisions(players)
        self.assertTrue(set(["BUY", "SELL", "KEEP_REVIEW", "WATCH", "INSUFFICIENT_PRICE_DATA"]).intersection(decisions.decision))
        self.assertIn("above_role_replacement", decisions)
        self.assertGreater(len(model.attribute_values()), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            MoneyballModel().fit_league(league).fit_drivers(team).fit_players(players).save(path)
            loaded = MoneyballModel.load(path)
            self.assertTrue(loaded.readiness()["player_values"])

    def test_player_import_accepts_displayed_money_and_raw_count_stats(self):
        rows = []
        for i in range(8):
            rows.append({"player_name": str(i), "league": "A", "role_group": "ST", "minutes": "1,000",
                         "goals": str(8 + i), "shots": str(40 + i), "finishing": str(10 + i % 5),
                         "off_the_ball": "12", "composure": "12", "technique": "12", "passing": "10",
                         "vision": "10", "crossing": "10", "pace": "14", "marking": "4", "tackling": "4",
                         "positioning": "4", "anticipation": "4", "decisions": "4", "concentration": "4",
                         "strength": "8", "jumping_reach": "8", "reflexes": "4", "handling": "4",
                         "market_value": "£1.5m"})
        model = PlayerValuationModel().fit(pd.DataFrame(rows))
        scored = model.score(pd.DataFrame(rows))
        self.assertAlmostEqual(scored.market_price.iloc[0], 1_500_000)
        self.assertTrue(np.isfinite(scored.goals_p90.iloc[0]))


if __name__ == "__main__":
    unittest.main()
