from __future__ import annotations

import io
import unittest

from fm_model.app_support import (
    build_model,
    infer_season_from_name,
    load_example_frames,
    read_player_for_app,
)


class NamedBytesIO(io.BytesIO):
    def __init__(self, value: bytes, name: str):
        super().__init__(value)
        self.name = name


class AppSupportTests(unittest.TestCase):
    def test_example_data_builds_every_model_layer(self):
        frames = load_example_frames()
        model = build_model(
            league_data=frames["league"],
            team_data=frames["team"],
            historical_player_data=frames["player_history"],
            player_data=frames["players"],
        )
        readiness = model.readiness()
        self.assertTrue(readiness["league_targets"])
        self.assertTrue(readiness["goal_drivers"])
        self.assertTrue(readiness["player_values"])
        self.assertTrue(readiness["attribute_outcomes"])
        self.assertTrue(readiness["squad_plan"])

        decisions = model.player_decisions(frames["players"])
        self.assertEqual(len(decisions), 72)
        self.assertIn("expected_market_price_for_contribution", decisions)
        self.assertIn("decision", decisions)

    def test_player_import_resolves_display_ranges_and_infers_ownership(self):
        raw = (
            "Name,Position,Club,Guide Value,Minutes Played,Goals,Shots\n"
            "One,ST,Club A,£1m-£3m,900,10,50\n"
            "Two,ST,Club B,£2m-£4m,1000,12,60\n"
        ).encode("utf-8")
        frame = read_player_for_app(
            NamedBytesIO(raw, "players.csv"),
            league="Example League",
            season=2026,
            ownership_mode="club",
            owned_club="Club A",
            range_policy="midpoint",
        )
        self.assertEqual(frame.loc[0, "market_value"], 2_000_000)
        self.assertEqual(frame.loc[1, "market_value"], 3_000_000)
        self.assertTrue(frame.loc[0, "owned"])
        self.assertFalse(frame.loc[1, "owned"])
        self.assertEqual(frame.loc[0, "role_group"], "ST")


    def test_complete_squad_plan_connects_target_forecast_and_transfers(self):
        frames = load_example_frames()
        model = build_model(
            league_data=frames["league"],
            team_data=frames["team"],
            historical_player_data=frames["player_history"],
            player_data=frames["players"],
        )
        plan = model.squad_plan(
            "Example League",
            4,
            frames["players"],
            club="Club F",
            probability=0.60,
            formation="4-2-3-1",
            max_recruits=3,
        )
        self.assertIn("current_forecast", plan)
        self.assertIn("gap", plan)
        self.assertGreater(plan["gap"]["additional_goals_for"] + plan["gap"]["fewer_goals_against"], 0)
        self.assertGreater(len(plan["position_opportunities"]), 0)
        self.assertGreater(len(plan["recommended_transfers"]), 0)
        self.assertIn("after_transfer_forecast", plan)
        self.assertTrue(
            any(row.get("minimum_profile") for row in plan["position_opportunities"])
        )

    def test_season_can_be_inferred_from_export_filename(self):
        self.assertEqual(infer_season_from_name("players_2025_26.csv"), 2025)
        self.assertEqual(infer_season_from_name("team-2024-25.csv"), 2024)
        self.assertIsNone(infer_season_from_name("players.csv"))

    def test_season_is_inferred_from_export_filename(self):
        self.assertEqual(infer_season_from_name("players_2025_26.csv"), 2025)
        self.assertEqual(infer_season_from_name("league-2024-25.csv"), 2024)
        self.assertIsNone(infer_season_from_name("players.csv"))


if __name__ == "__main__":
    unittest.main()
