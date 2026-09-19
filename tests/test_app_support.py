from __future__ import annotations

import io
import unittest

from fm_model.app_support import build_model, load_example_frames, read_player_for_app


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
            player_data=frames["players"],
        )
        readiness = model.readiness()
        self.assertTrue(readiness["league_targets"])
        self.assertTrue(readiness["goal_drivers"])
        self.assertTrue(readiness["player_values"])

        decisions = model.player_decisions(frames["players"])
        self.assertEqual(len(decisions), 36)
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


if __name__ == "__main__":
    unittest.main()
