from __future__ import annotations

import unittest

import pandas as pd

from fm_model.app_support import build_model, load_example_frames
from fm_model.collection import collection_plan
from fm_model.data import ATTRIBUTES
from fm_model.roles import canonical_role


class SquadPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = load_example_frames()
        cls.model = build_model(
            league_data=cls.frames["league"],
            team_data=cls.frames["team"],
            historical_player_data=cls.frames["player_history"],
            player_data=cls.frames["players"],
        )

    def test_collection_plan_requests_broad_attributes_clubs_and_positions(self):
        plan = collection_plan(historical_seasons=3)
        self.assertEqual(set(plan["historical_players"]["attribute_fields"]), set(ATTRIBUTES))
        self.assertIn("ST", plan["historical_players"]["positions"])
        self.assertIn("GK", plan["historical_players"]["positions"])
        self.assertIn("every club", plan["historical_players"]["clubs"].lower())
        self.assertIn("Current Ability (CA)", plan["exclude_from_performance_models"])

    def test_native_fm_position_labels_map_to_planner_groups(self):
        self.assertEqual(canonical_role("D (C)"), "CB")
        self.assertEqual(canonical_role("D (R)"), "FB_WB")
        self.assertEqual(canonical_role("WB (L)"), "FB_WB")
        self.assertEqual(canonical_role("M (C)"), "CM_DM")
        self.assertEqual(canonical_role("AM (R)"), "AM_W")
        self.assertEqual(canonical_role("ST (C)"), "ST")
        self.assertEqual(canonical_role("GK"), "GK")

    def test_attribute_outcome_model_uses_chronological_validation(self):
        outcomes = self.model.player_outcome_model
        self.assertTrue(outcomes.available)
        self.assertIn("chance_generation", outcomes.models)
        self.assertIn("finishing", outcomes.models)
        self.assertIn("ST", outcomes.models["chance_generation"])
        self.assertIn("CB", outcomes.models["defensive_output"])
        self.assertIn("GK", outcomes.models["goalkeeping"])
        importance = outcomes.attribute_importance("chance_generation", role="ST")
        self.assertFalse(importance.empty)
        self.assertIn("validation_mae_increase", importance.columns)

    def test_complete_squad_plan_connects_target_forecast_sales_and_recruitment(self):
        plan = self.model.squad_plan(
            "Example League",
            4,
            self.frames["players"],
            club="Club F",
            probability=0.60,
            formation="4-2-3-1",
            max_recruits=3,
        )
        self.assertIn("target", plan)
        self.assertIn("current_forecast", plan)
        self.assertIn("gap", plan)
        self.assertIn("sale_candidates", plan)
        self.assertIn("position_opportunities", plan)
        self.assertIn("recommended_transfers", plan)
        self.assertIn("after_transfer_forecast", plan)
        self.assertEqual(plan["missing_formation_slots"], {})
        self.assertIn("selection_reason", plan["after_transfer_forecast"])
        self.assertGreaterEqual(plan["after_transfer_forecast"]["packages_considered"], 1)
        self.assertTrue(plan["model_evidence"]["attribute_outcomes"])
        self.assertTrue(plan["current_forecast"]["squad_adjustment"]["applied"])
        self.assertGreater(plan["current_forecast"]["squad_adjustment"]["replacements"], 0)

        target = plan["target"]["recommended"]
        self.assertGreaterEqual(target["goals_for"], 0)
        self.assertGreaterEqual(target["goals_against"], 0)
        self.assertGreater(plan["current_forecast"]["goals_for"], 0)
        self.assertGreater(plan["current_forecast"]["goals_against"], 0)
        self.assertTrue(plan["current_forecast"]["squad_adjustment"]["applied"])
        self.assertGreater(plan["current_forecast"]["squad_adjustment"]["replacements"], 0)

        opportunities = plan["position_opportunities"]
        if opportunities:
            first = opportunities[0]
            self.assertIn("minimum_profile", first)
            self.assertIn("candidate_shortlist", first)
            self.assertGreater(len(first["candidate_shortlist"]), 0)
            self.assertGreater(len(first["candidate_options"]), 0)
            profile = pd.DataFrame(first["minimum_profile"])
            if not profile.empty:
                self.assertTrue(set(profile["type"]).issubset({"stat", "attribute"}))


if __name__ == "__main__":
    unittest.main()
