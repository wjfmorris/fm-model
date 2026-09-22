"""Behavioural regression tests for the staged FMST recruitment workflow."""
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fm_model.app_support import read_player_for_app
from fm_model.errors import DataError
from fm_model.data import prepare_player_export, read_table
from fm_model.recruitment import (
    apply_intended_positions, assess_squad, clean_statistics, compare_targets, cost_comparison,
    default_assignments, empty_costs, finish_target_percentile, identify, merge_costs,
    prepare_inputs, replacement_cost,
)
from fm_model.metric_learning import (
    available_metrics, learn_metric_importance, metric_inventory, role_outcomes,
    select_learned_metrics,
)


def export():
    rows = []
    for role, position in [('GK', 'GK'), ('CB', 'D (C)'), ('FB_WB', 'D (R)'),
                           ('CM_DM', 'DM'), ('AM_W', 'AM (L)'), ('ST', 'ST')]:
        for i in range(10):
            rows.append({
                'index': i, 'Name': f'{role} Player {i}', 'Position': position,
                'Club': 'My Club' if i < 3 else f'Club {i}',
                'Appearances': 22, 'Minutes Played': 1500 + 30*i,
                'Goals': i, 'Goals per 90': .02 + .03*i, 'xG per 90': .08 + .035*i,
                'xA per 90': .01 + .02*i, 'NP-xG per 90': .1 + .04*i,
                'xG Overperformance': -.05 + .01*i,
                'Shots per 90': 1+.3*i, 'Shots on Target per 90': .2+.1*i,
                'CCC per 90': .1+.03*i, 'Shots Outside Box per 90': .3+.02*i,
                'Pass Completion %': f'{60+i}%', 'Passes Completed per 90': 18+2*i,
                'Passes Attempted per 90': 25+2.5*i, 'Key Passes per 90': .4+.1*i,
                'Progressive Passes per 90': 2+.3*i, 'Crosses Attempted': 12+i,
                'Cross Completion %': f'{20+i}%', 'Open Play Crosses Att': 10+i,
                'Open Play Cross %': f'{18+i}%', 'Open Play Key Passes p90': .5+.1*i,
                'Tackles per 90': .8+.12*i, 'Tackle Success %': f'{55+i}%',
                'Interceptions per 90': 1+.1*i, 'Clearances per 90': .5+.08*i,
                'Blocks per 90': .2+.04*i, 'Key Tackles per 90': .1+.01*i,
                'Pressures Attempted per 90': 5+.4*i, 'Pressure Success %': f'{45+i}%',
                'Possession Won per 90': 2+.2*i, 'Possession Lost per 90': 5-.2*i,
                'Shots Blocked per 90': .1+.03*i, 'Distance per 90 (km)': 9.5+.1*i,
                'Sprints per 90': 10+.3*i, 'Dribbles per 90': .2+.1*i,
                'Header Win %': f'{30+3*i}%',
                'Save Percentage': f'{50+3*i}%' if role=='GK' else '',
                'Saves per 90': 2+.1*i if role=='GK' else '',
                'Clean Sheets per 90': .1+.02*i if role=='GK' else '',
                'Goals Conceded per 90': 2.0-.08*i if role=='GK' else '',
                'xG Prevented': i-2 if role=='GK' else '',
                'Goal Contributions per 90': .2, 'Non-Penalty Contributions per 90': .2,
                'Defensive Actions per 90': 50, 'Attacking Actions per 90': 99,
                'Creative Actions per 90': 30, 'Goalkeeping Actions per 90': 40 if role=='GK' else '',
                'Goals Outside Box': i+1 if i==0 else 0, 'Guide Value': '£300M',
            })
    return io.BytesIO(pd.DataFrame(rows).to_csv(index=False).encode())


def frames():
    raw = read_player_for_app(export(), league='League A', season=2025)
    return prepare_inputs(raw, raw[raw.team_id.eq('My Club')])


class RecruitmentTests(unittest.TestCase):
    def test_unparsed_numeric_fields_with_an_older_importer_registry(self):
        # Reproduces the Cloud traceback: new recruitment code with an older
        # numeric registry leaves appearances as text unless this layer parses it.
        raw = prepare_player_export(read_table(export()), league='League A', season=2025)
        raw.loc[0, 'appearances'] = '51'
        raw.loc[1, 'appearances'] = ''
        with patch('fm_model.recruitment.FMST_NUMERIC_COLUMNS', frozenset()):
            clean, issues = clean_statistics(raw, league_matches='34')
            squad = raw[raw.team_id.eq('My Club')]
            pool, squad, _ = prepare_inputs(raw, squad)
            ranking = learn_metric_importance(pool)
            selected = select_learned_metrics(pool, ranking, max_metrics=3)
            _, profiles, _ = assess_squad(
                pool, squad, default_assignments(squad), metrics=selected, metric_evidence=ranking
            )
        self.assertEqual(clean.appearances.iloc[0], 51.)
        self.assertTrue(np.isnan(clean.appearances.iloc[1]))
        self.assertTrue(pd.api.types.is_float_dtype(clean.appearances))
        self.assertTrue(issues.field.eq('competition_scope').any())
        self.assertFalse(profiles.empty)
        self.assertEqual(raw.appearances.iloc[0], '51')

    def test_invalid_appearance_text_is_an_actionable_input_error(self):
        raw = prepare_player_export(read_table(export()), league='League A', season=2025)
        raw.loc[0, 'appearances'] = 'not a count'
        with patch('fm_model.recruitment.FMST_NUMERIC_COLUMNS', frozenset()):
            with self.assertRaisesRegex(DataError, 'Invalid numeric values in appearances'):
                clean_statistics(raw)
        for value in ('', 0, -1, np.inf):
            with self.assertRaisesRegex(DataError, 'League match count'):
                clean_statistics(raw, league_matches=value)

    def test_new_schema_and_squad_overlap(self):
        pool, squad, issues = frames()
        self.assertEqual(len(pool), 60)
        self.assertEqual(len(squad), 18)
        self.assertEqual(pool.owned.sum(), 18)
        self.assertIn('distance_km_p90', pool)
        self.assertNotIn('market_value', pool)
        self.assertEqual(pool.fmst_guide_value.iloc[0], 300_000_000)
        self.assertEqual(pool.distance_km_p90.iloc[0], 9.5)
        self.assertTrue(pool.attacking_actions_p90.notna().all())
        self.assertTrue(pool.goals_outside_box.notna().any())
        self.assertTrue(issues.field.eq('goals_outside_box').any())
        self.assertAlmostEqual(pool.xg_prevented_p90.iloc[0], -2*90/1500)

    def test_quality_checks_do_not_change_source(self):
        raw = read_player_for_app(export(), league='League A', season=2025)
        raw.loc[0, 'appearances'] = 51
        raw.loc[0, 'pass_completion_pct'] = 140
        result, issues = clean_statistics(raw)
        self.assertEqual(raw.pass_completion_pct.iloc[0], 140)
        self.assertTrue(np.isnan(result.pass_completion_pct.iloc[0]))
        self.assertTrue(issues.field.eq('competition_scope').any())
        self.assertTrue(np.isnan(result.save_pct.iloc[10]))

    def test_duplicate_identity_and_mixed_seasons_blocked(self):
        pool, squad, _ = frames()
        conflicting = pool.iloc[[0]].copy()
        conflicting['minutes'] = 10
        with self.assertRaises(DataError):
            identify(pd.concat([pool, conflicting]))
        pool.loc[0, 'season'] = 2024
        with self.assertRaises(DataError):
            identify(pool)
        with self.assertRaises(DataError):
            prepare_inputs(squad, squad.assign(season=2024))

    def test_intended_position_is_editable_and_drives_model_group(self):
        pool, squad, _ = frames()
        assignments = default_assignments(squad)
        self.assertIn("intended_position", assignments.columns)
        self.assertIn("listed_positions", assignments.columns)

        striker = assignments[assignments.role_group.eq("ST")].index[0]
        assignments.loc[striker, "intended_position"] = "AMR"
        changed = apply_intended_positions(assignments)
        self.assertEqual(changed.loc[striker, "intended_position"], "AMR")
        self.assertEqual(changed.loc[striker, "role_group"], "AM_W")

        changed.loc[striker, "intended_position"] = "ST"
        restored = apply_intended_positions(changed)
        self.assertEqual(restored.loc[striker, "role_group"], "ST")

    def test_multifunctional_player_can_be_assigned_to_one_exact_position(self):
        pool, squad, _ = frames()
        row = squad.iloc[[0]].copy()
        row["position"] = "M (L), AM (RL), ST"
        row["role_group"] = "ST"
        assignment = default_assignments(row)
        self.assertEqual(assignment.intended_position.iloc[0], "ST")
        self.assertIn("AML", assignment.listed_positions.iloc[0])
        self.assertIn("AMR", assignment.listed_positions.iloc[0])
        assignment.loc[assignment.index[0], "intended_position"] = "AML"
        assignment = apply_intended_positions(assignment)
        self.assertEqual(assignment.role_group.iloc[0], "AM_W")

    def test_finish_target_sets_recruitment_percentile(self):
        self.assertAlmostEqual(finish_target_percentile(1, 18), 97.2222222222)
        self.assertAlmostEqual(finish_target_percentile(6, 18), 69.4444444444)
        self.assertAlmostEqual(finish_target_percentile(18, 18), 2.7777777778)
        with self.assertRaises(DataError):
            finish_target_percentile(19, 18)

        pool, squad, _ = frames()
        ranking = learn_metric_importance(pool)
        selected = select_learned_metrics(pool, ranking, max_metrics=3)
        _, ambitious, _ = assess_squad(
            pool, squad, default_assignments(squad),
            target_position=1, n_teams=10,
            metrics=selected, metric_evidence=ranking,
        )
        _, modest, _ = assess_squad(
            pool, squad, default_assignments(squad),
            target_position=8, n_teams=10,
            metrics=selected, metric_evidence=ranking,
        )
        self.assertTrue(ambitious.target_percentile.eq(95.0).all())
        self.assertTrue(modest.target_percentile.eq(25.0).all())
        self.assertTrue(ambitious.season_target_position.eq(1).all())
        self.assertTrue(modest.season_target_position.eq(8).all())

    def test_position_changes_are_ordered_least_to_most_important(self):
        pool, squad, _ = frames()
        ranking = learn_metric_importance(pool)
        selected = select_learned_metrics(pool, ranking, max_metrics=3)
        priorities, _, _ = assess_squad(
            pool, squad, default_assignments(squad),
            target_position=3, n_teams=10,
            metrics=selected, metric_evidence=ranking,
        )
        self.assertEqual(priorities.change_importance_order.tolist(), list(range(1, len(priorities) + 1)))
        sort_view = priorities[["unfilled_starter_slots", "weighted_percentile_shortfall"]].copy()
        expected = sort_view.sort_values(
            ["unfilled_starter_slots", "weighted_percentile_shortfall"],
            ascending=[True, True], na_position="first",
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(sort_view.reset_index(drop=True), expected)

    def test_learned_profiles_and_fixed_benchmark_candidate_comparison(self):
        pool, squad, _ = frames()
        assignments = default_assignments(squad)
        ranking = learn_metric_importance(pool)
        selected = select_learned_metrics(pool, ranking, max_metrics=3)
        priorities, profile, reviews = assess_squad(
            pool, squad, assignments, metrics=selected, metric_evidence=ranking
        )
        self.assertEqual(len(reviews), len(squad))
        self.assertEqual(len(priorities), 6)
        self.assertEqual(assignments.starter.sum(), 11)
        st_profile = profile[profile.role_group.eq('ST')]
        self.assertFalse(st_profile.empty)
        self.assertTrue(st_profile.importance_score.notna().all())
        self.assertAlmostEqual(float(st_profile.importance_weight.sum()), 1.0)
        self.assertTrue(st_profile.learned_outcome.eq('scoring').all())
        self.assertTrue(st_profile.tested_outcomes.eq('scoring').all())

        target = pool[~pool.owned & pool.role_group.eq('ST')].copy()
        first, _ = compare_targets(target, squad, profile, 'ST', league='League A')
        changed_metric = st_profile.metric.iloc[0]
        target.loc[target.index[0], 'league'] = 'League B'
        target.loc[target.index[0], changed_metric] = np.nan
        second, detail = compare_targets(target, squad, profile, 'ST', league='League A')
        self.assertTrue(second.evidence.str.contains('cross-league').any())
        self.assertEqual(second.metrics_required.unique().tolist(), [len(st_profile)])
        player = second[second.player_key.eq(target.player_key.iloc[0])].iloc[0]
        self.assertEqual(player.metrics_observed, len(st_profile) - 1)
        pd.testing.assert_frame_equal(
            first[first.source.eq('My squad')].reset_index(drop=True),
            second[second.source.eq('My squad')].reset_index(drop=True),
        )
        self.assertEqual(len(detail), len(second) * len(st_profile))

    def test_unsupported_metrics_and_low_minutes_not_zero_performance(self):
        pool, squad, _ = frames()
        ranking = learn_metric_importance(pool)
        selected = select_learned_metrics(pool, ranking, max_metrics=3)
        squad.loc[squad.role_group.eq('ST'), 'minutes'] = 30
        _, profile, review = assess_squad(
            pool, squad, default_assignments(squad), metrics=selected, metric_evidence=ranking
        )
        self.assertTrue(review[review.role_group.eq('ST')].review.eq('Insufficient minutes').all())
        self.assertTrue(profile[profile.role_group.eq('ST')].current_starter_median.isna().all())
        pool['shots_p90'] = 0
        ranking = learn_metric_importance(pool)
        selected = select_learned_metrics(pool, ranking, max_metrics=3)
        _, profile, _ = assess_squad(
            pool, squad, default_assignments(squad), metrics=selected, metric_evidence=ranking
        )
        self.assertNotIn('shots_p90', profile.metric.tolist())

    def test_metric_inventory_covers_full_fmst_style_export_and_excludes_only_by_reason(self):
        pool, _, _ = frames()
        inventory = metric_inventory(pool)
        known = set(inventory.column)
        expected = {
            'goals', 'goals_p90', 'xg_p90', 'xa_p90', 'non_penalty_xg_p90',
            'xg_overperformance', 'shots_p90', 'shots_on_target_p90',
            'clear_cut_chances_p90', 'shots_outside_box_p90',
            'pass_completion_pct', 'passes_completed_p90', 'passes_attempted_p90',
            'key_passes_p90', 'progressive_passes_p90',
            'crosses_attempted', 'crosses_attempted_p90', 'cross_completion_pct',
            'crosses_completed_p90', 'open_play_crosses_attempted',
            'open_play_crosses_attempted_p90', 'open_play_cross_pct',
            'open_play_crosses_completed_p90', 'open_play_key_passes_p90',
            'tackles_p90', 'tackle_success_pct', 'interceptions_p90',
            'clearances_p90', 'blocks_p90', 'key_tackles_p90',
            'pressures_attempted_p90', 'pressure_success_pct',
            'possession_won_p90', 'possession_lost_p90', 'shots_blocked_p90',
            'distance_km_p90', 'sprints_p90', 'dribbles_p90', 'header_win_pct',
            'save_pct', 'saves_p90', 'clean_sheets_p90', 'goals_conceded_p90',
            'xg_prevented', 'xg_prevented_p90', 'goal_contributions_p90',
            'non_penalty_contributions_p90', 'defensive_actions_p90',
            'attacking_actions_p90', 'creative_actions_p90',
            'goalkeeping_actions_p90',
        }
        self.assertTrue(expected.issubset(known), sorted(expected - known))
        eligibility = inventory.set_index('column').eligible.to_dict()
        self.assertFalse(eligibility['index'])
        self.assertFalse(eligibility['goals_p90'])
        self.assertFalse(eligibility['goals_conceded_p90'])
        self.assertFalse(eligibility['clean_sheets_p90'])
        self.assertFalse(eligibility['attacking_actions_p90'])
        self.assertFalse(eligibility['xg_overperformance'])
        self.assertFalse(eligibility['crosses_attempted'])
        self.assertTrue(eligibility['xg_p90'])
        self.assertTrue(eligibility['passes_completed_p90'])
        self.assertTrue(eligibility['crosses_attempted_p90'])
        self.assertTrue(eligibility['possession_lost_p90'])
        self.assertIn('outcome', inventory.loc[inventory.column.eq('goals_p90'), 'reason'].iloc[0].lower())

    def test_unknown_future_numeric_metric_is_discovered_automatically(self):
        source = pd.read_csv(export())
        source['Brand New FMST Metric per 90'] = [str(1 + i / 10) for i in range(len(source))]
        raw = read_player_for_app(
            io.BytesIO(source.to_csv(index=False).encode()),
            league='League A',
            season=2025,
        )
        cleaned, _ = clean_statistics(raw)
        self.assertIn('brand_new_fmst_metric_per_90', cleaned.columns)
        self.assertIn('brand_new_fmst_metric_per_90', available_metrics(cleaned))

    def test_metric_learning_uses_only_role_relevant_outcomes(self):
        self.assertEqual(role_outcomes("ST"), ("scoring",))
        self.assertEqual(role_outcomes("AM_W"), ("scoring",))
        self.assertEqual(role_outcomes("CB"), ("preventing goals",))
        self.assertEqual(role_outcomes("GK"), ("preventing goals",))
        self.assertEqual(role_outcomes("FB_WB"), ("scoring", "preventing goals"))
        self.assertEqual(role_outcomes("CM_DM"), ("scoring", "preventing goals"))

        pool, _, _ = frames()
        ranking = learn_metric_importance(pool)
        self.assertTrue(ranking[ranking.role_group.eq("ST")].learned_outcome.eq("scoring").all())
        self.assertTrue(ranking[ranking.role_group.eq("AM_W")].learned_outcome.eq("scoring").all())
        self.assertTrue(ranking[ranking.role_group.eq("CB")].learned_outcome.eq("preventing goals").all())
        self.assertTrue(ranking[ranking.role_group.eq("GK")].learned_outcome.eq("preventing goals").all())

    def test_metric_selection_is_learned_not_position_hard_coded(self):
        # Balanced one-player-per-club construction: the only strong scoring
        # relationship is deliberately placed in distance_km_p90.
        rows = []
        for i in range(10):
            rows.append({
                'player_name': f'Striker {i}', 'team_id': f'Club {i}',
                'league': 'League A', 'season': 2025, 'role_group': 'ST',
                'minutes': 1800, 'goals': 4 + 2*i,
                'distance_km_p90': 8.0 + .4*i,
                'shots_p90': 2.0 + (.1 if i % 2 else -.1),
                'xg_p90': .30 + (.01 if i % 3 else -.01),
            })
        learned_pool = pd.DataFrame(rows)
        ranking = learn_metric_importance(learned_pool)
        selected = select_learned_metrics(learned_pool, ranking, max_metrics=1)
        self.assertEqual(selected['ST'], ['distance_km_p90'])

    def test_costs_follow_identity_and_missing_remains_unknown(self):
        pool, squad, _ = frames()
        cost = empty_costs(pool.iloc[[0, 3]])
        cost.loc[0, ['weekly_wage', 'contract_years', 'additional_fees', 'sale_proceeds']] = [1000, 3, 0, 100000]
        cost.loc[1, ['weekly_wage', 'contract_years', 'additional_fees', 'purchase_fee']] = [1500, 3, 20000, 200000]
        reverse = merge_costs(pool.iloc[[3, 0]], cost)
        self.assertEqual(reverse.weekly_wage.tolist(), [1500, 1000])
        result = cost_comparison(cost)
        self.assertEqual(result.total_cost.tolist(), [156000, 454000])
        self.assertEqual(replacement_cost(result, cost.player_key.iloc[0], cost.player_key.iloc[1]), 198000)
        cost.loc[1, 'weekly_wage'] = np.nan
        result = cost_comparison(cost)
        self.assertTrue(np.isnan(result.total_cost.iloc[1]))
        self.assertIsNone(replacement_cost(result, cost.player_key.iloc[0], cost.player_key.iloc[1]))

    def test_currency_contract_and_negative_cost_validation(self):
        pool, _, _ = frames()
        cost = empty_costs(pool.iloc[[3]])
        cost.loc[0, list(('weekly_wage','purchase_fee','contract_years','additional_fees'))] = [100, 10000, 2, 0]
        self.assertTrue(cost_comparison(cost).total_cost.isna().all())
        cost.loc[0, 'contract_years'] = 3
        cost.loc[0, 'currency'] = 'EUR'
        self.assertEqual(cost_comparison(cost).cost_status.iloc[0], 'Currency differs')
        cost.loc[0, 'purchase_fee'] = -1
        with self.assertRaises(DataError):
            cost_comparison(cost)

    def test_too_many_starters_rejected(self):
        pool, squad, _ = frames()
        assignments = default_assignments(squad)
        assignments.loc[assignments.role_group.eq('GK'), 'starter'] = True
        with self.assertRaises(DataError):
            assess_squad(pool, squad, assignments)


if __name__ == '__main__':
    unittest.main()
