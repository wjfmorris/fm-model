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
    assess_squad, clean_statistics, compare_targets, cost_comparison,
    default_assignments, empty_costs, identify, merge_costs, prepare_inputs, replacement_cost,
)


def export():
    rows = []
    for role, position in [('GK', 'GK'), ('CB', 'D (C)'), ('FB_WB', 'D (R)'),
                           ('CM_DM', 'DM'), ('AM_W', 'AM (L)'), ('ST', 'ST')]:
        for i in range(10):
            rows.append({'Name': f'{role} Player {i}', 'Position': position,
                         'Club': 'My Club' if i < 3 else f'Club {i}',
                         'Minutes Played': 1500 + 30*i, 'Appearances': 22,
                         'Goals': i, 'Goals Outside Box': i+1 if i==0 else 0,
                         'NP-xG per 90': .1 + .04*i, 'Shots per 90': 1+.3*i,
                         'Shots on Target per 90': .2+.1*i, 'xA per 90': .01+.02*i,
                         'Open Play Key Passes p90': .5+.1*i, 'Progressive Passes per 90': 2+.3*i,
                         'Header Win %': f'{30+3*i}%', 'Interceptions per 90': 1+.1*i,
                         'Key Tackles per 90': .1+.01*i, 'Dribbles per 90': .2+.1*i,
                         'Pass Completion %': f'{60+i}%',
                         'Save Percentage': f'{50+3*i}%' if role=='GK' else '',
                         'xG Prevented': i-2 if role=='GK' else '',
                         'Guide Value': '£300M', 'Distance per 90 (km)': 10.3,
                         'Sprints per 90': 11.2, 'Clean Sheets per 90': .2 if role=='GK' else '',
                         'Goal Contributions per 90': .2, 'Non-Penalty Contributions per 90': .2,
                         'Defensive Actions per 90': 50, 'Attacking Actions per 90': 99,
                         'Creative Actions per 90': 30, 'Goalkeeping Actions per 90': 40 if role=='GK' else ''})
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
            _, profiles, _ = assess_squad(pool, squad, default_assignments(squad))
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
        self.assertEqual(pool.distance_km_p90.iloc[0], 10.3)
        self.assertTrue(pool.attacking_actions_p90.isna().all())
        self.assertTrue(pool.goals_outside_box.isna().all())
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

    def test_profiles_and_fixed_benchmark_candidate_comparison(self):
        pool, squad, _ = frames()
        assignments = default_assignments(squad)
        priorities, profile, reviews = assess_squad(pool, squad, assignments)
        self.assertEqual(len(reviews), len(squad))
        self.assertEqual(len(priorities), 6)
        self.assertEqual(assignments.starter.sum(), 11)
        st_profile = profile[profile.role_group.eq('ST')]
        self.assertAlmostEqual(st_profile.screening_target.iloc[0], .316)
        target = pool[~pool.owned & pool.role_group.eq('ST')].copy()
        first, _ = compare_targets(target, squad, profile, 'ST', league='League A')
        target.loc[target.index[0], 'league'] = 'League B'
        target.loc[target.index[0], 'shots_p90'] = np.nan
        second, detail = compare_targets(target, squad, profile, 'ST', league='League A')
        self.assertTrue(second.evidence.str.contains('cross-league').any())
        self.assertEqual(second.metrics_required.unique().tolist(), [3])
        player = second[second.player_key.eq(target.player_key.iloc[0])].iloc[0]
        self.assertEqual(player.metrics_observed, 2)
        pd.testing.assert_frame_equal(first[first.source.eq('My squad')].reset_index(drop=True), second[second.source.eq('My squad')].reset_index(drop=True))
        self.assertEqual(len(detail), len(second)*3)

    def test_unsupported_metrics_and_low_minutes_not_zero_performance(self):
        pool, squad, _ = frames()
        squad.loc[squad.role_group.eq('ST'), 'minutes'] = 30
        _, profile, review = assess_squad(pool, squad, default_assignments(squad))
        self.assertTrue(review[review.role_group.eq('ST')].review.eq('Insufficient minutes').all())
        self.assertTrue(profile[profile.role_group.eq('ST')].current_starter_median.isna().all())
        pool['shots_p90'] = 0
        _, profile, _ = assess_squad(pool, squad, default_assignments(squad))
        self.assertNotIn('shots_p90', profile.metric.tolist())

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
