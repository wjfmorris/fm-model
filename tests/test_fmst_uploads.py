"""Regression tests for the supplied FMST26 display schema (synthetic players)."""
import io
import unittest

import numpy as np

from fm_model.app_support import (
    build_model, load_example_frames, player_upload_report,
    read_player_for_app, read_player_history_for_app,
)
from fm_model.data import read_table
from fm_model.errors import DataError
from fm_model.pipeline import MoneyballModel
from fm_model.roles import canonical_role

HEADER = ('Name,Position,Club,Guide Value,Minutes Played,Goals,Assists,Goals per 90,'
          'xG per 90,xA per 90,NP-xG per 90,xG Overperformance,Shots per 90,'
          'Shots on Target per 90,Shot Accuracy %,CCC per 90,Shots Outside Box per 90,'
          'Pass Completion %,Passes Completed per 90,Passes Attempted per 90,'
          'Key Passes per 90,Progressive Passes per 90,Crosses Attempted,Crosses Completed,'
          'Cross Completion %,Open Play Crosses Att,Open Play Crosses Comp,Open Play Cross %,'
          'Open Play Key Passes p90,Tackles per 90,Tackles Attempted,Tackle Success %,'
          'Interceptions per 90,Clearances per 90,Blocks per 90,Key Tackles per 90,'
          'Pressures per 90,Pressures Attempted per 90,Pressure Success %,Fouls Made,'
          'Possession Won per 90,Possession Lost per 90,Shots Blocked per 90,'
          'Mistakes Leading to Goals,Dribbles per 90,Headers Attempted,Headers Won,'
          'Header Win %,Save Percentage,Saves per 90,Clean Sheets,Goals Conceded per 90,'
          'xG Prevented,Penalty Save %')


def upload(name='player_export.csv'):
    import csv
    stream = io.StringIO()
    writer = csv.writer(stream)
    headings = HEADER.split(',')
    writer.writerow(headings)
    for i in range(12):
        row = {key: '' for key in headings}
        row.update({'Name': f'Example {i}', 'Position': 'M (RC), AM (RLC), ST',
                    'Club': 'Club A' if i < 3 else 'Club B', 'Guide Value': '£1.1M',
                    'Minutes Played': '2198', 'Goals': '5', 'Assists': '6',
                    'Goals per 90': '0.20', 'xG per 90': '0.29', 'NP-xG per 90': '0.26',
                    'xA per 90': '0.30', 'xG Overperformance': "'-2.07",
                    'Shots per 90': '2.8', 'Key Passes per 90': '2.01',
                    'Progressive Passes per 90': '4.5', 'Shot Accuracy %': '39.7%',
                    'Interceptions per 90': '1.76', 'Pressures per 90': '2.0',
                    'Open Play Crosses Att': '0', 'Open Play Crosses Comp': '12'})
        writer.writerow([row[key] for key in headings])
    raw = io.BytesIO(stream.getvalue().encode('utf-8-sig'))
    raw.name = name
    return raw


class FMSTUploadTests(unittest.TestCase):
    def test_exact_display_schema_and_units(self):
        frame = read_player_for_app(upload(), league='A', season=2025, owned_club='Club A')
        report = player_upload_report(frame)
        self.assertEqual(len(report['numeric_columns']), 51)
        self.assertEqual(report['other_columns'], [])
        self.assertEqual(frame.loc[0, 'xg_p90'], .29)
        self.assertEqual(frame.loc[0, 'non_penalty_xg_p90'], .26)
        self.assertEqual(frame.loc[0, 'shot_accuracy_pct'], 39.7)
        self.assertEqual(frame.loc[0, 'xg_overperformance'], -2.07)
        self.assertEqual(frame.loc[0, 'fmst_guide_value'], 1_100_000)
        self.assertTrue(np.isnan(frame.loc[0, 'save_pct']))
        self.assertEqual(int(frame.owned.sum()), 3)
        self.assertEqual(frame.loc[0, 'position'], 'M (RC), AM (RLC), ST')
        model = build_model(player_data=frame)
        scored = model.player_decisions(frame)
        self.assertEqual(scored.loc[0, 'goals_p90'], .20)  # Do not derive over a supplied rate.
        self.assertEqual(scored.loc[0, 'shots_p90'], 2.8)
        self.assertTrue(scored.decision_confidence.eq('descriptive_proxy').all())
        self.assertTrue(any('greater than' in w for w in report['warnings']))

    def test_one_season_default_and_duplicate_uploads(self):
        history = read_player_history_for_app([upload('player_history.csv'), upload('copy.csv')],
                                              league='A', season=2025)
        self.assertEqual(len(history), 12)
        self.assertEqual(history.season.unique().tolist(), [2025])
        model = build_model(historical_player_data=history)
        self.assertTrue(model.readiness()['player_values'])
        self.assertFalse(model.readiness()['attribute_outcomes'])
        self.assertFalse(model.readiness()['goal_drivers'])
        explicit = read_player_history_for_app([upload('players_2024_25.csv')], league='A', season=2025)
        self.assertEqual(explicit.season.unique().tolist(), [2024])

    def test_invalid_league_does_not_block_players_or_get_silently_fixed(self):
        league = load_example_frames()['league']
        league.loc[0, 'goals_for'] = '0'
        players = read_player_for_app(upload(), league='A', season=2025)
        with self.assertRaises(DataError):
            build_model(league_data=league, player_data=players)
        model = build_model(league_data=league, player_data=players, allow_partial=True)
        self.assertIsNone(model.league_model)
        self.assertTrue(model.readiness()['player_values'])
        self.assertIn('total goals for', model.upload_issues[0])
        restored = MoneyballModel.from_dict(model.to_dict())
        self.assertEqual(restored.upload_issues, model.upload_issues)

    def test_valid_tables_and_one_player_season_need_no_team_file(self):
        model = build_model(league_data=load_example_frames()['league'],
                            player_data=read_player_for_app(upload(), league='Example League', season=2025))
        self.assertTrue(model.readiness()['league_targets'])
        self.assertTrue(model.readiness()['market_decisions'])
        self.assertFalse(model.readiness()['squad_plan'])
        self.assertNotIn('team metrics', model.readiness()['next_step'])
        self.assertIn('recommended', model.target_scenario('Example League', 4, probability=.6))

    def test_compound_position_labels_are_recognised(self):
        self.assertEqual(canonical_role('D (RC)'), 'CB')
        self.assertEqual(canonical_role('D (RL)'), 'FB_WB')
        self.assertEqual(canonical_role('M (RC)'), 'CM_DM')
        self.assertEqual(canonical_role('M (RL)'), 'AM_W')


if __name__ == '__main__':
    unittest.main()
