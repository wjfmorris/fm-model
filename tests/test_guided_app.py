import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from fm_model.app_support import load_example_frames
from fm_model.recruitment import empty_costs
from test_recruitment_workflow import export, frames

APP = Path(__file__).resolve().parents[1] / 'streamlit_app.py'


class GuidedAppTests(unittest.TestCase):
    def test_guided_demo_and_target_invalidation(self):
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button(key='guided_example').click().run()
        self.assertFalse(at.exception)
        at.button(key='calculate_target').click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any(m.label == 'Score at least' for m in at.metric))
        at.number_input(key='target_position_Example League').set_value(8).run()
        self.assertFalse(at.exception)
        self.assertFalse(any(m.label == 'Score at least' for m in at.metric))

    def test_three_uploads_targets_and_finance_restore(self):
        import pandas as pd
        raw = pd.read_csv(export())
        pool, _, _ = frames()
        finances = empty_costs(pool)
        finances['weekly_wage'] = 1000.
        finances['purchase_fee'] = 100000.
        finances['sale_proceeds'] = 50000.
        finances['contract_years'] = 3.
        finances['additional_fees'] = 0.
        uploads = {
            'guided_league': load_example_frames()['league'].to_csv(index=False).encode(),
            'guided_pool': export().getvalue(),
            'guided_squad': raw[raw.Club.eq('My Club')].to_csv(index=False).encode(),
            'guided_candidates': export().getvalue(),
            'restore_costs': finances.to_csv(index=False).encode(),
        }
        source = f'''
import io
from pathlib import Path
from unittest.mock import patch
import streamlit as st
uploads = {uploads!r}
original = st.file_uploader
def upload(label, *args, **kwargs):
    key=kwargs.get('key')
    if key not in uploads:
        return original(label, *args, **kwargs)
    f=io.BytesIO(uploads[key]);f.name=key+'.csv'
    return [f] if kwargs.get('accept_multiple_files') else f
with patch('streamlit.file_uploader', side_effect=upload):
    exec(Path({str(APP)!r}).read_text())
'''
        at = AppTest.from_string(source, default_timeout=30).run()
        at.button(key='guided_import').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state['workflow']['pool']), 60)
        self.assertEqual(len(at.session_state['workflow']['squad']), 18)
        at.button(key='add_targets').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state['workflow_targets']), 42)
        at.selectbox(key='compare_role').set_value('ST').run()
        at.button(key='import_costs').click().run()
        self.assertFalse(at.exception)
        namespace = at.session_state['workflow']['namespace']
        self.assertEqual(at.session_state['manual_costs_' + namespace].weekly_wage.iloc[0], 1000)
        self.assertTrue(any('Extra cost of replacing' in m.label for m in at.metric))
        at.button(key='guided_import').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state['manual_costs_' + namespace].weekly_wage.iloc[0], 1000)
        self.assertNotIn('workflow_targets', at.session_state)


if __name__ == '__main__':
    unittest.main()
