from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_app_boots_without_data(self):
        at = AppTest.from_file(APP, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertTrue(any("FM26 Moneyball Recruitment Lab" in item.value for item in at.title))

    def test_example_data_runs_full_dashboard_and_target(self):
        at = AppTest.from_file(APP, default_timeout=40).run()
        at.button(key="load_examples").click().run(timeout=40)
        self.assertFalse(at.exception)
        self.assertGreaterEqual(len(at.metric), 4)

        at.button(key="calculate_target").click().run(timeout=40)
        self.assertFalse(at.exception)
        self.assertTrue(any("Score at least" in item.label for item in at.metric))
        self.assertTrue(any("Concede at most" in item.label for item in at.metric))


if __name__ == "__main__":
    unittest.main()
