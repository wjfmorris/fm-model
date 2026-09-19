# Streamlit integration contract

The modelling backend is intentionally independent from Streamlit. The eventual app should be a thin interface over the public API in `fm_model`, rather than duplicating modelling logic inside the UI.

## Recommended entry point

Create `streamlit_app.py` at the repository root. Streamlit Community Cloud can then use that file as the app entry point and install `requirements.txt`.

The app should import from the public package:

```python
from fm_model import (
    DATA_CATALOGUE,
    DataError,
    MoneyballModel,
    read_player_export,
    read_table,
)
```

Avoid importing private helpers from individual model modules unless a new public API is genuinely required.

## Intended app flow

1. Upload completed league-table history and call `read_table()`.
2. Fit `MoneyballModel.fit_league()`.
3. Upload team-performance history and call `fit_drivers()`.
4. Upload an FMST26/player-search export with `read_player_export()`.
5. Fit `fit_players()`.
6. Use `readiness()` to show which model layers are available.
7. Use `target_scenario()` for finishing-position goal targets.
8. Use `explain_scenario()` for goal-driver explanations and an improvement route.
9. Use `player_decisions()` for recruitment/squad decisions.
10. Keep the fitted model in Streamlit session state and optionally expose `MoneyballModel.save()` / `load()` for persistence.

## Upload handling

`read_table()` and `read_player_export()` accept file-like objects, so Streamlit's `UploadedFile` can be passed directly.

FMST26 exports may omit league, season and ownership context. Supply those explicitly:

```python
players = read_player_export(
    uploaded_file,
    league=selected_league,
    season=selected_season,
    owned=False,
)
```

Displayed money ranges are deliberately not guessed. If an uploaded value is a range, the UI should make the user choose lower, upper or midpoint handling and pass that policy through the importer before fitting.

## Error handling

Catch `DataError` for user-facing validation messages. Do not catch every exception and silently continue: unexpected exceptions should remain visible during development.

## State boundaries

A practical session state split is:

- raw uploaded files / parsed frames;
- `MoneyballModel`;
- selected league / season / club;
- target position and confidence;
- current filter state for the recruitment table.

Do not store mutable modelling state in module-level globals.

## Deployment

Local development:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The repository is ready for the UI layer once `streamlit_app.py` is added; no backend files need to be moved or repackaged at that point.
