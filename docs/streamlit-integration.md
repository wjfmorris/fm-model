# Streamlit app architecture

The Streamlit layer is deliberately thin. Statistical logic belongs in `src/fm_model/`; `streamlit_app.py` handles uploads, controls, visualisation and downloads.

## Entry point

Streamlit Community Cloud should use:

- repository: `wjfmorris/fm-model`
- branch: `main`
- entry point: `streamlit_app.py`

Dependencies are installed from `requirements.txt`.

## App workflow

The standard path uses **Season target** and **Recruitment**:

1. Upload several completed league-table seasons.
2. Upload the current squad and market statistics for one recent season.
3. Set the player statistics season start year and club. No team metrics are required.
4. Build available layers and inspect input diagnostics.
5. Use league GF/GA targets and descriptive player/price comparisons.

Additional player files and team history live under Optional additional data. Without a team-goal model, the complete squad forecast remains unavailable and the UI explains why. It never manufactures team metrics by summing player statistics or invents another season from duplicate uploads.

A bad league table is reported with its actual GF/GA totals. The app can still load player comparisons, while the invalid league layer stays unavailable. Library callers remain strict by default; only the UI explicitly opts into `build_model(..., allow_partial=True)`.

## Upload handling

The app supports multiple historical files. A file may contain its own `season` column, or the season can be inferred from a filename such as:

```text
league_2024_25.csv
team_2024_25.csv
players_2024_25.csv
```

Both player uploads can omit league/season when that context is supplied in the sidebar. A filename such as `player_history.csv` does not mean it contains multiple seasons. Ownership can come from an `owned` column or be inferred from the user's club.

Displayed money ranges are never silently averaged. The user explicitly chooses lower, midpoint, upper or error handling.

## Data layers

Keep the following concepts separate:

- **League targets:** GF/GA → finishing position.
- **Team goal drivers:** team process statistics → goals for/against.
- **Historical player outcomes:** attributes → observable player outcomes such as xG/90, xA/key passes, finishing over xG, defensive output and goalkeeper xG prevention.
- **Player valuation:** role-relative football contribution → comparable market price.
- **Squad planning:** current team baseline + player replacement scenarios → transfer package.

CA, PA, reputation, market value and wages are not predictors in the football-performance layers. Price is introduced only after contribution is estimated.

## Evidence and fallbacks

Historical player seasons are optional. Without enough chronological evidence, the app still produces recruitment profiles, but attribute thresholds are labelled **descriptive** rather than learned.

The current team forecast assumes the latest observed team-process profile repeats next season. Replacement simulations modify that baseline using mapped player statistics and, where validated history supports it, attribute-predicted chance generation, finishing-over-xG and goalkeeper prevention. All outputs are scenario estimates, not causal guarantees.

## State boundaries

Session state stores:

- parsed league/team/player frames;
- fitted `MoneyballModel`;
- recruitment decisions;
- selected target scenario;
- driver route;
- complete squad-plan result.

Do not put mutable model state in module globals.

## Development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Automated tests boot the Streamlit app headlessly and exercise the synthetic end-to-end squad-plan workflow.
