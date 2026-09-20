# Streamlit app architecture

The Streamlit layer is deliberately thin. Statistical logic belongs in `src/fm_model/`; `streamlit_app.py` handles uploads, controls, visualisation and downloads.

## Entry point

Streamlit Community Cloud should use:

- repository: `wjfmorris/fm-model`
- branch: `main`
- entry point: `streamlit_app.py`

Dependencies are installed from `requirements.txt`.

## App workflow

The primary path is now the **Complete squad plan**:

1. Upload several completed league-table seasons.
2. Upload matching team-performance history for every club.
3. Optionally upload several historical player-season exports. These train the attribute → player-outcome layer with the latest historical season held out for validation.
4. Upload the current squad plus the broadest realistic transfer-market pool.
5. Fit the model and choose club, formation, target position, evidence threshold and maximum signings.
6. `MoneyballModel.squad_plan()` returns:
   - required GF/GA;
   - current next-season GF/GA forecast from the latest team-process profile;
   - attack/defence gap;
   - players to consider selling;
   - position upgrade opportunities;
   - minimum stat/attribute screening profiles;
   - current-market shortlists;
   - a value-aware transfer package;
   - combined post-transfer GF/GA, predicted position and target probability.

The separate Season target, Goal drivers, Recruitment and Diagnostics tabs remain available for inspection.

## Upload handling

The app supports multiple historical files. A file may contain its own `season` column, or the season can be inferred from a filename such as:

```text
league_2024_25.csv
team_2024_25.csv
players_2024_25.csv
```

Current player exports can omit league/season when that context is supplied in the sidebar. Ownership can come from an `owned` column or be inferred from the user's club.

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
