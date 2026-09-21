# FM Model

A Football Manager 26 tool for league goal targets and player/price comparisons, with optional research models for team goal drivers and squad planning.

## What the model does

The standard upload workflow is:

1. Upload completed league tables covering several seasons.
2. Upload one FMST26 player file covering the most recent season. Select its season start year and your club in the sidebar.
3. Build the available layers, review import diagnostics, and use **Season target** and **Recruitment**.

A separate team-metrics file is **not required**. `player_history.csv` is optional and may cover the same recent season. Its missing season uses the sidebar setting; explicit season columns or dated filenames take priority. Repeating the same player observations never creates another historical season.

With these inputs, league targets and descriptive player/price comparisons work independently. The existing player comparison uses statistical proxies when no team-goal model exists. It does **not** establish which stats cause goals, forecast goals added by signings, or learn attribute effects from a statistics-only export. BUY/SELL labels in this mode are review candidates with `descriptive_proxy` confidence. See [FMST26 upload contract](docs/fmst26-uploads.md).

The optional research workflow retains team goal-driver models and complete squad planning for datasets that support them. These features require team outcomes/process data; validated attribute models also require attributes and multiple player seasons. They remain unavailable when that evidence is absent.

The app deliberately keeps **CA, PA, reputation and price out of the football-performance models**. Price/wage information is only used after football contribution has been estimated, so the model can search for market inefficiencies instead of reproducing FM's hidden ability ratings.

## Repository layout

```text
fm-model/
├── streamlit_app.py
├── .streamlit/
│   └── config.toml
├── src/
│   └── fm_model/
│       ├── __init__.py
│       ├── app_support.py
│       ├── catalogue.py
│       ├── collection.py
│       ├── data.py
│       ├── drivers.py
│       ├── league.py
│       ├── learning.py
│       ├── outcomes.py
│       ├── pipeline.py
│       ├── planning.py
│       ├── players.py
│       └── roles.py
├── tests/
│   ├── test_model.py
│   ├── test_app_support.py
│   ├── test_squad_planning.py
│   └── test_streamlit_app.py
├── data/
│   └── templates/
│       ├── league_table.csv
│       ├── team_metrics.csv
│       ├── player_history.csv
│       └── player_export.csv
├── docs/
│   ├── fm26-data.md
│   └── streamlit-integration.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Implemented model layers

### LeagueModel

Uses uploaded completed seasons, applies recency weighting, tests a damped scoring-environment trend on historical seasons, and models normalized finishing rank from goals for/goals against. Where points are available it also models points. `targets()` returns a non-dominated goal frontier, a balanced recommendation, predicted position/points and an empirical target probability.

### GoalDriverModel

Fits separate attacking and defensive models using available team metrics such as xG, shots, shots on target, chance creation, penalty-area entries, set pieces, pressing, interceptions, blocks and goalkeeper measures. Driver importance is reported from chronological holdout performance and local model sensitivity. `route()` searches observed metric ranges for the smallest standardized changes that move toward a goal target.

### PlayerValuationModel

Links learned team-driver fields to player statistics, derives per-90 rates from labelled raw totals, compares players with league-and-role references, and produces contribution components. `decisions()` compares contribution with local market evidence and returns `BUY`, `SELL`, `KEEP_REVIEW`, `WATCH`, `INSUFFICIENT_PRICE_DATA` or `INSUFFICIENT_PERFORMANCE_DATA`, together with reasons and confidence flags.

### PlayerOutcomeModel

Uses historical player-season rows to learn how attributes predict observable player outcomes. It separates chance generation, chance creation, finishing relative to xG, defensive output and goalkeeper xG prevention where the available columns support those targets. The final historical season is held out for validation. When there is not enough history, the app falls back to descriptive candidate profiles rather than pretending attribute weights were learned.

### SquadPlanner

Connects the model layers into one planning result. It selects a formation-aware current XI, adjusts the latest team-process profile from the previous XI to the current XI when historical player data allows, forecasts next-season GF/GA, calculates the goal gap, evaluates sales, simulates market replacements by position, builds minimum stat/attribute profiles and candidate shortlists, and re-runs the team goal model after the proposed transfer package.

### MoneyballModel

Orchestrates the league, goal-driver, historical attribute, player-valuation and squad-planning layers and serializes/deserializes fitted state for the Streamlit app.

## Install

Backend only:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .
```

Backend plus spreadsheet/HTML import support:

```bash
pip install -e ".[imports]"
```

Backend plus the Streamlit app dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit app

Install the app environment and launch from the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dashboard includes a synthetic **Load example data** path, so every main screen can be tested before importing a real FM26 save. For real data, the sidebar accepts multiple league-table files, multiple team-history files, multiple historical player-season files, and one current squad/market export. The **Data collection & setup** tab gives the exact columns, clubs, positions and seasons to collect before you start exporting.

## Verify the project

```bash
python -m unittest discover -s tests -v
python -m fm_model --help
```

GitHub Actions runs the same backend checks on Python 3.11 and 3.12.

## Python API

```python
from fm_model import MoneyballModel, read_table

model = MoneyballModel()
model.fit_league(read_table("data/templates/league_table.csv"))
model.fit_drivers(read_table("data/templates/team_metrics.csv"))
model.fit_player_outcomes(read_table("data/templates/player_history.csv"))
players = read_table("data/templates/player_export.csv")
model.fit_players(players)

scenario = model.target_scenario(
    "Example League",
    position=6,
    probability=0.70,
)

print(scenario["recommended"])
print(scenario["frontier"])

plan = model.squad_plan(
    "Example League",
    position=4,
    players=players,
    club="Club F",
    probability=0.70,
    formation="4-2-3-1",
    max_recruits=3,
)
print(plan["current_forecast"])
print(plan["sale_candidates"])
print(plan["position_opportunities"])
print(plan["after_transfer_forecast"])
```

The public package also exposes `read_player_export`, `DATA_CATALOGUE`, `DataError`, `LeagueModel`, `GoalDriverModel`, `PlayerValuationModel`, `PlayerOutcomeModel`, `SquadPlanner` and the formation/position helpers.

## FMST26 player exports

An FMST26 statistics export can be passed directly after adding save context that may not be present in the table:

```python
from fm_model import MoneyballModel, read_player_export

players = read_player_export(
    "league_statistics.csv",
    league="Example League",
    season=2026,
    owned=False,
)

model = MoneyballModel()
model.fit_players(players)
```

The importer recognizes labels including `Guide Value`, `Minutes Played`, `Non-Penalty xG`, `Tackles Completed`, `Saves per 90` and `xG Prevented`. Rows without minutes are retained but cannot receive the same performance confidence as players with meaningful playing-time samples.

Displayed price/attribute ranges are never silently averaged. The UI makes the user choose how a range is handled rather than hiding that assumption.

## Input contracts

### League table

Required:

`league, season, position, matches, goals_for, goals_against`

Recommended:

`points, points_adjustment, team_id`

Each `league + season` group must contain one completed comparable league phase with every position from 1 through N exactly once. Total goals for must equal total goals against.

### Team metrics

Use the same league/season/team identifiers as the league table and add as many consistent team metrics as the save exposes. Example attacking fields include `xg`, `shots`, `shots_on_target`, `chances_created`, `final_third_entries`, `penalty_area_entries` and `set_piece_goals`. Example defensive fields include `xg_against`, `shots_against`, `big_chances_against`, `pressures`, `tackles_won`, `interceptions`, `goals_prevented` and `saves`.

Do not use `goals_for`, `goals_against`, `points` or `position` as explanatory features.

### Player export

Recommended identity fields:

`player_name, player_id, league, team_id, role_group, minutes, age`

Useful performance fields include:

`xg_p90, non_penalty_xg_p90, xa_p90, goals_p90, assists_p90, shots_p90, key_passes_p90, progressive_passes_p90, dribbles_p90, tackles_won_p90, interceptions_p90, headers_won_p90, pressures_p90, possession_won_p90, possession_lost_p90, saves_p90, xg_prevented_p90`

Useful market fields include:

`market_value, asking_price, transfer_fee, wage, annual_wage, contract_months, owned`

The planner accepts native FM position labels such as `D (C)`, `D (R)`, `M (C)`, `AM (R)` and `ST (C)` and maps them into stable planning groups. Keep the original position text in the export; the app handles the grouping.

## CLI

Fit the available layers and save the model:

```bash
fm-model fit \
  --league-table data/templates/league_table.csv \
  --team-data data/templates/team_metrics.csv \
  --player-data data/templates/player_export.csv \
  --out outputs/model.json
```

Run a finishing target:

```bash
fm-model scenario \
  --model outputs/model.json \
  --league "Example League" \
  --position 6 \
  --probability 0.70
```

For an FMST26 export that omits competition/season context:

```bash
fm-model fit \
  --league-table data/templates/league_table.csv \
  --player-data league_statistics.csv \
  --player-league "Example League" \
  --player-season 2026 \
  --player-owned false \
  --out outputs/model.json
```

## Statistical safeguards

The league model uses chronological validation. Driver models reserve the latest season as a holdout and choose regularization using earlier expanding-window validation. Predictions expose empirical residual uncertainty and out-of-range flags. Route optimization stays within historical metric ranges.

These are predictive relationships from game data, not proof that changing one metric alone causes a specific goal change. The model is designed to report weak or incomplete evidence rather than invent certainty.

## Streamlit handoff

The Streamlit UI is implemented in `streamlit_app.py`. See `docs/streamlit-integration.md` for the UI/backend boundary, upload flow, state handling and deployment notes.
