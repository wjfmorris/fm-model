# FM Model

This repository contains the modelling backend for a data-driven Football Manager 26 recruitment tool. The intended user flow is:

1. upload completed league tables for several seasons;
2. choose a league, season context, finishing position and confidence level;
3. estimate the goals-scored/goals-conceded combinations associated with that finish;
4. upload team metrics to learn which attacking and defensive processes predict those goal totals;
5. upload a player export to score players against role- and league-relative replacement levels;
6. compare owned players with the market and recalculate the effect of possible recruitment decisions.

The repository is a Python library. It has no Streamlit dependency yet so that the modelling API can be tested independently before the interface is connected.

## What is implemented

`LeagueModel` is the first Moneyball layer. It uses all uploaded seasons, applies recency weights, tests whether a damped scoring-environment trend helps on historical seasons, and forecasts the next season's goal pace. It models normalized finishing rank from goals for and goals against and uses points when the table contains them. `targets()` returns a non-dominated goal frontier, a balanced recommendation, predicted position/points and an empirical target probability.

`GoalDriverModel` fits separate attack and defence models. It can use xG, shots, shots on target, chance creation, penalty-area entries, set pieces, pressing, interceptions, blocks, goalkeeper measures and other columns. Feature importance is reported from chronological holdout performance and local model sensitivity. `route()` searches observed metric ranges for the smallest standardised change that reaches a target; an infeasible target returns the best in-range attempt and marks it as infeasible.

`PlayerValuationModel` links team-driver fields to player statistics, derives per-90 rates from labelled season totals, compares players with league-and-role references, and produces contribution components. It estimates a cautious local goal sensitivity when a team-driver model is available. `decisions()` builds an empirical local price frontier and returns `BUY`, `SELL`, `KEEP_REVIEW`, `WATCH`, `INSUFFICIENT_PRICE_DATA` or `INSUFFICIENT_PERFORMANCE_DATA` with a reason and confidence flag.

`MoneyballModel` orchestrates the three layers and serialises a fitted model to JSON for the later Streamlit app.

## Install and use

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -e .
python -m unittest discover -s tests -v
```

The model can be used directly:

```python
from fm_model import MoneyballModel
from fm_model.data import read_table

model = MoneyballModel()
model.fit_league(read_table("data/templates/league_table.csv"))
model.fit_drivers(read_table("data/templates/team_metrics.csv"))
model.fit_players(read_table("data/templates/player_export.csv"))

scenario = model.target_scenario("Example League", position=6, probability=0.70)
print(scenario["recommended"])
print(scenario["frontier"])
```

An FMST26 statistics export can be read directly after supplying the save
context that is not present in the table:

```python
from fm_model import read_player_export

players = read_player_export(
    "league_statistics.csv", league="Example League", season=2026, owned=False
)
model.fit_players(players)
```

The importer recognises FMST26 labels such as `Guide Value`, `Minutes Played`,
`Non-Penalty xG`, `Tackles Completed`, `Saves per 90` and `xG Prevented`. Rows
without minutes are retained but marked as `no_minutes` and cannot receive a
confident performance-based buy/sell decision.

The command line wrapper can fit available layers and save the fitted model:

```bash
fm-model fit --league-table league.csv --team-data team_metrics.csv \\
  --player-data players.csv --out outputs/model.json
fm-model scenario --model outputs/model.json --league "Example League" \\
  --position 6 --probability 0.70
```

For an FMST26 export that omits the competition, attach its context on the
command line:

```bash
fm-model fit --league-table league.csv --player-data league_statistics.csv \\
  --player-league "Example League" --player-season 2026 --player-owned false \\
  --out outputs/model.json
```

## Input contracts

The templates in `data/templates/` contain headers and examples. The importer accepts CSV, semicolon-delimited CSV, TSV, local HTML tables and XLSX. It retains unknown columns, maps common FM labels to canonical names and refuses ambiguous ranges unless the caller chooses `lower`, `upper` or `midpoint`.

### League table

Required columns:

`league, season, position, matches, goals_for, goals_against`

Recommended columns:

`points, points_adjustment, team_id`

Each `league + season` group must be a completed comparable league phase with every position from 1 through N exactly once. Total goals for must equal total goals against. The optional points adjustment is useful for deductions or playoff adjustments when the raw points column needs explaining.

### Team metrics

Required columns are the league-table columns. Add season/team metrics wherever the FM26 view provides them. Counts are converted to per-match rates; fields named `_p90`, `_pct`, `_rate` or `_ratio` are treated as already normalised. Example attacking fields are `xg, shots, shots_on_target, chances_created, final_third_entries, penalty_area_entries, set_piece_goals`; example defensive fields are `xg_against, shots_against, big_chances_against, pressures, tackles_won, interceptions, goals_prevented, saves`.

Do not include `goals_for`, `goals_against`, `points` or `position` as explanatory features. The model protects those fields when `include_all_available=True`.

### Player export

Recommended identity columns are `player_name, player_id, league, team_id, role_group, minutes, age`. Performance columns can include `xg_p90, non_penalty_xg_p90, xa_p90, goals_p90, assists_p90, shots_p90, key_passes_p90, progressive_passes_p90, dribbles_p90, tackles_won_p90, interceptions_p90, headers_won_p90, pressures_p90, possession_won_p90, possession_lost_p90, saves_p90, xg_prevented_p90` plus any custom metrics. Add `market_value`, `asking_price`, `transfer_fee`, `wage`, `annual_wage`, `contract_months` and a Boolean `owned` column for recruitment decisions.

FM26 has changed role behaviour and separates in-possession and out-of-possession roles, so `role_group` should describe the role or role family that is actually being evaluated. Positional familiarity and the relevant attributes can be included as additional columns; they are retained by the importer and are ready for the next player-outcome training layer.

## Data availability for FM26

The official FM26 site describes the enhanced Recruitment Hub and TransferRoom integration, and the official tactical documentation describes separate in-possession and out-of-possession roles. In practice, community tools currently provide additional ways to extract data: FMST26 reports player statistics, attributes, contracts and club data and exports CSV; FM26 Player Export provides CSV/HTML export from squad and player-search screens. The repository does not bundle or redistribute either tool. See `docs/fm26-data.md` for a collection checklist and the exact columns this model can use.

## Statistical safeguards

The league model uses a chronological validation report. The driver models reserve the latest season as a holdout and choose regularisation using earlier expanding-window validation. Predictions include empirical residual intervals and an out-of-range flag. The route optimiser searches within historical metric ranges. These are predictive relationships from game data; they do not establish that changing one metric alone causes a particular number of goals. Player scores are role-relative contribution units, and money decisions require prices or wages from the user's own save.

The model is designed to learn an opportunity rather than assume one. It will therefore report a weak or unvalidated result when the uploaded data cannot support a conclusion.
