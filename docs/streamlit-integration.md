# Streamlit app architecture

The Streamlit layer is deliberately thin. Statistical logic belongs in `src/fm_model/`; `streamlit_app.py` handles uploads, controls, visualisation and downloads.

## Entry point

Streamlit Community Cloud should use:

- repository: `wjfmorris/fm-model`
- branch: `main`
- entry point: `streamlit_app.py`

Dependencies are installed from `requirements.txt`.

## App workflow

The default entry point calls `fm_model.recruitment_ui.render_workflow`. It provides five tabs:

1. Import checks: league-table history, league player statistics, own squad.
2. Season target: a GF/GA frontier and explicitly historical club actuals.
3. Squad assessment: editable roles/starters, purpose-specific descriptive profiles.
4. Potential signings: subsequent CSV uploads against fixed league thresholds.
5. Costs & replacements: editable wages/fees, CSV persistence and joint profile/cost comparison.

`fm_model.recruitment` owns identity matching, data exclusions, empirical profiles,
candidate comparisons and cost arithmetic. See [the upload contract](fmst26-uploads.md)
for units, formulas, validation and evidence limits. A bad league table disables only
the league model; it does not silently discard seasons or block player assessment.

The **Advanced research dashboard** checkbox retains the previous team-driver,
attribute-history and squad-planning interfaces for richer compatible datasets.
The guided workflow does not pretend to fit those layers from player totals.

Upload parsing supports CSV, TSV, HTML and XLSX. Original parsed frames remain
separate from cleaned analysis frames. Each player file covers one season; candidate
context must match that season. Club spelling is confirmed explicitly rather than
silently fuzzy-matched. Guide Value is isolated from all financial calculations.

Guided state stores a transactional assessment and a separate candidate pool.
Refreshing the three-file assessment clears candidates and stale target scenarios;
manual costs are kept by save/league/club namespace and stable player identity.
Targets also invalidate when finish/probability/context changes. Financial edits
need CSV download/restore to survive session resets; no shared server persistence
is implied. Optional advanced model state remains independent of the guided model.

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

Automated tests boot the Streamlit app headlessly, exercise the guided uploads/targets/finance-restore flow, verify stale-target invalidation and retain the advanced synthetic squad-plan regression test.
