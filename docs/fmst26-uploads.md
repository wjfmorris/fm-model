# FMST26 guided recruitment workflow

## Start with three files

| Upload | Contents | Use |
| --- | --- | --- |
| League table history | Complete tables from multiple seasons: league, season, position, club, matches, GF, GA, points | League-specific GF/GA target frontier |
| League player statistics | Broad player pool for one recent season | Fixed positional reference distributions |
| My squad | Your club's players for that same season | Ownership, starting assignments and recruitment needs |

File names are not significant. The supplied 52-column FMST view and earlier
54-column view are recognised. Select the statistics season start year (2025 for
2025/26) and competition scope in the sidebar. League is inferred from a single
league in the tables, or supplied explicitly. Explicit player seasons take priority;
mixed player seasons or mismatched squad/pool seasons are rejected.

Squad rows are identified by normalised name and club, never appended to the league
pool. Exact repeated rows are deduplicated; conflicting identities are rejected.
Changing club/name requires reviewing identity and re-entering or rematching costs.
No fuzzy player matching is performed.

## Assess first, then upload targets

1. Import the three files and inspect source issues.
2. Calculate the finishing-position GF/GA frontier. Pairs are joint alternatives,
   not two independently guaranteed cut-offs. Latest club actuals are labelled as
   historical data, not next-season forecasts. Confirm club spelling explicitly.
3. Select formation; edit inferred broad position groups and intended starters.
   Each player occupies one group. Missing starters flag a gap; excess starters
   for the chosen formation must be corrected.
4. Choose recruitment purposes and, if needed, screening metrics. Defaults are
   explicit scouting menus, **not learned importance weights**. Percentile
   thresholds are computed from positional league peers above the minimum minutes.
   A metric needs at least five eligible peers and more than one distinct value.
5. Review individual profile shortfalls and download the recruitment profiles.
6. Upload candidate exports later. Add one league at a time when league is absent
   from the CSV. Re-uploading updates an existing candidate; owned players are
   excluded. Confirm the candidate's comparison position.
7. Enter costs and compare an outgoing player with a candidate in that position.

Priorities sort by unfilled starting slots, then the mean positive percentile
shortfall across supported metrics. Individual reviews show the number of
thresholds met, observed and required. Profile match is the fraction of thresholds
met. It is not a learned football-value score; missing fields are not successes.
Low/unknown minutes are flagged and placed after candidates with adequate minutes.
Cross-league output is explicitly unadjusted; the app does not invent league-strength
factors. Defensive activity is context-dependent and does not measure goals prevented.

The league target percentile and the recruitment screening percentile are separate:
there is no validated mapping from a league finish to player-stat thresholds here.
Candidate uploads never refit or move the original league benchmarks.

## Manual costs and persistence

Owned players need current weekly wage, remaining contract years, realistic sale
proceeds and any additional future costs. Targets need expected wage demand, total
purchase fee including instalments, proposed contract years and additional fees
(agent/signing costs and expected bonuses). Use the notes column for estimates or
renewal assumptions. Blank means unknown; explicitly enter 0 for genuinely zero fees.

For horizon H years, the app uses:

- Target cost = purchase fee + weekly wage × 52 × H + additional fees.
- Keeping cost = weekly wage × 52 × H + additional future fees (original acquisition
  fees are sunk).
- Extra replacement cost = target cost − sale proceeds − keeping cost.

Contracts must cover the selected horizon. Choose a shorter horizon or record an
explicit renewal assumption otherwise. Currency is explicit; mixed-currency rows
cannot produce a comparable total. No speculative resale proceeds are included.
Budget timing/instalment schedules are not modelled by this total-cost comparison.

Save edits with **Save financial entries**. They persist during the session and
through assessment refreshes for the same save label, league and club. **Download
financial entries** to keep them across browser/session resets; restore the CSV
on the next visit. Keys rather than row order join restored prices. Use different
save labels for different careers and restore only that career's finance file.
The app does not write private data into GitHub or a shared server database.

## Units and source checks

- `Guide Value` maps to `fmst_guide_value`, reference only. It is never an asking
  price, a fair-price training label or an input to financial recommendations.
- `£1.1M` parses as 1,100,000; `'-£1` is missing. Signed `'-2.07` is -2.07.
- Percentage fields stay on the displayed 0–100 scale. Missing GK fields stay missing.
- Per-90 fields remain rates. Total xG prevented is divided by minutes/90 only if
  its rate is absent. This assumes the total and minutes cover the same scope.
- Appearances, distance in km/90, sprints/90, clean sheets/90 and composite action
  headings are recognised. Recognition does not imply use in a model.
- Outside-box goals and undocumented composite action scores are excluded from
  the analysis copy. Outside-box goals greater than total goals are reported.
- Impossible percentages and negative non-signed statistics are excluded from
  analysis, with issues shown. Negative xG prevented is valid.
- Completed crosses/headers exceeding attempts are reported and those completed
  values excluded. Source observations are retained separately.
- Appearances above league fixtures warn about competition coverage. Player
  statistics are never automatically summed into league team metrics, even if
  the user chooses League only.
- Complete league tables must balance GF/GA. An invalid league layer is reported
  while squad comparisons remain available.

## What is not established by these files

One player season and league tables do not establish causal metric importance,
attribute effects, a next-season squad goal forecast or exact signing goal gains.
The advanced research dashboard retains the previous optional team/history models.
Those models need their own compatible inputs and validation. A statistics-only
FMST export cannot train attribute models.

Synthetic fixtures test both export schemas, matching, missing data, finance
arithmetic and the full guided Streamlit flow. User save files are not published.
