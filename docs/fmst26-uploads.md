# FMST26 upload contract

The standard inputs are multi-season completed league tables and one recent-season player export. Team metrics are optional and not part of the required workflow.

## Display formats

The 54-column FMST26 statistics view is supported unchanged: three identity columns (`Name`, `Position`, `Club`) and 51 numeric columns. The importer maps displayed headings to canonical fields and retains all input fields. Recognising a column does not mean the present scoring model uses it.

| Display | Interpretation |
| --- | --- |
| `Guide Value`: `£1.1M`, `£946K` | 1,100,000 and 946,000 in the source currency; not an asking price |
| `'-£1` | Missing value |
| `xG Overperformance`: `'-2.07` | Signed number -2.07 |
| `Shot Accuracy %`: `39.7%` | 39.7 on a 0–100 percentage scale |
| `xG per 90`, `NP-xG per 90`, `Shots per 90` | Already rates; never divide these by minutes again |
| `Goals`, `Assists`, `Headers Won`, `xG Prevented` | Totals; derive recognised per-90 metrics only when the rate is absent and minutes are available |
| Blank goalkeeper statistics for outfield players | Missing, not zero |
| `M (RC), AM (RLC), ST` | Original position text retained; broad comparison role inferred using the existing position priority |

CCC and tackles retain distinct canonical fields rather than being silently relabelled as another event. All currency values must use a consistent currency; no currency conversion is performed.

## Season and ownership

Set **Player statistics season (start year)** to the season the figures describe: 2025 for 2025/26. This supplies missing context to both player files. An explicit year or dated history filename takes priority. Identical additional history files are deduplicated. Identical current and history uploads are used in their respective roles, not appended into a larger training sample.

Use the exact exported club name to infer owned players, or supply an `owned` column. The standard workflow needs only `player_export.csv`; uploading the same data as `player_history.csv` adds no information.

## Diagnostics and evidence limits

- Completed league tables must include every position and have equal total GF and GA. An invalid table disables league targets with a specific error, while player analysis can still load. Source values are never silently repaired or seasons omitted.
- Report completed crosses/headers exceeding attempts and unusually repeated guide prices for review. These values are retained; warnings do not prove which value is wrong.
- No attributes in the file means no attribute-learning result. No wages means price comparisons exclude employment costs.
- One player season supports descriptive comparisons. It does not supply a held-out future season for the historical attribute models.
- No team metrics means no learned team goal drivers or transfer goal forecast. Player proxy rankings are explicitly labelled. Defensive event volumes alone do not measure goals prevented.
- Real club identifiers in league tables are required for any future club-level joins; template IDs such as `example-2023-1` do not match exported club names.

The supplied format has regression tests using synthetic players. Private save exports are not included in the repository.
