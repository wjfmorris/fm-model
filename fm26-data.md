# FM26 data collection checklist

The backend accepts ordinary exports and does not depend on a particular extraction tool. Because FM26 may not provide a native statistics export in every edition/platform, collect the maximum information available from the save and preserve the source view name and season in each file.

## League history

For every completed season and target competition, capture one row per club with:

`League, Season, Position, Played, GF, GA, Points`

Use the final table after any split phase. Do not combine a regular season table with a championship/relegation group unless the group is modelled separately. Enter point deductions in `points_adjustment` rather than silently changing the football results.

## Team performance history

Use the same `league, season, team_id, matches, goals_for, goals_against` identifiers as the league table. Add as many team-level metrics as the save exposes:

- attack: xG, shots, shots on target, shots in the box, big chances, chances created, key passes, progressive passes, final-third entries, penalty-area entries, touches in the box, completed crosses, set-piece goals and possession;
- defence: xG against, shots against, shots on target against, big chances against, penalty-area entries against, pressures, successful pressures, tackles, interceptions, blocks, clearances, aerial wins, possession won/lost, errors, set-piece goals against, saves, save percentage and clean sheets.

Use one consistent definition and unit for a column across every season. The importer converts labelled count totals into per-match rates. A value such as `xG/90` should be labelled `xg_p90`; it is not treated as a season total.

## Player pool

Export the whole relevant player pool, not only obvious targets. Include a minimum-minute field so the model can distinguish a genuine season sample from a three-appearance spike. Keep the player's league, club, role/position, current season minutes, age, contract situation, market/asking value and wage.

The optional FM26 Player Export community tool is reported to create CSV and HTML from the squad and player-search screens. Its CSV may be semicolon-delimited, so import it with delimiter detection and check that `Best Pos` has not split across columns. A separate FMST26 community tool is reported to expose statistics, attributes, CA/PA, contracts and club data with CSV export. These tools are external to this repository; verify that their current version and platform support match the user's installation before using them.

The FMST26 statistics export may look like `Name, Position, Club, Guide Value,
Minutes Played, Goals, Assists, xA, Non-Penalty xG, ...`. The importer maps those
headers to the canonical player schema, including `Guide Value -> market_value`,
`Tackles Completed -> tackles_won`, `Saves per 90 -> saves_p90` and
`xG Prevented -> xg_prevented`. Because the export does not necessarily contain a
league or season column, call `read_player_export(..., league=..., season=...)` or
add those fields in the front end. FMST can include non-playing players with blank
minutes; those rows remain available for screening but are marked as lacking
performance evidence rather than being treated as zero-output players.

## Fields that make the later models stronger

Historical player seasons joined to team seasons by `team_id + season` allow a later attribution model to learn how player aggregates relate to team drivers. Current-save player data alone can provide role-relative performance and market comparisons, but it cannot prove a player's causal contribution to team goals. For that reason the current code labels sensitivity outputs and keeps the underlying driver weights visible.
