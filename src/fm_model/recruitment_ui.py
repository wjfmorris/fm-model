"""Guided three-file Streamlit workflow; statistical functions live in recruitment."""
from __future__ import annotations

import hashlib
import io

import pandas as pd
import streamlit as st

from .app_support import clone_upload, frame_to_csv_bytes, load_example_frames, merge_table_uploads, read_player_for_app
from .data import read_table, season_number
from .errors import DataError
from .pipeline import MoneyballModel
from .roles import FORMATION_PRESETS, ROLE_LABELS
from .metric_learning import learn_metric_importance, metric_inventory, select_learned_metrics
from .recruitment import (
    COST_FIELDS, assess_squad, available_metrics, clean_statistics,
    compare_targets, cost_comparison, default_assignments, empty_costs, merge_costs,
    normal, prepare_inputs, replacement_cost,
)

FORMATS = ["csv", "tsv", "html", "htm", "xlsx", "xlsm"]


def display_frame(frame):
    """Readable labels; downloaded files keep stable machine-readable headings."""
    f = frame.drop(columns=["player_key", "evidence"], errors="ignore").copy()
    if "role_group" in f:
        f["role_group"] = f.role_group.map(ROLE_LABELS).fillna(f.role_group)
    if "metric" in f:
        f["metric"] = f.metric.str.replace("_p90", " /90", regex=False).str.replace("_pct", " %", regex=False).str.replace("_", " ", regex=False)
    f.columns = [c.replace("_", " ").capitalize() for c in f.columns]
    return f


def install_workflow(league, pool, squad, *, save_label="My save", scope="Unknown / mixed competitions"):
    """Transactional load: malformed files cannot replace a working assessment."""
    model = MoneyballModel()
    try:
        model.fit_league(league)
    except DataError as exc:
        model.upload_issues.append("League targets unavailable: " + str(exc))
    selected_league = str(pool.league.iloc[0]) if len(pool) else ""
    matches = 34
    if model.league_model is not None and selected_league in model.league_model.leagues:
        matches = int(model.league_model.context(selected_league)["matches"])
    clean_pool, clean_squad, issues = prepare_inputs(pool, squad, league_matches=matches)
    if scope == "League only" and issues.field.eq("competition_scope").any():
        issues.loc[issues.field.eq("competition_scope"), "message"] += " This contradicts the selected League only scope."
    namespace = hashlib.sha256((save_label + '|' + selected_league + '|' + str(clean_squad.team_id.iloc[0])).encode()).hexdigest()[:16]
    revision = st.session_state.get("workflow_revision", 0) + 1
    st.session_state["workflow"] = {
        "league": league, "raw_pool": pool, "raw_squad": squad,
        "pool": clean_pool, "squad": clean_squad, "issues": issues,
        "scope": scope, "namespace": namespace, "revision": revision, "model": model,
    }
    st.session_state["workflow_revision"] = revision
    st.session_state["moneyball_model"] = model
    st.session_state.pop("target_scenario", None)
    st.session_state.pop("guided_target_signature", None)
    st.session_state.pop("workflow_targets", None)
    st.session_state.pop("workflow_target_issues", None)


def render_uploads():
    with st.sidebar:
        st.title("FM26 Moneyball")
        st.caption("Import → assess squad → scout targets → compare costs")
        save_label = st.text_input("Save label", value="My save", key="guided_save_label",
                                   help="Use a different name for each FM save to keep financial entries separate.")
        league_upload = st.file_uploader("1. League table history", type=FORMATS, key="guided_league", accept_multiple_files=True)
        pool_upload = st.file_uploader("2. League player statistics", type=FORMATS, key="guided_pool")
        squad_upload = st.file_uploader("3. My squad", type=FORMATS, key="guided_squad")
        st.caption("Use league_table, league_statistics and my_team. No team-metrics file required.")
        league_name = st.text_input("League (leave blank to infer from table)", key="guided_league_name")
        season = st.number_input("Statistics season start year", 1900, 2200, 2025, key="guided_season",
                                 help="2025 means 2025/26. Both player files cover this season; tables retain their own years.")
        scope = st.selectbox("Player statistics cover", ["Unknown / mixed competitions", "All competitions", "League only"], key="guided_scope")
        if st.button("Import and assess squad", type="primary", key="guided_import",
                     disabled=not (league_upload and pool_upload and squad_upload)):
            try:
                tables = merge_table_uploads(league_upload)
                leagues = tables.league.dropna().unique().tolist() if "league" in tables else []
                context = league_name.strip() or (leagues[0] if len(leagues) == 1 else None)
                if context is None:
                    raise DataError("Enter the league name when the tables contain more than one league.")
                pool = read_player_for_app(pool_upload, league=context, season=season, ownership_mode="none")
                squad = read_player_for_app(squad_upload, league=context, season=season, ownership_mode="all")
                install_workflow(tables, pool, squad, save_label=save_label, scope=scope)
                st.success("Files imported. Confirm roles and starters in Squad assessment.")
            except (DataError, ValueError, KeyError) as exc:
                st.error(str(exc))
        if st.button("Load guided example", key="guided_example"):
            frames = load_example_frames()
            pool = read_player_for_app(io.BytesIO(frames["players"].to_csv(index=False).encode()), ownership_mode="none")
            # The example's owned players belong to Club F; use the complete club.
            squad = pool[pool.team_id.eq("Club F")].copy()
            install_workflow(frames["league"], pool, squad, save_label="Synthetic demo", scope="Unknown / mixed competitions")
        if st.button("Clear assessment", key="guided_clear"):
            for key in ("workflow", "workflow_targets", "workflow_target_issues", "moneyball_model", "target_scenario"):
                st.session_state.pop(key, None)
            st.rerun()
        st.caption("Financial edits last for this session. Download their CSV to keep them after closing or restarting the app.")


def render_checks(work):
    st.subheader("Imported files")
    pool, squad = work["pool"], work["squad"]
    a, b, c, d = st.columns(4)
    a.metric("League players", len(pool))
    b.metric("Squad players", len(squad))
    c.metric("Historical seasons", work["league"].season.nunique() if "season" in work["league"] else 0)
    d.metric("Squad rows already in pool", int(squad.player_key.isin(pool.player_key).sum()))
    st.write(f"Your squad: **{squad.team_id.iloc[0]}** · Statistics scope: **{work['scope']}**")
    st.caption("Overlapping squad rows are matched once. Statistics and contracts are never added together across the two uploads.")
    for issue in st.session_state["moneyball_model"].upload_issues:
        st.error(issue)
    if not work["issues"].empty:
        st.dataframe(work["issues"], hide_index=True, width="stretch")
    inventory = metric_inventory(pool)
    eligible = int(inventory["eligible"].sum()) if not inventory.empty else 0
    performance_rows = int(inventory["category"].isin(["performance", "raw total", "outcome/composite"]).sum()) if not inventory.empty else 0
    st.info(
        f"Metric learning sees {performance_rows} imported/derived performance fields; "
        f"{eligible} are currently eligible independent predictors. It tests them rather than assigning metrics to positions in advance."
    )
    st.caption(
        "Goals/90, goals conceded/90, clean sheets and undocumented FMST composite action scores remain visible in the inventory "
        "but are excluded as predictors to prevent outcome leakage or circular scoring. Raw count fields use per-90 derivatives where possible."
    )
    with st.expander("Metric inventory: every imported/derived field"):
        st.dataframe(display_frame(inventory), hide_index=True, width="stretch")
    table = work["league"]
    if {"team_id", "season", "league"}.issubset(table):
        latest = table[(table.league.astype(str) == str(pool.league.iloc[0]))].copy()
        if not latest.empty:
            latest = latest[latest.season.map(season_number).eq(latest.season.map(season_number).max())]
            unmatched = sorted(set(latest.team_id.map(normal)) - set(pool.team_id.map(normal)))
            if unmatched:
                st.warning("Latest-table club names without an exact player-export match: " + ", ".join(unmatched) + ". Confirm your club below; other clubs are not silently fuzzy-matched.")
    with st.expander("Inspect original imported data"):
        for label, key in (("League tables", "league"), ("League player statistics", "raw_pool"), ("My squad", "raw_squad")):
            st.write(label)
            st.dataframe(work[key], hide_index=True, width="stretch")


def render_baseline(work):
    model = st.session_state["moneyball_model"].league_model
    if model is None:
        return
    league = str(work["pool"].league.iloc[0])
    f = model.table[model.table.league.eq(league)]
    if f.empty:
        st.warning("No historical table matches the player pool's league.")
        return
    f = f[f.season.eq(f.season.max())]
    clubs = sorted(f.team_id.astype(str).unique()) if "team_id" in f else []
    own = str(work["squad"].team_id.iloc[0])
    match = next((c for c in clubs if normal(c) == normal(own)), "Not in this table")
    options = ["Not in this table", *clubs]
    club = st.selectbox("Confirm your club in the latest league table", options, index=options.index(match),
                        key=f"club_match_{work['revision']}")
    if club == "Not in this table":
        return
    row = f[f.team_id.eq(club)].iloc[0]
    st.write(f"Latest actual season: **{int(row.season)}/{str(int(row.season)+1)[-2:]}** · **{int(row.goals_for)} scored**, **{int(row.goals_against)} conceded**, **position {int(row.position)}**.")
    st.caption("Historical actuals are a reference point, not a forecast of the current squad.")


def render_assessment(work):
    st.subheader("Confirm your formation and player roles")
    formation = st.selectbox("Formation", list(FORMATION_PRESETS), key="guided_formation")
    minimum = st.number_input("Minimum minutes for a dependable player comparison", 90, 5000, 900, step=90, key="guided_minimum")
    percentile = st.slider(
        "Recruitment screening percentile", 25, 90, 60, step=5, key="guided_percentile",
        help="After the model learns which metrics matter, this sets how strong a candidate should be relative to positional peers.",
    )
    max_metrics = st.slider(
        "Learned metrics per position", 1, 8, 4, step=1, key="guided_learned_metric_count",
        help="The strongest out-of-sample, non-redundant metrics are selected. They are not treated as equally important.",
    )
    st.caption(
        "Starting assignments use the most-played players in each broad position group. "
        "Edit multifunctional players and confirm your intended starters."
    )
    revision = work["revision"]
    assignment = st.data_editor(
        default_assignments(work["squad"], formation),
        hide_index=True, width="stretch",
        disabled=["player_key", "player_name", "position", "minutes"],
        column_config={
            "player_key": None,
            "role_group": st.column_config.SelectboxColumn("Intended position group", options=list(ROLE_LABELS), required=True),
            "starter": st.column_config.CheckboxColumn("Intended starter", required=True),
        },
        key=f"assignments_{revision}_{formation}",
    )

    try:
        ranking = learn_metric_importance(
            work["pool"], work["league"], scope=work.get("scope"),
            minimum_role_minutes=max(450, min(int(minimum), 900)),
        )
        learned_metrics = select_learned_metrics(
            work["pool"], ranking, max_metrics=int(max_metrics),
            minimum_validation_gain=0.0, redundancy_threshold=0.85,
        )
    except DataError as exc:
        st.error("The app cannot learn a defensible recruitment focus from this export yet: " + str(exc))
        st.caption("No hard-coded positional fallback is used. Broaden the league player export or include the requested outcome fields.")
        return None

    st.subheader("Learned recruitment evidence")
    st.write(
        "Every eligible performance metric is tested for each position against both team scoring and team conceding. "
        "Importance is based on leave-one-club-out improvement over a league-average baseline; near-duplicate metrics are pruned."
    )
    st.caption(
        "A positive result is a predictive association in this save, not proof of causation. "
        "Metrics with validation gain at or below zero are not selected."
    )
    active_roles = list(FORMATION_PRESETS[formation])
    visible = ranking[ranking["role_group"].isin(active_roles)].copy()
    pairs = {(r, m) for r, ms in learned_metrics.items() for m in ms}
    visible["selected"] = [(str(r), str(m)) in pairs for r, m in zip(visible["role_group"], visible["metric"])]
    st.dataframe(display_frame(visible), hide_index=True, width="stretch")

    manual = st.checkbox(
        "Manually override the learned metric selection", value=False,
        key=f"manual_metric_override_{revision}_{formation}",
        help="Off by default. Manual choices replace only the screening list, not the learned evidence.",
    )
    metrics = {role: list(learned_metrics.get(role, [])) for role in active_roles}
    if manual:
        st.warning("Manual override is active. The ranking above remains the learned evidence.")
        all_metrics = available_metrics(work["pool"])
        for role in active_roles:
            metrics[role] = st.multiselect(
                f"Statistics for {ROLE_LABELS[role]}", all_metrics,
                default=metrics[role], key=f"manual_metrics_{revision}_{role}",
            )

    no_signal = [ROLE_LABELS[r] for r in active_roles if not metrics.get(r)]
    if no_signal:
        st.warning(
            "No metric produced positive out-of-sample evidence for: " + ", ".join(no_signal)
            + ". The app will not invent a profile for those positions."
        )

    focuses = {role: "Learned from uploaded league data" for role in active_roles}
    try:
        priorities, profile, squad_review = assess_squad(
            work["pool"], work["squad"], assignment,
            formation=formation, minimum_minutes=minimum,
            target_percentile=percentile, focuses=focuses,
            metrics=metrics, metric_evidence=ranking,
        )
    except DataError as exc:
        st.error(str(exc))
        return None

    st.subheader("Where to look for improvements")
    st.dataframe(display_frame(priorities), hide_index=True, width="stretch")
    st.caption(
        "Priority order uses missing starters first, then your starters' shortfall from the screening percentile "
        "on metrics that earned positive out-of-sample evidence."
    )
    st.subheader("Your players")
    st.dataframe(display_frame(squad_review), hide_index=True, width="stretch")
    st.subheader("What the learned model says to look for")
    if profile.empty:
        st.warning("No learned metric currently has enough positional peer data to create a stable screening threshold.")
    else:
        st.dataframe(display_frame(profile), hide_index=True, width="stretch")
        st.download_button("Download learned recruitment profiles", frame_to_csv_bytes(profile), "learned_recruitment_profiles.csv", "text/csv")
        st.download_button("Download all metric evidence", frame_to_csv_bytes(ranking), "learned_metric_importance.csv", "text/csv")
    st.download_button("Download squad review", frame_to_csv_bytes(squad_review), "squad_review.csv", "text/csv")
    assigned_squad = work["squad"].drop(columns="role_group").merge(
        assignment[["player_key", "role_group"]], on="player_key", validate="one_to_one"
    )
    return profile, assigned_squad, int(minimum)


def render_targets(work, assessment):
    st.subheader("Upload potential signings after reviewing your squad")
    uploaded = st.file_uploader("Potential signings", type=FORMATS, key="guided_candidates", accept_multiple_files=True)
    league = st.text_input("Candidates' league if absent from export", value=str(work["pool"].league.iloc[0]), key="candidate_league")
    st.caption("Upload one league at a time, then add another if needed. A file containing several leagues must identify each player's league. Cross-league statistics are not strength-adjusted.")
    if st.button("Add / update potential signings", disabled=not uploaded, key="add_targets"):
        try:
            batches, checks = [], []
            for source in uploaded:
                f = read_player_for_app(source, league=league, season=int(work["pool"].season.iloc[0]), ownership_mode="none")
                f, issues = clean_statistics(f)
                if int(f.season.iloc[0]) != int(work["pool"].season.iloc[0]):
                    raise DataError("Candidates must use the same statistics season as the assessment.")
                batches.append(f)
                checks.append(issues.assign(source=getattr(source, "name", "Targets")))
            new = pd.concat(batches, ignore_index=True)
            if new.player_key.duplicated().any():
                raise DataError("A candidate occurs in more than one uploaded file. Remove overlapping rows first.")
            existing = st.session_state.get("workflow_targets", new.iloc[:0])
            existing = existing[~existing.player_key.isin(new.player_key)]
            combined = pd.concat([existing, new], ignore_index=True)
            overlap = combined.player_key.isin(work["squad"].player_key)
            st.session_state["workflow_targets"] = combined[~overlap].copy()
            st.session_state["workflow_target_issues"] = pd.concat(checks, ignore_index=True)
            st.success(f"{len(combined) - int(overlap.sum())} targets loaded; {int(overlap.sum())} owned players excluded.")
        except (DataError, ValueError) as exc:
            st.error(str(exc))
    targets = st.session_state.get("workflow_targets")
    if targets is None or targets.empty:
        st.info("Your initial assessment does not require a target upload. Export promising candidates using the same statistics columns when ready.")
        return
    if st.button("Clear potential signings", key="clear_targets"):
        st.session_state.pop("workflow_targets", None)
        st.rerun()
    issues = st.session_state.get("workflow_target_issues")
    if issues is not None and not issues.empty:
        st.dataframe(issues, hide_index=True)
    digest = hashlib.sha256(targets.to_csv(index=False).encode()).hexdigest()[:12]
    edited = st.data_editor(targets[["player_key", "player_name", "team_id", "position", "role_group"]] if "position" in targets else targets[["player_key", "player_name", "team_id", "role_group"]],
        hide_index=True, disabled=["player_key", "player_name", "team_id", "position"],
        column_config={"player_key": None, "role_group": st.column_config.SelectboxColumn("Compare in position", options=list(ROLE_LABELS), required=True)},
        key=f"target_roles_{digest}")
    targets = targets.drop(columns="role_group").merge(edited[["player_key", "role_group"]], on="player_key", validate="one_to_one")
    if assessment is None:
        st.info("Resolve starter assignments to compare candidates with a recruitment profile.")
        return
    profile, squad, minimum = assessment
    role_options = profile["role_group"].dropna().astype(str).unique().tolist()
    if not role_options:
        st.info("No learned positional profile is available for target comparison.")
        return targets
    role = st.selectbox("Position to compare", role_options, format_func=ROLE_LABELS.get, key="compare_role")
    result, details = compare_targets(targets, squad, profile, role, minimum_minutes=minimum, league=str(work["pool"].league.iloc[0]))
    st.caption("Profile match is the fraction of screening thresholds met, not a football-value score. Missing metrics do not count as successes. Check minutes and league context before acting.")
    if result.empty:
        st.info("No supported profile or players for this position.")
    else:
        # Keep the evidence column here: league and minutes caveats must be visible.
        st.dataframe(display_frame(result.rename(columns={"evidence": "comparison_context"})), hide_index=True, width="stretch")
        st.dataframe(display_frame(details), hide_index=True, width="stretch")
        st.download_button("Download target comparison", frame_to_csv_bytes(result), "target_comparison.csv", "text/csv")
    return targets


def render_finances(work, assessment=None, comparison_targets=None):
    st.subheader("Enter costs for your squad and shortlisted targets")
    st.write("Own players: current weekly wage, remaining contract years, achievable sale proceeds. Targets: expected wage demand, total purchase fee including instalments, proposed contract years. Additional fees cover future agent/signing/exit fees and expected bonuses. Enter 0 only when genuinely zero.")
    targets = st.session_state.get("workflow_targets", work["squad"].iloc[:0]).copy()
    targets["owned"] = False
    players = pd.concat([work["squad"], targets], ignore_index=True).drop_duplicates("player_key")
    currency = st.selectbox("Currency for this comparison", ["GBP", "EUR", "USD"], key="cost_currency")
    horizon = st.number_input("Comparison horizon (years)", 0.5, 10.0, 3.0, step=0.5, key="cost_horizon")
    store_key = "manual_costs_" + work["namespace"]
    saved = st.session_state.get(store_key)
    uploaded = st.file_uploader("Restore saved financial entries", type=["csv"], key="restore_costs")
    if st.button("Import financial entries", disabled=uploaded is None, key="import_costs"):
        try:
            restored = read_table(clone_upload(uploaded))
            current = merge_costs(players, saved, currency)
            new = merge_costs(players, restored, currency)
            # Import only matching identities, preserving other session entries.
            keys = set(restored.player_key)
            merged = pd.concat([current[~current.player_key.isin(keys)], new[new.player_key.isin(keys)]], ignore_index=True)
            st.session_state[store_key] = merged
            saved = merged
            st.session_state[store_key + "_revision"] = st.session_state.get(store_key + "_revision", 0) + 1
            st.success(f"Restored {int(players.player_key.isin(keys).sum())} matching entries. Other save/club identities were not joined.")
        except (DataError, ValueError) as exc:
            st.error(str(exc))
    try:
        costs = merge_costs(players, saved, currency)
    except DataError as exc:
        st.error(str(exc))
        return
    digest = hashlib.sha256(("|".join(costs.player_key) + currency).encode()).hexdigest()[:10]
    with st.form("finance_form"):
        edited = st.data_editor(costs, hide_index=True, width="stretch",
            disabled=["player_key", "player_name", "team_id", "owned"],
            column_config={"player_key": None, "owned": st.column_config.CheckboxColumn("My player"),
                **{c: st.column_config.NumberColumn(c.replace("_", " ").title(), min_value=0.0) for c in COST_FIELDS},
                "currency": st.column_config.SelectboxColumn(options=["GBP", "EUR", "USD"], required=True)},
            key=f"finance_{work['namespace']}_{digest}_{st.session_state.get(store_key + '_revision', 0)}")
        if st.form_submit_button("Save financial entries"):
            try:
                costs = merge_costs(players, edited, currency)
                # Retain costs for temporarily removed candidates for subsequent re-import.
                archive = saved[~saved.player_key.isin(costs.player_key)] if saved is not None else costs.iloc[:0]
                st.session_state[store_key] = pd.concat([archive, costs], ignore_index=True)
                st.success("Saved for this session. Download below to retain a permanent copy.")
            except DataError as exc:
                st.error(str(exc))
    saved_all = st.session_state.get(store_key, costs)
    st.download_button("Download financial entries", frame_to_csv_bytes(saved_all), "manual_finances.csv", "text/csv")
    result = cost_comparison(costs, horizon=horizon, currency=currency)
    st.caption("Total cost = purchase fee (targets only) + 52 × weekly wage × horizon + additional fees. All figures must use the selected currency. Contracts shorter than the horizon need a shorter horizon or an explicit renewal assumption. No resale value is invented.")
    st.dataframe(display_frame(result), hide_index=True, width="stretch")
    owned, market = result[result.owned], result[~result.owned]
    if assessment is not None and comparison_targets is not None:
        profile, squad, minimum = assessment
        role = st.session_state.get("compare_role", "ST")
        football, _ = compare_targets(comparison_targets, squad, profile, role,
                                     minimum_minutes=minimum, league=str(work["pool"].league.iloc[0]))
        if not football.empty:
            combined = football.merge(result[["player_key", "total_cost", "currency", "cost_status"]],
                                      on="player_key", validate="one_to_one", how="left")
            st.subheader(f"Football profile and costs: {ROLE_LABELS[role]}")
            st.dataframe(display_frame(combined.rename(columns={"evidence": "comparison_context"})), hide_index=True, width="stretch")
            st.download_button("Download football and cost comparison", frame_to_csv_bytes(combined), "recruitment_cost_comparison.csv", "text/csv")
            owned = owned[owned.player_key.isin(football.player_key)]
            market = market[market.player_key.isin(football.player_key)]
    if len(owned) and len(market):
        names = result.set_index("player_key").player_name.to_dict()
        outgoing = st.selectbox("Player you would sell", owned.player_key.tolist(), format_func=names.get, key="cost_outgoing")
        incoming = st.selectbox("Player you would sign", market.player_key.tolist(), format_func=names.get, key="cost_incoming")
        delta = replacement_cost(result, outgoing, incoming)
        if delta is None:
            st.info("Complete both players' horizon costs and the outgoing player's sale proceeds to compare replacing versus keeping.")
        else:
            st.metric(f"Extra cost of replacing vs keeping ({currency})", f"{delta:,.0f}")
            st.caption("Negative means savings over the selected horizon. Check the position-specific football comparison above; this financial scenario does not predict goals or automatically recommend a sale. Transfer and wage budget timing must also fit your club.")


def render_workflow(render_target):
    render_uploads()
    st.title("FM26 Moneyball Recruitment Lab")
    st.write("Assess your squad against the league, define recruitment profiles, then compare possible signings and their real costs.")
    work = st.session_state.get("workflow")
    if work is None:
        st.info("Start with league_table.csv, league_statistics.csv and my_team.csv in the sidebar. Potential signings come later.")
        return
    st.session_state["moneyball_model"] = work["model"]
    setup, season, squad, targets, costs = st.tabs(["1. Import checks", "2. Season target", "3. Squad assessment", "4. Potential signings", "5. Costs & replacements"])
    with setup:
        render_checks(work)
    with season:
        render_baseline(work)
        render_target()
    with squad:
        assessment = render_assessment(work)
    with targets:
        comparison_targets = render_targets(work, assessment)
    with costs:
        render_finances(work, assessment, comparison_targets)
