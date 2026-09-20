from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fm_model import DATA_CATALOGUE, DataError, FORMATION_PRESETS, ROLE_LABELS, MoneyballModel, read_table
from fm_model.collection import collection_plan
from fm_model.app_support import (
    OWNERSHIP_MODES,
    build_model,
    clone_upload,
    frame_to_csv_bytes,
    load_example_frames,
    merge_table_uploads,
    read_player_for_app,
    read_player_history_for_app,
    serialise_model,
)


st.set_page_config(
    page_title="FM26 Moneyball Recruitment Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_KEY = "moneyball_model"
FRAMES_KEY = "model_frames"
DECISIONS_KEY = "player_decisions"
SOURCE_KEY = "model_source"
SCENARIO_KEY = "target_scenario"
ROUTE_KEY = "driver_route"
SQUAD_PLAN_KEY = "squad_plan"


def _fmt(value: Any, digits=1):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:,.0f}"


def _ordinal(value):
    value = int(value)
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _fmt_money(value: Any, symbol="£"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if abs(number) >= 1_000_000:
        return f"{symbol}{number / 1_000_000:,.2f}m"
    if abs(number) >= 1_000:
        return f"{symbol}{number / 1_000:,.0f}k"
    return f"{symbol}{number:,.0f}"


def _clear_state():
    for key in (MODEL_KEY, FRAMES_KEY, DECISIONS_KEY, SOURCE_KEY, SCENARIO_KEY, ROUTE_KEY, SQUAD_PLAN_KEY):
        st.session_state.pop(key, None)


def _store_model(model: MoneyballModel, frames: dict[str, pd.DataFrame | None], source: str):
    st.session_state[MODEL_KEY] = model
    st.session_state[FRAMES_KEY] = frames
    st.session_state[SOURCE_KEY] = source
    st.session_state.pop(SCENARIO_KEY, None)
    st.session_state.pop(ROUTE_KEY, None)
    st.session_state.pop(SQUAD_PLAN_KEY, None)

    decisions = None
    players = frames.get("players")
    if model.player_model is not None and players is not None:
        decisions = model.player_decisions(players)
    st.session_state[DECISIONS_KEY] = decisions


def _infer_single_league(frame: pd.DataFrame | None):
    if frame is None or "league" not in frame:
        return None
    values = frame["league"].dropna().astype(str).str.strip()
    unique = values[values.ne("")].unique().tolist()
    return unique[0] if len(unique) == 1 else None


def _build_from_sidebar(
    league_uploads,
    team_uploads,
    historical_player_uploads,
    player_upload,
    *,
    player_league,
    player_season,
    ownership_mode,
    owned_club,
    range_policy,
    include_all_available,
):
    league_frame = merge_table_uploads(league_uploads) if league_uploads else None
    team_frame = merge_table_uploads(team_uploads) if team_uploads else None

    league_context = player_league.strip() or _infer_single_league(league_frame)
    historical_player_frame = read_player_history_for_app(
        historical_player_uploads,
        league=league_context,
        range_policy=range_policy,
    ) if historical_player_uploads else None
    player_frame = None
    if player_upload:
        player_frame = read_player_for_app(
            clone_upload(player_upload),
            league=league_context,
            season=int(player_season),
            ownership_mode=ownership_mode,
            owned_club=owned_club.strip() or None,
            range_policy=range_policy,
        )

    model = build_model(
        league_data=league_frame,
        team_data=team_frame,
        historical_player_data=historical_player_frame,
        player_data=player_frame,
        include_all_available=include_all_available,
    )
    _store_model(
        model,
        {"league": league_frame, "team": team_frame, "player_history": historical_player_frame, "players": player_frame},
        "Uploaded FM26 data",
    )


def _load_examples():
    frames = load_example_frames()
    model = build_model(
        league_data=frames["league"],
        team_data=frames["team"],
        historical_player_data=frames.get("player_history"),
        player_data=frames["players"],
    )
    _store_model(model, frames, "Synthetic example data")


def _readiness_label(model, key):
    ready = model.readiness()[key]
    return "Ready" if ready else "Not loaded"


def _current_model():
    return st.session_state.get(MODEL_KEY)


def _current_frames():
    return st.session_state.get(FRAMES_KEY, {})


def _current_decisions():
    return st.session_state.get(DECISIONS_KEY)


def _download_json(label, payload, filename, key):
    st.download_button(
        label,
        data=json.dumps(payload, default=float, indent=2).encode("utf-8"),
        file_name=filename,
        mime="application/json",
        key=key,
    )


def render_sidebar():
    with st.sidebar:
        st.title("FM26 Moneyball")
        st.caption("Build the model from your save exports, then use the dashboard to set targets and find mispriced players.")

        c1, c2 = st.columns(2)
        if c1.button("Load example data", key="load_examples", width="stretch"):
            try:
                _load_examples()
                st.success("Example model loaded.")
            except DataError as exc:
                st.error(str(exc))
        if c2.button("Clear", key="clear_model", width="stretch"):
            _clear_state()
            st.rerun()

        st.divider()
        st.subheader("1. Upload data")
        league_upload = st.file_uploader(
            "League table history",
            type=["csv", "tsv", "html", "htm", "xlsx", "xlsm"],
            key="league_upload",
            accept_multiple_files=True,
            help="Upload several completed seasons at once. Each file needs a season column, or a filename such as league_2025_26.csv.",
        )
        team_upload = st.file_uploader(
            "Team performance history",
            type=["csv", "tsv", "html", "htm", "xlsx", "xlsm"],
            key="team_upload",
            accept_multiple_files=True,
            help="Upload all clubs for several seasons: xG, shots, chance creation, pressing, defensive and goalkeeping metrics.",
        )
        historical_player_upload = st.file_uploader(
            "Historical player-season exports",
            type=["csv", "tsv", "html", "htm", "xlsx", "xlsm"],
            key="historical_player_upload",
            accept_multiple_files=True,
            help="All players, all clubs, several completed seasons. This is what lets the model learn which attributes predict player outcomes.",
        )
        player_upload = st.file_uploader(
            "Player pool / FMST26 export",
            type=["csv", "tsv", "html", "htm", "xlsx", "xlsm"],
            key="player_upload",
            help="The broadest relevant player pool is best; include price, minutes and role/position.",
        )

        with st.expander("Player export context", expanded=bool(player_upload)):
            player_league = st.text_input(
                "League if the export omits it",
                key="player_league_context",
                help="Leave blank when the export already includes league, or when your uploaded league history has exactly one league.",
            )
            player_season = st.number_input(
                "Save season",
                min_value=1900,
                max_value=2200,
                value=2026,
                step=1,
                key="player_season_context",
            )
            ownership_label = st.selectbox(
                "Ownership handling",
                options=list(OWNERSHIP_MODES.values()),
                index=0,
                key="ownership_label",
            )
            label_to_mode = {label: mode for mode, label in OWNERSHIP_MODES.items()}
            ownership_mode = label_to_mode[ownership_label]
            owned_club = st.text_input(
                "Your club name",
                key="owned_club_input",
                help="Used when ownership is inferred from the player's club column.",
            )
            range_policy = st.selectbox(
                "Displayed value ranges",
                ["error", "lower", "midpoint", "upper"],
                index=0,
                key="range_policy",
                help="The model never silently averages a displayed range. Choose how ranges should be resolved, or leave error to force explicit cleanup.",
            )

        include_all_available = st.checkbox(
            "Let the goal-driver model test all numeric team metrics",
            value=False,
            key="include_all_available",
            help="Protected outcome fields are still excluded. Leave this off for the most interpretable model.",
        )

        has_upload = bool(league_upload or team_upload or historical_player_upload or player_upload)
        if st.button(
            "Build / refresh model",
            type="primary",
            disabled=not has_upload,
            key="build_model",
            width="stretch",
        ):
            try:
                with st.spinner("Fitting model layers..."):
                    _build_from_sidebar(
                        league_upload,
                        team_upload,
                        historical_player_upload,
                        player_upload,
                        player_league=player_league,
                        player_season=player_season,
                        ownership_mode=ownership_mode,
                        owned_club=owned_club,
                        range_policy=range_policy,
                        include_all_available=include_all_available,
                    )
                st.success("Model built.")
            except (DataError, ValueError) as exc:
                st.error(str(exc))

        model = _current_model()
        if model is not None:
            st.divider()
            st.subheader("Current model")
            st.caption(st.session_state.get(SOURCE_KEY, "Loaded data"))
            readiness = model.readiness()
            st.write(
                f"League targets: {'✅' if readiness['league_targets'] else '—'}  "
                f"Goal drivers: {'✅' if readiness['goal_drivers'] else '—'}  "
                f"Players: {'✅' if readiness['player_values'] else '—'}  "
                f"Attributes learned: {'✅' if readiness['attribute_outcomes'] else '—'}"
            )


def render_header():
    st.title("FM26 Moneyball Recruitment Lab")
    st.write(
        "Use your save's own data to estimate the goal profile associated with a target finish, "
        "learn which team processes predict those goals, and identify players whose market price "
        "looks different from their role-relative contribution."
    )
    st.caption(
        "Model outputs are predictive evidence from your FM universe, not guaranteed results or causal football laws."
    )

    model = _current_model()
    if model is None:
        st.info("Start in the sidebar: upload your exports or load the synthetic example data.")
        return

    readiness = model.readiness()
    frames = _current_frames()
    cols = st.columns(4)
    cols[0].metric("League target model", _readiness_label(model, "league_targets"))
    cols[1].metric("Goal-driver model", _readiness_label(model, "goal_drivers"))
    cols[2].metric("Player valuation model", _readiness_label(model, "player_values"))
    player_rows = len(frames.get("players", [])) if frames.get("players") is not None else 0
    cols[3].metric("Players loaded", f"{player_rows:,}")
    st.caption(f"Source: {st.session_state.get(SOURCE_KEY, 'Loaded data')} · Next step: {readiness['next_step']}")



def render_collection_wizard():
    st.subheader("Data Collection Wizard")
    st.write(
        "Use this before exporting anything. The model is designed to collect broadly, then learn what matters "
        "instead of asking you to guess which attributes or statistics are important."
    )
    seasons = st.number_input(
        "Historical seasons to collect",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
        key="collection_seasons",
    )
    plan = collection_plan(historical_seasons=int(seasons))

    st.info(
        "**Club scope:** historical league tables, team metrics and player history should cover **every club in the league**, "
        "not just your club or the strongest sides. The current-market file should contain your full squad plus the broadest "
        "set of players you could realistically sign."
    )

    section_labels = {
        "league_tables": "1. League tables",
        "team_history": "2. Team performance history",
        "historical_players": "3. Historical player seasons",
        "current_market": "4. Current squad + transfer market",
    }
    manifest_rows = []
    for key, title in section_labels.items():
        item = plan[key]
        with st.expander(title, expanded=(key == "historical_players")):
            st.write(f"**Seasons:** {item['seasons']}")
            st.write(f"**Clubs:** {item['clubs']}")
            if isinstance(item.get("positions"), dict):
                position_table = pd.DataFrame(
                    [{"Position group": code, "Include": scope} for code, scope in item["positions"].items()]
                )
                st.dataframe(position_table, width="stretch", hide_index=True)
            elif item.get("positions"):
                st.write(f"**Positions:** {item['positions']}")
            st.write(f"**Why:** {item['purpose']}")

            field_groups = [
                ("Required / identity", item.get("required_fields") or item.get("identity_fields") or item.get("fields", [])),
                ("Attacking", item.get("attacking_fields", [])),
                ("Defensive", item.get("defensive_fields", [])),
                ("Performance", item.get("performance_fields", [])),
                ("Attributes", item.get("attribute_fields", [])),
                ("Market", item.get("market_fields", [])),
            ]
            for group_name, fields in field_groups:
                if not fields:
                    continue
                st.markdown(f"**{group_name} columns**")
                st.code(", ".join(fields), language=None)
                for field in fields:
                    manifest_rows.append({"File": title, "Group": group_name, "Column": field})

    st.warning(
        "**Do not feed CA, PA, price, wages or reputation into the football-performance models.** "
        "They are kept separate for valuation/validation so the model has to discover performance value from observable football data."
    )
    manifest = pd.DataFrame(manifest_rows).drop_duplicates()
    st.download_button(
        "Download complete export-column checklist",
        frame_to_csv_bytes(manifest),
        "fm26_data_collection_checklist.csv",
        "text/csv",
        key="download_collection_manifest",
    )


def render_setup():
    st.header("Setup & readiness")
    render_collection_wizard()
    st.divider()
    model = _current_model()
    frames = _current_frames()

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recommended save workflow")
        st.markdown(
            """
1. Export several completed league tables.
2. Export the matching team-performance history.
3. Export the broad player pool from FMST26, including minutes, role/position, value and wages where available.
4. Build the model in the sidebar.
5. Set a finishing target, inspect the learned drivers, then use Recruitment to search the market.
            """
        )
    with right:
        st.subheader("Minimum useful evidence")
        st.write("**League target:** at least two completed seasons.")
        st.write("**Goal drivers:** at least 24 team observations across at least two seasons.")
        st.write("**Recruitment:** broad role groups with meaningful minutes and market prices.")
        st.write("**Best version:** historical player seasons linked to team seasons.")

    if model is None:
        st.warning("No model is loaded yet.")
    else:
        readiness = model.readiness()
        ready_table = pd.DataFrame(
            [
                {"Layer": "League targets", "Status": "Ready" if readiness["league_targets"] else "Not loaded"},
                {"Layer": "Goal drivers", "Status": "Ready" if readiness["goal_drivers"] else "Not loaded"},
                {"Layer": "Historical attribute outcomes", "Status": "Ready" if readiness["attribute_outcomes"] else "Not loaded"},
                {"Layer": "Player valuation", "Status": "Ready" if readiness["player_values"] else "Not loaded"},
                {"Layer": "Market decisions", "Status": "Ready" if readiness["market_decisions"] else "Not loaded"},
                {"Layer": "End-to-end squad plan", "Status": "Ready" if readiness["squad_plan"] else "Not loaded"},
            ]
        )
        st.dataframe(ready_table, width="stretch", hide_index=True)

    st.subheader("Loaded data")
    if not frames:
        st.caption("Nothing loaded.")
    for key, label in (
        ("league", "League tables"),
        ("team", "Team metrics"),
        ("player_history", "Historical player seasons"),
        ("players", "Current player pool"),
    ):
        frame = frames.get(key)
        if frame is None:
            continue
        with st.expander(f"{label}: {len(frame):,} rows · {len(frame.columns)} columns"):
            st.dataframe(frame.head(50), width="stretch", hide_index=True)

    st.subheader("Example templates")
    try:
        examples = load_example_frames()
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button(
            "League table example",
            frame_to_csv_bytes(examples["league"]),
            "league_table.csv",
            "text/csv",
            key="download_league_template",
            width="stretch",
        )
        c2.download_button(
            "Team metrics example",
            frame_to_csv_bytes(examples["team"]),
            "team_metrics.csv",
            "text/csv",
            key="download_team_template",
            width="stretch",
        )
        c3.download_button(
            "Player history example",
            frame_to_csv_bytes(examples["player_history"]),
            "player_history.csv",
            "text/csv",
            key="download_player_history_template",
            width="stretch",
        )
        c4.download_button(
            "Current market example",
            frame_to_csv_bytes(examples["players"]),
            "player_export.csv",
            "text/csv",
            key="download_player_template",
            width="stretch",
        )
    except Exception:
        st.caption("Example files are unavailable in this environment.")

    with st.expander("Recognised data catalogue"):
        st.json(DATA_CATALOGUE)


def render_target():
    st.header("Season target")
    model = _current_model()
    if model is None or model.league_model is None:
        st.info("Upload league-table history first.")
        return

    league_model = model.league_model
    league = st.selectbox("League", league_model.leagues, key="target_league")
    try:
        default_context = league_model.context(league)
    except DataError as exc:
        st.error(str(exc))
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        target_position = st.number_input(
            "Target finish",
            min_value=1,
            max_value=int(default_context["n_teams"]),
            value=min(6, int(default_context["n_teams"])),
            step=1,
            key=f"target_position_{league}",
        )
    with c2:
        probability = st.slider(
            "Evidence threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.70,
            step=0.05,
            key="target_probability",
            help="Empirical estimated probability threshold used to construct the goal frontier.",
        )
    with c3:
        st.metric("Forecast season", default_context["season"])
        st.caption(
            f"{default_context['matches']} matches · {default_context['n_teams']} teams · "
            f"{_fmt(default_context['goals_per_team_match'], 2)} goals/team-match"
        )

    custom = st.checkbox("Override next-season context", value=False, key="custom_context")
    matches = n_teams = next_season = None
    if custom:
        a, b, c = st.columns(3)
        matches = a.number_input(
            "Matches",
            min_value=1,
            max_value=100,
            value=int(default_context["matches"]),
            step=1,
            key="custom_matches",
        )
        n_teams = b.number_input(
            "Teams",
            min_value=4,
            max_value=100,
            value=int(default_context["n_teams"]),
            step=1,
            key="custom_teams",
        )
        next_season = c.number_input(
            "Next season start year",
            min_value=int(default_context["season"]),
            max_value=2200,
            value=int(default_context["season"]),
            step=1,
            key="custom_next_season",
        )

    if st.button("Calculate target", type="primary", key="calculate_target"):
        try:
            with st.spinner("Searching the non-dominated goal frontier..."):
                scenario = model.target_scenario(
                    league,
                    int(target_position),
                    probability=float(probability),
                    matches=int(matches) if matches is not None else None,
                    n_teams=int(n_teams) if n_teams is not None else None,
                    next_season=int(next_season) if next_season is not None else None,
                )
            st.session_state[SCENARIO_KEY] = scenario
            st.session_state.pop(ROUTE_KEY, None)
        except DataError as exc:
            st.error(str(exc))

    scenario = st.session_state.get(SCENARIO_KEY)
    if not scenario or scenario.get("context", {}).get("league") != str(league):
        st.caption("Calculate a target to see the recommended goal pair and frontier.")
        return

    recommended = scenario["recommended"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score at least", f"{_fmt_int(recommended.get('goals_for'))} GF")
    m2.metric("Concede at most", f"{_fmt_int(recommended.get('goals_against'))} GA")
    m3.metric("Modelled position", _fmt(recommended.get("predicted_position"), 1))
    if "predicted_points" in recommended:
        m4.metric("Modelled points", _fmt(recommended.get("predicted_points"), 1))
    else:
        m4.metric("Estimated target probability", f"{100 * float(recommended.get('estimated_target_probability', np.nan)):.0f}%")

    st.caption(
        f"Requested empirical threshold: {scenario['requested_probability']:.0%}. "
        "GF and GA must be taken from the same frontier pair."
    )

    frontier = pd.DataFrame(scenario["frontier"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["goals_for"],
            y=frontier["goals_against"],
            mode="lines+markers",
            name="Target frontier",
            customdata=np.column_stack(
                [
                    frontier.get("estimated_target_probability", pd.Series(np.nan, index=frontier.index)),
                    frontier.get("predicted_position", pd.Series(np.nan, index=frontier.index)),
                ]
            ),
            hovertemplate="GF %{x}<br>GA %{y}<br>Probability %{customdata[0]:.1%}<br>Position %{customdata[1]:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[recommended["goals_for"]],
            y=[recommended["goals_against"]],
            mode="markers",
            marker={"size": 16, "symbol": "star"},
            name="Balanced recommendation",
        )
    )
    fig.update_layout(
        xaxis_title="Goals for — minimum",
        yaxis_title="Goals against — maximum",
        legend_title=None,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", key="frontier_chart")

    with st.expander("Frontier table"):
        st.dataframe(frontier, width="stretch", hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download frontier CSV",
        frame_to_csv_bytes(frontier),
        "goal_target_frontier.csv",
        "text/csv",
        key="download_frontier",
        width="stretch",
    )
    d2.download_button(
        "Download target JSON",
        json.dumps(scenario, default=float, indent=2).encode("utf-8"),
        "goal_target_scenario.json",
        "application/json",
        key="download_scenario",
        width="stretch",
    )


def _driver_validation_table(model):
    rows = []
    for direction in ("attack", "defence"):
        report = model.driver_model.driver_report(direction)["model_report"]
        rows.append(
            {
                "Direction": direction.title(),
                "Holdout season": report.get("test_season"),
                "MAE / match": report.get("mae"),
                "Baseline MAE / match": report.get("baseline_mae"),
                "RMSE / match": report.get("rmse"),
                "Beats baseline": report.get("beats_baseline"),
                "Training rows": report.get("training_rows"),
            }
        )
    return pd.DataFrame(rows)


def render_drivers():
    st.header("Goal drivers")
    model = _current_model()
    if model is None or model.driver_model is None:
        st.info("Upload team performance history to learn attacking and defensive goal drivers.")
        return

    st.dataframe(_driver_validation_table(model), width="stretch", hide_index=True)
    st.caption("Driver effects are predictive associations. Correlated metrics can act as proxies for one another.")

    direction = st.radio(
        "Direction",
        ["attack", "defence"],
        horizontal=True,
        key="driver_direction",
        format_func=lambda x: "Attack / goals for" if x == "attack" else "Defence / goals against",
    )
    report = model.driver_model.driver_report(direction)
    driver_df = pd.DataFrame(report["drivers"])

    if driver_df.empty:
        st.warning("No driver effects are available.")
    else:
        chart_df = driver_df.sort_values("benefit", ascending=True).tail(20)
        fig = px.bar(
            chart_df,
            x="benefit",
            y="feature",
            orientation="h",
            hover_data=["predicted_rate_change", "validation_mae_increase"],
            labels={
                "benefit": "Predicted benefit from +1 training SD",
                "feature": "Metric",
                "predicted_rate_change": "Predicted rate change",
                "validation_mae_increase": "Validation importance",
            },
        )
        fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
        st.plotly_chart(fig, width="stretch", key=f"driver_chart_{direction}")
        st.dataframe(
            driver_df[
                ["feature", "benefit", "predicted_rate_change", "validation_mae_increase", "one_training_sd_change"]
            ],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Route from a current team profile to the target")
    scenario = st.session_state.get(SCENARIO_KEY)
    raw = getattr(model.driver_model, "raw_frame", None)
    if not scenario:
        st.info("Calculate a Season target first. The route optimiser then searches within historically observed metric ranges.")
        return
    if raw is None or "team_id" not in raw.columns:
        st.info("The team export needs a team/club identifier to choose a baseline team.")
        return

    candidate = raw.copy()
    if "league" in candidate and scenario["context"].get("league") is not None:
        same = candidate["league"].astype(str).eq(str(scenario["context"]["league"]))
        if same.any():
            candidate = candidate[same]
    teams = sorted(candidate["team_id"].dropna().astype(str).unique().tolist())
    if not teams:
        st.info("No baseline teams are available.")
        return

    preferred = st.session_state.get("owned_club_input", "")
    default_idx = teams.index(preferred) if preferred in teams else 0
    team = st.selectbox("Baseline team", teams, index=default_idx, key="route_team")
    subset = candidate[candidate["team_id"].astype(str).eq(str(team))].copy()
    if "season" in subset.columns:
        subset = subset.sort_values("season")
    baseline = subset.tail(1)

    if st.button("Calculate improvement route", key="calculate_route"):
        try:
            rec = scenario["recommended"]
            route = model.driver_model.route(
                baseline,
                target_goals_for=rec["goals_for"],
                target_goals_against=rec["goals_against"],
                matches=scenario["context"]["matches"],
            )
            st.session_state[ROUTE_KEY] = {"team": team, "route": route}
        except DataError as exc:
            st.error(str(exc))

    route_state = st.session_state.get(ROUTE_KEY)
    if not route_state or route_state.get("team") != team:
        return

    for side in ("attack", "defence"):
        route = route_state["route"].get(side)
        if not route:
            continue
        label = "Attack" if side == "attack" else "Defence"
        if route["feasible"]:
            st.success(
                f"{label}: in-range route found · projected {_fmt(route['predicted_goals'], 1)} "
                f"vs target {_fmt(route['target_goals'], 1)}."
            )
        else:
            st.warning(
                f"{label}: target is not reachable inside the historical metric envelope; "
                f"showing the best in-range attempt ({_fmt(route['predicted_goals'], 1)} goals)."
            )
        changes = pd.DataFrame(route["changes"])
        changes = changes[changes["standardised_change"].abs() > 0.01]
        if changes.empty:
            st.caption(f"{label}: no material metric movement is required.")
            continue
        fig = px.bar(
            changes.sort_values("standardised_change"),
            x="standardised_change",
            y="feature",
            orientation="h",
            labels={"standardised_change": "Required change (training SD)", "feature": "Metric"},
        )
        fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
        st.plotly_chart(fig, width="stretch", key=f"route_chart_{side}")
        st.dataframe(
            changes[
                ["feature", "baseline", "target_value", "change", "standardised_change", "training_min", "training_max"]
            ],
            width="stretch",
            hide_index=True,
        )


def _filter_recruitment(frame):
    filtered = frame.copy()

    search = st.text_input("Search player / club", key="player_search").strip().casefold()
    if search:
        name = filtered.get("player_name", pd.Series("", index=filtered.index)).astype(str).str.casefold()
        club = filtered.get("team_id", pd.Series("", index=filtered.index)).astype(str).str.casefold()
        filtered = filtered[name.str.contains(search, regex=False) | club.str.contains(search, regex=False)]

    roles = sorted(frame.get("role_group", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    leagues = sorted(frame.get("league", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    decisions = sorted(frame.get("decision", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

    c1, c2, c3 = st.columns(3)
    selected_roles = c1.multiselect("Roles", roles, default=roles, key="filter_roles")
    selected_leagues = c2.multiselect("Leagues", leagues, default=leagues, key="filter_leagues")
    selected_decisions = c3.multiselect("Decisions", decisions, default=decisions, key="filter_decisions")

    if roles:
        filtered = filtered[filtered["role_group"].astype(str).isin(selected_roles)]
    if leagues:
        filtered = filtered[filtered["league"].astype(str).isin(selected_leagues)]
    if decisions:
        filtered = filtered[filtered["decision"].astype(str).isin(selected_decisions)]

    c4, c5, c6 = st.columns(3)
    max_minutes = int(pd.to_numeric(frame.get("minutes"), errors="coerce").fillna(0).max()) if "minutes" in frame else 0
    min_minutes = c4.slider(
        "Minimum minutes",
        min_value=0,
        max_value=max(max_minutes, 1),
        value=0,
        step=90 if max_minutes >= 90 else 1,
        key="filter_minutes",
    )
    filtered = filtered[pd.to_numeric(filtered.get("minutes"), errors="coerce").fillna(0) >= min_minutes]

    max_value_m = c5.number_input(
        "Maximum market value (millions, 0 = no cap)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="filter_max_value",
    )
    if max_value_m > 0 and "market_price" in filtered:
        filtered = filtered[
            pd.to_numeric(filtered["market_price"], errors="coerce").fillna(np.inf) <= max_value_m * 1_000_000
        ]

    ownership = c6.selectbox("Ownership", ["All", "Market only", "My squad only"], key="filter_ownership")
    if "owned" in filtered:
        if ownership == "Market only":
            filtered = filtered[~filtered["owned"].astype(bool)]
        elif ownership == "My squad only":
            filtered = filtered[filtered["owned"].astype(bool)]

    if "age" in frame:
        ages = pd.to_numeric(frame["age"], errors="coerce").dropna()
        if not ages.empty:
            amin, amax = int(ages.min()), int(ages.max())
            selected_age = st.slider("Age", amin, amax, (amin, amax), key="filter_age")
            fa = pd.to_numeric(filtered["age"], errors="coerce")
            filtered = filtered[fa.between(*selected_age) | fa.isna()]

    confidence = st.selectbox("Decision confidence", ["All", "Higher only"], key="filter_confidence")
    if confidence == "Higher only" and "decision_confidence" in filtered:
        filtered = filtered[filtered["decision_confidence"].eq("higher")]

    return filtered


def render_recruitment():
    st.header("Recruitment")
    model = _current_model()
    if model is None or model.player_model is None:
        st.info("Upload a player pool to build role-relative contribution and market comparisons.")
        return

    decisions = _current_decisions()
    if decisions is None:
        players = _current_frames().get("players")
        if players is None:
            st.info("No player rows are available.")
            return
        try:
            decisions = model.player_decisions(players)
            st.session_state[DECISIONS_KEY] = decisions
        except DataError as exc:
            st.error(str(exc))
            return

    frame = decisions.copy()
    frame["value_edge"] = frame["expected_market_price_for_contribution"] - frame["market_price"]
    frame["value_edge_pct"] = (
        frame["expected_market_price_for_contribution"] / frame["market_price"].replace(0, np.nan) - 1
    )

    filtered = _filter_recruitment(frame)
    sort_options = {
        "Largest estimated value edge": ("value_edge", False),
        "Highest contribution": ("football_contribution_units", False),
        "Highest above replacement": ("above_role_replacement", False),
        "Lowest market price": ("market_price", True),
        "Player name": ("player_name", True),
    }
    sort_label = st.selectbox("Sort", list(sort_options), key="recruitment_sort")
    sort_col, ascending = sort_options[sort_label]
    if sort_col in filtered:
        filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players shown", f"{len(filtered):,}")
    m2.metric("BUY signals", f"{int((filtered.get('decision') == 'BUY').sum()) if 'decision' in filtered else 0:,}")
    finite_edge = pd.to_numeric(filtered.get("value_edge"), errors="coerce").dropna()
    m3.metric("Best estimated value edge", _fmt_money(finite_edge.max()) if not finite_edge.empty else "—")
    high_conf = int(filtered.get("decision_confidence", pd.Series(index=filtered.index, dtype=str)).eq("higher").sum())
    m4.metric("Higher-confidence decisions", f"{high_conf:,}")

    st.caption(
        "Expected market price is the median price of nearby contribution comparables in this save. "
        "It is not an external transfer-value forecast."
    )

    plot = filtered[
        np.isfinite(pd.to_numeric(filtered.get("market_price"), errors="coerce"))
        & np.isfinite(pd.to_numeric(filtered.get("expected_market_price_for_contribution"), errors="coerce"))
    ].copy()
    if len(plot) >= 2:
        fig = px.scatter(
            plot,
            x="market_price",
            y="expected_market_price_for_contribution",
            color="decision",
            hover_name="player_name" if "player_name" in plot else None,
            hover_data=[c for c in ["role_group", "team_id", "football_contribution_units", "decision_confidence"] if c in plot],
            labels={
                "market_price": "Current market price",
                "expected_market_price_for_contribution": "Comparable price for contribution",
            },
        )
        lo = float(min(plot["market_price"].min(), plot["expected_market_price_for_contribution"].min()))
        hi = float(max(plot["market_price"].max(), plot["expected_market_price_for_contribution"].max()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dash"})
        fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
        st.plotly_chart(fig, width="stretch", key="valuation_scatter")

    display_cols = [
        c
        for c in [
            "player_name",
            "team_id",
            "league",
            "role_group",
            "age",
            "minutes",
            "market_price",
            "expected_market_price_for_contribution",
            "value_edge",
            "price_vs_comparables",
            "football_contribution_units",
            "above_role_replacement",
            "approx_goals_for_impact",
            "approx_goals_against_reduction",
            "decision",
            "decision_confidence",
        ]
        if c in filtered.columns
    ]
    display = filtered[display_cols].copy()
    st.dataframe(
        display,
        width="stretch",
        height=520,
        hide_index=True,
        column_config={
            "market_price": st.column_config.NumberColumn("Market price", format="%.0f"),
            "expected_market_price_for_contribution": st.column_config.NumberColumn(
                "Comparable price", format="%.0f"
            ),
            "value_edge": st.column_config.NumberColumn("Estimated value edge", format="%.0f"),
            "price_vs_comparables": st.column_config.NumberColumn("Price / comparable", format="%.2f"),
            "football_contribution_units": st.column_config.NumberColumn("Contribution", format="%.2f"),
            "above_role_replacement": st.column_config.NumberColumn("Above replacement", format="%.2f"),
            "approx_goals_for_impact": st.column_config.NumberColumn("Approx GF impact", format="%.2f"),
            "approx_goals_against_reduction": st.column_config.NumberColumn("Approx GA change", format="%.2f"),
        },
    )

    st.download_button(
        "Download filtered recruitment CSV",
        frame_to_csv_bytes(filtered),
        "fm26_recruitment_results.csv",
        "text/csv",
        key="download_recruitment",
    )

    if filtered.empty:
        st.info("No players match the current filters.")
        return

    st.subheader("Player explanation")
    options = filtered.index.tolist()

    def player_label(index):
        row = filtered.loc[index]
        return " · ".join(
            [
                str(row.get("player_name", index)),
                str(row.get("role_group", "")),
                str(row.get("team_id", "")),
            ]
        )

    selected = st.selectbox("Player", options, format_func=player_label, key="selected_player")
    row = filtered.loc[selected]
    st.write(f"**Decision:** {row.get('decision', '—')} — {row.get('decision_reason', '')}")
    a, b, c, d = st.columns(4)
    a.metric("Market price", _fmt_money(row.get("market_price")))
    b.metric("Comparable price", _fmt_money(row.get("expected_market_price_for_contribution")))
    c.metric("Contribution units", _fmt(row.get("football_contribution_units"), 2))
    d.metric("Above replacement", _fmt(row.get("above_role_replacement"), 2))
    st.caption(
        f"Evidence: {row.get('profile_evidence', '—')} · minutes: {row.get('data_quality', '—')} · "
        f"team-impact method: {row.get('team_impact_evidence', '—')}"
    )

    component_cols = [c for c in filtered.columns if c.endswith("_component")]
    if component_cols:
        components = pd.DataFrame(
            {
                "component": [c.replace("_component", "") for c in component_cols],
                "value": [pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0] for c in component_cols],
            }
        ).dropna()
        components = components[components["value"].abs() > 1e-6]
        if not components.empty:
            components = components.reindex(components["value"].abs().sort_values().index).tail(20)
            fig = px.bar(
                components,
                x="value",
                y="component",
                orientation="h",
                labels={"value": "Contribution to role-relative score", "component": "Component"},
            )
            fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
            st.plotly_chart(fig, width="stretch", key="player_components")



def render_squad_plan():
    st.header("Complete squad plan")
    st.write(
        "This is the main output: where the team needs to get to, where it is currently projected to be, "
        "which squad value can be sold, which positions offer the biggest improvement, and which players/profile thresholds close the gap."
    )

    model = _current_model()
    frames = _current_frames()
    players = frames.get("players")
    if model is None or not model.readiness().get("squad_plan") or players is None:
        st.info(
            "Build all three core layers first: completed league tables, historical team metrics and the current player pool. "
            "Historical player seasons are optional but make the attribute recommendations much stronger."
        )
        return

    league = st.selectbox("League", model.league_model.leagues, key="plan_league")
    context = model.league_model.context(league)
    clubs = model.available_planning_clubs(league)
    if not clubs:
        st.warning("No club identifiers are available in the team-performance history.")
        return
    preferred = st.session_state.get("owned_club_input", "")
    if preferred not in clubs and players is not None and "owned" in players and "team_id" in players:
        owned_values = players["owned"]
        if pd.api.types.is_bool_dtype(owned_values):
            owned_mask = owned_values.fillna(False)
        else:
            owned_mask = owned_values.astype(str).str.strip().str.lower().isin(
                {"true", "1", "yes", "y", "owned", "ours", "current", "squad"}
            )
        owned_clubs = players.loc[owned_mask, "team_id"].dropna().astype(str).unique().tolist()
        if len(owned_clubs) == 1 and owned_clubs[0] in clubs:
            preferred = owned_clubs[0]
    club_index = clubs.index(preferred) if preferred in clubs else 0

    a, b, c1, d = st.columns(4)
    club = a.selectbox("Your club", clubs, index=club_index, key="plan_club")
    target_position = b.number_input(
        "Target finish",
        min_value=1,
        max_value=int(context["n_teams"]),
        value=min(4, int(context["n_teams"])),
        step=1,
        key="plan_target_position",
    )
    probability = c1.slider(
        "Target evidence threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        key="plan_probability",
    )
    formation = d.selectbox("Formation", list(FORMATION_PRESETS), key="plan_formation")

    max_recruits = st.slider(
        "Maximum recommended signings in this plan",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        key="plan_max_recruits",
    )

    if st.button("Build complete squad plan", type="primary", key="build_squad_plan"):
        try:
            with st.spinner("Forecasting the team and simulating replacement scenarios across the market..."):
                result = model.squad_plan(
                    league,
                    int(target_position),
                    players,
                    club=club,
                    probability=float(probability),
                    formation=formation,
                    max_recruits=int(max_recruits),
                )
            st.session_state[SQUAD_PLAN_KEY] = result
        except (DataError, ValueError) as exc:
            st.error(str(exc))

    result = st.session_state.get(SQUAD_PLAN_KEY)
    if not result or result.get("league") != str(league) or result.get("club") != str(club):
        st.caption("Build the plan to generate the full target → squad → transfers result.")
        return

    target = result["target"]["recommended"]
    current = result["current_forecast"]
    gap = result["gap"]
    after = result["after_transfer_forecast"]

    st.subheader("1. What the target requires")
    st.success(
        f"To target **{_ordinal(target_position)} or better** at the selected evidence threshold, "
        f"the balanced model target is **at least {_fmt_int(target['goals_for'])} goals scored** and "
        f"**at most {_fmt_int(target['goals_against'])} conceded**."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target GF", _fmt_int(target.get("goals_for")))
    m2.metric("Target GA", _fmt_int(target.get("goals_against")))
    m3.metric("Current forecast GF", _fmt(current.get("goals_for"), 1))
    m4.metric("Current forecast GA", _fmt(current.get("goals_against"), 1))

    st.write(
        f"Starting from **{club}'s** latest underlying team-performance profile and adjusting it from the previous XI "
        f"to the current modelled XI where historical player seasons allow, the model forecasts approximately "
        f"**{_fmt(current['goals_for'], 1)} scored and {_fmt(current['goals_against'], 1)} conceded** next season, "
        f"a modelled finishing position of about **{_fmt(current['predicted_position'], 1)}**."
    )
    st.caption(current.get("evidence", ""))
    squad_adjustment = current.get("squad_adjustment", {})
    if squad_adjustment:
        if squad_adjustment.get("applied"):
            st.caption(
                f"Current-squad adjustment applied across {int(squad_adjustment.get('replacements', 0))} "
                "position-matched starter replacements from the latest historical XI."
            )
        else:
            st.caption(f"Current-squad adjustment not applied: {squad_adjustment.get('reason', 'insufficient history')}")

    g1, g2 = st.columns(2)
    g1.metric("Attacking gap", f"+{_fmt(gap['additional_goals_for'], 1)} goals")
    g2.metric("Defensive gap", f"-{_fmt(gap['fewer_goals_against'], 1)} goals conceded")

    st.subheader("2. Players to consider selling")
    sales = pd.DataFrame(result.get("sale_candidates", []))
    if sales.empty:
        st.info("No owned player currently meets the planner's overvaluation / replaceability review threshold.")
    else:
        st.dataframe(
            sales,
            width="stretch",
            hide_index=True,
            column_config={
                "market_price": st.column_config.NumberColumn("Market price", format="%.0f"),
                "expected_comparable_price": st.column_config.NumberColumn("Comparable price", format="%.0f"),
                "price_vs_comparables": st.column_config.NumberColumn("Price / comparable", format="%.2f"),
                "above_role_replacement": st.column_config.NumberColumn("Above replacement", format="%.2f"),
            },
        )

    st.subheader("3. Where to recruit")
    opportunities = result.get("position_opportunities", [])
    if not opportunities:
        st.success("The current forecast already meets the selected balanced GF/GA target, or no measurable market upgrade was found.")
    else:
        opp_table = pd.DataFrame([
            {
                "Position": row["position_label"],
                "Current player replaced": row["outgoing_player"],
                "Best-value target": row["recommended_player"],
                "Target club": row["recommended_club"],
                "Price": row["market_price"],
                "Comparable price": row["expected_comparable_price"],
                "Projected GF change": row["gf_gain"],
                "Projected GA reduction": row["ga_reduction"],
                "Gap impact score": row["impact_score"],
            }
            for row in opportunities
        ])
        st.dataframe(
            opp_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="%.0f"),
                "Comparable price": st.column_config.NumberColumn(format="%.0f"),
                "Projected GF change": st.column_config.NumberColumn(format="%.2f"),
                "Projected GA reduction": st.column_config.NumberColumn(format="%.2f"),
                "Gap impact score": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.subheader("4. Minimum stat / attribute profiles")
        st.caption(
            "These are lower-quartile values among candidates producing at least half of the best available modelled impact "
            "for that position. They are screening thresholds, not magic hard cut-offs."
        )
        for row in opportunities:
            with st.expander(
                f"{row['position_label']} · target profile around {row['recommended_player']}",
                expanded=False,
            ):
                profile = pd.DataFrame(row.get("minimum_profile", []))
                if profile.empty:
                    st.write("Not enough comparable player evidence to create a stable profile.")
                else:
                    st.dataframe(
                        profile,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "minimum_typical_value": st.column_config.NumberColumn(
                                "Suggested minimum", format="%.2f"
                            ),
                            "median_successful_candidate": st.column_config.NumberColumn(
                                "Successful-candidate median", format="%.2f"
                            ),
                        },
                    )

                shortlist = pd.DataFrame(row.get("candidate_shortlist", []))
                if not shortlist.empty:
                    st.markdown("**Players in the current export who fit this upgrade route**")
                    st.dataframe(
                        shortlist,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "market_price": st.column_config.NumberColumn("Market price", format="%.0f"),
                            "expected_market_price_for_contribution": st.column_config.NumberColumn(
                                "Comparable price", format="%.0f"
                            ),
                            "estimated_value_edge": st.column_config.NumberColumn(
                                "Estimated value edge", format="%.0f"
                            ),
                            "gf_gain": st.column_config.NumberColumn("GF change", format="%.2f"),
                            "ga_reduction": st.column_config.NumberColumn("GA reduction", format="%.2f"),
                            "impact_score": st.column_config.NumberColumn("Gap impact", format="%.2f"),
                        },
                    )

    st.subheader("5. Recommended transfer scenario")
    selection_reason = after.get("selection_reason")
    if selection_reason:
        st.write(selection_reason)
        st.caption(
            f"Transfer packages evaluated: {int(after.get('packages_considered', 0)):,} · "
            f"Requested threshold reached: {'yes' if after.get('target_reached') else 'no'}."
        )

    transfers = pd.DataFrame(result.get("recommended_transfers", []))
    if transfers.empty:
        st.info("No transfer was required or no candidate had measurable positive impact under the available evidence.")
    else:
        show = [
            col for col in (
                "position_label", "outgoing_player", "recommended_player", "recommended_club",
                "market_price", "outgoing_market_price", "gf_gain", "ga_reduction", "mapped_features_used",
            ) if col in transfers
        ]
        st.dataframe(
            transfers[show],
            width="stretch",
            hide_index=True,
            column_config={
                "market_price": st.column_config.NumberColumn("Buy price", format="%.0f"),
                "outgoing_market_price": st.column_config.NumberColumn("Outgoing value", format="%.0f"),
                "gf_gain": st.column_config.NumberColumn("Individual GF scenario", format="%.2f"),
                "ga_reduction": st.column_config.NumberColumn("Individual GA scenario", format="%.2f"),
            },
        )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("After-plan GF", _fmt(after.get("goals_for"), 1))
    p2.metric("After-plan GA", _fmt(after.get("goals_against"), 1))
    p3.metric("After-plan position", _fmt(after.get("predicted_position"), 1))
    p4.metric("Modelled net spend", _fmt_money(after.get("net_spend")))

    probability_after = after.get("estimated_target_probability")
    if probability_after is not None:
        st.write(
            f"After applying the recommended replacements together and re-running the team goal model, "
            f"the estimated chance of meeting the {_ordinal(target_position)}-place threshold is "
            f"**{float(probability_after):.0%}** under the model."
        )
    st.caption(after.get("interpretation", ""))

    for warning in result.get("model_evidence", {}).get("warnings", []):
        st.warning(warning)

    st.download_button(
        "Download complete squad plan JSON",
        json.dumps(result, default=float, indent=2).encode("utf-8"),
        "fm26_complete_squad_plan.json",
        "application/json",
        key="download_squad_plan",
    )


def render_diagnostics():
    st.header("Model diagnostics")
    model = _current_model()
    if model is None:
        st.info("Build a model first.")
        return

    if model.league_model is not None:
        st.subheader("League target validation")
        report = model.league_model.report
        c1, c2, c3 = st.columns(3)
        c1.metric("Historical seasons", report.get("seasons", "—"))
        c2.metric("League tables", report.get("tables", "—"))
        c3.metric("Rank MAE", _fmt(report.get("rank_mae"), 2))
        folds = pd.DataFrame(report.get("folds", []))
        if not folds.empty:
            st.dataframe(folds, width="stretch", hide_index=True)

    if model.driver_model is not None:
        st.subheader("Goal-driver holdout")
        st.dataframe(_driver_validation_table(model), width="stretch", hide_index=True)
        with st.expander("Raw goal-driver report"):
            st.json(model.driver_model.report)

    if model.player_outcome_model is not None:
        st.subheader("Historical attribute → outcome evidence")
        outcome_report = model.player_outcome_model.report
        if outcome_report.get("available"):
            st.success(
                f"Validated attribute models available from {outcome_report.get('rows', 0):,} player-season rows "
                f"across {outcome_report.get('seasons', 0)} seasons."
            )
            outcomes = list(model.player_outcome_model.models)
            if outcomes:
                chosen_outcome = st.selectbox("Attribute outcome", outcomes, key="diagnostic_outcome")
                importance = model.player_outcome_model.attribute_importance(chosen_outcome)
                st.dataframe(importance.head(25), width="stretch", hide_index=True)
        else:
            st.info(outcome_report.get("reason", "Historical attribute model is not available."))

    if model.player_model is not None:
        st.subheader("Player evidence map")
        st.caption(model.player_model.report.get("evidence", ""))
        values = model.player_model.attribute_values()
        st.dataframe(values, width="stretch", hide_index=True)

        decisions = _current_decisions()
        if decisions is not None and "data_quality" in decisions:
            quality = (
                decisions["data_quality"]
                .value_counts(dropna=False)
                .rename_axis("Data quality")
                .reset_index(name="Players")
            )
            st.dataframe(quality, width="stretch", hide_index=True)

    st.subheader("Export fitted model")
    st.download_button(
        "Download model JSON",
        serialise_model(model),
        "fm26_moneyball_model.json",
        "application/json",
        key="download_model",
    )


render_sidebar()
render_header()

setup_tab, plan_tab, target_tab, drivers_tab, recruitment_tab, diagnostics_tab = st.tabs(
    ["Data collection & setup", "Complete squad plan", "Season target", "Goal drivers", "Recruitment", "Diagnostics"]
)

with setup_tab:
    render_setup()
with plan_tab:
    render_squad_plan()
with target_tab:
    render_target()
with drivers_tab:
    render_drivers()
with recruitment_tab:
    render_recruitment()
with diagnostics_tab:
    render_diagnostics()

st.divider()
st.caption(
    "FM26 Moneyball Recruitment Lab · The app keeps modelling logic in the fm_model package so the statistical backend can be tested independently."
)
