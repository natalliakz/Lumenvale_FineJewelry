"""Channel Investment Planner: Lumenvale Jewelers (fictional).

A Streamlit app on Posit Connect that puts the marketing mix model in the hands of
the VP of Growth. It reads precomputed model outputs (Snowflake, or local files
offline) and never refits, so every interaction is instant. The Scenario tab runs
the registered model (Snowflake Model Registry, see mmm_registry.py) in a Snowflake
warehouse, and falls back to the same math locally when it is not registered.

Run locally:  uv run streamlit run app.py

DISCLAIMER: This project contains synthetic data and analysis created for
demonstration purposes only.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "ml"))

from brand import CHANNEL_COLORS, brand, css, plotly_template  # noqa: E402
from mmm_data import SCENARIO_TABLE, TableReader  # noqa: E402
from mmm_registry import RegisteredModel, call, default_model, scoring_mode  # noqa: E402
from model_utils import CHANNELS  # noqa: E402
from planner import (  # noqa: E402
    QUARTER_WEEKS, Curves, current_plan, optimize, plan_columns, project, recommend_columns,
)
from planner import recommend as recommend_locally  # noqa: E402

DISCLAIMER = ("This project contains synthetic data and analysis created for "
              "demonstration purposes only.")

st.set_page_config(page_title="Channel Investment Planner", page_icon="❄️", layout="wide")
st.markdown(css(), unsafe_allow_html=True)
TEMPLATE = plotly_template()
COLORS = brand()["colors"]
PLOT_CONFIG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def money(x: float, digits: int = 2) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.{digits}f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.0f}K"
    return f"{sign}${x:.0f}"


def signed(x: float) -> str:
    return ("+" if x >= 0 else "") + money(x)


def md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not read them as LaTeX."""
    return text.replace("$", "\\$")


def html(text: str) -> str:
    """The same for raw HTML blocks, where a backslash escape would show."""
    return text.replace("$", "&#36;")


# ---------------------------------------------------------------------------
# Data: read once per viewer, then everything is in memory
# ---------------------------------------------------------------------------
def viewer_token() -> str | None:
    """On Posit Connect, the viewer's session token (for viewer-level Snowflake access)."""
    try:
        return st.context.headers.get("Posit-Connect-User-Session-Token")
    except Exception:  # noqa: BLE001
        return None


def viewer_name() -> str:
    """Who is saving a scenario: the Connect viewer, or the local user."""
    try:
        creds = json.loads(st.context.headers.get("Rstudio-Connect-Credentials") or "{}")
    except Exception:  # noqa: BLE001
        creds = {}
    return creds.get("user") or os.getenv("USER", "viewer")


@st.cache_resource(show_spinner=False)
def reader_for(token: str | None) -> TableReader:
    return TableReader(user_session_token=token)


@st.cache_data(show_spinner="Loading model outputs ...")
def load(token: str | None) -> dict:
    reader = reader_for(token)
    # Connected to Snowflake, each read is a query such as
    #   SELECT * FROM LUMENVALE_MMM.PUBLIC.MMM_CHANNEL_SUMMARY
    summary = reader.read("MMM_CHANNEL_SUMMARY").set_index("channel").loc[CHANNELS]
    contrib = reader.read("MMM_CONTRIBUTIONS_WEEKLY")
    curves = reader.read("MMM_RESPONSE_CURVES")
    draws = reader.read("MMM_POSTERIOR_DRAWS")
    meta = reader.model_run()
    return dict(summary=summary, contrib=contrib, curves=curves, draws=draws, meta=meta,
                model_source=reader.origin("MMM_MODEL_RUN"), source=reader.label,
                error=reader.error)


@st.cache_data(ttl=60, show_spinner=False)
def registered_model(token: str | None) -> RegisteredModel | None:
    """The model's default version in the Snowflake Model Registry (None if not registered).

    Re-checked every minute, so a new default version reaches running sessions quickly.
    """
    return default_model(reader_for(token))


@st.cache_data(show_spinner="Running the scenario on the registered model in Snowflake ...")
def recommend_in_snowflake(token: str | None, model: RegisteredModel, extra: float,
                           caps: tuple) -> dict:
    """SELECT MODEL(LUMENVALE_MMM.PUBLIC.LUMENVALE_MMM, <default>)!RECOMMEND(...)"""
    inputs = dict(zip(recommend_columns(CHANNELS), (extra, *caps)))
    return call(reader_for(token), model, "recommend", inputs)


def saved_scenarios(token: str | None) -> dict:
    """Scenarios saved earlier (by anyone), latest version of each name, as {name: plan}."""
    try:
        saved = reader_for(token).read(SCENARIO_TABLE)
    except Exception:  # noqa: BLE001 - the app works without them
        return {}
    if saved.empty:
        return {}
    saved = saved.sort_values("saved_at")
    latest = saved.groupby("scenario_name")["saved_at"].transform("max") == saved["saved_at"]
    plans = {}
    for name, rows in saved[latest].groupby("scenario_name", sort=False):
        by_channel = rows.set_index("channel")["quarterly_budget"]
        if set(CHANNELS) <= set(by_channel.index):
            plans[name] = by_channel.loc[CHANNELS].to_numpy(float)
    return plans


TOKEN = viewer_token()
DATA = load(TOKEN)
SUMMARY: pd.DataFrame = DATA["summary"]
REGISTERED = registered_model(TOKEN)
CURVES = Curves.from_draws(DATA["draws"], CHANNELS)
# $ per quarter at the current run rate, rounded to $10K (the registered model uses the same).
CURRENT_PLAN = current_plan(SUMMARY["current_quarter_budget"].to_numpy())
CURRENT_K = {ch: int(round(v / 1e3)) for ch, v in zip(CHANNELS, CURRENT_PLAN)}
SLIDER_MAX_K = {ch: max(500, int(np.ceil(4 * CURRENT_K[ch] / 50) * 50)) for ch in CHANNELS}


def next_quarter(today: date) -> str:
    q = (today.month - 1) // 3 + 1
    year, q = (today.year + 1, 1) if q == 4 else (today.year, q + 1)
    return f"Q{q} {year}"


# The quarter being planned. The projection is a steady-state quarter, so it does
# not depend on which quarter this is; set MMM_HORIZON to label it differently.
HORIZON = os.getenv("MMM_HORIZON") or next_quarter(date.today())
EXTRA_MAX_K = 500
DEFAULT_EXTRA_K = 200
DEFAULT_EXTRA_CAPS = {"Connected TV": 60}  # $K: CTV inventory is nearly sold out
CAP_COL = "Cap on extra ($K)"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def default_constraints() -> pd.DataFrame:
    return pd.DataFrame({
        "Channel": CHANNELS,
        "Min ($K)": [round(CURRENT_K[ch] * 0.5 / 10) * 10 for ch in CHANNELS],
        "Max ($K)": [min(SLIDER_MAX_K[ch], CURRENT_K[ch] * 3) for ch in CHANNELS],
    })


def default_extra_caps() -> pd.DataFrame:
    return pd.DataFrame({"Channel": CHANNELS,
                         CAP_COL: [DEFAULT_EXTRA_CAPS.get(ch, np.nan) for ch in CHANNELS]})


def extra_caps(caps: pd.DataFrame) -> np.ndarray:
    """Per-channel cap on the additional budget in $ (NaN = no cap), in CHANNELS order."""
    return caps.set_index("Channel").loc[CHANNELS, CAP_COL].to_numpy(float) * 1e3


def recommend(extra: float, caps: np.ndarray) -> dict:
    """Best split of an additional budget on top of the current plan, within the caps.

    Scored by the registered model in Snowflake when there is one; otherwise (or if the
    call fails in auto mode) by the same code locally, on MMM_POSTERIOR_DRAWS.
    """
    if REGISTERED is not None and extra > 0:
        try:
            out = recommend_in_snowflake(TOKEN, REGISTERED, float(extra), tuple(map(float, caps)))
        except Exception as exc:  # noqa: BLE001
            if scoring_mode() == "registry":
                raise
            st.session_state["scn_flash"] = ("warning", f"The registered model could not be run "
                                             f"({exc}); scored locally instead.")
        else:
            add = np.array([out[f"ADD_{c}"] for c in plan_columns(CHANNELS)], float)
            plan = CURRENT_PLAN + add
            proj = project(CURVES, plan, CURRENT_PLAN)  # revenue by channel, for saving
            proj["delta"] = (out["INCREMENTAL"], out["INCREMENTAL_P05"], out["INCREMENTAL_P95"])
            return dict(extra=extra, caps=caps, add=add, plan=plan, proj=proj,
                        note=out.get("NOTE") or "",
                        scored_by=f"{REGISTERED.fqn}!RECOMMEND · {REGISTERED.version}, in Snowflake")
    res = recommend_locally(CURVES, CURRENT_PLAN, extra, caps)
    return res | dict(scored_by="locally, from MMM_POSTERIOR_DRAWS")


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("extra_k", DEFAULT_EXTRA_K)
    ss.setdefault("extra_caps_base", default_extra_caps())
    ss.setdefault("extra_caps_live", ss["extra_caps_base"])
    if "scn_result" not in ss:
        ss["scn_result"] = recommend(DEFAULT_EXTRA_K * 1e3, extra_caps(ss["extra_caps_base"]))
    ss.setdefault("budget_k", int(sum(CURRENT_K.values())))
    for ch in CHANNELS:
        ss.setdefault(f"plan_{ch}", CURRENT_K[ch])
    ss.setdefault("constraints_base", default_constraints())
    ss.setdefault("constraints_live", ss["constraints_base"])
    if "scenarios" not in ss:
        # Current plan first, then everything saved to MMM_SAVED_SCENARIOS so far.
        plans = {"Current plan": CURRENT_PLAN, **saved_scenarios(TOKEN)}
        plans["Current plan"] = CURRENT_PLAN
        ss["scenarios"] = {name: dict(plan=plan, proj=project(CURVES, plan, CURRENT_PLAN))
                           for name, plan in plans.items()}
    ss.setdefault("flash", None)


def plan_now() -> np.ndarray:
    return np.array([st.session_state[f"plan_{ch}"] * 1e3 for ch in CHANNELS])


def reset_plan() -> None:
    for ch in CHANNELS:
        st.session_state[f"plan_{ch}"] = CURRENT_K[ch]
    st.session_state["budget_k"] = int(sum(CURRENT_K.values()))
    st.session_state["flash"] = ("info", "Back to the current plan.")


def reset_constraints() -> None:
    st.session_state["constraints_base"] = default_constraints()
    st.session_state["constraints_live"] = st.session_state["constraints_base"]
    st.session_state.pop("constraints_editor", None)


def run_optimizer() -> None:
    cons = st.session_state["constraints_live"].set_index("Channel").loc[CHANNELS]
    lower = cons["Min ($K)"].fillna(0).clip(lower=0).to_numpy(float) * 1e3
    cap = np.array([SLIDER_MAX_K[ch] for ch in CHANNELS]) * 1e3
    upper = np.minimum(cons["Max ($K)"].fillna(np.inf).to_numpy(float) * 1e3, cap)
    plan, note = optimize(CURVES, st.session_state["budget_k"] * 1e3, lower, upper, plan_now())
    for ch, v in zip(CHANNELS, plan):
        st.session_state[f"plan_{ch}"] = int(round(v / 1e3 / 10) * 10)
    bound = [ch for ch, v, u in zip(CHANNELS, plan, upper) if v >= u - 5e3 and u < cap[CHANNELS.index(ch)]]
    extra = f" At a cap you set: {', '.join(bound)}." if bound else ""
    st.session_state["flash"] = ("success", note + extra)


def run_scenario() -> None:
    ss = st.session_state
    ss["scn_result"] = recommend(ss["extra_k"] * 1e3, extra_caps(ss["extra_caps_live"]))


def save_plan(name: str, plan: np.ndarray) -> tuple[str, str]:
    """Save a plan in the session and to MMM_SAVED_SCENARIOS; returns a flash message."""
    proj = project(CURVES, plan, CURRENT_PLAN)
    st.session_state["scenarios"][name] = dict(plan=plan, proj=proj)
    rows = pd.DataFrame({
        "scenario_name": name,
        "saved_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "saved_by": viewer_name(),
        "channel": CHANNELS,
        "quarterly_budget": plan,
        "projected_revenue": proj["by_channel"],
        "total_budget": proj["spend"],
        "total_revenue": proj["revenue"][0],
        "total_revenue_p05": proj["revenue"][1],
        "total_revenue_p95": proj["revenue"][2],
        "blended_roi": proj["roi"][0],
        "change_vs_current": proj["delta"][0],
    })
    try:
        where = reader_for(TOKEN).write(SCENARIO_TABLE, rows, overwrite=False)
        return "success", f"Saved “{name}” to {where}."
    except Exception as exc:  # noqa: BLE001 - keep the scenario in the session regardless
        return "warning", f"Saved “{name}” for this session only ({exc})."


def save_scenario() -> None:
    name = st.session_state.get("scenario_name", "").strip() or f"Scenario {len(st.session_state['scenarios']) + 1}"
    st.session_state["scenario_name"] = ""
    st.session_state["flash"] = save_plan(name, plan_now())


def save_recommendation() -> None:
    res = st.session_state["scn_result"]
    name = f"Recommended +{money(res['add'].sum())} ({HORIZON})"
    st.session_state["scn_flash"] = save_plan(name, res["plan"])


init_state()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_contributions(show_baseline: bool) -> go.Figure:
    df = DATA["contrib"].copy()
    df = df[df["component"].isin(CHANNELS + (["Baseline"] if show_baseline else []))]
    order = (["Baseline"] if show_baseline else []) + CHANNELS
    fig = go.Figure()
    for comp in order:
        d = df[df["component"] == comp].sort_values("week_start")
        fig.add_scatter(
            x=d["week_start"], y=d["contribution"] / 1e3, name=comp, stackgroup="one",
            mode="lines", line=dict(width=0.6, color=CHANNEL_COLORS[comp]),
            fillcolor=CHANNEL_COLORS[comp],
            hovertemplate=f"{comp}: $%{{y:,.0f}}K<extra></extra>",
        )
    fig.update_layout(template=TEMPLATE, height=360, hovermode="x unified",
                      legend=dict(traceorder="normal"),
                      yaxis_title="Weekly revenue driven ($K)", xaxis_title=None)
    return fig


def fig_roi() -> go.Figure:
    s = SUMMARY.sort_values("marginal_roi")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("ROI: revenue per $ spent (3 years)",
                                        "Marginal ROI: revenue from the next $"))
    for col, (m, lo, hi) in enumerate([("roi", "roi_p05", "roi_p95"),
                                       ("marginal_roi", "marginal_roi_p05", "marginal_roi_p95")], 1):
        fig.add_bar(
            y=s.index, x=s[m], orientation="h", showlegend=False,
            marker=dict(color=[CHANNEL_COLORS[c] for c in s.index], cornerradius=4),
            error_x=dict(type="data", symmetric=False, array=s[hi] - s[m],
                         arrayminus=s[m] - s[lo], color="#7C8B96", thickness=1.2, width=3),
            customdata=np.column_stack([s[lo], s[hi]]),
            hovertemplate="%{y}: %{x:.2f}× (90%: %{customdata[0]:.2f}–%{customdata[1]:.2f})<extra></extra>",
            row=1, col=col,
        )
        # Value labels sit just past the whisker so they never collide with it.
        fig.add_scatter(y=s.index, x=s[hi], mode="text", text=[f"  {v:.1f}×" for v in s[m]],
                        textposition="middle right", showlegend=False, hoverinfo="skip",
                        textfont=dict(color=COLORS["foreground"], size=12), cliponaxis=False,
                        row=1, col=col)
        fig.add_vline(x=1, line=dict(color=COLORS["danger"], dash="dot", width=1.5), row=1, col=col)
    fig.add_annotation(x=1, y=-0.9, text="break-even", showarrow=False, xanchor="left",
                       font=dict(size=11, color=COLORS["danger"]), row=1, col=2)
    fig.update_layout(template=TEMPLATE, height=380, bargap=0.35)
    fig.update_annotations(font=dict(family=brand()["heading_font"], size=16))
    return fig


def fig_change(plan: np.ndarray) -> go.Figure:
    delta = (plan - CURRENT_PLAN) / 1e3
    order = np.argsort(delta)
    colors = [COLORS["success"] if d >= 0 else COLORS["danger"] for d in delta[order]]
    fig = go.Figure(go.Bar(
        y=np.array(CHANNELS)[order], x=delta[order], orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[("+" if d >= 0 else "−") + f"${abs(d):,.0f}K" if abs(d) >= 1 else "no change"
              for d in delta[order]],
        textposition="outside", cliponaxis=False, textfont=dict(color=COLORS["foreground"]),
        hovertemplate="%{y}: %{x:+,.0f}K vs current<extra></extra>",
    ))
    lim = max(50, np.abs(delta).max() * 1.35)
    fig.update_layout(template=TEMPLATE, height=300, title="Budget change vs current plan",
                      xaxis=dict(range=[-lim, lim], title="$K per quarter", zeroline=True,
                                 zerolinecolor="#A9B8C4"), showlegend=False)
    return fig


def fig_curves(plan: np.ndarray) -> go.Figure:
    cur = DATA["curves"]
    fig = make_subplots(rows=2, cols=4, subplot_titles=CHANNELS,
                        horizontal_spacing=0.07, vertical_spacing=0.2)
    for i, ch in enumerate(CHANNELS):
        r, c = i // 4 + 1, i % 4 + 1
        d = cur[cur["channel"] == ch]
        x = d["weekly_spend"] * QUARTER_WEEKS / 1e6  # $M per quarter
        y, lo, hi = (d[k] * QUARTER_WEEKS / 1e6 for k in ("revenue", "revenue_p05", "revenue_p95"))
        color = CHANNEL_COLORS[ch]
        fig.add_scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([hi, lo[::-1]]), fill="toself",
                        fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip",
                        showlegend=False, row=r, col=c)
        observed = d["within_observed_range"].to_numpy()
        fig.add_scatter(x=x[observed], y=y[observed], mode="lines", line=dict(color=color, width=2),
                        showlegend=False, hovertemplate="$%{x:.2f}M → $%{y:.2f}M<extra></extra>",
                        row=r, col=c)
        fig.add_scatter(x=x[~observed], y=y[~observed], mode="lines",
                        line=dict(color=color, width=2, dash="dot"), showlegend=False,
                        hovertemplate="$%{x:.2f}M → $%{y:.2f}M (beyond observed spend)<extra></extra>",
                        row=r, col=c)
        for spend, label, symbol, fill in ((CURRENT_PLAN[i], "Current", "circle", "white"),
                                           (plan[i], "Proposed", "diamond", color)):
            rev = CURVES.revenue(np.where(np.arange(len(CHANNELS)) == i, spend, 0))[:, i].mean() / 1e6
            fig.add_scatter(x=[spend / 1e6], y=[rev], mode="markers", name=label,
                            legendgroup=label, showlegend=(i == 0),
                            marker=dict(symbol=symbol, size=11, color=fill,
                                        line=dict(color=COLORS["foreground"], width=1.5)),
                            hovertemplate=f"{label}: $%{{x:.2f}}M → $%{{y:.2f}}M<extra></extra>",
                            row=r, col=c)
    fig.update_layout(template=TEMPLATE, height=520, margin=dict(t=70),
                      legend=dict(y=1.1))
    fig.update_xaxes(tickprefix="$", ticksuffix="M", nticks=3)
    fig.update_yaxes(tickprefix="$", ticksuffix="M", nticks=4)
    fig.update_annotations(font=dict(size=13))
    return fig


def compare_height(names: list[str]) -> int:
    return 200 + 55 * len(names)


def fig_compare(names: list[str]) -> go.Figure:
    fig = go.Figure()
    for ch in CHANNELS:
        i = CHANNELS.index(ch)
        fig.add_bar(y=names, x=[st.session_state["scenarios"][n]["plan"][i] / 1e6 for n in names],
                    name=ch, orientation="h", marker=dict(color=CHANNEL_COLORS[ch],
                                                          line=dict(color=COLORS["background"], width=2)),
                    hovertemplate=f"{ch}: $%{{x:.2f}}M<extra>%{{y}}</extra>")
    fig.update_layout(template=TEMPLATE, barmode="stack", height=compare_height(names),
                      title="Quarterly budget by channel",
                      xaxis=dict(tickprefix="$", ticksuffix="M"),
                      yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, traceorder="normal"))
    return fig


def fig_compare_revenue(names: list[str]) -> go.Figure:
    proj = [st.session_state["scenarios"][n]["proj"] for n in names]
    mean = np.array([p["revenue"][0] for p in proj]) / 1e6
    lo = np.array([p["revenue"][1] for p in proj]) / 1e6
    hi = np.array([p["revenue"][2] for p in proj]) / 1e6
    fig = go.Figure(go.Scatter(
        y=names, x=mean, mode="markers+text", text=[f"${m:.1f}M" for m in mean],
        textposition="top center", marker=dict(size=13, color=COLORS["primary"],
                                               line=dict(color=COLORS["foreground"], width=1)),
        error_x=dict(type="data", symmetric=False, array=hi - mean, arrayminus=mean - lo,
                     color=COLORS["secondary"], thickness=1.5, width=5),
        hovertemplate="%{y}: $%{x:.2f}M<extra></extra>"))
    fig.update_layout(template=TEMPLATE, height=compare_height(names),
                      title="Projected media revenue per quarter (90% interval)",
                      xaxis=dict(tickprefix="$", ticksuffix="M"),
                      # Explicit reversed range leaves headroom for the value label above the top row.
                      yaxis=dict(range=[len(names) - 0.5, -0.9]))
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
meta = DATA["meta"]
st.markdown('<div class="lv-topbar"></div>', unsafe_allow_html=True)
head_l, head_r = st.columns([3, 1.3], vertical_alignment="bottom")
with head_l:
    st.markdown('<div class="lv-eyebrow">Marketing mix model · what-if analysis for channel '
                'investment · VP of Growth</div>', unsafe_allow_html=True)
    st.title("Channel Investment Planner")
with head_r:
    # The badge names the model the scenarios run on: the default version in the Snowflake
    # Model Registry or, when the model is not registered, the fit in MMM_MODEL_RUN.
    fitted = meta.get("fitted_at", "")[:10]
    if REGISTERED is not None:
        version, source = REGISTERED.model_version, REGISTERED.label
        fitted = str(REGISTERED.metrics.get("fitted_at") or fitted)[:10]
    else:
        version, source = meta.get("model_version", "?"), DATA["model_source"]
    st.markdown(
        f'<div style="text-align:right"><span class="lv-badge" title="{source}">model {version}'
        f'</span><br><span class="lv-source" style="margin-top:.35rem">{source} · fit {fitted}'
        f'</span><br><span class="lv-source" style="margin-top:.3rem">Data: {DATA["source"]}'
        f'</span></div>', unsafe_allow_html=True)
if REGISTERED is not None and REGISTERED.model_version != meta.get("model_version"):
    st.warning(f"The registry's default version is {REGISTERED.model_version}, but the model "
               f"outputs in Snowflake are from {meta.get('model_version')}. The Scenario tab runs "
               f"{REGISTERED.model_version}; the other tabs use {meta.get('model_version')}. Run "
               "`uv run ml/register_model.py` after refitting.")
st.caption(f"⚠️ {DISCLAIMER} Lumenvale Jewelers is a fictional company.")

tab0, tab1, tab2, tab3 = st.tabs(["Scenario", "Where the money goes today", "Plan the full mix",
                                  "Compare scenarios"])


def allocation_rows(add: np.ndarray) -> str:
    """HTML bars for the additional budget by channel, largest first."""
    top = max(add.max(), 1.0)
    rows = []
    for i in np.argsort(-add, kind="stable"):
        ch, v = CHANNELS[i], add[i]
        bar = (f'<div class="bar" style="width:{100 * v / top:.1f}%;background:{CHANNEL_COLORS[ch]}">'
               '</div>' if v > 0 else "")
        value = f"+${v:,.0f}" if v > 0 else "$0"
        rows.append(f'<div class="lv-alloc"><span>{ch}</span><div class="track">{bar}</div>'
                    f'<span class="val{"" if v > 0 else " zero"}">{value}</span></div>')
    return f'<div style="margin-bottom:1.1rem">{"".join(rows)}</div>'


# --- Scenario: the VP of Growth's view ----------------------------------------------
with tab0:
    ss = st.session_state
    res = ss["scn_result"]
    panel, results = st.columns([1, 1.55], gap="large")
    with panel:
        with st.container(border=True):
            st.markdown('<div class="lv-section">Scenario</div>', unsafe_allow_html=True)
            st.slider("Additional budget", min_value=0, max_value=EXTRA_MAX_K, step=10,
                      key="extra_k", format="$%dK",
                      help="Budget on top of the current plan for the quarter.")
            h1, h2 = st.columns(2)
            h1.markdown(f'<div class="lv-kv">Horizon</div><div class="lv-kv-value">{HORIZON}</div>',
                        unsafe_allow_html=True)
            h2.markdown(html(f'<div class="lv-kv">Current plan</div><div class="lv-kv-value">'
                           f'{money(CURRENT_PLAN.sum())} / qtr</div>'), unsafe_allow_html=True)
            with st.expander("Constraints"):
                ss["extra_caps_live"] = st.data_editor(
                    ss["extra_caps_base"], key="extra_caps_editor", hide_index=True,
                    width="stretch", disabled=["Channel"],
                    column_config={CAP_COL: st.column_config.NumberColumn(
                        min_value=0, step=10, format="$%d",
                        help="Most extra budget the channel can take. Empty = no cap.")})
            st.button("Run scenario", type="primary", on_click=run_scenario, width="stretch")
            live = extra_caps(ss["extra_caps_live"])
            if ss["extra_k"] * 1e3 != res["extra"] or not np.array_equal(live, res["caps"], equal_nan=True):
                st.caption("Settings changed: click **Run scenario** to update the recommendation.")

    with results:
        p = res["proj"]
        spent = res["add"].sum()
        st.markdown(html(f'<div class="lv-headline">Recommended allocation of <b>+{money(spent)}</b> '
                       f'<span class="lv-tag">illustrative</span></div>'), unsafe_allow_html=True)
        st.markdown(html(allocation_rows(res["add"])), unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        k1.metric("Projected incremental revenue", money(p["delta"][0]),
                  help=f"Extra media-driven revenue in {HORIZON} vs the current plan (posterior mean).")
        k1.caption(md(f"90% interval: {money(p['delta'][1])} – {money(p['delta'][2])}"))
        k2.metric("Blended ROAS", f"{p['delta'][0] / spent:.1f}×" if spent else "–",
                  help="Incremental revenue per additional dollar of budget.")
        k2.caption(md(f"on the additional {money(spent)}"))
        capped = [f"{ch} capped at +{money(c)}"
                  for ch, c in zip(CHANNELS, res["caps"]) if not np.isnan(c)]
        note = f" · {res['note']}" if res["note"] else ""
        st.markdown(html(f'<div class="lv-note"><b>Constraints:</b> {"; ".join(capped) or "none"}{note}'
                       '</div>'), unsafe_allow_html=True)
        if spent and res["add"][CHANNELS.index("Paid Search")] == 0:
            st.markdown(f'<div class="lv-note">Paid Search gets none of it: its next dollar returns '
                        f'only {SUMMARY.loc["Paid Search", "marginal_roi"]:.2f}×.</div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="lv-note">Scored {res["scored_by"]}</div>',
                    unsafe_allow_html=True)
        st.button("Save to compare", on_click=save_recommendation, disabled=not spent)
        flash = ss.pop("scn_flash", None)
        if flash:
            getattr(st, flash[0])(md(flash[1]))

# --- Tab 1 ------------------------------------------------------------------------
with tab1:
    contrib = DATA["contrib"]
    last52 = contrib["week_start"] >= contrib["week_start"].max() - np.timedelta64(51 * 7, "D")
    media = contrib[last52 & contrib["component"].isin(CHANNELS)]["contribution"].sum()
    total = contrib[last52]["contribution"].sum()
    spend52 = SUMMARY["run_rate_weekly_spend"].sum() * 52
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Media spend · last 52 weeks", money(spend52))
    k2.metric("Revenue driven by media", money(media))
    k3.metric("Blended ROI", f"{media / spend52:.1f}×")
    k4.metric("Media share of revenue", f"{media / total:.0%}")

    ps, ctv = SUMMARY.loc["Paid Search"], SUMMARY.loc["Connected TV"]
    share = ps["run_rate_weekly_spend"] / SUMMARY["run_rate_weekly_spend"].sum()
    st.markdown(
        f'<div class="lv-callout"><b>Paid Search</b> takes {share:.0%} of the budget and returns '
        f'{ps["roi"]:.1f}× on average, but the <i>next</i> dollar returns only '
        f'<b>{ps["marginal_roi"]:.2f}×</b>: it is saturated. <b>Connected TV</b> gets '
        f'{ctv["run_rate_weekly_spend"] / SUMMARY["run_rate_weekly_spend"].sum():.0%} and its next '
        f'dollar returns <b>{ctv["marginal_roi"]:.1f}×</b>.</div>', unsafe_allow_html=True)

    st.subheader("Revenue driven by each channel, week by week")
    show_base = st.toggle("Include baseline (non-media) revenue", value=False)
    st.plotly_chart(fig_contributions(show_base), width="stretch", config=PLOT_CONFIG)

    st.subheader("Return on spend by channel")
    st.plotly_chart(fig_roi(), width="stretch", config=PLOT_CONFIG)
    st.caption("Bars show the posterior mean; whiskers the 90% credible interval. Marginal ROI is "
               "measured at the trailing 52-week spend rate. Below the dotted line, the next dollar "
               "returns less than it costs.")

    with st.expander("Table view"):
        table = SUMMARY[["run_rate_weekly_spend", "roi", "roi_p05", "roi_p95", "marginal_roi",
                         "marginal_roi_p05", "marginal_roi_p95", "decay"]].copy()
        table["run_rate_weekly_spend"] *= QUARTER_WEEKS
        table.columns = ["Quarterly spend", "ROI", "ROI p5", "ROI p95", "Marginal ROI",
                         "mROI p5", "mROI p95", "Carryover"]
        st.dataframe(table.style.format({"Quarterly spend": "${:,.0f}", "Carryover": "{:.2f}"}
                                        | {c: "{:.2f}×" for c in table.columns[1:7]}),
                     width="stretch")

# --- Tab 2 ------------------------------------------------------------------------
with tab2:
    left, right = st.columns([1, 2.1], gap="large")
    with left:
        st.markdown(f"#### {HORIZON} plan")
        st.number_input("Total quarterly budget ($K)", min_value=500, max_value=30_000, step=100,
                        key="budget_k")
        for ch in CHANNELS:
            st.slider(ch, min_value=0, max_value=SLIDER_MAX_K[ch], step=10, key=f"plan_{ch}",
                      format="$%dK")
        allocated = plan_now().sum() / 1e3
        gap = st.session_state["budget_k"] - allocated
        if abs(gap) >= 1:
            st.caption(md(f"Sliders allocate **${allocated:,.0f}K**: "
                          f"${abs(gap):,.0f}K {'under' if gap > 0 else 'over'} the total budget. "
                          f"Optimize spends exactly the total."))
        st.button("Reset to current plan", on_click=reset_plan, width="stretch")

        st.markdown("#### Business constraints")
        st.caption(md("What you know that the model doesn't. Example: *Connected TV inventory "
                      "sells out past $900K a quarter.*"))
        edited = st.data_editor(
            st.session_state["constraints_base"], key="constraints_editor", hide_index=True,
            width="stretch", disabled=["Channel"],
            column_config={
                "Min ($K)": st.column_config.NumberColumn(min_value=0, step=10, format="$%d"),
                "Max ($K)": st.column_config.NumberColumn(min_value=0, step=10, format="$%d"),
            })
        st.session_state["constraints_live"] = edited
        st.button("✦ Optimize within my constraints", type="primary", on_click=run_optimizer,
                  width="stretch")
        st.button("Reset constraints", on_click=reset_constraints, width="stretch")

    with right:
        flash = st.session_state.pop("flash", None)
        if flash:
            getattr(st, flash[0])(md(flash[1]))
        plan = plan_now()
        cur = project(CURVES, CURRENT_PLAN, CURRENT_PLAN)
        prop = project(CURVES, plan, CURRENT_PLAN)
        m1, m2, m3 = st.columns(3)
        m1.metric("Projected media revenue", money(prop["revenue"][0]),
                  help="Revenue driven by media next quarter (posterior mean).")
        m1.caption(md(f"90% interval: {money(prop['revenue'][1])} – {money(prop['revenue'][2])}"))
        m2.metric("Blended ROAS", f"{prop['roi'][0]:.2f}×",
                  delta=f"{prop['roi'][0] - cur['roi'][0]:+.2f}× vs current")
        m2.caption(md(f"on {money(prop['spend'])} of media spend"))
        m3.metric("Change vs current plan", signed(prop["delta"][0]),
                  delta=f"{prop['delta'][0] / cur['revenue'][0]:+.1%}")
        m3.caption(md(f"90% interval: {signed(prop['delta'][1])} to {signed(prop['delta'][2])}"))

        st.plotly_chart(fig_change(plan), width="stretch", config=PLOT_CONFIG)
        st.markdown("##### Response curves: where each channel sits today and under your plan")
        st.plotly_chart(fig_curves(plan), width="stretch", config=PLOT_CONFIG)
        st.caption("Quarterly revenue driven by each channel vs quarterly spend. Shaded: 90% "
                   "credible band. Dotted: beyond any spend level seen in the data, so treat "
                   "with care.")

        s1, s2 = st.columns([3, 1], vertical_alignment="bottom")
        s1.text_input("Name this scenario", key="scenario_name",
                      placeholder="e.g. Shift to Connected TV")
        s2.button("Save scenario", on_click=save_scenario, width="stretch")

# --- Tab 3 ------------------------------------------------------------------------
with tab3:
    scenarios = st.session_state["scenarios"]
    if "Current plan" not in scenarios:
        scenarios["Current plan"] = dict(plan=CURRENT_PLAN, proj=project(CURVES, CURRENT_PLAN, CURRENT_PLAN))
    if len(scenarios) == 1:
        st.info("Save a scenario on the **Scenario** or **Plan the full mix** tab to compare it with the current plan.")
    names = st.multiselect("Scenarios to compare", list(scenarios), default=list(scenarios))
    if names:
        rows = []
        for n in names:
            p = scenarios[n]["proj"]
            rows.append({"Scenario": n, "Budget": money(p["spend"]),
                         "Projected revenue": money(p["revenue"][0]),
                         "90% interval": f"{money(p['revenue'][1])} – {money(p['revenue'][2])}",
                         "Blended ROAS": f"{p['roi'][0]:.2f}×",
                         "vs current plan": signed(p["delta"][0])}
                        | {ch: money(v) for ch, v in zip(CHANNELS, scenarios[n]["plan"])})
        compare = pd.DataFrame(rows).set_index("Scenario").T
        st.dataframe(compare, width="stretch", height=35 * (len(compare) + 1) + 3)
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_compare_revenue(names), width="stretch", config=PLOT_CONFIG)
        c2.plotly_chart(fig_compare(names), width="stretch", config=PLOT_CONFIG)

st.markdown(f'<div class="lv-disclaimer">{DISCLAIMER} All company names, channels, budgets and '
            f'results are fictional and AI-generated. Model {meta.get("model_version", "")}: Bayesian '
            f'marketing mix model fit on {meta.get("weeks", "")} weeks of synthetic data '
            f'(hold-out error {meta.get("holdout_mape", 0):.1%}).</div>', unsafe_allow_html=True)
