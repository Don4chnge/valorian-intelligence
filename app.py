"""
Valorian Intelligence dashboard.

The page is built around the three questions the library keeps separate:

    Did the batch arrive intact?      data quality
    Have the inputs changed?          input shift
    Is the model still right?         measured performance

The strip at the top shows all three for every batch, so the shape of a
model's history is readable before any number is: a row of amber over a row of
calm is "the world moved and the model coped"; a single red cell under a calm
row is the case this project exists for. The sections below it take the same
three questions in the same order for whichever batch is selected.

Colour follows the library's rule that input drift is not failure. Red is
reserved for a batch that arrived broken and for performance measured as
degraded. A significant input shift is orange, however large.

Database resolution, in order:
    valorian.db   written by demo/run_demo.py
    qlfs.db       written by demo/run_qlfs.py
    demo.db       committed fixture, so the deployed app has something to show

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from valorian import PSI_MODERATE, PSI_SIGNIFICANT
from valorian.driftscore import (
    BAND_DEGRADED,
    BAND_HEALTHY,
    BAND_WATCH,
    band_for,
    compute_driftscore,
)
from valorian.store import MonitoringStore

st.set_page_config(
    page_title="Valorian Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="auto",
)

CANDIDATES = ["valorian.db", "qlfs.db", "demo.db"]

# Petrol blue rather than black: the page is a measuring instrument, and the
# signal colours need a ground that is clearly a colour of its own.
GROUND = "#0C2A33"
PANEL = "#123A45"
RULE = "#1F4D59"
INK = "#EAF3F0"
MUTED = "#9DB8BC"

GOOD = "#7ADCC4"
WATCH = "#F4C95D"
SHIFT = "#F29A4A"
BAD = "#F0645A"
NONE = "#5B7A82"

# state -> (css class, colour, glyph, plain-language label). The glyph is there
# so the strip can be read without relying on colour.
STATES = {
    "healthy": ("good", GOOD, "", "healthy"),
    "stable": ("good", GOOD, "", "stable"),
    "within_tolerance": ("good", GOOD, "", "within tolerance"),
    "warning": ("watch", WATCH, "△", "warning"),
    "watch": ("watch", WATCH, "△", "watch"),
    "significant": ("shift", SHIFT, "▲", "significant shift"),
    "failed": ("bad", BAD, "✕", "failed"),
    "degraded": ("bad", BAD, "✕", "degraded"),
    "insufficient_evidence": ("thin", NONE, "?", "too little evidence"),
    "not_observed": ("none", NONE, "–", "not measured"),
    "unknown": ("none", NONE, "", "not recorded"),
}
BAND_COLOUR = {"healthy": GOOD, "watch": WATCH, "degraded": SHIFT, "critical": BAD}
SEVERITY_COLOUR = {"stable": GOOD, "moderate": WATCH, "significant": SHIFT}

METRIC_NAMES = {
    "roc_auc": "ROC-AUC", "accuracy": "Accuracy", "f1": "F1",
    "log_loss": "Log loss", "mae": "MAE", "rmse": "RMSE",
}


def metric_name(metric) -> str:
    return METRIC_NAMES.get(str(metric), html.escape(str(metric)))


DIMENSIONS = [
    ("data_quality", "Data quality"),
    ("input_shift", "Input shift"),
    ("performance_status", "Performance"),
]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stSidebar"] {{
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
.stApp {{ background: {GROUND}; }}
div.block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1280px; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {RULE}; }}
hr {{ border-color: {RULE}; }}

h1, h2, h3, .v-verdict, .v-brand {{
    font-family: 'Bricolage Grotesque', 'IBM Plex Sans', system-ui, sans-serif;
}}
h2 {{ font-size: 1.55rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 .2rem; }}

.v-brand {{ font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em; color: {INK}; }}
.v-batch {{ color: {MUTED}; font-size: .95rem; margin-bottom: .5rem; }}
.v-verdict {{
    font-size: clamp(2rem, 4.4vw, 3.5rem); line-height: 1.04; font-weight: 700;
    letter-spacing: -0.025em; color: {INK}; margin: 0 0 .9rem; max-width: 20ch;
    text-wrap: balance;
}}
.v-detail {{ color: {MUTED}; font-size: 1.05rem; line-height: 1.55; max-width: 68ch; margin: 0 0 2rem; }}
.v-detail b {{ color: {INK}; font-weight: 600; font-variant-numeric: tabular-nums; }}
.v-lede {{ color: {MUTED}; font-size: .98rem; line-height: 1.55; max-width: 72ch; margin: 0 0 1rem; }}
.v-lede b {{ color: {INK}; font-weight: 600; }}
.v-section {{ border-top: 1px solid {RULE}; padding-top: 1.6rem; margin-top: 2.4rem; }}
.v-note {{
    border-left: 3px solid var(--tone, {MUTED}); background: {PANEL};
    padding: .75rem 1rem; border-radius: 0 6px 6px 0; margin: 0 0 1rem;
    color: {INK}; font-size: .95rem; line-height: 1.5; max-width: 80ch;
}}
.v-provenance {{
    display: inline-block; border: 1px dashed {NONE}; border-radius: 4px;
    padding: .3rem .7rem; color: {MUTED}; font-size: .85rem; margin-bottom: 1.4rem;
}}

/* ---- The health strip ------------------------------------------------ */
.v-strip {{
    display: grid; grid-template-columns: 8.5rem repeat(var(--n), minmax(1.1rem, 1fr));
    gap: 5px; align-items: stretch; margin: 0 0 .5rem;
}}
.v-row-label {{ color: {INK}; font-size: .92rem; font-weight: 500; align-self: center; }}
.v-cell {{
    height: 2.9rem; border-radius: 4px; display: grid; place-items: center;
    font-size: .8rem; font-weight: 600; color: {GROUND};
    transform-origin: left center;
    animation: v-draw .45s cubic-bezier(.2,.7,.2,1) both;
    animation-delay: calc(var(--i) * 28ms);
}}
.v-cell.good  {{ background: {GOOD}; }}
.v-cell.watch {{ background: {WATCH}; }}
.v-cell.shift {{ background: {SHIFT}; }}
.v-cell.bad   {{ background: {BAD}; }}
.v-cell.thin  {{ background: transparent; border: 1.5px dashed {NONE}; color: {MUTED}; }}
.v-cell.none  {{
    background: repeating-linear-gradient(135deg, {PANEL} 0 5px, {RULE} 5px 7px);
    color: {MUTED};
}}
.v-cell.picked {{ outline: 2px solid {INK}; outline-offset: 2px; }}
.v-tick {{
    color: {MUTED}; font-size: .74rem; text-align: center; padding-top: .25rem;
    white-space: nowrap; overflow: hidden; text-overflow: clip;
    font-variant-numeric: tabular-nums;
}}
.v-tick.picked {{ color: {INK}; font-weight: 600; }}
.v-legend {{ color: {MUTED}; font-size: .85rem; margin: .9rem 0 0; display: flex; flex-wrap: wrap; gap: .4rem 1.3rem; }}
.v-key {{ display: inline-flex; align-items: center; gap: .45rem; }}
.v-key i {{ width: .85rem; height: .85rem; border-radius: 3px; display: inline-block; }}
@keyframes v-draw {{ from {{ transform: scaleX(0); opacity: 0; }} to {{ transform: scaleX(1); opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{ .v-cell {{ animation: none; }} }}
@media (max-width: 760px) {{
    .v-strip {{ grid-template-columns: 6rem repeat(var(--n), minmax(.7rem, 1fr)); gap: 3px; }}
    .v-tick {{ display: none; }}
    .v-cell {{ font-size: 0; height: 2.2rem; }}
}}

/* ---- Contribution bars ------------------------------------------------ */
.v-bars {{ display: grid; grid-template-columns: max-content 1fr max-content; gap: .35rem .8rem;
           align-items: center; max-width: 34rem; margin: .4rem 0 1.2rem; font-size: .9rem; }}
.v-bars span.t {{ background: {RULE}; border-radius: 3px; height: .55rem; overflow: hidden; }}
.v-bars span.t i {{ display: block; height: 100%; background: {SHIFT}; border-radius: 3px; }}
.v-bars span.n {{ color: {MUTED}; font-variant-numeric: tabular-nums; }}

[data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 6px; }}
a {{ color: {WATCH}; }}
:focus-visible {{ outline: 2px solid {WATCH}; outline-offset: 2px; }}
</style>
"""


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def resolve_db() -> tuple[Path | None, bool]:
    """Return the first database that exists, and whether it is the fixture."""
    for name in CANDIDATES:
        path = Path(name)
        if path.exists():
            return path, name == "demo.db"
    return None, False


@st.cache_data(ttl=30)
def load_history(db: str, model: str | None) -> pd.DataFrame:
    """Runs for one model, oldest first, with a DriftScore for each."""
    store = MonitoringStore(db)
    runs = store.history(model, limit=500).sort_values("run_id").reset_index(drop=True)
    perf = store.performance_history(model)

    change = {}
    for _, row in perf.iterrows():
        evaluable = row.get("evaluable")
        if pd.notna(row["relative_change"]) and not (pd.notna(evaluable) and evaluable == 0):
            change[row["run_id"]] = float(row["relative_change"])

    scores, modes = [], []
    for _, run in runs.iterrows():
        result = compute_driftscore(
            max_psi=float(run["max_psi"] or 0.0),
            n_drifted=int(run["n_drifted"] or 0),
            n_features=int(run["n_features"] or 0),
            relative_change=change.get(run["run_id"]),
        )
        scores.append(result.score)
        modes.append(result.mode)
    runs["driftscore"], runs["score_mode"] = scores, modes
    runs["label"] = [
        r.batch_label if r.batch_label else f"run {r.run_id}" for r in runs.itertuples()
    ]
    return runs


@st.cache_data(ttl=30)
def load_performance(db: str, model: str | None) -> pd.DataFrame:
    return MonitoringStore(db).performance_history(model)


def state_of(run: pd.Series, column: str) -> str:
    value = run.get(column)
    if isinstance(value, str) and value in STATES:
        return value
    return "unknown"


def severity_for(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


# --------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------

def verdict(run: pd.Series) -> str:
    """One sentence for the batch, written from its three findings."""
    quality = state_of(run, "data_quality")
    shift = state_of(run, "input_shift")
    perf = state_of(run, "performance_status")
    moved = shift in ("watch", "significant")

    if quality == "failed":
        return "This batch arrived broken. Nothing measured on it can be trusted."
    if perf == "degraded" and not moved:
        return "The model got worse, and the inputs don't show it."
    if perf == "degraded":
        return "The model got worse, and its inputs have shifted."
    if perf == "insufficient_evidence":
        return "There are too few outcomes to judge the model yet."
    if perf == "not_observed" and moved:
        return "The inputs have shifted. Performance hasn't been measured."
    if perf == "not_observed" and shift == "stable":
        return "The inputs look like the reference. Performance hasn't been measured."
    if moved:
        return "The inputs have shifted. Measured performance is holding."
    if quality == "warning":
        return "The model is holding, but the batch has data-quality warnings."
    if "unknown" in (quality, shift, perf):
        return "This run predates the three-part health record."
    return "Nothing has moved. The model performs as it did at baseline."


def detail_line(run: pd.Series, features: pd.DataFrame, perf: pd.DataFrame) -> str:
    parts = []
    if not perf.empty and pd.notna(perf.iloc[0]["current"]):
        row = perf.iloc[0]
        parts.append(
            f"{metric_name(row['metric'])} is <b>{row['current']:.3f}</b> against a "
            f"baseline of <b>{row['baseline']:.3f}</b> (<b>{row['relative_change']:+.1%}</b>)."
        )
    if not features.empty:
        top = features.sort_values("psi", ascending=False).iloc[0]
        drifted = int((features["severity"] != "stable").sum())
        parts.append(
            f"<b>{drifted}</b> of <b>{len(features)}</b> features drifted; the largest shift is "
            f"{html.escape(str(top['feature']))} at PSI <b>{top['psi']:.3f}</b>."
        )
    mode = "inputs only, since performance is unmeasured" if run["score_mode"] == "inputs_only" else band_for(run["driftscore"])
    parts.append(f"DriftScore <b>{run['driftscore']:.0f}</b> of 100 ({mode}).")
    return " ".join(parts)


# --------------------------------------------------------------------------
# The health strip
# --------------------------------------------------------------------------

def health_strip(runs: pd.DataFrame, picked_run_id: int, limit: int = 30) -> str:
    """Three rows of cells, one column per batch, newest on the right."""
    shown = runs.tail(limit)
    cells = []
    for column, title in DIMENSIONS:
        cells.append(f"<div class='v-row-label'>{title}</div>")
        for i, (_, run) in enumerate(shown.iterrows()):
            css, _, glyph, words = STATES[state_of(run, column)]
            picked = " picked" if run["run_id"] == picked_run_id else ""
            label = html.escape(f"{run['label']}: {title.lower()} {words}")
            cells.append(
                f"<div class='v-cell {css}{picked}' style='--i:{i}' "
                f"role='img' aria-label='{label}' title='{label}'>{glyph}</div>"
            )
    cells.append("<div></div>")
    step = max(1, len(shown) // 12)
    for i, (_, run) in enumerate(shown.iterrows()):
        picked = " picked" if run["run_id"] == picked_run_id else ""
        text = html.escape(str(run["label"])) if i % step == 0 or picked else ""
        cells.append(f"<div class='v-tick{picked}'>{text}</div>")

    def key(colour: str, text: str, style: str = "") -> str:
        return f"<span class='v-key'><i style='background:{colour};{style}'></i>{text}</span>"

    legend = "".join([
        key(GOOD, "as expected"),
        key(WATCH, "△ worth a look"),
        key(SHIFT, "▲ inputs shifted significantly"),
        key(BAD, "✕ batch broken, or performance degraded"),
        key("transparent", "? too little evidence", f"border:1.5px dashed {NONE}"),
        key(RULE, "– not measured"),
    ])
    return (
        f"<div class='v-strip' style='--n:{len(shown)}'>{''.join(cells)}</div>"
        f"<div class='v-legend'>{legend}</div>"
    )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def base_layout(fig: go.Figure, height: int, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=64, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=MUTED, size=12),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(color=INK)),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE, type="category", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=RULE, zeroline=False, title=ytitle)
    return fig


def threshold(fig: go.Figure, y: float, text: str, colour: str) -> None:
    """A reference line labelled at its right-hand end, where the eye finishes."""
    fig.add_hline(y=y, line=dict(color=colour, width=1, dash="dot"))
    fig.add_annotation(
        x=1, xref="paper", y=y, text=text, showarrow=False, xanchor="left",
        xshift=6, font=dict(color=colour, size=11),
    )


def psi_chart(labels: list[str], values: list[float], picked: str | None = None,
              height: int = 280) -> go.Figure:
    ceiling = max(max(values, default=0) * 1.2, PSI_SIGNIFICANT * 1.35)
    fig = go.Figure(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color=INK, width=2),
        marker=dict(
            size=[13 if lab == picked else 9 for lab in labels],
            color=[SEVERITY_COLOUR[severity_for(v)] for v in values],
            line=dict(color=GROUND, width=2),
        ),
        hovertemplate="PSI %{y:.4f}<extra></extra>",
    ))
    threshold(fig, PSI_MODERATE, "look", WATCH)
    threshold(fig, PSI_SIGNIFICANT, "significant", SHIFT)
    fig.update_yaxes(range=[0, ceiling])
    return base_layout(fig, height, "Largest PSI in the batch")


def feature_bar(features: pd.DataFrame, height: int = 280) -> go.Figure:
    frame = features.sort_values("psi")
    fig = go.Figure(go.Bar(
        x=frame["psi"], y=frame["feature"], orientation="h",
        marker=dict(color=[SEVERITY_COLOUR.get(s, GOOD) for s in frame["severity"]],
                    line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>PSI %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=PSI_MODERATE, line=dict(color=WATCH, width=1, dash="dot"))
    fig.add_vline(x=PSI_SIGNIFICANT, line=dict(color=SHIFT, width=1, dash="dot"))
    fig = base_layout(fig, height)
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=16, t=12, b=8))
    fig.update_xaxes(title="PSI", type="linear", range=[0, max(frame["psi"].max() * 1.15, PSI_SIGNIFICANT * 1.2)])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return fig


def performance_chart(perf: pd.DataFrame, labels: dict, picked: str | None,
                      height: int = 300) -> go.Figure:
    """The metric per batch with its 95% interval, against the baseline."""
    frame = perf[perf["current"].notna()].copy()
    frame["label"] = frame["run_id"].map(labels)
    has_ci = "ci_low" in frame.columns and frame["ci_low"].notna().any()
    fig = go.Figure(go.Scatter(
        x=frame["label"], y=frame["current"], mode="lines+markers",
        line=dict(color=INK, width=2),
        marker=dict(
            size=[13 if lab == picked else 9 for lab in frame["label"]],
            color=[BAD if d else GOOD for d in frame["degraded"]],
            line=dict(color=GROUND, width=2),
        ),
        error_y=dict(
            type="data", symmetric=False,
            array=(frame["ci_high"] - frame["current"]).fillna(0),
            arrayminus=(frame["current"] - frame["ci_low"]).fillna(0),
            color=MUTED, thickness=1.2, width=4,
        ) if has_ci else None,
        hovertemplate="%{y:.4f}<extra></extra>",
    ))
    if not frame.empty:
        threshold(fig, float(frame["baseline"].iloc[0]), "baseline", MUTED)
    return base_layout(fig, height, metric_name(frame["metric"].iloc[0]) if not frame.empty else "")


def score_chart(runs: pd.DataFrame, picked: str | None, height: int = 260) -> go.Figure:
    full = runs["score_mode"] == "full"
    fig = go.Figure(go.Scatter(
        x=runs["label"], y=runs["driftscore"], mode="lines+markers",
        line=dict(color=INK, width=2),
        marker=dict(
            size=[13 if lab == picked else 9 for lab in runs["label"]],
            color=[BAND_COLOUR[band_for(s)] if f else GROUND for s, f in zip(runs["driftscore"], full)],
            line=dict(
                color=[GROUND if f else BAND_COLOUR[band_for(s)] for s, f in zip(runs["driftscore"], full)],
                width=2,
            ),
        ),
        customdata=["" if f else " (inputs only)" for f in full],
        hovertemplate="DriftScore %{y:.0f}%{customdata}<extra></extra>",
    ))
    threshold(fig, BAND_HEALTHY, "healthy", GOOD)
    threshold(fig, BAND_WATCH, "watch", WATCH)
    threshold(fig, BAND_DEGRADED, "degraded", SHIFT)
    fig.update_yaxes(range=[0, 104])
    return base_layout(fig, height, "DriftScore")


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def note(text: str, tone: str = MUTED) -> None:
    st.markdown(f"<div class='v-note' style='--tone:{tone}'>{text}</div>", unsafe_allow_html=True)


def section(title: str, lede: str) -> None:
    st.markdown(
        f"<div class='v-section'><h2>{title}</h2></div><p class='v-lede'>{lede}</p>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    db_path, is_fixture = resolve_db()
    if db_path is None:
        st.markdown("<div class='v-verdict'>No monitoring runs to show yet.</div>", unsafe_allow_html=True)
        st.markdown(
            "<p class='v-detail'>Run <code>python demo/run_demo.py</code> to generate six "
            "months of monitored batches, then reload this page.</p>",
            unsafe_allow_html=True,
        )
        st.stop()

    store = MonitoringStore(db_path)
    all_runs = store.history(limit=1000)
    if all_runs.empty:
        st.markdown("<div class='v-verdict'>The database has no runs yet.</div>", unsafe_allow_html=True)
        st.markdown(
            "<p class='v-detail'>Call <code>ModelMonitor.check()</code> on a batch and it "
            "will appear here.</p>",
            unsafe_allow_html=True,
        )
        st.stop()

    models = sorted(all_runs["model_name"].unique())
    with st.sidebar:
        st.markdown("<div class='v-brand'>Valorian Intelligence</div>", unsafe_allow_html=True)
        st.caption("Watches a deployed model and keeps three questions apart.")
        model = st.selectbox("Model", models)
        runs = load_history(str(db_path), model)
        labels = runs["label"].tolist()
        picked_label = st.selectbox("Batch", labels[::-1], help="Newest first.")
        st.divider()
        st.caption(f"Reading `{db_path.name}`")
        st.caption("[Source on GitHub](https://github.com/Don4chnge/valorian-intelligence)")

    run = runs[runs["label"] == picked_label].iloc[-1]
    run_id = int(run["run_id"])
    detail = store.run_detail(run_id)
    features = detail["features"].drop(columns=["run_id"])
    perf = detail["performance"]
    perf_history = load_performance(str(db_path), model)

    # ---- Verdict and strip ------------------------------------------------
    if is_fixture:
        st.markdown(
            "<div class='v-provenance'>Synthetic demo data. Drift was injected on purpose, "
            "so the detector can be checked against a known answer.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div class='v-batch'>{html.escape(str(model))}, batch {html.escape(picked_label)}</div>"
        f"<div class='v-verdict'>{verdict(run)}</div>"
        f"<p class='v-detail'>{detail_line(run, features, perf)}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(health_strip(runs, run_id), unsafe_allow_html=True)

    # ---- 1. Did the batch arrive intact? ----------------------------------
    section(
        "Did the batch arrive intact?",
        "Missing, extra and wrong-typed columns, and jumps in null rates. Drift statistics "
        "are computed on the values that arrived, so they cannot see any of this.",
    )
    issues = json.loads(detail["run"].get("quality_issues") or "[]")
    quality = state_of(run, "data_quality")
    if issues:
        tone = BAD if quality == "failed" else WATCH
        note("<br>".join(html.escape(i) for i in issues), tone)
    elif quality == "unknown":
        note("This run was stored before data-quality checks existed.")
    else:
        note("Every expected column arrived, with null rates in line with the reference.", GOOD)

    # ---- 2. Have the inputs changed? --------------------------------------
    section(
        "Have the inputs changed?",
        "Each feature against the data the model was trained on. A shift here is a reason "
        "to look, <b>not evidence the model is wrong</b>; that is the next question.",
    )
    if features.empty:
        note("No feature could be compared in this batch.", BAD)
    else:
        left, right = st.columns([1, 1.15])
        with left:
            st.caption(f"Batch {picked_label}, feature by feature")
            plot(feature_bar(features))
        with right:
            st.caption("Largest shift in each batch")
            plot(psi_chart(labels, runs["max_psi"].fillna(0).tolist(), picked_label))

        outside = features[features.get("out_of_range", pd.Series(dtype=float)).fillna(0) > 0] \
            if "out_of_range" in features.columns else features.iloc[0:0]
        for _, r in outside.iterrows():
            note(
                f"<b>{html.escape(str(r['feature']))}</b>: {r['out_of_range']:.1%} of values lie "
                "beyond anything in the reference data. PSI barely registers values like "
                "these, so they are counted separately.", SHIFT,
            )

        mv = detail.get("multivariate")
        if mv:
            drivers = [(k, v) for k, v in mv["contributions"].items() if v >= 0.005][:4]
            p_text = f", permutation p = {mv['p_value']:.3f}" if mv["p_value"] is not None else ""
            st.markdown(
                f"<p class='v-lede'><b>Taken together</b>, the features of this batch are "
                f"<b>{html.escape(mv['separability'])}</b> from the reference to a classifier "
                f"trained to tell them apart (AUC {mv['auc']:.3f}, where 0.50 is chance{p_text})."
                + (" What it relied on:" if drivers else "") + "</p>",
                unsafe_allow_html=True,
            )
            if drivers:
                rows = "".join(
                    f"<span>{html.escape(str(k))}</span>"
                    f"<span class='t'><i style='width:{v * 100:.0f}%'></i></span>"
                    f"<span class='n'>{v:.0%}</span>"
                    for k, v in drivers
                )
                st.markdown(f"<div class='v-bars'>{rows}</div>", unsafe_allow_html=True)

        with st.expander("Every feature in this batch"):
            columns = [c for c in ["feature", "psi", "severity", "out_of_range",
                                   "missing_current", "p_value"] if c in features.columns]
            st.dataframe(
                features[columns], use_container_width=True, hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("Feature"),
                    "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
                    "severity": st.column_config.TextColumn("Severity"),
                    "out_of_range": st.column_config.NumberColumn("Beyond reference", format="percent"),
                    "missing_current": st.column_config.NumberColumn("Null rate", format="percent"),
                    "p_value": st.column_config.NumberColumn("p-value", format="%.2e"),
                },
            )
            feature = st.selectbox("Trace one feature across batches", sorted(features["feature"]))
            trace = store.feature_history(feature, model)
            if not trace.empty:
                trace_labels = trace["run_id"].map(dict(zip(runs["run_id"], runs["label"]))).tolist()
                plot(psi_chart(trace_labels, trace["psi"].tolist(), picked_label, height=240))

    # ---- 3. Is the model still right? -------------------------------------
    section(
        "Is the model still right?",
        "The model's metric on real outcomes, against its baseline. This is the only one "
        "of the three that is evidence about the model, and it arrives last.",
    )
    perf_state = state_of(run, "performance_status")
    if perf.empty:
        note("No outcomes have been recorded for this batch, so performance is unmeasured. "
             "That is not the same as healthy.")
    else:
        row = perf.iloc[0]
        unevaluable = "evaluable" in perf.columns and row["evaluable"] == 0
        if unevaluable:
            so_far = f" So far it reads {row['current']:.3f}." if pd.notna(row["current"]) else ""
            note(
                f"<b>{metric_name(row['metric'])} was not judged</b>: "
                f"{html.escape(str(row['reason']))}.{so_far} Performance is unverified for "
                "this batch, not confirmed healthy.", NONE,
            )
        else:
            interval = ""
            if "ci_low" in perf.columns and pd.notna(row["ci_low"]):
                interval = f" The 95% interval is {row['ci_low']:.3f} to {row['ci_high']:.3f}."
            words = "degraded" if row["degraded"] else "within tolerance"
            note(
                f"<b>{metric_name(row['metric'])} {row['current']:.3f}</b> against a baseline "
                f"of {row['baseline']:.3f}, a change of <b>{row['relative_change']:+.1%}</b>: "
                f"{words}.{interval}", BAD if row["degraded"] else GOOD,
            )
            if row["degraded"] and state_of(run, "input_shift") == "stable":
                note(
                    "Every input feature looks normal in this batch. A monitor that watched "
                    "inputs alone would have called it healthy.", BAD,
                )
        if "label_coverage" in perf.columns and pd.notna(row["label_coverage"]):
            delay = (
                f", arriving a median of {row['median_label_delay_days']:.0f} days after scoring"
                if pd.notna(row["median_label_delay_days"]) else ""
            )
            st.caption(f"Measured on outcomes for {row['label_coverage']:.0%} of this batch's predictions{delay}.")

    if perf_history["current"].notna().any():
        plot(performance_chart(perf_history, dict(zip(runs["run_id"], runs["label"])), picked_label))
        st.caption("Whiskers are 95% bootstrap intervals. A point is red when it fell past the tolerance.")

    cohorts = detail.get("cohorts")
    if cohorts is not None and not cohorts.empty:
        st.markdown("<p class='v-lede'><b>By cohort.</b> Each one is judged against its own "
                    "baseline, never the overall one.</p>", unsafe_allow_html=True)
        st.dataframe(
            cohorts[["cohort", "n", "current", "ci_low", "ci_high", "baseline", "status"]],
            use_container_width=True, hide_index=True,
            column_config={
                "cohort": st.column_config.TextColumn("Cohort"),
                "n": st.column_config.NumberColumn("Rows"),
                "current": st.column_config.NumberColumn("Score", format="%.4f"),
                "ci_low": st.column_config.NumberColumn("95% low", format="%.4f"),
                "ci_high": st.column_config.NumberColumn("95% high", format="%.4f"),
                "baseline": st.column_config.NumberColumn("Own baseline", format="%.4f"),
                "status": st.column_config.TextColumn("Status"),
            },
        )

    # ---- One number --------------------------------------------------------
    section(
        "One number, if you need one",
        "DriftScore folds the evidence above into 0 to 100. Hollow points had no outcomes "
        "yet and are scored on inputs alone, which answers a different question; don't "
        "read them on the same scale as the filled ones.",
    )
    plot(score_chart(runs, picked_label))


if __name__ == "__main__":
    main()
