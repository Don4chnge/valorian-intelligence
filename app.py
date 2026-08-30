"""
Valorian Intelligence dashboard.

Reads whichever monitoring database is available and lets you browse runs,
drill into a single run's feature table, and trace one feature's PSI over time.

Database resolution, in order:
    valorian.db   written by demo/run_demo.py
    qlfs.db       written by demo/run_qlfs.py
    demo.db       committed fixture, so the deployed app has something to show

The fixture exists because the generated databases are gitignored. Without it a
hosted instance would greet every visitor with an empty state. It holds the
synthetic demo run and nothing else — the banner says so rather than letting
anyone mistake it for real data.

Design note: the charts draw the PSI interpretation thresholds as shaded bands
rather than leaving them to a caption. A drift line at 0.07 tells the reader
nothing unless they also know that 0.10 is where "investigate" begins, and
putting that below the chart makes them do the work themselves.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from valorian import PSI_MODERATE, PSI_SIGNIFICANT
from valorian.driftscore import band_for, compute_driftscore
from valorian.store import MonitoringStore

st.set_page_config(
    page_title="Valorian Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CANDIDATES = ["valorian.db", "qlfs.db", "demo.db"]

MUTED = "#8B93A7"
GRID = "#232838"
PANEL = "#171B26"
ACCENT = "#4C8DFF"

STATUS = {
    "stable": "#2FBF71",
    "moderate": "#E8A33D",
    "significant": "#E5484D",
}
BANDS = {
    "healthy": "#2FBF71",
    "watch": "#E8A33D",
    "degraded": "#F0803C",
    "critical": "#E5484D",
}

CSS = """
<style>
  /* Streamlit renders metrics as flat numbers on a flat background. Cards give
     the four headline figures visual weight matching how much they matter
     relative to the tables below. */
  div[data-testid="stMetric"] {
      background: #171B26;
      border: 1px solid #232838;
      border-radius: 10px;
      padding: 16px 18px;
  }
  div[data-testid="stMetricLabel"] p {
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #8B93A7;
  }
  div[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 600; }

  h1 { letter-spacing: -0.02em; font-weight: 700; }
  h2, h3 { letter-spacing: -0.01em; }

  section[data-testid="stSidebar"] { border-right: 1px solid #232838; }

  /* Streamlit is generous with vertical whitespace; the page reads better
     denser. */
  div.block-container { padding-top: 2.4rem; max-width: 1400px; }
  hr { border-color: #232838; }
</style>
"""

PILL = """
<span style="
    display:inline-block; padding:5px 14px; border-radius:999px;
    background:{bg}22; border:1px solid {bg}55; color:{bg};
    font-size:0.82rem; font-weight:600; letter-spacing:0.05em;
    text-transform:uppercase;">{label}</span>
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
    return MonitoringStore(db).history(model)


def severity_for(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def driftscore_for(detail: dict) -> float | None:
    features = detail["features"]
    if features.empty:
        return None
    perf = detail["performance"]
    rel = float(perf.iloc[0]["relative_change"]) if not perf.empty else None
    return compute_driftscore(
        max_psi=float(features["psi"].max()),
        n_drifted=int((features["severity"] != "stable").sum()),
        n_features=len(features),
        relative_change=rel,
    ).score


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def base_layout(fig: go.Figure, height: int, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle)
    return fig


def psi_chart(labels: list[str], values: list[float], height: int = 300) -> go.Figure:
    """PSI over time, with the interpretation thresholds drawn as bands."""
    ceiling = max(max(values, default=0) * 1.25, PSI_SIGNIFICANT * 1.4)
    fig = go.Figure()

    for y0, y1, colour in [
        (0, PSI_MODERATE, STATUS["stable"]),
        (PSI_MODERATE, PSI_SIGNIFICANT, STATUS["moderate"]),
        (PSI_SIGNIFICANT, ceiling, STATUS["significant"]),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=colour, opacity=0.07,
                      layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color=ACCENT, width=2.5, shape="spline", smoothing=0.4),
        marker=dict(
            size=9,
            color=[STATUS[severity_for(v)] for v in values],
            line=dict(color="#0E1117", width=2),
        ),
        hovertemplate="<b>%{x}</b><br>PSI %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=PSI_MODERATE, line=dict(color=STATUS["moderate"], width=1, dash="dot"))
    fig.add_hline(y=PSI_SIGNIFICANT, line=dict(color=STATUS["significant"], width=1, dash="dot"))
    fig.update_yaxes(range=[0, ceiling])
    return base_layout(fig, height, "Max PSI")


def feature_bar(features: pd.DataFrame, height: int = 300) -> go.Figure:
    """Horizontal bars ranked by PSI, coloured against the thresholds."""
    frame = features.sort_values("psi")

    fig = go.Figure(go.Bar(
        x=frame["psi"], y=frame["feature"], orientation="h",
        marker=dict(
            color=[STATUS[severity_for(v)] for v in frame["psi"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>PSI %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=PSI_MODERATE, line=dict(color=STATUS["moderate"], width=1, dash="dot"))
    fig.add_vline(x=PSI_SIGNIFICANT, line=dict(color=STATUS["significant"], width=1, dash="dot"))

    fig = base_layout(fig, height)
    fig.update_xaxes(title="PSI")
    fig.update_yaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    db_path, is_fixture = resolve_db()
    if db_path is None:
        st.title("Valorian Intelligence")
        st.warning("No monitoring database found. Run `python demo/run_demo.py` first.")
        st.stop()

    store = MonitoringStore(db_path)
    all_runs = store.history()
    models = sorted(all_runs["model_name"].unique()) if not all_runs.empty else []

    with st.sidebar:
        st.markdown("### Valorian Intelligence")
        st.caption("Drift detection and performance monitoring for production ML models")
        st.divider()
        model = st.selectbox("Model", models) if models else None
        st.divider()
        st.caption(f"Source `{db_path.name}`")
        st.caption("[View on GitHub](https://github.com/Don4chnge/valorian-intelligence)")

    if all_runs.empty:
        st.title("Valorian Intelligence")
        st.info("The database exists but has no runs yet.")
        st.stop()

    runs = load_history(str(db_path), model)
    latest = runs.iloc[0]
    detail = store.run_detail(int(latest["run_id"]))
    score = driftscore_for(detail)

    # ---- Header ----------------------------------------------------------
    left, right = st.columns([3, 1])
    with left:
        st.title("Valorian Intelligence")
    with right:
        if score is not None:
            band = band_for(score)
            st.markdown(
                f"<div style='text-align:right;padding-top:26px'>"
                f"{PILL.format(bg=BANDS[band], label=band)}</div>",
                unsafe_allow_html=True,
            )

    if is_fixture:
        st.info(
            "Showing the bundled synthetic demo run — drift was injected on purpose "
            "so the detector could be checked against a known ground truth. "
            "Run `python demo/run_qlfs.py` locally for the same monitoring against "
            "real Stats SA labour force data."
        )

    # ---- Headline --------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest batch", latest["batch_label"] or "—")
    c2.metric(
        "DriftScore", f"{score:.0f}" if score is not None else "—",
        help="0–100 composite health score. 100 is a model behaving as it did at training time.",
    )
    c3.metric("Features drifted", int(latest["n_drifted"]))
    c4.metric("Max PSI", f"{latest['max_psi']:.3f}")

    st.write("")

    # ---- Trend -----------------------------------------------------------
    ordered = runs.sort_values("run_id")
    st.subheader("Drift over time")
    st.plotly_chart(
        psi_chart(ordered["batch_label"].tolist(), ordered["max_psi"].tolist()),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(
        "Shaded bands mark the PSI interpretation thresholds. A high PSI does not "
        "by itself mean the model is broken — check the performance line below."
    )

    st.divider()

    # ---- Run detail ------------------------------------------------------
    st.subheader("Run detail")
    labels = {
        f"#{r.run_id} · {r.batch_label or 'unlabelled'}": r.run_id
        for r in runs.itertuples()
    }
    chosen = st.selectbox("Select a run", list(labels), label_visibility="collapsed")
    chosen_detail = store.run_detail(labels[chosen])
    features = chosen_detail["features"].drop(columns=["run_id"])

    chart_col, table_col = st.columns([1, 1.3])
    with chart_col:
        st.plotly_chart(feature_bar(features), use_container_width=True,
                        config={"displayModeBar": False})
    with table_col:
        st.dataframe(
            features[["feature", "psi", "severity", "p_value"]],
            use_container_width=True, hide_index=True, height=300,
            column_config={
                "feature": st.column_config.TextColumn("Feature"),
                "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
                "severity": st.column_config.TextColumn("Severity"),
                "p_value": st.column_config.NumberColumn("p-value", format="%.2e"),
            },
        )

    perf = chosen_detail["performance"]
    if not perf.empty:
        row = perf.iloc[0]
        colour = STATUS["significant"] if row["degraded"] else STATUS["stable"]
        verdict = "DEGRADED" if row["degraded"] else "within tolerance"
        st.markdown(
            f"<div style='background:{PANEL};border:1px solid {GRID};"
            f"border-radius:10px;padding:14px 18px'>"
            f"<b>{row['metric']}</b> &nbsp;·&nbsp; baseline {row['baseline']:.4f} "
            f"&nbsp;→&nbsp; current {row['current']:.4f} &nbsp;"
            f"<span style='color:{colour};font-weight:600'>"
            f"({row['relative_change']:+.1%}) {verdict}</span></div>",
            unsafe_allow_html=True,
        )

        if row["degraded"] and features["severity"].eq("stable").all():
            st.warning(
                "**Performance dropped while every input feature looks stable.** "
                "Input-only monitoring would report this batch as healthy. "
                "Suspect concept drift or a data pipeline fault — this is the case "
                "univariate drift detection cannot see."
            )

    st.divider()

    # ---- Feature trace ---------------------------------------------------
    st.subheader("Feature trace")
    feature = st.selectbox("Feature", sorted(features["feature"]),
                           label_visibility="collapsed")
    trace = store.feature_history(feature, model)
    if not trace.empty:
        st.plotly_chart(
            psi_chart(trace["batch_label"].tolist(), trace["psi"].tolist(), height=260),
            use_container_width=True,
            config={"displayModeBar": False},
        )


if __name__ == "__main__":
    main()
