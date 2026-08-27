"""
Valorian Intelligence dashboard.

Reads the SQLite database written by ModelMonitor and lets you browse runs,
drill into a single run's feature table, and trace one feature's PSI over time.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from valorian import PSI_MODERATE, PSI_SIGNIFICANT
from valorian.store import MonitoringStore

st.set_page_config(page_title="Valorian Intelligence", page_icon="~", layout="wide")

DB_PATH = Path("valorian.db")

STATUS_COLOUR = {"ok": "#1a7f4b", "warning": "#b8860b", "critical": "#b3261e"}


@st.cache_data(ttl=30)
def load_history(model: str | None) -> pd.DataFrame:
    return MonitoringStore(DB_PATH).history(model)


def main() -> None:
    st.title("Valorian Intelligence")
    st.caption("Model drift and performance monitoring")

    if not DB_PATH.exists():
        st.warning("No monitoring database found. Run `python demo/run_demo.py` first.")
        st.stop()

    store = MonitoringStore(DB_PATH)
    all_runs = store.history()

    if all_runs.empty:
        st.info("The database exists but has no runs yet.")
        st.stop()

    models = sorted(all_runs["model_name"].unique())
    model = st.sidebar.selectbox("Model", models)
    runs = load_history(model)

    # ---- Headline numbers ------------------------------------------------
    latest = runs.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest batch", latest["batch_label"] or "—")
    c2.metric("Status", str(latest["status"]).upper())
    c3.metric("Features drifted", int(latest["n_drifted"]))
    c4.metric("Max PSI", f"{latest['max_psi']:.3f}")

    if latest["status"] == "critical":
        st.error("Latest run is critical — inspect the feature table below.")
    elif latest["status"] == "warning":
        st.warning("Latest run shows moderate drift.")

    st.divider()

    # ---- Max PSI over time ----------------------------------------------
    st.subheader("Drift over time")
    trend = runs.sort_values("run_id").set_index("batch_label")[["max_psi", "n_drifted"]]
    st.line_chart(trend["max_psi"], height=260)
    st.caption(
        f"PSI thresholds — below {PSI_MODERATE} stable, "
        f"{PSI_MODERATE}–{PSI_SIGNIFICANT} moderate, above {PSI_SIGNIFICANT} significant."
    )

    st.divider()

    # ---- Single run drill-down ------------------------------------------
    st.subheader("Run detail")
    labels = {
        f"#{r.run_id} · {r.batch_label or 'unlabelled'} · {r.status}": r.run_id
        for r in runs.itertuples()
    }
    chosen = st.selectbox("Select a run", list(labels))
    detail = store.run_detail(labels[chosen])

    features = detail["features"].drop(columns=["run_id"])
    st.dataframe(
        features.style.background_gradient(subset=["psi"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
    )

    perf = detail["performance"]
    if not perf.empty:
        row = perf.iloc[0]
        verdict = "DEGRADED" if row["degraded"] else "within tolerance"
        st.write(
            f"**{row['metric']}** — baseline {row['baseline']:.4f}, "
            f"current {row['current']:.4f} ({row['relative_change']:+.1%}) · {verdict}"
        )

    st.divider()

    # ---- Per-feature trace ----------------------------------------------
    st.subheader("Feature trace")
    feature = st.selectbox("Feature", sorted(features["feature"]))
    trace = store.feature_history(feature, model)
    if not trace.empty:
        st.line_chart(trace.set_index("batch_label")["psi"], height=240)
        st.dataframe(trace, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
