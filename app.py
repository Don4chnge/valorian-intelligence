"""
Valorian Intelligence dashboard.

Reads whichever monitoring database is available and lets you browse runs,
drill into a single run's feature table, and trace one feature's PSI over time.

Database resolution, in order:
    valorian.db   written by demo/run_demo.py
    qlfs.db       written by demo/run_qlfs.py
    demo.db       committed fixture, so the deployed app has something to show

The fixture exists because the generated databases are gitignored. Without it a
hosted instance would greet every visitor with an empty state. It contains the
synthetic demo run and nothing else — the banner says so rather than letting
anyone mistake it for real data.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from valorian import PSI_MODERATE, PSI_SIGNIFICANT
from valorian.driftscore import BAND_DEGRADED, BAND_HEALTHY, BAND_WATCH, band_for
from valorian.store import MonitoringStore

st.set_page_config(page_title="Valorian Intelligence", page_icon="📉", layout="wide")

CANDIDATES = ["valorian.db", "qlfs.db", "demo.db"]

SEVERITY_COLOUR = {
    "stable": "#1a7f4b",
    "moderate": "#b8860b",
    "significant": "#b3261e",
}
BAND_COLOUR = {
    "healthy": "#1a7f4b",
    "watch": "#b8860b",
    "degraded": "#c85a19",
    "critical": "#b3261e",
}


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


def psi_colour(value: float) -> str:
    """
    Colour a PSI cell against the interpretation thresholds, not against the
    other values in the table. A relative gradient makes the largest number in
    a run look alarming even when every feature is stable — which is exactly
    backwards for the case this project exists to show.
    """
    return f"color: {SEVERITY_COLOUR[band_for_psi(value)]}"


def band_for_psi(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def main() -> None:
    st.title("Valorian Intelligence")
    st.caption("Drift detection and performance monitoring for production ML models")

    db_path, is_fixture = resolve_db()
    if db_path is None:
        st.warning(
            "No monitoring database found. Run `python demo/run_demo.py` to generate one."
        )
        st.stop()

    if is_fixture:
        st.info(
            "Showing the bundled synthetic demo run. Drift was injected on purpose "
            "so the detector could be checked against a known ground truth. "
            "Run `python demo/run_qlfs.py` locally to see the same monitoring "
            "against real Stats SA labour force data."
        )

    store = MonitoringStore(db_path)
    all_runs = store.history()

    if all_runs.empty:
        st.info("The database exists but has no runs yet.")
        st.stop()

    models = sorted(all_runs["model_name"].unique())
    model = st.sidebar.selectbox("Model", models)
    runs = load_history(str(db_path), model)

    st.sidebar.caption(f"Reading `{db_path.name}` · {len(runs)} runs")

    # ---- Headline ---------------------------------------------------------
    latest = runs.iloc[0]
    detail = store.run_detail(int(latest["run_id"]))
    perf = detail["performance"]

    score = driftscore_for(detail)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest batch", latest["batch_label"] or "—")
    c2.metric("DriftScore", f"{score:.0f}/100" if score is not None else "—")
    c3.metric("Features drifted", int(latest["n_drifted"]))
    c4.metric("Max PSI", f"{latest['max_psi']:.3f}")

    if score is not None:
        band = band_for(score)
        st.markdown(
            f"<span style='color:{BAND_COLOUR[band]};font-weight:600;font-size:1.1rem'>"
            f"{band.upper()}</span>",
            unsafe_allow_html=True,
        )

    if latest["status"] == "critical":
        st.error("Latest run is critical — inspect the feature table below.")
    elif latest["status"] == "warning":
        st.warning("Latest run shows moderate drift.")

    st.divider()

    # ---- Trend ------------------------------------------------------------
    st.subheader("Drift over time")
    trend = runs.sort_values("run_id").set_index("batch_label")
    st.line_chart(trend["max_psi"], height=260)
    st.caption(
        f"PSI thresholds — below {PSI_MODERATE} stable, "
        f"{PSI_MODERATE}–{PSI_SIGNIFICANT} moderate, above {PSI_SIGNIFICANT} significant. "
        "A high PSI does not by itself mean the model is broken; check the "
        "performance line in the run detail below."
    )

    st.divider()

    # ---- Run detail -------------------------------------------------------
    st.subheader("Run detail")
    labels = {
        f"#{r.run_id} · {r.batch_label or 'unlabelled'} · {r.status}": r.run_id
        for r in runs.itertuples()
    }
    chosen = st.selectbox("Select a run", list(labels))
    chosen_detail = store.run_detail(labels[chosen])

    features = chosen_detail["features"].drop(columns=["run_id"])
    st.dataframe(
        features.style.map(psi_colour, subset=["psi"]).format({
            "psi": "{:.4f}",
            "statistic": "{:.4f}",
            "p_value": "{:.2e}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    chosen_perf = chosen_detail["performance"]
    if not chosen_perf.empty:
        row = chosen_perf.iloc[0]
        colour = "#b3261e" if row["degraded"] else "#1a7f4b"
        verdict = "DEGRADED" if row["degraded"] else "within tolerance"
        st.markdown(
            f"**{row['metric']}** — baseline {row['baseline']:.4f}, "
            f"current {row['current']:.4f} "
            f"(<span style='color:{colour}'>{row['relative_change']:+.1%}</span>) · "
            f"<span style='color:{colour}'>{verdict}</span>",
            unsafe_allow_html=True,
        )

        if row["degraded"] and features["severity"].eq("stable").all():
            st.warning(
                "Performance has dropped while every input feature looks stable. "
                "Input-only monitoring would report this batch as healthy. "
                "Suspect concept drift or a pipeline fault."
            )

    st.divider()

    # ---- Feature trace ----------------------------------------------------
    st.subheader("Feature trace")
    feature = st.selectbox("Feature", sorted(features["feature"]))
    trace = store.feature_history(feature, model)
    if not trace.empty:
        st.line_chart(trace.set_index("batch_label")["psi"], height=240)
        st.dataframe(trace, use_container_width=True, hide_index=True)


def driftscore_for(detail: dict) -> float | None:
    """Recompute DriftScore from a stored run."""
    from valorian.driftscore import compute_driftscore

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


if __name__ == "__main__":
    main()
