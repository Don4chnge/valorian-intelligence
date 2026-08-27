# Roadmap

The scaffold in this repo is roughly the end of week 1. What follows is what turns it from a working prototype into something defensible in an interview.

## Week 1 — make it yours

The code here is a starting point, not a deliverable. Before anything else:

- [ ] Read `drift.py` line by line until you can explain PSI from memory — why the log term, why bin edges are frozen at training time, why empty bins get clipped. **This is the part you will be questioned on.** Being unable to derive your own headline metric in an interview is worse than not having the project.
- [ ] Rewrite at least one function in your own style so the codebase reads like you wrote it
- [ ] Push to GitHub with the demo output pasted into the README

## Week 2 — validate on real data

Synthetic data proves the detector catches planted drift. It proves nothing about the real world. Pick one:

- [ ] **UCI Bank Marketing / Credit Default** — well-known, easy to explain, has a natural time ordering
- [ ] **A South African dataset** — StatsSA, Eskom, or data.gov.za. This is the stronger option: it connects to your load shedding project and almost nobody else's portfolio has it. Train on 2022 data, monitor 2023–2024, see what actually moved.

Then:

- [ ] Add a `notebooks/` walkthrough — load, train, monitor, interpret. Notebooks are how reviewers skim a project in 90 seconds.
- [ ] Write up one real finding: a feature that genuinely drifted and what it meant

## Week 3 — depth and polish

Pick **one** of these and do it properly, rather than all three badly:

- [ ] **Multivariate drift** — train a classifier to distinguish reference from current data. If it can tell them apart (AUC well above 0.5), something drifted that univariate tests missed. Roughly 40 lines and it closes the biggest gap in the limitations section.
- [ ] **Drift explanations** — when performance drops, rank features by contribution to the drop rather than just by PSI
- [ ] **Alert routing** — email or Slack webhook on critical status, plus a `valorian check` CLI so it can run from cron

Then finish:

- [ ] GitHub Actions running `pytest` on push (a green badge signals you test your work)
- [ ] Deploy the dashboard to Streamlit Community Cloud — free, and a live link beats a screenshot
- [ ] Add screenshots to the README

## Resume rewrite

Once week 2 is done, replace the current Valorian bullets. The existing ones ("scalable AI solutions", "responsible and ethical AI") describe nothing specific and read as filler next to your load shedding entry. Something closer to:

> **Valorian Intelligence — ML Observability Toolkit** (2026)
> - Built a Python library that detects data and concept drift in deployed ML models using PSI, Kolmogorov–Smirnov and chi-square tests, with SQLite-backed run history and a Streamlit dashboard.
> - Validated against [real dataset], catching an 11% ROC-AUC degradation that univariate input monitoring alone did not surface.
> - 23-test suite covering both detection accuracy and false-positive resistance on stable data.

Numbers and named methods. That is what makes your load shedding entry (r = −0.649, 49 months of Eskom and StatsSA data) read as real, and it is the same thing this entry needs.

## Deliberately out of scope

Named so you do not drift into them — a finished narrow project beats an abandoned broad one:

- Model serving or a prediction API — different problem, use MLflow or BentoML
- A feature store
- Deep learning drift (embeddings, images) — different techniques entirely
- Multi-tenant auth, user accounts, billing
