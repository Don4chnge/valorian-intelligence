# Data

The QLFS microdata files are not committed — they are large and Stats SA is the
canonical source. Download them yourself:

**isibaloweb.statssa.gov.za/pages/surveys/pss/qlfs/qlfsp.php**

Select the year, expand the quarter, choose **Comma Separated Values**, and
extract the CSV into this folder. Six quarters are needed:

| File | Quarter | Role |
|---|---|---|
| `QLFS202104.csv` | 2021 Q4 | reference — model trains on this |
| `QLFS202201.csv` | 2022 Q1 | monitored |
| `QLFS202202.csv` | 2022 Q2 | monitored |
| `QLFS202203.csv` | 2022 Q3 | monitored |
| `QLFS202204.csv` | 2022 Q4 | monitored |
| `QLFS202301.csv` | 2023 Q1 | monitored |

Then run `python demo/run_qlfs.py`.

## Why this window

Stats SA collected the QLFS by telephone from 2020 Q2 to 2021 Q4 during COVID,
then reverted to face-to-face interviewing from 2022 Q1. The reference quarter
is the last telephone quarter; the first monitored quarter is the first
face-to-face one. Any drift detected at that boundary is a real methodology
break, not an injected one.

## Columns used

Stats SA ships around 160 columns per quarter, mostly raw survey codes. This
project uses the derived variables at the end of the file:

- `Q13GENDER`, `Q14AGE`, `Education_Status`, `Province`, `Geo_type_code` — features
- `Status` — 1 employed, 2 unemployed, 3 discouraged, 4 other not economically
  active. Collapsed to employed vs not.

Rows are filtered to ages 15-64 with a non-null status; under-15s carry no
labour market status.

`sector1` is deliberately **not** used. It is only recorded for people who
already have a job, so including it as a feature would leak the target.
