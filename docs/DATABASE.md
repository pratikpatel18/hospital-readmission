# Data & Storage

**There is no database server in this project.** Data lives in CSV files and model files on disk. That is enough for ~500k rows and a single-node service. A relational schema is given in section 6 for when the clinics want to host the data in PostgreSQL.

## 1. Storage map
| Location | Content | Created by | In zip? |
|---|---|---|---|
| `data/raw/admissions.csv`, `opd_visits.csv`, `lab_tests.csv`, `billing.csv` | Raw tables (messy) | `python -m src.data_generation` | no (regenerated, deterministic seed 42) |
| `data/modeling_table.csv` | Cleaned, merged, 1 row per admission | `src/train.py` | yes (~11 MB) |
| `artifacts/model.joblib` | Fitted sklearn pipeline (features + preprocessing + tuned XGBoost) | `src/train.py` | yes |
| `artifacts/threshold.json`, `metrics.json`, `model_comparison.csv`, `threshold_curve.csv` | Decision threshold + cost matrix, metrics, model comparison | `src/train.py` | yes |
| `artifacts/shap_summary.png`, `shap_importance.csv` | Global SHAP | `src/train.py` | yes |
| `artifacts/reference_sample.csv` | 5,000 training rows (raw features) for drift checks | `src/train.py` | yes |
| `logs/drift.log` | Drift warnings | `src/drift.py` | no |

## 2. Raw tables (default run: 494,692 rows)
Data is **synthetic and anonymised** - `patient_id` is an integer surrogate key, there are no names, phone numbers or addresses.

### admissions (90,000 rows) - one row per inpatient stay
| Column | Type | Notes |
|---|---|---|
| admission_id | int PK | |
| patient_id | int | surrogate; ~40,600 distinct patients appear |
| hospital_id | text | 4 hospitals |
| admission_date | date | 2023-01-01 -> 2024-12-31 |
| discharge_date | date | **~6 % missing** |
| age, sex, chronic_conditions | int / text / int | |
| insurance, admission_type, discharge_disposition | text | |
| diagnosis_text | text | **free text** (case noise, abbreviations, trailing spaces, ~2 % blank) |
| readmitted_30d | int 0/1 | **target** (~13 % positive) |

### opd_visits (180,000 rows)
`visit_id` PK, `patient_id`, `hospital_id`, `visit_date`, `department`, `fee`.

### lab_tests (134,692 rows) - long format, ~30 % of admissions have each test
`lab_id` PK, `admission_id` FK, `test_name` (HbA1c, Creatinine, Hemoglobin, Sodium, WBC), `value`, `is_abnormal` (0/1).

### billing (90,000 rows) - one row per admission
`admission_id` PK/FK, `base_amount`, `gst_amount` (18 % of base), `total_amount` - **GST counted twice on ~4 % of rows (~15 % at PUN-HINJEWADI)**: `total = base + 2 x GST`.

```
patients (implicit via patient_id) 1---* admissions 1---* lab_tests
                                   1---* opd_visits     1---1 billing
```

## 3. Data problems and how they are handled
| Problem (from brief) | Detection / fix | Code |
|---|---|---|
| GST counted twice | `abs(total - (base + 2 x gst)) <= 1` -> set `billing_total = base + gst`; flag kept as `gst_corrected` | `cleaning.fix_gst` |
| Missing discharge dates | `los_days`, `discharge_dow` stay NaN; pipeline imputes (median) and adds missing-indicators | `features.build_preprocessor` |
| Diagnosis as free text | Lower-case + regex rules -> 9 groups (`heart_failure`, `copd`, `diabetes`, `pneumonia`, `kidney_disease`, `injury`, `surgical`, `infection_fever`, `other`); unknown -> `other` | `cleaning.map_diagnosis` |
| Incomplete follow-up at end of data | Drop admissions in the last 45 days | `build_modeling_table` |
| Leakage via history | Prior-admission / OPD counts use events strictly before admission date | `count_events_before` |

## 4. Modeling table (84,516 rows, 26 columns)
Split chronologically: **train 59,161** (to 2024-04-24) / **valid 12,677** (to 2024-08-05) / **test 12,678** (to 2024-11-16).

| Group | Columns |
|---|---|
| Keys / time | admission_id, patient_id, admission_date |
| Demographic / admin (categorical) | hospital_id, sex, insurance, admission_type, discharge_disposition, diagnosis_group |
| Numeric | age, chronic_conditions, los_days, prior_adm_180d, prior_adm_30d, opd_visits_90d, n_lab_tests, n_abnormal_labs, hba1c, creatinine, hemoglobin, sodium, wbc, billing_total, discharge_dow, admission_month |
| Target | readmitted_30d |

Engineered inside the pipeline (`features.FeatureEngineer`): `charge_per_day`, `log_billing`, `is_weekend_discharge`, `abnormal_lab_ratio`, `utilisation_score` (2 x prior_adm_180d + 0.5 x opd_visits_90d), `is_elderly` (>= 65).

Typical missingness: lab values ~70 % (test not ordered), `los_days`/`discharge_dow` 6 %.

## 5. Config for paths
Environment variables: `DATA_DIR` (default `./data`), `ARTIFACT_DIR` (`./artifacts`), `LOG_DIR` (`./logs`).

## 6. Optional: PostgreSQL schema for production
```sql
CREATE TABLE admissions (
  admission_id BIGINT PRIMARY KEY, patient_id BIGINT NOT NULL, hospital_id TEXT NOT NULL,
  admission_date DATE NOT NULL, discharge_date DATE, age SMALLINT, sex CHAR(1),
  insurance TEXT, admission_type TEXT, discharge_disposition TEXT, diagnosis_text TEXT,
  chronic_conditions SMALLINT, readmitted_30d SMALLINT);
CREATE INDEX ON admissions (patient_id, admission_date);   -- history features
CREATE TABLE opd_visits (visit_id BIGINT PRIMARY KEY, patient_id BIGINT NOT NULL, hospital_id TEXT,
  visit_date DATE NOT NULL, department TEXT, fee NUMERIC(10,2));
CREATE INDEX ON opd_visits (patient_id, visit_date);
CREATE TABLE lab_tests (lab_id BIGINT PRIMARY KEY, admission_id BIGINT REFERENCES admissions, test_name TEXT,
  value NUMERIC(8,2), is_abnormal SMALLINT);
CREATE TABLE billing (admission_id BIGINT PRIMARY KEY REFERENCES admissions, base_amount NUMERIC(12,2),
  gst_amount NUMERIC(12,2), total_amount NUMERIC(12,2));
CREATE TABLE predictions_log (id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT now(),
  admission_id BIGINT, probability REAL, flagged BOOLEAN, model_version TEXT, features JSONB);  -- audit + drift
```
To use it: load with `pandas.read_sql` into the same four DataFrames that `cleaning.load_raw()` returns; nothing else changes.

## 7. Privacy
Keep real data pseudonymised, never log raw requests containing identifiers, and restrict access to `data/` and `logs/` (see `DEPLOYMENT.md`).
