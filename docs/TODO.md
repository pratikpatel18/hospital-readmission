# TODO & Requirements Traceability

## A. The brief, line by line
Status: ✅ done and verified · ⚠️ done with a caveat you should know about.

| # | Line in the brief | Status | Where / evidence |
|---|---|---|---|
| 1 | Fictional Pune clinic network wants an early-warning system for 30-day readmission, usable from a simple web screen | ✅ | Streamlit app `app/streamlit_app.py` (form -> score, flag, reasons, global SHAP); 4 Pune hospital ids in `src/config.py` |
| 2a | Anonymised OPD visits, admissions, lab tests, billing from 4 hospitals | ✅ | 4 tables, integer surrogate patient ids (`docs/DATABASE.md`) |
| 2b | About 500,000 rows | ✅ | 494,692 rows (90k admissions, 180k OPD, 134.7k labs, 90k billing) |
| 2c | GST counted twice | ✅ | `cleaning.fix_gst` + unit test `test_gst_double_count_is_fixed` |
| 2d | Missing discharge dates | ✅ | ~6 % missing; kept NaN, imputed in pipeline; test `test_pipeline_handles_missing_values...` |
| 2e | Diagnosis in free text instead of codes | ✅ | `cleaning.map_diagnosis` (9 groups); test `test_diagnosis_free_text_mapping` |
| 3a | Reproducible scikit-learn pipeline | ✅ | `src/features.py::build_pipeline`; fixed seeds; `requirements-lock.txt` |
| 3b | Imputation, encoding, scaling, feature engineering | ✅ | `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `FeatureEngineer` (6 engineered features) |
| 3c | Time-based split to avoid leakage | ✅ | `cleaning.time_split` (assert in training; test `test_time_split_is_chronological...`); history features use strictly earlier events (`test_history_counts_...`); last 45 days dropped |
| 4a | At least four models incl. logistic/linear, random forest, XGBoost or LightGBM | ✅ | Logistic regression, random forest, XGBoost **and** LightGBM (+ dummy) - `artifacts/model_comparison.csv` |
| 4b | Tune the best with randomised search | ✅ | `RandomizedSearchCV`, 20 iterations, `TimeSeriesSplit`, scoring = PR-AUC; winner XGBoost |
| 5a | Handle class imbalance with class weights or resampling | ✅ | `class_weight="balanced"` / `scale_pos_weight` |
| 5b | Decision threshold from a cost matrix agreed with the business | ⚠️ | Implemented (`train.pick_threshold`, `config.py`). The cost numbers (INR 40,000 / 1,500 / 30 % effectiveness) are **my assumptions** - confirm with your instructor / "business" |
| 5c | SHAP global and individual | ✅ | Global: `artifacts/shap_summary.png`, `/explain/global`, Streamlit expander. Individual: top-5 factors in every `/predict` response and in Streamlit |
| 6a | FastAPI endpoint with input validation | ✅ | `api/main.py`, pydantic ranges/enums, `extra="forbid"`, 422 tests |
| 6b | Streamlit front end | ✅ | `app/streamlit_app.py` |
| 6c | Unit tests for pipeline and API | ✅ | 22 tests in `tests/` (`python -m pytest -q`) |
| 7 | Stretch: drift monitoring that logs a warning when key features shift | ✅ | `src/drift.py` (PSI + KS), `/drift`, `logs/drift.log`; tests in `tests/test_drift.py`; demo `python -m src.drift --demo` |
| 8a | Beats the baseline by >= 20 % on the chosen metric | ⚠️ | PR-AUC 0.462 vs **0.191** baseline (rank by prior admissions) = +141 %. But vs an untuned logistic regression (0.443) it is only +4 %. The brief does not define "baseline" - be ready to state which you used (see `MODEL_CARD.md`) |
| 8b | API responds in under 300 ms | ✅ | ~15-25 ms incl. SHAP; test `test_latency_under_300ms` |
| 8c | Whole project runs from a fresh clone with one command | ✅ | `python run_all.py` (verified on a clean copy) / `docker compose up --build` (⚠️ compose not built in my sandbox) |
| 9a | GitHub repository | ✅ | git repo initialised with first commit; `.github/workflows/ci.yml`; you only need `git remote add` + `git push` (`docs/DEPLOYMENT.md` s.8) |
| 9b | Training notebook | ✅ | `notebooks/training.ipynb` (executed, with outputs) |
| 9c | Saved model | ✅ | `artifacts/model.joblib` |
| 9d | API, Streamlit app, Dockerfile, model card | ✅ | `api/`, `app/`, `Dockerfile` + `docker-compose.yml`, `MODEL_CARD.md` |
| 10 | Be ready to justify the threshold in business terms | ✅ | `MODEL_CARD.md` "Threshold choice" + the talking points below |

### Threshold talking points (line 10)
1. A missed readmission costs ~INR 40,000; a follow-up call costs ~INR 1,500 and prevents ~30 % of readmissions -> a call is worth it when risk > 1,500 / (0.3 x 40,000) = **12.5 %**.
2. Class weighting inflates raw scores, so I scanned thresholds on the validation set and minimised total cost -> **0.51**.
3. On the unseen later test set: flags 34 % of discharges, catches 73 % of readmissions (precision 30 %), total cost INR 61.7 M vs INR 70.8 M if nobody is flagged -> **INR ~9.1 M saved** (~INR 721 k per 1,000 discharges).
4. The cost curve is shallow around the optimum (INR 61.66 M at 0.51 vs 61.68 M at 0.50), so small changes in capacity do not hurt much. If staff capacity is limited, raise the threshold; if calls are cheaper, lower it.

## B. Done (summary)
Synthetic data generator · cleaning · leakage-safe features · sklearn pipeline · 4 models + dummy · randomised search · cost-based threshold · SHAP (global + local) · FastAPI · Streamlit · 22 tests · drift monitoring · notebook · Dockerfile + compose · model card · CI workflow · lock file · docs.

## C. Open items (suggested order)
**Before you submit / present**
- [ ] Run `python run_all.py` on your own machine and `docker compose up --build` once (not built in my environment).
- [ ] Confirm the cost-matrix numbers with your instructor and edit `src/config.py` if they differ; retrain.
- [ ] Decide and state your baseline definition (rule baseline vs logistic regression).
- [ ] `git remote add origin ... && git push` to create the GitHub repo; add screenshots to README.

**Model / data**
- [ ] Replace synthetic data with real data (same 4 tables) via a loader in `cleaning.load_raw`.
- [ ] Calibrate probabilities (isotonic on validation) so scores can be shown as percentages.
- [ ] Subgroup evaluation (hospital, age band, sex) and fairness review.
- [ ] Improve diagnosis mapping (fuzzy matching / ICD-10 lookup) for misspellings.
- [ ] Re-fit on train + validation with the frozen threshold for the final production model.
- [ ] Capture readmissions to *other* hospitals (currently invisible).

**Engineering**
- [ ] `/predict/batch` endpoint and CSV upload page in Streamlit.
- [ ] API authentication + HTTPS (`docs/DEPLOYMENT.md` s.6).
- [ ] Persist predictions (`predictions_log` table in `docs/DATABASE.md`) so drift checks survive restarts.
- [ ] Scheduled drift job + alert (email/Slack) instead of log-only.
- [ ] Model registry / versioning (MLflow or tagged artifacts) and a `model_version` field in the API.
- [ ] Run containers as non-root; add a Docker build job to CI.
- [ ] Prometheus metrics (latency, flag rate) and dashboards.
