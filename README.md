# Hospital 30-Day Readmission Model (with API)

Early-warning system for a **fictional Pune clinic network**: predicts 30-day patient readmission, explains it with SHAP, serves it via **FastAPI**, and gives staff a **Streamlit** screen.
Data is **synthetic** (~495k rows across OPD visits, admissions, lab tests, billing from 4 hospitals) with the problems from the brief: GST counted twice, missing discharge dates, free-text diagnosis.

## Run everything with ONE command (fresh clone)
```bash
python run_all.py            # installs deps, generates data, trains+tunes, runs tests (~3 min)
python run_all.py --fast     # same but small/quick (~1 min) for a smoke test
# or, with Docker (trains inside the image, then serves API + UI):
docker compose up --build    # API: http://localhost:8000/docs   UI: http://localhost:8501
```

## Run / test step by step
```bash
pip install -r requirements.txt

python -m src.data_generation          # raw CSVs -> data/raw/
python -m src.train                    # clean -> time split -> 4 models -> random search -> threshold -> SHAP -> artifacts/
python -m pytest -q                    # unit tests: cleaning, pipeline, time split, API, drift

uvicorn api.main:app --port 8000       # API   (docs at http://localhost:8000/docs)
streamlit run app/streamlit_app.py     # UI    (http://localhost:8501) - start the API first

python -m src.drift --demo             # drift monitoring demo: shifted population -> WARNING in logs/drift.log
```

### Call the API
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
 "age":72,"sex":"F","hospital_id":"PUN-WAKAD","insurance":"Self-pay","admission_type":"Emergency",
 "discharge_disposition":"Left against advice","diagnosis_group":"heart_failure","chronic_conditions":3,
 "los_days":8,"prior_adm_180d":2,"prior_adm_30d":1,"opd_visits_90d":4,"creatinine":1.9,
 "billing_total":98000,"discharge_dow":5,"admission_month":8}'

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":-3}'   # -> 422 validation error
curl http://localhost:8000/drift        # compares the last requests with the training data
```
Windows PowerShell: use `curl.exe`, or just use the Swagger UI at `/docs` (easiest).

## Layout
| Path | What |
|---|---|
| `src/data_generation.py` | synthetic 4-table data with the messy problems |
| `src/cleaning.py` | GST fix, diagnosis text -> groups, leakage-safe history features, 45-day censoring, time split |
| `src/features.py` | one sklearn `Pipeline`: feature engineering -> imputation -> one-hot -> scaling -> model |
| `src/train.py` | LR / RandomForest / XGBoost / LightGBM, RandomizedSearchCV (time-ordered CV), cost-matrix threshold, SHAP, artifacts |
| `src/explain.py`, `src/drift.py`, `src/model_card.py` | SHAP, drift (PSI+KS, logs warnings), model card generator |
| `api/main.py` | FastAPI: `/predict`, `/health`, `/model-info`, `/explain/global`, `/drift`, validated by pydantic |
| `app/streamlit_app.py` | staff UI with SHAP top factors |
| `tests/` | 22 unit tests (pipeline + API + drift) |
| `artifacts/` | `model.joblib`, `threshold.json`, `metrics.json`, `shap_summary.png`, ... |
| `MODEL_CARD.md`, `notebooks/training.ipynb`, `Dockerfile`, `docker-compose.yml` | deliverables |

## Documentation
| Doc | Content |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | components, data flow diagram, design decisions |
| [`docs/API.md`](docs/API.md) | every endpoint, field rules, examples, errors |
| [`docs/DATABASE.md`](docs/DATABASE.md) | raw/modeling tables, data problems + fixes, artifacts, optional PostgreSQL schema |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | local, Docker, config, hardening checklist, drift/retraining ops, GitHub |
| [`docs/TODO.md`](docs/TODO.md) | **every line of the brief mapped to evidence**, caveats, open items |
| [`MODEL_CARD.md`](MODEL_CARD.md) | metrics, threshold justification, limitations |

## Success criteria
See `MODEL_CARD.md` and `artifacts/metrics.json` (written by every training run) for PR-AUC vs baseline, API latency incl. SHAP, and the threshold justification in business terms.
