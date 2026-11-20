# Deployment Guide

Verified in the build environment: Python 3.12, `python run_all.py`, `uvicorn`, `streamlit`, `pytest`.
**Not verified here:** `docker build` / `docker compose` (no Docker daemon available when this was built). The Dockerfile runs exactly the commands that were verified; run it once on your machine and fix anything environment-specific.

## 1. Options at a glance
| Option | Command | Use for |
|---|---|---|
| One command (local) | `python run_all.py` | Fresh clone: install -> data -> train -> tests |
| Manual local | see 2 | Development |
| Docker Compose | `docker compose up --build` | Demo / simple server; API :8000, UI :8501 |
| Single container | see 4 | API only |

## 2. Local (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # or requirements-lock.txt for exact versions
python -m src.train                                      # ~90 s; creates artifacts/
python -m pytest -q
uvicorn api.main:app --host 0.0.0.0 --port 8000          # terminal 1
streamlit run app/streamlit_app.py                       # terminal 2 -> http://localhost:8501
```
Windows tip: if `python` is not found use `py`; use `curl.exe` or the Swagger page `/docs`.

## 3. Docker Compose
```bash
docker compose up --build        # first build trains the model inside the image (~3-5 min)
docker compose logs -f api
docker compose down
```
- `api` service: `uvicorn api.main:app` on 8000, health check hits `/health`.
- `ui` service: same image, runs Streamlit on 8501 with `API_URL=http://api:8000`.
- The image already contains `artifacts/` because `RUN python -m src.train && python -m pytest -q` runs at build time, so a failing test fails the build.

## 4. Single container (API only)
```bash
docker build -t readmission-api .
docker run -p 8000:8000 readmission-api
```

## 5. Configuration
| Variable | Default | Meaning |
|---|---|---|
| `ARTIFACT_DIR` | `./artifacts` | where `model.joblib`, `threshold.json`... are read from / written to |
| `DATA_DIR` | `./data` | raw + modeling tables |
| `LOG_DIR` | `./logs` | `drift.log` |
| `API_URL` | `http://localhost:8000` | used by Streamlit to find the API |

Thresholds/costs are in `src/config.py` (`READMISSION_COST`, `INTERVENTION_COST`, `INTERVENTION_EFFECTIVENESS`); change them and retrain.

## 6. Production hardening checklist
| Item | Status | What to do |
|---|---|---|
| Authentication | **not implemented** | Put an API key / OAuth proxy in front (FastAPI `Depends` or nginx); clinic staff only |
| HTTPS | not implemented | Terminate TLS at a reverse proxy (nginx, Caddy, cloud LB) |
| CORS | default | Restrict to the UI origin if browsers call the API directly |
| Workers | 1 | `uvicorn api.main:app --workers 4` (each loads the model; ~15-25 ms/request, so one worker is plenty for a clinic network) |
| Request logging with PHI | none | Do not log raw payloads; log only ids and scores |
| Rate limiting | none | At the proxy |
| Non-root container | not set | Add `USER` in the Dockerfile |
| Secrets | none needed | Keep it that way; use env vars if you add a DB |
| Backups | n/a | Version `artifacts/` (git-lfs or object storage) |

## 7. Operations
**Health:** `GET /health` (200 + model name). **Latency:** header `X-Process-Time-ms` on every response; target < 300 ms (measured ~15-25 ms with SHAP).

**Drift monitoring**
- Online: `GET /drift` (needs >= 50 recent requests; buffer is per process).
- Batch: `python -m src.drift --new new_discharges.csv` (columns = API field names). Warnings appear in `logs/drift.log` as `DRIFT detected on '<feature>': PSI=...`.
- Suggested policy: run weekly via cron / scheduler; if any key feature drifts for 2 weeks in a row, investigate data quality first, then retrain.

**Retraining**
```bash
python -m src.train          # regenerates data (synthetic) - for real data replace generate() with your loader
```
Then restart the API (`docker compose up --build` or restart uvicorn). Keep the previous `artifacts/` folder until the new metrics in `artifacts/metrics.json` and `MODEL_CARD.md` are reviewed.

**CI:** `.github/workflows/ci.yml` runs the tests and a fast training smoke test on each push.

## 8. Publishing to GitHub
```bash
cd hospital_readmission
git init -b main && git add . && git commit -m "Hospital readmission model with API"   # already done in the zip
git remote add origin https://github.com/<you>/hospital-readmission.git
git push -u origin main
```
Add a screenshot of `artifacts/shap_summary.png` and the Streamlit screen to the README before sharing.

## 9. Troubleshooting
| Symptom | Fix |
|---|---|
| `RuntimeError: No trained model ...` at API start | run `python -m src.train` first |
| Streamlit shows "API not reachable" | start uvicorn first; check `API_URL` |
| `libgomp` / LightGBM import error on Linux | `apt-get install libgomp1` (already in Dockerfile) |
| Port already in use | `uvicorn ... --port 8001` and set `API_URL` |
| Tests slow the first time | the test fixture trains a tiny model (~10-20 s) |
| 422 from `/predict` | read the `loc` / `msg` in the body; see `docs/API.md` for ranges and enums |
