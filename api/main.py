"""FastAPI service: validated input -> risk probability, flag, and SHAP top factors."""
from __future__ import annotations

import json
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src import config as C
from src.drift import check_drift
from src.explain import local_explanation, make_explainer

STATE: dict = {}
RECENT = deque(maxlen=2000)   # recent requests, used by /drift


@asynccontextmanager
async def lifespan(app: FastAPI):
    art = C.artifact_dir()
    if not (art / "model.joblib").exists():
        raise RuntimeError(f"No trained model in {art}. Run: python -m src.train")
    STATE["model"] = joblib.load(art / "model.joblib")
    STATE["threshold"] = json.loads((art / "threshold.json").read_text())["threshold"]
    STATE["metrics"] = json.loads((art / "metrics.json").read_text())
    ref = pd.read_csv(art / "reference_sample.csv")
    STATE["reference"] = ref
    STATE["explainer"] = make_explainer(STATE["model"], ref.sample(min(300, len(ref)), random_state=0))
    STATE["model"].predict_proba(ref.head(2))      # warm-up so the first request is fast
    yield
    STATE.clear()


app = FastAPI(title="Hospital 30-day Readmission API", version="1.0.0", lifespan=lifespan)


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int = Field(ge=0, le=110, examples=[72])
    sex: Literal["F", "M", "O"]
    hospital_id: Literal["PUN-KOTHRUD", "PUN-HADAPSAR", "PUN-WAKAD", "PUN-HINJEWADI"]
    insurance: Literal["Self-pay", "Private", "Government scheme", "Corporate"]
    admission_type: Literal["Emergency", "Elective", "Referral"]
    discharge_disposition: Literal["Home", "Home with follow-up", "Transferred", "Left against advice"]
    diagnosis_group: Literal["heart_failure", "copd", "diabetes", "pneumonia", "kidney_disease",
                             "injury", "surgical", "infection_fever", "other"]
    chronic_conditions: int = Field(ge=0, le=20)
    los_days: Optional[float] = Field(default=None, ge=0, le=365, description="None if discharge date missing")
    prior_adm_180d: int = Field(ge=0, le=50)
    prior_adm_30d: int = Field(ge=0, le=30)
    opd_visits_90d: int = Field(ge=0, le=100)
    n_lab_tests: int = Field(default=0, ge=0, le=100)
    n_abnormal_labs: int = Field(default=0, ge=0, le=100)
    hba1c: Optional[float] = Field(default=None, ge=2, le=20)
    creatinine: Optional[float] = Field(default=None, ge=0.1, le=20)
    hemoglobin: Optional[float] = Field(default=None, ge=2, le=25)
    sodium: Optional[float] = Field(default=None, ge=100, le=180)
    wbc: Optional[float] = Field(default=None, ge=0.5, le=100)
    billing_total: float = Field(ge=0, le=5_000_000, description="INR incl. GST once")
    discharge_dow: Optional[int] = Field(default=None, ge=0, le=6, description="0=Mon ... 6=Sun")
    admission_month: int = Field(ge=1, le=12)


class Factor(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str


class Prediction(BaseModel):
    readmission_probability: float
    risk_flag: bool
    risk_band: Literal["High", "Medium", "Low"]
    threshold: float
    top_factors: list[Factor]
    model: str
    latency_ms: float


def _to_frame(p: Patient) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump()])[C.RAW_FEATURES]
    return df.astype({c: float for c in C.NUMERIC_FEATURES})   # None -> NaN


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t) * 1000:.1f}"
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["metrics"]["best_model"], "threshold": STATE["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient, explain: bool = True):
    t = time.perf_counter()
    X = _to_frame(patient)
    prob = float(STATE["model"].predict_proba(X)[0, 1])
    thr = STATE["threshold"]
    RECENT.append(X.iloc[0].to_dict())
    factors = local_explanation(STATE["model"], STATE["explainer"], X) if explain else []
    band = "High" if prob >= thr else ("Medium" if prob >= 0.6 * thr else "Low")
    return Prediction(readmission_probability=round(prob, 4), risk_flag=prob >= thr, risk_band=band,
                      threshold=thr, top_factors=factors, model=STATE["metrics"]["best_model"],
                      latency_ms=round((time.perf_counter() - t) * 1000, 2))


@app.get("/model-info")
def model_info():
    """Model name, tuned params, test metrics, threshold and cost matrix (for the UI / audits)."""
    m = STATE["metrics"]
    return {"model": m["best_model"], "metric": m["metric"], "test": m["test"], "best_params": m["best_params"],
            "threshold": STATE["threshold"], "improvement_over_baseline": m["improvement_over_baseline"],
            "cost_matrix_inr": json.loads((C.artifact_dir() / "threshold.json").read_text())["cost_matrix_inr"]}


@app.get("/explain/global")
def explain_global(top: int = 15):
    """Global SHAP importance (mean |SHAP| over a test sample)."""
    imp = pd.read_csv(C.artifact_dir() / "shap_importance.csv").head(max(1, min(top, 50)))
    return json.loads(imp.round(5).to_json(orient="records"))


@app.get("/drift")
def drift(min_rows: int = 50):
    """Compare the last requests with the training sample; WARNINGs go to logs/drift.log."""
    if len(RECENT) < min_rows:
        return {"status": "not_enough_data", "n_requests": len(RECENT), "needed": min_rows}
    rep = check_drift(STATE["reference"], pd.DataFrame(list(RECENT)))
    return {"status": "drift" if rep["drifted"].any() else "ok", "n_requests": len(RECENT),
            "report": json.loads(rep.round(4).to_json(orient="records"))}
