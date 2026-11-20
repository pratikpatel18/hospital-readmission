"""Streamlit front end for clinic staff. Talks to the FastAPI service (API_URL)."""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Readmission risk", page_icon="🏥", layout="wide")
st.title("🏥 30-day readmission risk")
st.caption("Early-warning screen for the Pune community-clinic network. Decision support only - not a diagnosis.")

try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"API online · model: {h['model']} · threshold {h['threshold']:.2f}")
except Exception:
    st.sidebar.error(f"API not reachable at {API_URL}. Start it with: uvicorn api.main:app --port 8000")
    st.stop()

with st.form("patient"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 0, 110, 72)
        sex = st.selectbox("Sex", ["F", "M", "O"])
        hospital_id = st.selectbox("Hospital", ["PUN-KOTHRUD", "PUN-HADAPSAR", "PUN-WAKAD", "PUN-HINJEWADI"])
        insurance = st.selectbox("Insurance", ["Self-pay", "Private", "Government scheme", "Corporate"])
        admission_type = st.selectbox("Admission type", ["Emergency", "Elective", "Referral"])
        diagnosis_group = st.selectbox("Diagnosis group", ["heart_failure", "copd", "diabetes", "pneumonia",
                                       "kidney_disease", "injury", "surgical", "infection_fever", "other"])
    with c2:
        discharge_disposition = st.selectbox("Discharge disposition", ["Home", "Home with follow-up", "Transferred", "Left against advice"])
        chronic = st.number_input("Chronic conditions", 0, 20, 2)
        los_known = st.checkbox("Discharge date known", True)
        los = st.number_input("Length of stay (days)", 0.0, 365.0, 6.0)
        prior180 = st.number_input("Admissions in last 180 days", 0, 50, 2)
        prior30 = st.number_input("Admissions in last 30 days", 0, 30, 1)
        opd90 = st.number_input("OPD visits in last 90 days", 0, 100, 3)
    with c3:
        billing = st.number_input("Final bill, INR (GST counted once)", 0.0, 5_000_000.0, 85000.0, step=1000.0)
        month = st.number_input("Admission month", 1, 12, 8)
        dow = st.selectbox("Discharge weekday (0=Mon)", [None, 0, 1, 2, 3, 4, 5, 6], index=5)
        st.markdown("**Labs (leave 0 if not done)**")
        creat = st.number_input("Creatinine", 0.0, 20.0, 1.6)
        hba1c = st.number_input("HbA1c", 0.0, 20.0, 0.0)
        hb = st.number_input("Hemoglobin", 0.0, 25.0, 0.0)
        na = st.number_input("Sodium", 0.0, 180.0, 0.0)
        wbc = st.number_input("WBC", 0.0, 100.0, 0.0)
        n_abn = st.number_input("Abnormal lab results", 0, 100, 1)
    go = st.form_submit_button("Predict risk", type="primary")

if go:
    nz = lambda v: float(v) if v and v > 0 else None
    labs = [creat, hba1c, hb, na, wbc]
    payload = dict(age=age, sex=sex, hospital_id=hospital_id, insurance=insurance, admission_type=admission_type,
                   discharge_disposition=discharge_disposition, diagnosis_group=diagnosis_group,
                   chronic_conditions=chronic, los_days=float(los) if los_known else None,
                   prior_adm_180d=prior180, prior_adm_30d=prior30, opd_visits_90d=opd90,
                   n_lab_tests=sum(1 for v in labs if v and v > 0), n_abnormal_labs=n_abn,
                   hba1c=nz(hba1c), creatinine=nz(creat), hemoglobin=nz(hb), sodium=nz(na), wbc=nz(wbc),
                   billing_total=billing, discharge_dow=dow if los_known else None, admission_month=month)
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    if r.status_code != 200:
        st.error(f"Input rejected: {r.json()}")
    else:
        out = r.json()
        a, b, c = st.columns(3)
        a.metric("Risk score", f"{out['readmission_probability']:.0%}")
        b.metric("Risk band", out["risk_band"])
        c.metric("Response time", f"{out['latency_ms']:.0f} ms")
        if out["risk_flag"]:
            st.error(f"⚠️ FLAG for follow-up (score ≥ {out['threshold']:.2f}). Suggest a nurse call / early OPD slot.")
        else:
            st.success(f"Below the follow-up threshold ({out['threshold']:.2f}).")
        st.subheader("Why? Top factors (SHAP)")
        df = pd.DataFrame(out["top_factors"])
        st.bar_chart(df.set_index("feature")["shap"])
        st.dataframe(df[["feature", "direction", "shap"]], hide_index=True, use_container_width=True)


st.divider()
with st.expander("🌍 Global explanation - what drives readmission risk across all patients (SHAP)", expanded=False):
    try:
        g = pd.DataFrame(requests.get(f"{API_URL}/explain/global", params={"top": 12}, timeout=5).json())
        st.bar_chart(g.set_index("feature")["mean_abs_shap"])
        mi = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.caption(f"Model: {mi['model']} · test PR-AUC {mi['test']['pr_auc']:.3f} · threshold {mi['threshold']:.2f} "
                   f"(cost matrix INR: {mi['cost_matrix_inr']})")
    except Exception as e:
        st.warning(f"Global explanation unavailable: {e}")
