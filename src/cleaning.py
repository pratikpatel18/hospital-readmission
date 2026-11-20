"""Cleaning + table merging. Everything here uses only information available at discharge."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# ---- diagnosis text -> code group -----------------------------------------------
DX_RULES = [
    (r"heart failure|cardiac failure|\bchf\b|\bhf\b", "heart_failure"),
    (r"copd|chronic obstructive|emphysema", "copd"),
    (r"diabet|t2dm|\bdm2\b|hyperglyc|\bdka\b", "diabetes"),
    (r"pneumon|lrti|chest infection", "pneumonia"),
    (r"kidney|renal|\bckd\b|\baki\b", "kidney_disease"),
    (r"fracture|trauma|injur", "injury"),
    (r"appendic|hernia|cholecyst|surgical|post-op", "surgical"),
    (r"fever|dengue|malaria|typhoid|viral", "infection_fever"),
]


def map_diagnosis(text) -> str:
    """Map free-text diagnosis to a coarse group (rule-based; 'other' if nothing matches)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "other"
    t = str(text).lower().strip()
    for pattern, group in DX_RULES:
        if re.search(pattern, t):
            return group
    return "other"


# ---- billing --------------------------------------------------------------------
def fix_gst(billing: pd.DataFrame, tol: float = 1.0) -> pd.DataFrame:
    """Detect rows where total == base + 2*GST (GST counted twice) and correct to base + GST."""
    b = billing.copy()
    double = (b["gst_amount"] > 0) & ((b["total_amount"] - (b["base_amount"] + 2 * b["gst_amount"])).abs() <= tol)
    b["gst_corrected"] = double.astype(int)
    b["billing_total"] = np.where(double, b["base_amount"] + b["gst_amount"], b["total_amount"])
    return b


# ---- patient history (leakage-safe: strictly earlier events) ---------------------
def count_events_before(q_pid, q_dates, e_pid, e_dates, window_days: int) -> np.ndarray:
    """For each query (patient, date) count that patient's events in [date-window, date)."""
    BIG = 100_000
    day0 = np.datetime64("2000-01-01", "D")
    qd = (np.asarray(q_dates).astype("datetime64[D]") - day0).astype(np.int64)
    ed = (np.asarray(e_dates).astype("datetime64[D]") - day0).astype(np.int64)
    ekeys = np.sort(np.asarray(e_pid, dtype=np.int64) * BIG + ed)
    qkeys = np.asarray(q_pid, dtype=np.int64) * BIG + qd
    return (np.searchsorted(ekeys, qkeys, "left") - np.searchsorted(ekeys, qkeys - window_days, "left")).astype(int)


def prior_counts(pid, dates, window_days: int) -> np.ndarray:
    """Prior admissions of the same patient in the window (current admission excluded)."""
    return count_events_before(pid, dates, pid, dates, window_days)


# ---- main entry -----------------------------------------------------------------
def load_raw(raw_dir: Path | None = None) -> dict:
    raw_dir = Path(raw_dir) if raw_dir else C.data_dir() / "raw"
    return {
        "admissions": pd.read_csv(raw_dir / "admissions.csv", parse_dates=["admission_date", "discharge_date"]),
        "opd_visits": pd.read_csv(raw_dir / "opd_visits.csv", parse_dates=["visit_date"]),
        "lab_tests": pd.read_csv(raw_dir / "lab_tests.csv"),
        "billing": pd.read_csv(raw_dir / "billing.csv"),
    }


def build_modeling_table(raw: dict, censor_days: int = 45) -> pd.DataFrame:
    """Clean + merge raw tables into one row per admission."""
    adm = raw["admissions"].copy()
    adm["diagnosis_group"] = adm["diagnosis_text"].map(map_diagnosis)

    # LOS / discharge: missing discharge dates stay NaN here; the pipeline imputes them.
    adm["los_days"] = (adm["discharge_date"] - adm["admission_date"]).dt.days.astype(float)
    adm["discharge_dow"] = adm["discharge_date"].dt.dayofweek.astype(float)
    adm["admission_month"] = adm["admission_date"].dt.month

    # right-censoring: the last `censor_days` have an incomplete 30-day follow-up window
    cutoff = adm["admission_date"].max() - pd.Timedelta(days=censor_days)
    adm = adm[adm["admission_date"] <= cutoff].copy()

    pid, d = adm["patient_id"].values, adm["admission_date"].values
    full = raw["admissions"]
    adm["prior_adm_180d"] = count_events_before(pid, d, full["patient_id"].values, full["admission_date"].values, 180)
    adm["prior_adm_30d"] = count_events_before(pid, d, full["patient_id"].values, full["admission_date"].values, 30)
    opd = raw["opd_visits"]
    adm["opd_visits_90d"] = count_events_before(pid, d, opd["patient_id"].values, opd["visit_date"].values, 90)

    labs = raw["lab_tests"]
    piv = labs.pivot_table(index="admission_id", columns="test_name", values="value", aggfunc="mean")
    piv = piv.rename(columns={"HbA1c": "hba1c", "Creatinine": "creatinine", "Hemoglobin": "hemoglobin",
                              "Sodium": "sodium", "WBC": "wbc"}).reset_index()
    agg = labs.groupby("admission_id").agg(n_lab_tests=("value", "size"), n_abnormal_labs=("is_abnormal", "sum")).reset_index()
    adm = adm.merge(piv, on="admission_id", how="left").merge(agg, on="admission_id", how="left")
    for c in ["hba1c", "creatinine", "hemoglobin", "sodium", "wbc"]:
        if c not in adm:
            adm[c] = np.nan
    adm[["n_lab_tests", "n_abnormal_labs"]] = adm[["n_lab_tests", "n_abnormal_labs"]].fillna(0)

    bill = fix_gst(raw["billing"])[["admission_id", "billing_total", "gst_corrected"]]
    adm = adm.merge(bill, on="admission_id", how="left")

    adm = adm.sort_values(C.TIME_COL).reset_index(drop=True)
    keep = ["admission_id", "patient_id", C.TIME_COL] + C.RAW_FEATURES + [C.TARGET]
    return adm[keep]


def time_split(df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15):
    """Chronological split on admission date: train < valid < test, no shuffling."""
    df = df.sort_values(C.TIME_COL).reset_index(drop=True)
    i1, i2 = int(len(df) * train_frac), int(len(df) * (train_frac + valid_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()
