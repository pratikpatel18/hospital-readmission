"""Synthetic generator for the fictional Pune clinic network.

Produces four raw CSVs (~500k rows total at default size) with the problems named in the
brief: GST counted twice in billing, missing discharge dates, free-text diagnosis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

GST_RATE = 0.18

DX_TEXT = {
    "heart_failure": ["Congestive heart failure", "CHF exacerbation", "acute decompensated cardiac failure", "HF with fluid overload"],
    "copd": ["COPD exacerbation", "chronic obstructive pulmonary disease", "emphysema flare", "COPD w/ resp infection"],
    "diabetes": ["Type 2 diabetes mellitus", "T2DM uncontrolled", "diabetic ketoacidosis", "DM2 w/ neuropathy", "hyperglycaemia"],
    "pneumonia": ["Community acquired pneumonia", "LRTI", "pneumonia right lower lobe", "chest infection"],
    "kidney_disease": ["CKD stage 4", "acute kidney injury", "renal failure", "AKI on CKD"],
    "injury": ["Fracture neck of femur", "road traffic trauma", "head injury", "fracture radius"],
    "surgical": ["Appendicitis - surgical", "Inguinal hernia repair", "Cholecystectomy", "post-op surgical follow up"],
    "infection_fever": ["Dengue fever", "viral fever", "Typhoid", "malaria"],
    "other": ["Abdominal pain", "Back pain", "Gastroenteritis", "Hypertension urgency"],
}
DX_EFFECT = {"heart_failure": 0.9, "copd": 0.8, "diabetes": 0.4, "pneumonia": 0.3,
             "kidney_disease": 0.7, "injury": -0.1, "surgical": -0.5, "infection_fever": -0.4, "other": -0.2}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n_admissions: int = 90_000, seed: int = C.SEED, out_dir: Path | None = None,
             start: str = "2023-01-01", end: str = "2024-12-31", drift: bool = False) -> dict:
    """Generate raw tables. `drift=True` shifts the population (older, sicker) for drift demos."""
    from .cleaning import count_events_before, prior_counts  # local import avoids cycle

    rng = np.random.default_rng(seed)
    n = n_admissions
    n_pat = max(int(n * 0.55), 10)

    # patients ----------------------------------------------------------------
    age_mu = 64 if drift else 52
    p_age = np.clip(rng.normal(age_mu, 18, n_pat), 1, 95).astype(int)
    p_sex = rng.choice(C.SEXES, n_pat, p=[0.49, 0.50, 0.01])
    p_chronic = np.clip(rng.poisson(p_age / 35), 0, 8)

    # admissions ----------------------------------------------------------------
    pid = rng.integers(0, n_pat, n)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    adm_date = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days + 1, n), unit="D")
    age = p_age[pid]
    severity = rng.normal(0.8 if drift else 0.0, 1.0, n)

    old = (age > 60).astype(float)
    probs = np.tile(np.array([0.06, 0.05, 0.12, 0.09, 0.05, 0.08, 0.15, 0.12, 0.28]), (n, 1))
    probs[:, 0] += 0.07 * old; probs[:, 1] += 0.05 * old; probs[:, 4] += 0.04 * old
    probs /= probs.sum(1, keepdims=True)
    cum = probs.cumsum(1)
    dx_idx = np.minimum((rng.random(n)[:, None] > cum).sum(1), len(C.DX_GROUPS) - 1)
    dx_group = np.array(C.DX_GROUPS)[dx_idx]

    los = np.clip(np.round(rng.gamma(2.0, 1.8, n) + 1 + 0.8 * np.clip(severity, 0, None)), 1, 30).astype(int)
    disch_date = adm_date + pd.to_timedelta(los, unit="D")
    hospital = rng.choice(C.HOSPITALS, n, p=[0.3, 0.25, 0.2, 0.25])
    insurance = rng.choice(C.INSURANCE, n, p=[0.3, 0.3, 0.25, 0.15])
    adm_type = rng.choice(C.ADMISSION_TYPES, n, p=[0.45, 0.3, 0.25])
    disp = rng.choice(C.DISPOSITIONS, n, p=[0.45, 0.38, 0.09, 0.08])

    text = np.empty(n, dtype=object)
    for g, templates in DX_TEXT.items():
        idx = np.where(dx_group == g)[0]
        text[idx] = rng.choice(templates, len(idx))
    # messy free text: case noise, trailing spaces, blanks
    case_r = rng.random(n)
    text = np.array([t.upper() if r < 0.25 else (t.lower() if r < 0.45 else t) for t, r in zip(text, case_r)], dtype=object)
    text = np.where(rng.random(n) < 0.02, "", text)
    text = np.array([t + " " if i % 17 == 0 else t for i, t in enumerate(text)], dtype=object)

    adm = pd.DataFrame({
        "admission_id": np.arange(n), "patient_id": pid, "hospital_id": hospital,
        "admission_date": adm_date, "discharge_date": disch_date, "age": age,
        "sex": p_sex[pid], "insurance": insurance, "admission_type": adm_type,
        "discharge_disposition": disp, "diagnosis_text": text, "chronic_conditions": p_chronic[pid],
    })

    # OPD visits ------------------------------------------------------------------
    n_opd = 2 * n
    opd = pd.DataFrame({
        "visit_id": np.arange(n_opd), "patient_id": rng.integers(0, n_pat, n_opd),
        "hospital_id": rng.choice(C.HOSPITALS, n_opd),
        "visit_date": pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days + 1, n_opd), unit="D"),
        "department": rng.choice(["General", "Cardiology", "Pulmonology", "Endocrine", "Ortho", "Nephrology"], n_opd),
        "fee": np.round(rng.gamma(4, 150, n_opd), 2),
    })

    # labs (latent severity drives the values) --------------------------------------
    lab_spec = {  # name: (mean, slope on severity, sd, abnormal-test)
        "HbA1c": (6.4, 0.5, 1.0, lambda v: v > 6.5),
        "Creatinine": (1.1, 0.35, 0.3, lambda v: v > 1.3),
        "Hemoglobin": (12.5, -0.9, 1.5, lambda v: v < 11),
        "Sodium": (138, -1.5, 3.0, lambda v: v < 135),
        "WBC": (8.0, 1.2, 2.5, lambda v: v > 11),
    }
    kidney = (dx_group == "kidney_disease").astype(float)
    diab = (dx_group == "diabetes").astype(float)
    frames = []
    for name, (mu, slope, sd, abn) in lab_spec.items():
        ids = np.where(rng.random(n) < 0.30)[0]
        v = mu + slope * severity[ids] + rng.normal(0, sd, len(ids))
        if name == "Creatinine":
            v += 0.6 * kidney[ids]
        if name == "HbA1c":
            v += 1.2 * diab[ids]
        frames.append(pd.DataFrame({"admission_id": ids, "test_name": name, "value": np.round(v, 2),
                                    "is_abnormal": abn(v).astype(int)}))
    labs = pd.concat(frames, ignore_index=True)
    labs["lab_id"] = np.arange(len(labs))

    # billing: GST counted twice on a share of rows --------------------------------
    base = np.round(np.exp(rng.normal(np.log(12000 + 7500 * los + 6000 * np.clip(severity, 0, None)), 0.25)), 2)
    gst = np.round(base * GST_RATE, 2)
    dbl = rng.random(n) < np.where(hospital == "PUN-HINJEWADI", 0.15, 0.04)
    billing = pd.DataFrame({"admission_id": np.arange(n), "base_amount": base, "gst_amount": gst,
                            "total_amount": np.round(np.where(dbl, base + 2 * gst, base + gst), 2)})

    # target -----------------------------------------------------------------------
    chronic = p_chronic[pid]
    adm_np = adm_date.values
    prior180 = prior_counts(pid, adm_np, 180)
    opd90 = count_events_before(pid, adm_np, opd["patient_id"].values, opd["visit_date"].values, 90)
    creat_hi = (severity > 0.7).astype(float)
    month = adm_date.month.values
    logit = (0.022 * (age - 50) + 0.38 * np.minimum(prior180, 4) + 0.10 * np.minimum(opd90, 6)
             + 0.12 * chronic + 0.60 * severity + 0.04 * (los - 4)
             + np.array([DX_EFFECT[g] for g in dx_group])
             + 0.55 * (disp == "Left against advice") - 0.35 * (disp == "Home with follow-up")
             + 0.30 * (adm_type == "Emergency") + 0.15 * (hospital == "PUN-WAKAD")
             + 0.25 * np.isin(month, [7, 8, 9])  # monsoon-season bump
             + 1.2 * ((age > 70) & np.isin(dx_group, ["heart_failure", "copd"]))
             + 0.7 * (creat_hi * np.isin(dx_group, ["diabetes", "kidney_disease"]))
             + 0.5 * ((prior180 >= 2) & (disp != "Home with follow-up"))
             # non-linear / interaction effects (typical of real clinical risk)
             + 0.35 * np.minimum(prior180, 4) * np.minimum(chronic, 4) / 2
             + 0.9 * (los > 9) + 0.8 * ((age < 6) | (age > 82))
             + 0.7 * ((hospital == "PUN-HINJEWADI") & (insurance == "Self-pay"))
             + 0.9 * ((adm_type == "Emergency") & (disp == "Left against advice"))
             + 0.8 * (np.isin(dx_group, ["heart_failure", "copd"]) & (severity > 0.5))
             - 0.6 * ((adm_type == "Elective") & (age < 60) & (severity < 0))
             + rng.normal(0, 0.25, n))
    lo, hi, mid = -10.0, 5.0, 0.0
    for _ in range(40):  # solve intercept for ~13 % prevalence
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(logit + mid).mean() < 0.13 else (lo, mid)
    adm[C.TARGET] = (rng.random(n) < _sigmoid(logit + mid)).astype(int)

    # inject missing discharge dates (LOS is derived from them, so it goes missing too)
    adm.loc[rng.random(n) < 0.06, "discharge_date"] = pd.NaT

    tables = {"admissions": adm, "opd_visits": opd, "lab_tests": labs, "billing": billing}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, df in tables.items():
            df.to_csv(out_dir / f"{k}.csv", index=False)
    return tables


def main():
    ap = argparse.ArgumentParser(description="Generate raw synthetic hospital data")
    ap.add_argument("--n-admissions", type=int, default=90_000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else C.data_dir() / "raw"
    t = generate(a.n_admissions, a.seed, out)
    print({k: len(v) for k, v in t.items()}, "total rows:", sum(len(v) for v in t.values()), "->", out)


if __name__ == "__main__":
    main()
