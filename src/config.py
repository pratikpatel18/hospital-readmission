"""Central configuration: paths, schema and business costs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", ROOT / "data"))


def artifact_dir() -> Path:
    return Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))


def log_dir() -> Path:
    return Path(os.getenv("LOG_DIR", ROOT / "logs"))


HOSPITALS = ["PUN-KOTHRUD", "PUN-HADAPSAR", "PUN-WAKAD", "PUN-HINJEWADI"]
SEXES = ["F", "M", "O"]
INSURANCE = ["Self-pay", "Private", "Government scheme", "Corporate"]
ADMISSION_TYPES = ["Emergency", "Elective", "Referral"]
DISPOSITIONS = ["Home", "Home with follow-up", "Transferred", "Left against advice"]
DX_GROUPS = ["heart_failure", "copd", "diabetes", "pneumonia", "kidney_disease",
             "injury", "surgical", "infection_fever", "other"]

TARGET = "readmitted_30d"
TIME_COL = "admission_date"

NUMERIC_FEATURES = [
    "age", "chronic_conditions", "los_days", "prior_adm_180d", "prior_adm_30d",
    "opd_visits_90d", "n_lab_tests", "n_abnormal_labs", "hba1c", "creatinine",
    "hemoglobin", "sodium", "wbc", "billing_total", "discharge_dow", "admission_month",
]
CATEGORICAL_FEATURES = ["hospital_id", "sex", "insurance", "admission_type",
                        "discharge_disposition", "diagnosis_group"]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- Business cost matrix (INR) -------------------------------------------------
# Agreed with (fictional) clinic network finance team:
READMISSION_COST = 40_000.0     # avg cost of an unplanned readmission
INTERVENTION_COST = 1_500.0     # nurse follow-up call + early OPD slot
INTERVENTION_EFFECTIVENESS = 0.30  # share of readmissions a follow-up prevents
COST_FN = READMISSION_COST                                            # missed patient
COST_FP = INTERVENTION_COST                                           # wasted follow-up
COST_TP = INTERVENTION_COST + (1 - INTERVENTION_EFFECTIVENESS) * READMISSION_COST
COST_TN = 0.0

# ---- Success criteria (from the brief) ------------------------------------------
MIN_IMPROVEMENT_OVER_BASELINE = 0.20   # >= 20 % relative on PR-AUC
MAX_API_LATENCY_MS = 300
