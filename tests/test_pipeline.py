import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config as C
from src.cleaning import (build_modeling_table, count_events_before, fix_gst, map_diagnosis, prior_counts, time_split)
from src.data_generation import generate
from src.features import build_pipeline


def test_gst_double_count_is_fixed():
    b = pd.DataFrame({"admission_id": [1, 2], "base_amount": [1000.0, 1000.0], "gst_amount": [180.0, 180.0],
                      "total_amount": [1360.0, 1180.0]})   # row 1: GST counted twice
    out = fix_gst(b)
    assert out["billing_total"].tolist() == [1180.0, 1180.0]
    assert out["gst_corrected"].tolist() == [1, 0]


def test_diagnosis_free_text_mapping():
    assert map_diagnosis("CHF exacerbation") == "heart_failure"
    assert map_diagnosis("  T2DM uncontrolled ") == "diabetes"
    assert map_diagnosis("ACUTE KIDNEY INJURY") == "kidney_disease"   # kidney wins over 'injury'
    assert map_diagnosis("") == "other" and map_diagnosis(None) == "other"


def test_history_counts_exclude_current_and_future():
    pid = np.array([1, 1, 1, 2])
    d = np.array(["2024-01-01", "2024-01-20", "2024-06-01", "2024-01-20"], dtype="datetime64[D]")
    assert prior_counts(pid, d, 30).tolist() == [0, 1, 0, 0]
    assert prior_counts(pid, d, 180).tolist() == [0, 1, 2, 0]


def test_time_split_is_chronological_without_overlap():
    df = build_modeling_table(generate(4000, seed=1))
    tr, va, te = time_split(df)
    assert tr[C.TIME_COL].max() <= va[C.TIME_COL].min()
    assert va[C.TIME_COL].max() <= te[C.TIME_COL].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_censoring_drops_last_45_days():
    raw = generate(3000, seed=2)
    df = build_modeling_table(raw)
    assert df[C.TIME_COL].max() <= raw["admissions"][C.TIME_COL].max() - pd.Timedelta(days=45)


def test_pipeline_handles_missing_values_and_unseen_category():
    df = build_modeling_table(generate(4000, seed=3))
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(df[C.RAW_FEATURES], df[C.TARGET])
    X = df[C.RAW_FEATURES].head(5).copy()
    X["los_days"] = np.nan
    X.loc[X.index[0], "hospital_id"] = "UNSEEN-HOSPITAL"
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (5,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
