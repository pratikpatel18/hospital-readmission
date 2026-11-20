"""Drift monitoring: compare new data with training data (PSI + KS) and log warnings."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config as C

NUMERIC_KEYS = ["age", "los_days", "creatinine", "billing_total", "prior_adm_180d", "opd_visits_90d"]
CATEGORICAL_KEYS = ["diagnosis_group", "discharge_disposition"]
PSI_WARN = 0.2       # common rule of thumb: >0.1 moderate, >0.2 significant shift
KS_ALPHA = 0.01

logger = logging.getLogger("drift")


def _setup_logger() -> logging.Logger:
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        try:
            C.log_dir().mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(C.log_dir() / "drift.log"); fh.setFormatter(fmt); logger.addHandler(fh)
        except OSError:
            pass
    return logger


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(cur.dropna()))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4).values
    c = cur.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4).values
    return float(np.sum((c - r) * np.log(c / r)))


def check_drift(reference: pd.DataFrame, new: pd.DataFrame, psi_warn: float = PSI_WARN,
                ks_alpha: float = KS_ALPHA, log: bool = True) -> pd.DataFrame:
    rows = []
    for f in NUMERIC_KEYS:
        if f in reference and f in new:
            psi = psi_numeric(reference[f], new[f])
            ks = ks_2samp(reference[f].dropna(), new[f].dropna()) if new[f].notna().any() else None
            rows.append({"feature": f, "type": "numeric", "psi": psi,
                         "ks_stat": getattr(ks, "statistic", np.nan), "ks_pvalue": getattr(ks, "pvalue", np.nan)})
    for f in CATEGORICAL_KEYS:
        if f in reference and f in new:
            rows.append({"feature": f, "type": "categorical", "psi": psi_categorical(reference[f], new[f]),
                         "ks_stat": np.nan, "ks_pvalue": np.nan})
    rep = pd.DataFrame(rows)
    rep["drifted"] = (rep["psi"] > psi_warn) | (rep["ks_pvalue"] < ks_alpha) & (rep["ks_stat"] > 0.1)
    if log:
        lg = _setup_logger()
        for _, r in rep.iterrows():
            if r["drifted"]:
                lg.warning("DRIFT detected on '%s': PSI=%.3f KS=%.3f (n_new=%d)", r["feature"], r["psi"],
                           0 if np.isnan(r["ks_stat"]) else r["ks_stat"], len(new))
        if not rep["drifted"].any():
            lg.info("No significant drift on %d monitored features (n_new=%d)", len(rep), len(new))
    return rep


def main():
    ap = argparse.ArgumentParser(description="Compare new data with the training reference sample")
    ap.add_argument("--new", help="CSV with raw feature columns (same names as the API input)")
    ap.add_argument("--demo", action="store_true", help="simulate an older/sicker population and check it")
    a = ap.parse_args()
    ref = pd.read_csv(C.artifact_dir() / "reference_sample.csv")
    if a.demo:
        from .cleaning import build_modeling_table
        from .data_generation import generate
        new = build_modeling_table(generate(6000, seed=7, drift=True))
    elif a.new:
        new = pd.read_csv(a.new)
    else:
        ap.error("pass --new FILE.csv or --demo")
    rep = check_drift(ref, new)
    print(rep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
