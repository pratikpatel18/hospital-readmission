"""End-to-end training: data -> clean -> time split -> 4+ models -> tuning -> threshold -> SHAP -> artifacts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from . import config as C
from .cleaning import build_modeling_table, load_raw, time_split
from .data_generation import generate
from .explain import global_explanation, make_explainer
from .features import build_pipeline


# ---------------------------------------------------------------- models
def candidate_models(pos_weight: float, fast: bool) -> dict:
    n = 60 if fast else 300
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=60 if fast else 250, min_samples_leaf=20,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED),
        "xgboost": XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.08, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr",
                                 n_jobs=-1, random_state=C.SEED, verbosity=0),
        "lightgbm": LGBMClassifier(n_estimators=n, num_leaves=15, learning_rate=0.08, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                   n_jobs=-1, random_state=C.SEED, verbose=-1),
    }


SEARCH_SPACES = {
    "xgboost": {"model__n_estimators": randint(100, 500), "model__max_depth": randint(2, 7),
                "model__learning_rate": loguniform(0.02, 0.2), "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 20),
                "model__reg_lambda": loguniform(0.1, 20)},
    "lightgbm": {"model__n_estimators": randint(100, 500), "model__num_leaves": randint(7, 63),
                 "model__learning_rate": loguniform(0.02, 0.2), "model__subsample": uniform(0.6, 0.4),
                 "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_samples": randint(10, 100),
                 "model__reg_lambda": loguniform(0.1, 20)},
    "random_forest": {"model__n_estimators": randint(100, 400), "model__max_depth": randint(4, 20),
                      "model__min_samples_leaf": randint(5, 60), "model__max_features": uniform(0.2, 0.6)},
    "logistic_regression": {"model__C": loguniform(0.01, 10)},
}


# ---------------------------------------------------------------- metrics & threshold
def cost_at(y, p, t) -> float:
    pred = p >= t
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    return tp * C.COST_TP + fp * C.COST_FP + fn * C.COST_FN + tn * C.COST_TN


def pick_threshold(y, p) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.linspace(0.02, 0.98, 97), 2)
    curve = pd.DataFrame({"threshold": grid, "total_cost_inr": [cost_at(y, p, t) for t in grid]})
    return float(curve.loc[curve["total_cost_inr"].idxmin(), "threshold"]), curve


def evaluate(y, p, t=None) -> dict:
    m = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
         "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))}
    if t is not None:
        tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
        m.update({"threshold": t, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                  "recall": float(tp / max(tp + fn, 1)), "precision": float(tp / max(tp + fp, 1)),
                  "flag_rate": float((tp + fp) / len(y)), "total_cost_inr": float(cost_at(y, p, t))})
    return m


# ---------------------------------------------------------------- main
def run(n_admissions: int = 90_000, fast: bool = False, data_dir: Path | None = None,
        artifact_dir: Path | None = None, n_iter: int | None = None, regenerate: bool = True) -> dict:
    t0 = time.time()
    data_dir = Path(data_dir) if data_dir else C.data_dir()
    art = Path(artifact_dir) if artifact_dir else C.artifact_dir()
    art.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"

    if regenerate or not (raw_dir / "admissions.csv").exists():
        print(f"[1/7] Generating synthetic data ({n_admissions:,} admissions)...")
        generate(n_admissions, out_dir=raw_dir)
    print("[2/7] Cleaning + merging tables (GST fix, diagnosis mapping, history features)...")
    df = build_modeling_table(load_raw(raw_dir))
    df.to_csv(data_dir / "modeling_table.csv", index=False)

    train, valid, test = time_split(df)
    print(f"      rows train/valid/test = {len(train):,}/{len(valid):,}/{len(test):,}; "
          f"time ranges: train<= {train[C.TIME_COL].max().date()}, valid<= {valid[C.TIME_COL].max().date()}, "
          f"test<= {test[C.TIME_COL].max().date()}; prevalence={train[C.TARGET].mean():.3f}")
    assert train[C.TIME_COL].max() <= valid[C.TIME_COL].min() and valid[C.TIME_COL].max() <= test[C.TIME_COL].min()
    X_tr, y_tr = train[C.RAW_FEATURES], train[C.TARGET].values
    X_va, y_va = valid[C.RAW_FEATURES], valid[C.TARGET].values
    X_te, y_te = test[C.RAW_FEATURES], test[C.TARGET].values

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))   # class-imbalance handling
    print(f"[3/7] Training baseline models (class weights, scale_pos_weight={pos_weight:.1f})...")
    rows, fitted = [], {}
    dummy = build_pipeline(DummyClassifier(strategy="prior")).fit(X_tr, y_tr)
    rows.append({"model": "dummy_prior", **{f"valid_{k}": v for k, v in evaluate(y_va, dummy.predict_proba(X_va)[:, 1]).items()}})
    for name, mdl in candidate_models(pos_weight, fast).items():
        pipe = build_pipeline(mdl).fit(X_tr, y_tr)
        fitted[name] = pipe
        p = pipe.predict_proba(X_va)[:, 1]
        rows.append({"model": name, **{f"valid_{k}": v for k, v in evaluate(y_va, p).items()}})
        print(f"      {name:20s} valid PR-AUC={rows[-1]['valid_pr_auc']:.4f} ROC-AUC={rows[-1]['valid_roc_auc']:.4f}")
    comp = pd.DataFrame(rows)

    # untuned logistic regression is kept as a strong *model* benchmark (excluded from the tuning pick)
    baseline_name = "logistic_regression"
    cand = comp[~comp["model"].isin(["dummy_prior", baseline_name])]
    best_name = cand.loc[cand["valid_pr_auc"].idxmax(), "model"]

    print(f"[4/7] Tuning best model '{best_name}' with RandomizedSearchCV (time-ordered CV)...")
    search = RandomizedSearchCV(
        build_pipeline(candidate_models(pos_weight, fast)[best_name]), SEARCH_SPACES[best_name],
        n_iter=n_iter or (3 if fast else 20), scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=2 if fast else 3), random_state=C.SEED, n_jobs=1, refit=True)
    search.fit(X_tr, y_tr)   # rows are already time-sorted -> TimeSeriesSplit respects chronology
    best = search.best_estimator_
    print(f"      best CV PR-AUC={search.best_score_:.4f}; params={ {k.split('__')[1]: round(float(v), 4) for k, v in search.best_params_.items()} }")

    print("[5/7] Choosing decision threshold with the cost matrix on the validation set...")
    p_va = best.predict_proba(X_va)[:, 1]
    thr, curve = pick_threshold(y_va, p_va)
    curve.to_csv(art / "threshold_curve.csv", index=False)

    p_te = best.predict_proba(X_te)[:, 1]
    lr_te = fitted[baseline_name].predict_proba(X_te)[:, 1]
    prevalence_pr = float(y_te.mean())
    test_metrics = evaluate(y_te, p_te, thr)
    lr_metrics = evaluate(y_te, lr_te)
    # Baseline = today's rule of thumb: rank patients by number of admissions in the last 180 days.
    heur_te = X_te["prior_adm_180d"].astype(float).values
    base_metrics = {"roc_auc": float(roc_auc_score(y_te, heur_te)), "pr_auc": float(average_precision_score(y_te, heur_te))}
    improvement = test_metrics["pr_auc"] / base_metrics["pr_auc"] - 1
    improvement_lr = test_metrics["pr_auc"] / lr_metrics["pr_auc"] - 1
    cost_default = cost_at(y_te, p_te, 0.5)
    cost_none = cost_at(y_te, np.zeros_like(p_te), 0.5)          # flag nobody
    cost_all = cost_at(y_te, np.ones_like(p_te), 0.5)            # flag everybody
    business = {"cost_at_chosen_threshold": test_metrics["total_cost_inr"], "cost_threshold_0.5": cost_default,
                "cost_flag_nobody": cost_none, "cost_flag_everybody": cost_all,
                "saving_vs_flag_nobody_inr": cost_none - test_metrics["total_cost_inr"],
                "saving_per_1000_patients_inr": (cost_none - test_metrics["total_cost_inr"]) / len(y_te) * 1000}

    print("[6/7] SHAP explanations (global)...")
    explainer = make_explainer(best, X_tr.sample(min(500, len(X_tr)), random_state=0))
    imp = global_explanation(best, explainer, X_te, art)

    # latency check on the real model + explainer
    from .explain import local_explanation
    one = X_te.iloc[[0]]
    local_explanation(best, explainer, one)
    t_ = time.perf_counter()
    for _ in range(20):
        pr = best.predict_proba(one)[:, 1]; local_explanation(best, explainer, one)
    latency_ms = (time.perf_counter() - t_) / 20 * 1000

    print("[7/7] Saving artifacts...")
    joblib.dump(best, art / "model.joblib")
    X_tr.sample(min(5000, len(X_tr)), random_state=0).to_csv(art / "reference_sample.csv", index=False)
    comp.to_csv(art / "model_comparison.csv", index=False)
    (art / "threshold.json").write_text(json.dumps({
        "threshold": thr, "cost_matrix_inr": {"FN": C.COST_FN, "FP": C.COST_FP, "TP": C.COST_TP, "TN": C.COST_TN},
        "assumptions": {"readmission_cost": C.READMISSION_COST, "intervention_cost": C.INTERVENTION_COST,
                        "intervention_effectiveness": C.INTERVENTION_EFFECTIVENESS}}, indent=2))
    metrics = {
        "best_model": best_name, "metric": "PR-AUC (average precision)",
        "n_rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "prevalence_test": prevalence_pr, "cv_best_pr_auc": float(search.best_score_),
        "best_params": {k.split("__")[1]: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                        for k, v in search.best_params_.items()},
        "test": test_metrics, "baseline_rule_test": base_metrics, "baseline_logistic_test": lr_metrics,
        "improvement_over_logistic": improvement_lr,
        "improvement_over_baseline": improvement, "improvement_over_prevalence": test_metrics["pr_auc"] / prevalence_pr - 1,
        "success_improvement_met": bool(improvement >= C.MIN_IMPROVEMENT_OVER_BASELINE),
        "business": business, "inference_latency_ms_with_shap": latency_ms,
        "success_latency_met": bool(latency_ms < C.MAX_API_LATENCY_MS),
        "top_features": imp.head(10).to_dict("records"), "train_seconds": round(time.time() - t0, 1),
        "time_ranges": {"train_end": str(train[C.TIME_COL].max().date()), "valid_end": str(valid[C.TIME_COL].max().date()),
                        "test_end": str(test[C.TIME_COL].max().date())},
    }
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    from .model_card import write_model_card
    write_model_card(metrics, comp, C.ROOT / "MODEL_CARD.md" if artifact_dir is None else art / "MODEL_CARD.md")

    print("\n=== RESULT ===")
    print(f"best model       : {best_name}")
    print(f"test PR-AUC      : {test_metrics['pr_auc']:.4f}  (rule baseline {base_metrics['pr_auc']:.4f}, untuned LR {lr_metrics['pr_auc']:.4f}, prevalence {prevalence_pr:.4f})")
    print(f"improvement      : {improvement*100:.1f}% over baseline  -> {'PASS' if improvement >= C.MIN_IMPROVEMENT_OVER_BASELINE else 'FAIL'} (needs >= 20%)")
    print(f"threshold        : {thr:.2f}  recall={test_metrics['recall']:.2f} precision={test_metrics['precision']:.2f} flag_rate={test_metrics['flag_rate']:.2f}")
    print(f"cost saving      : INR {business['saving_vs_flag_nobody_inr']:,.0f} on test set vs flagging nobody")
    print(f"latency (predict+SHAP): {latency_ms:.1f} ms -> {'PASS' if latency_ms < C.MAX_API_LATENCY_MS else 'FAIL'} (< 300 ms)")
    print(f"total time       : {time.time()-t0:.0f}s; artifacts in {art}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train the 30-day readmission model")
    ap.add_argument("--n-admissions", type=int, default=90_000)
    ap.add_argument("--fast", action="store_true", help="small/quick settings for smoke tests")
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--reuse-data", action="store_true", help="do not regenerate raw CSVs if present")
    a = ap.parse_args()
    run(a.n_admissions, a.fast, n_iter=a.n_iter, regenerate=not a.reuse_data)


if __name__ == "__main__":
    main()
