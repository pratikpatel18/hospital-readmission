"""SHAP explanations: global (summary plot + importance table) and local (per patient)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from .features import feature_names


def _positive_class(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def make_explainer(pipe, background: pd.DataFrame):
    model = pipe.named_steps["model"]
    if isinstance(model, LogisticRegression):
        bg = pipe[:-1].transform(background)
        return shap.LinearExplainer(model, bg)
    return shap.TreeExplainer(model)


def shap_matrix(pipe, explainer, X: pd.DataFrame):
    Xt = pipe[:-1].transform(X)
    return _positive_class(explainer.shap_values(Xt)), Xt


def global_explanation(pipe, explainer, X: pd.DataFrame, out_dir, max_rows: int = 2000):
    """Save SHAP summary plot + mean(|SHAP|) table. Returns the importance DataFrame."""
    X = X.sample(min(max_rows, len(X)), random_state=0)
    sv, Xt = shap_matrix(pipe, explainer, X)
    names = feature_names(pipe)
    imp = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp.to_csv(f"{out_dir}/shap_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=names, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    return imp


def local_explanation(pipe, explainer, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """Top-k feature contributions (log-odds) for a single patient."""
    sv, Xt = shap_matrix(pipe, explainer, row)
    names = feature_names(pipe)
    contrib = sv[0]
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [{"feature": names[i], "value": float(Xt[0][i]), "shap": round(float(contrib[i]), 4),
             "direction": "raises risk" if contrib[i] > 0 else "lowers risk"} for i in order]
