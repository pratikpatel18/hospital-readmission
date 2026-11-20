"""Feature engineering + preprocessing + model pipeline (all inside one sklearn Pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

ENGINEERED = ["charge_per_day", "log_billing", "is_weekend_discharge", "abnormal_lab_ratio",
              "utilisation_score", "is_elderly"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering. NaNs are kept so the downstream imputer handles them."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        los = X["los_days"].astype(float)
        X["charge_per_day"] = X["billing_total"].astype(float) / los.clip(lower=1)
        X["log_billing"] = np.log1p(X["billing_total"].astype(float).clip(lower=0))
        dow = X["discharge_dow"].astype(float)
        X["is_weekend_discharge"] = np.where(dow.isna(), np.nan, dow.isin([5, 6]).astype(float))
        n_labs = X["n_lab_tests"].astype(float)
        X["abnormal_lab_ratio"] = np.where(n_labs > 0, X["n_abnormal_labs"] / n_labs.replace(0, np.nan), np.nan)
        X["utilisation_score"] = 2.0 * X["prior_adm_180d"] + 0.5 * X["opd_visits_90d"]
        X["is_elderly"] = (X["age"] >= 65).astype(float)
        return X.replace([np.inf, -np.inf], np.nan)


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer(
        [("num", num, C.NUMERIC_FEATURES + ENGINEERED), ("cat", cat, C.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False)


def build_pipeline(model) -> Pipeline:
    return Pipeline([("features", FeatureEngineer()), ("prep", build_preprocessor()), ("model", model)])


def feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())
