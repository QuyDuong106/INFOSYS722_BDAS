"""Fixed model settings extracted from stacking and error-analysis notebooks."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_NAMES = ("logistic", "random_forest", "xgboost", "lightgbm", "stacking")
CODED_FIELDS = {"gender", "comorbidity", "cad", "hypothyroidism", "hyperlipidemia", "dm", "hfa"}


def make_pipeline(X, model, scale_numeric=False):
    categorical = [c for c in X if c in CODED_FIELDS or not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X if c not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler() if scale_numeric else "passthrough"),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("select", SelectPercentile(f_classif, percentile=75)),
        ("model", model),
    ])


def make_model(name, X, seed):
    """Return a fresh estimator; all learned transformations stay inside CV."""
    if name == "stacking":
        return StackingClassifier(
            estimators=[(base, make_model(base, X, seed)) for base in MODEL_NAMES[:-1]],
            final_estimator=Pipeline([
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(C=1.0, max_iter=3000, random_state=seed)),
            ]),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
            stack_method="predict_proba", passthrough=False, n_jobs=1,
        )
    if name == "logistic":
        model = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_leaf=4,
                                       n_jobs=1, random_state=seed)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=60, learning_rate=0.05, max_depth=2,
                              reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", tree_method="hist", n_jobs=1,
                              random_state=seed)
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=60, learning_rate=0.05, num_leaves=7,
                               min_child_samples=20, colsample_bytree=0.8,
                               deterministic=True, force_col_wise=True, verbosity=-1,
                               n_jobs=1, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(X, model, scale_numeric=name == "logistic")
