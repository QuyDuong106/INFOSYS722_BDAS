"""Optuna search using only the holdout training partition."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from utils import make_evaluation_model, disease_probability
def suggest_parameters(trial, name):
    """Only classifier settings are searched; RFE retains 20 source fields."""
    if name == "logistic":
        return {"model__C": trial.suggest_float("model__C", .01, 100, log=True)}
    if name == "stacking":
        # Tune only the named layer-1 boosting pipelines. Both logistic
        # regressions (base and meta) and the forest retain their fixed settings.
        class BaseTrial:
            def __init__(self, model_name):
                self.prefix = model_name + "__"

            def suggest_float(self, key, *args, **kwargs):
                return trial.suggest_float(self.prefix + key, *args, **kwargs)

            def suggest_int(self, key, *args, **kwargs):
                return trial.suggest_int(self.prefix + key, *args, **kwargs)

        params = {}
        for base in ("xgboost", "lightgbm"):
            selected = suggest_parameters(BaseTrial(base), base)
            params.update({base + "__" + key: value for key, value in selected.items()})
        return params
    params = {"model__n_estimators": trial.suggest_int("model__n_estimators", 40, 160, step=20)}
    if name == "random_forest":
        params.update({"model__max_depth": trial.suggest_int("model__max_depth", 2, 8),
                       "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 2, 10),
                       "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", .5, 1.0])})
    elif name in {"xgboost", "lightgbm"}:
        params["model__learning_rate"] = trial.suggest_float("model__learning_rate", .01, .15, log=True)
        params["model__colsample_bytree"] = trial.suggest_float("model__colsample_bytree", .6, 1.0)
        params["model__reg_lambda"] = trial.suggest_float("model__reg_lambda", .1, 10, log=True)
        if name == "xgboost":
            params["model__max_depth"] = trial.suggest_int("model__max_depth", 1, 4)
            params["model__subsample"] = trial.suggest_float("model__subsample", .6, 1.0)
        else:
            params["model__num_leaves"] = trial.suggest_int("model__num_leaves", 3, 15)
            params["model__min_child_samples"] = trial.suggest_int("model__min_child_samples", 10, 40)
    else:
        raise ValueError(f"Unsupported tunable model: {name}")
    return params

def tune_training_fold(name, X_train, y_train, *, seed=42, trials=20, inner_folds=3,
                       top_k=20, sampler_seed=None, selection="rfe", scaling="yeo_standard"):
    """Return refitted best pipeline, best parameters, CV score, and trial evidence.

    No outer test features or labels are accepted by this function.
    """
    if name == "dummy_prior":
        fitted = DummyClassifier(strategy="prior", random_state=seed).fit(X_train, y_train)
        return fitted, {}, None, pd.DataFrame()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    splits = list(cv.split(X_train, y_train))
    if y_train.value_counts().min() < inner_folds:
        raise ValueError("Each class needs at least inner_folds training records")
    template = make_evaluation_model(name, X_train, seed, selection, top_k, scaling=scaling)

    def objective(trial):
        params = suggest_parameters(trial, name)
        scores = []
        for train_pos, valid_pos in splits:
            candidate = clone(template).set_params(**params)
            # Includes imputation, RFE, transformations and (if used) stacking CV.
            with threadpool_limits(limits=1):
                candidate.fit(X_train.iloc[train_pos], y_train.iloc[train_pos])
                probability = disease_probability(candidate, X_train.iloc[valid_pos])
            scores.append(roc_auc_score(y_train.iloc[valid_pos], probability))
        trial.set_user_attr("inner_auc", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=seed if sampler_seed is None else sampler_seed))
    study.optimize(objective, n_trials=trials, n_jobs=1)
    fitted = clone(template).set_params(**study.best_params)
    with threadpool_limits(limits=1):
        fitted.fit(X_train, y_train)
    return fitted, study.best_params, study.best_value, study.trials_dataframe()
