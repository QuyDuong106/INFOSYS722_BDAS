"""Nested Optuna tuning with Leave One Patient Out outer evaluation.

Each study receives outer training records only. All preparation is refitted in
inner folds. Completed outer folds can be resumed; original fixed results remain
in results/lopo. A partial smoke run never produces full-evaluation metrics.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from threadpoolctl import threadpool_limits

from utils import (ROOT, WORKFLOW_MODELS, workflow_data, frame_signature,
                   make_evaluation_model, disease_probability, result_row,
                   result_predictions, write_reports, dump_pickle)

SEARCH_VERSION = 1


def suggest_parameters(trial, name):
    """Only classifier settings are searched; RFE retains 20 source fields."""
    if name == "logistic":
        return {"model__C": trial.suggest_float("model__C", .01, 100, log=True)}
    if name == "stacking":
        # Base pipelines remain fixed; tune the logistic meta-learner only.
        return {"final_estimator__logistic__C": trial.suggest_float(
            "final_estimator__logistic__C", .01, 100, log=True)}
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
                       top_k=20, sampler_seed=None):
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
    template = make_evaluation_model(name, X_train, seed, "rfe", top_k)

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


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_optuna_lopo(*, seed=42, trials=20, inner_folds=3, top_k=20,
                    models=WORKFLOW_MODELS, output_dir=None, max_folds=None,
                    refit_final=False):
    """Run/resume nested evaluation. Returns (metrics, summary), or (None, None).

    max_folds limits the evaluated prefix for a smoke run, never the training
    dataset. Resume the same output directory without max_folds for all records.
    """
    if trials < 1 or inner_folds < 2 or (max_folds is not None and max_folds < 1):
        raise ValueError("Positive trials/max_folds and at least two inner folds are required")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    models = tuple(dict.fromkeys(models))
    if not models or any(name not in WORKFLOW_MODELS for name in models):
        raise ValueError("Choose at least one supported model")
    X, y = workflow_data()
    if not 1 <= top_k <= X.shape[1]:
        raise ValueError("top_k must be within the source feature count")
    out = Path(output_dir) if output_dir else ROOT / "results/lopo_optuna"
    out.mkdir(parents=True, exist_ok=True)
    config = dict(search_version=SEARCH_VERSION, signature=frame_signature(X, y),
                  seed=seed, trials=trials, inner_folds=inner_folds, top_k=top_k,
                  models=list(models), threshold=.5, selection="rfe",
                  objective="mean inner-fold ROC-AUC", rows=len(X),
                  positive_class="1 = gallstones present (source code 0)",
                  stacking_search="logistic meta-learner C only; fixed base learners",
                  packages={name: version(name) for name in (
                      "optuna", "scikit-learn", "numpy", "pandas", "xgboost", "lightgbm")})
    config_path = out / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory belongs to a different configuration/data/environment; choose another --output-dir")
    _write_json(config_path, config)
    progress_path = out / "progress.json"
    if (max_folds is not None and max_folds < len(X) and progress_path.exists()
            and json.loads(progress_path.read_text()).get("complete")):
        raise ValueError("A complete evaluation already exists here; choose another output directory for a smoke run")
    folds = out / "folds"
    folds.mkdir(exist_ok=True)
    completed = []
    limit = min(max_folds, len(X)) if max_folds is not None else len(X)
    for fold, (train_pos, test_pos) in enumerate(LeaveOneOut().split(X)):
        if fold >= limit:
            break
        source_row = int(X.index[test_pos[0]])
        X_train, y_train = X.iloc[train_pos], y.iloc[train_pos]
        for model_number, name in enumerate(models):
            checkpoint = folds / f"fold_{fold:04d}_{name}.json"
            if checkpoint.exists():
                evidence = json.loads(checkpoint.read_text())
                if (evidence["fold"] != fold or evidence["source_row"] != source_row
                        or evidence["model"] != name or evidence["actual"] != int(y.iloc[test_pos[0]])
                        or not 0 <= evidence["probability"] <= 1):
                    raise ValueError(f"Invalid checkpoint: {checkpoint}")
            else:
                print(f"Leave One Patient Out {fold + 1}/{len(X)}: tuning {name}", flush=True)
                started = perf_counter()
                sampler_seed = (seed + fold * len(models) + model_number) % 2**32
                fitted, params, score, trial_table = tune_training_fold(
                    name, X_train, y_train, seed=seed, trials=trials,
                    inner_folds=inner_folds, top_k=top_k, sampler_seed=sampler_seed)
                with threadpool_limits(limits=1):
                    probability = float(disease_probability(fitted, X.iloc[test_pos])[0])
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Nonfinite or invalid held-out probability")
                trial_table.to_csv(folds / f"fold_{fold:04d}_{name}_trials.csv", index=False)
                evidence = dict(fold=fold, source_row=source_row, model=name,
                                actual=int(y.iloc[test_pos[0]]), probability=probability,
                                best_params=params, inner_cv_auc=score, sampler_seed=sampler_seed,
                                fit_predict_seconds=perf_counter() - started)
                # Commit after all evidence is written; interrupted studies restart.
                _write_json(checkpoint, evidence)
            completed.append(evidence)
        _write_json(out / "progress.json", dict(complete=False,
                    evaluated_folds=fold + 1, total_folds=len(X), models=list(models)))
    pd.DataFrame(completed).to_csv(out / "fold_best_parameters.csv", index=False)
    if limit < len(X):
        print(f"Partial run: {limit}/{len(X)} folds. No full-evaluation metrics written.", flush=True)
        return None, None
    rows, predictions = [], []
    for name in models:
        evidence = sorted([row for row in completed if row["model"] == name], key=lambda row: row["fold"])
        if len(evidence) != len(X) or len({row["source_row"] for row in evidence}) != len(X):
            raise RuntimeError("Each source record must have exactly one held-out prediction")
        probability = np.array([row["probability"] for row in evidence])
        rows.append(result_row(seed, name, "lopo_optuna", y, probability,
                               sum(row["fit_predict_seconds"] for row in evidence), "rfe"))
        predictions.append(result_predictions(seed, name, "lopo_optuna", X, y, probability, "rfe"))
    metrics, summary = write_reports(rows, predictions, out)
    _write_json(out / "progress.json", dict(complete=True, evaluated_folds=len(X),
                total_folds=len(X), models=list(models)))
    print(metrics[["model", "roc_auc", "accuracy", "recall", "specificity", "fnr"]].to_string(index=False), flush=True)
    if refit_final:
        # A fresh full-data search is for inference only, never outer metrics.
        final_dir = out / "final_models"
        final_dir.mkdir(exist_ok=True)
        for name in models:
            fitted, params, score, trials_table = tune_training_fold(
                name, X, y, seed=seed, trials=trials, inner_folds=inner_folds, top_k=top_k)
            dump_pickle(fitted, final_dir / f"{name}.pkl")
            trials_table.to_csv(final_dir / f"{name}_trials.csv", index=False)
            _write_json(final_dir / f"{name}_metadata.json", dict(best_params=params,
                inner_cv_auc=score, usage="Full-data inference refit; not used for LOPO metrics"))
    return metrics, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--models", nargs="+", choices=WORKFLOW_MODELS, default=list(WORKFLOW_MODELS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/lopo_optuna")
    parser.add_argument("--max-folds", type=int, help="Smoke-run prefix; omit for complete evaluation")
    parser.add_argument("--refit-final", action="store_true", help="Also tune/refit full-data inference models")
    args = parser.parse_args()
    run_optuna_lopo(**vars(args))


if __name__ == "__main__":
    main()
