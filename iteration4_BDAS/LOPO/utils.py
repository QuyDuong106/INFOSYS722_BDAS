"""Notebook data exports, fold-safe modeling, pickle persistence and evaluation.

The earlier command-line helpers remain for compatibility with archived code.
New workflows use disease-positive targets and separately fitted fold exports.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from modeling import MODEL_NAMES

ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (42, 7, 21, 84, 123)
METRICS = ("roc_auc", "average_precision", "accuracy", "balanced_accuracy", "precision",
           "recall", "specificity", "f1", "log_loss", "TN", "FP", "FN", "TP")


def parse_args(strategy, argv=None):
    parser = argparse.ArgumentParser(description=f"Five-seed {strategy} model comparison")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/dataset_gallstone")
    parser.add_argument("--target", default="gallstone_status")
    parser.add_argument("--id-columns", nargs="*", default=["record_id"])
    parser.add_argument("--seeds", nargs=5, type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "modeling_results" / strategy)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != 5 or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Provide five distinct seeds between 0 and 2**32 - 1.")
    args.models = list(dict.fromkeys(args.models))
    return args


def load_data(path, target, id_columns):
    path = Path(path)
    data = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    exported_indices = []
    for c in data:
        if str(c).startswith("Unnamed:") or not str(c).strip():
            values = pd.to_numeric(data[c], errors="coerce").to_numpy()
            if any(np.array_equal(values, np.arange(start, len(data) + start)) for start in (0, 1)):
                exported_indices.append(c)
    data = data.drop(columns=exported_indices)
    if target not in data or data[target].isna().any() or set(data[target].unique()) != {0, 1}:
        raise ValueError("Target must exist, contain both 0 and 1, and have no missing values.")
    if target in id_columns:
        raise ValueError("Target cannot also be an ID column.")
    ids = [c for c in id_columns if c in data]
    if ids and (data[ids].isna().any().any() or data.duplicated(ids).any()):
        raise ValueError("Missing or repeated patient IDs require review or grouped evaluation.")
    if data.drop(columns=ids).duplicated().any():
        raise ValueError("Review duplicate records before row-level evaluation.")
    X = data.drop(columns=[target, *ids]).replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        raise ValueError("No predictors remain.")
    return X, data[target].astype(int)


def metric_row(seed, model, evaluation, target, probability, seconds):
    probability = np.asarray(probability)
    if probability.shape != (len(target),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Every evaluated row must have one finite probability in [0, 1].")
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return dict(seed=seed, model=model, evaluation=evaluation, rows=len(target),
                roc_auc=roc_auc_score(target, probability),
                average_precision=average_precision_score(target, probability),
                accuracy=accuracy_score(target, prediction),
                balanced_accuracy=balanced_accuracy_score(target, prediction),
                precision=precision_score(target, prediction, zero_division=0),
                recall=recall_score(target, prediction, zero_division=0),
                specificity=tn / (tn + fp), f1=f1_score(target, prediction, zero_division=0),
                log_loss=log_loss(target, probability, labels=[0, 1]),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), fit_predict_seconds=seconds)


def prediction_rows(seed, model, evaluation, X, y, probability):
    actual = np.asarray(y)
    predicted = (probability >= 0.5).astype(int)
    return pd.DataFrame(dict(seed=seed, model=model, evaluation=evaluation, source_row=X.index,
                             actual=actual, probability_1=probability, prediction=predicted,
                             error_type=np.select([(actual == 0) & (predicted == 1),
                                                   (actual == 1) & (predicted == 0)],
                                                  ["FP", "FN"], default="Correct")))


def summarize(metrics):
    long = metrics.melt(id_vars=["seed", "model", "evaluation"], value_vars=METRICS,
                        var_name="metric", value_name="value")
    return long.groupby(["model", "evaluation", "metric"], sort=False)["value"].agg(
        n_seeds="count", mean="mean", variance="var", std="std").reset_index()


def save_results(args, strategy, rows):
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics_per_seed.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = dict(strategy=strategy, data=str(args.data.resolve()), target=args.target,
                    id_columns=args.id_columns, seeds=args.seeds, models=args.models,
                    threshold=0.5, variance_ddof=1,
                    seed_scope="split, estimator and inner stacking CV randomness")
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)


def continuous_columns(frame):
    """Schema-based columns, including after selection changes the frame width."""
    from modeling import CODED_FIELDS
    return [c for c in frame if c not in CODED_FIELDS and pd.api.types.is_numeric_dtype(frame[c])]


def categorical_columns(frame):
    return [c for c in frame if c not in continuous_columns(frame)]


def make_evaluation_model(name, X, seed, selection="rfe", top_k=20):
    """Select raw predictors with CART, then transform within every training fold.

    The supplied dataset stores all categories as numeric codes. CART selection
    uses these codes; the final classifier receives one-hot encoded categories.
    Twenty means source predictors, not twenty expanded dummy columns.
    """
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import RFE, SelectFromModel
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from modeling import make_model

    if selection not in {"rfe", "cart"}:
        raise ValueError("Selection must be 'rfe' or 'cart'.")
    if not 1 <= top_k <= X.shape[1]:
        raise ValueError("top_k must be between 1 and the number of source predictors.")
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
        raise ValueError("CART selection expects the numeric category codes in the supplied dataset.")
    if name == "stacking":
        stack = make_model(name, X, seed)
        stack.set_params(estimators=[
            (base, make_evaluation_model(base, X, seed, selection, top_k))
            for base in MODEL_NAMES[:-1]
        ])
        return stack

    impute = ColumnTransformer([
        ("continuous", SimpleImputer(strategy="median", keep_empty_features=True), continuous_columns),
        ("categorical", SimpleImputer(strategy="most_frequent", keep_empty_features=True), categorical_columns),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=seed)
    selector = (RFE(tree, n_features_to_select=top_k, step=1) if selection == "rfe"
                else SelectFromModel(tree, threshold=-np.inf, max_features=top_k))
    selector.set_output(transform="pandas")
    transform = ColumnTransformer([
        ("continuous", Pipeline([
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("standardize", StandardScaler()),
        ]), continuous_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ])
    estimator = clone(make_model(name, X, seed).named_steps["model"])
    return Pipeline([("impute", impute), ("select", selector),
                     ("transform", transform), ("model", estimator)])


def selected_feature_table(fitted):
    """Return selected source fields, separately for each stacking base learner."""
    pipelines = ([(name, fitted.named_estimators_[name]) for name, _ in fitted.estimators]
                 if hasattr(fitted, "named_estimators_") else [("model", fitted)])
    rows = []
    for name, pipeline in pipelines:
        selector = pipeline.named_steps["select"]
        names = pipeline.named_steps["impute"].get_feature_names_out()
        ranks = getattr(selector, "ranking_", np.where(selector.get_support(), 1, 2))
        for feature, selected, rank in zip(names, selector.get_support(), ranks):
            rows.append(dict(base_model=name, feature=feature, selected=bool(selected), rank=int(rank)))
    return pd.DataFrame(rows)


# Shared workflow for the reorganized notebooks. Source code 0 is disease-positive.
import hashlib
import pickle
from time import perf_counter

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

WORKFLOW_MODELS = (*MODEL_NAMES, "dummy_prior")
REPORT_METRICS = ("accuracy", "roc_auc", "fnr", "recall", "specificity", "precision", "f1")
DATASET_DIR = ROOT / "data/processed"


def dump_pickle(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_pickle(path):
    """Load only trusted, locally generated model files."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def workflow_data():
    X, source_y = load_data(ROOT / "data/raw/dataset_gallstone", "gallstone_status", ["record_id"])
    return X, source_y.eq(0).astype(int).rename("gallstones_present")


def frame_signature(X, y):
    digest = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    digest.update(repr(list(zip(X.columns, X.dtypes.astype(str)))).encode())
    return digest.hexdigest()


def split_indices(X, y, seed):
    train, reserve = train_test_split(X.index, test_size=0.30, stratify=y, random_state=seed)
    valid, test = train_test_split(reserve, test_size=0.50, stratify=y.loc[reserve], random_state=seed)
    return {"train": train, "validation": valid, "test": test,
            "development": train.append(valid)}


def make_preparation(X, seed, selection="rfe", top_k=20):
    return make_evaluation_model("logistic", X, seed, selection, top_k)[:-1]


def export_partition(X, y, fit_indices, partitions, directory, seed, selection="rfe", top_k=20):
    """Persist transformations and matrices; only fit_indices may teach the transformer."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fit_indices = pd.Index(fit_indices)
    config = dict(version=1, signature=frame_signature(X, y), seed=int(seed), selection=selection,
                  top_k=top_k, fit_rows=fit_indices.tolist(),
                  partitions={name: pd.Index(rows).tolist() for name, rows in partitions.items()})
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) == config:
        if (directory / "transformer.pkl").exists() and all((directory / f"{name}.csv").exists() for name in partitions):
            return directory
    transformer = make_preparation(X, seed, selection, top_k).fit(X.loc[fit_indices], y.loc[fit_indices])
    names = transformer.get_feature_names_out()
    for name, indices in partitions.items():
        values = transformer.transform(X.loc[indices])
        if not np.isfinite(values).all():
            raise ValueError("Transformed matrix contains nonfinite values.")
        table = pd.DataFrame(values, columns=names, index=pd.Index(indices, name="source_row"))
        table["gallstones_present"] = y.loc[indices].to_numpy()
        table.to_csv(directory / f"{name}.csv")
    selector = transformer.named_steps["select"]
    selected = pd.DataFrame({"feature": transformer.named_steps["impute"].get_feature_names_out(),
                             "selected": selector.get_support()})
    if selected.selected.sum() != top_k:
        raise ValueError("Feature selection did not retain the requested number of source fields.")
    selected.to_csv(directory / "selected_features.csv", index=False)
    dump_pickle(transformer, directory / "transformer.pkl")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    return directory


def read_transformed(directory, partition):
    frame = pd.read_csv(Path(directory) / f"{partition}.csv", index_col="source_row")
    target = frame.pop("gallstones_present").astype(int)
    return frame, target


def prepare_holdout_exports(seeds=DEFAULT_SEEDS, selections=("rfe", "cart"), top_k=20):
    X, y = workflow_data()
    for seed in seeds:
        indices = split_indices(X, y, seed)
        membership = pd.concat([pd.DataFrame({"source_row": indices[name], "split": name})
                                for name in ("train", "validation", "test")])
        for selection in selections:
            directory = DATASET_DIR / "holdout" / f"seed_{seed}" / selection
            export_partition(X, y, indices["train"],
                             {k: indices[k] for k in ("train", "validation")},
                             directory / "training", seed, selection, top_k)
            export_partition(X, y, indices["development"],
                             {k: indices[k] for k in ("development", "test")},
                             directory / "development", seed, selection, top_k)
        membership.to_csv(DATASET_DIR / "holdout" / f"seed_{seed}" / "split_membership.csv", index=False)
        print(f"Transformed holdout partitions: seed {seed}", flush=True)
    full = export_partition(X, y, X.index, {"transformed_dataset": X.index},
                            DATASET_DIR / "full_fit", seeds[0], selections[0], top_k)
    # This export is for exploration/deployment only, never an evaluation input.
    (DATASET_DIR / "transformed_dataset.csv").write_bytes((full / "transformed_dataset.csv").read_bytes())
    config = dict(signature=frame_signature(X, y), seeds=list(seeds), selections=list(selections), top_k=top_k,
                  source_labels={"0": "present", "1": "absent"},
                  modeling_labels={"0": "absent", "1": "present"},
                  full_export_usage="Exploration and final fitting only; use fold-specific files for evaluation.")
    (DATASET_DIR / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def require_exports():
    path = DATASET_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run transformation_feature_engineering.ipynb first.")
    config = json.loads(path.read_text())
    X, y = workflow_data()
    if config["signature"] != frame_signature(X, y):
        raise ValueError("Source data changed. Rerun transformation_feature_engineering.ipynb.")
    return X, y, config


def bare_model(name, seed):
    from modeling import make_model
    if name == "dummy_prior":
        return DummyClassifier(strategy="prior", random_state=seed)
    if name == "stacking":
        raise ValueError("Stacking requires complete pipelines for its inner folds.")
    return clone(make_model(name, pd.DataFrame(), seed).named_steps["model"])


def fit_exported_model(name, seed, directory, fit_partition, raw_X, raw_y,
                       selection="rfe", top_k=20, parameters=None):
    matrix, target = read_transformed(directory, fit_partition)
    if name == "stacking":
        # Supervised selection must also be refitted inside stacking's inner CV.
        model = make_evaluation_model(name, raw_X, seed, selection, top_k)
        model.fit(raw_X.loc[matrix.index], raw_y.loc[matrix.index])
        return model
    estimator = bare_model(name, seed)
    if parameters:
        estimator.set_params(**parameters)
    estimator.fit(matrix, target)
    preparation = load_pickle(Path(directory) / "transformer.pkl")
    # Keep feature names consistent when the saved raw-input pipeline is used.
    preparation.set_output(transform="pandas")
    return Pipeline([("preparation", preparation), ("model", estimator)])


def disease_probability(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def result_row(seed, model, evaluation, y, probability, seconds, selection):
    row = metric_row(seed, model, evaluation, y, probability, seconds)
    row.update(selection=selection, fnr=row["FN"] / (row["FN"] + row["TP"]))
    return row


def result_predictions(seed, model, evaluation, X, y, probability, selection):
    frame = prediction_rows(seed, model, evaluation, X, y, probability).rename(
        columns={"actual": "actual_present", "prediction": "predicted_present", "probability_1": "probability_present"})
    frame["source_target"] = 1 - frame.actual_present
    frame["selection"] = selection
    return frame


def write_reports(rows, predictions, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(directory / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
    long = metrics.melt(id_vars=["seed", "model", "selection", "evaluation"], value_vars=REPORT_METRICS,
                        var_name="metric", value_name="value")
    summary = long.groupby(["model", "selection", "evaluation", "metric"]).value.agg(
        n_runs="count", mean="mean", variance="var", std="std").reset_index()
    summary.to_csv(directory / "summary.csv", index=False)
    return metrics, summary


def run_holdout_workflow(models=WORKFLOW_MODELS):
    X, y, config = require_exports()
    rows, predictions, selected = [], [], []
    for seed in config["seeds"]:
        indices = split_indices(X, y, seed)
        for selection in config["selections"]:
            directory = DATASET_DIR / "holdout" / f"seed_{seed}" / selection
            for name in models:
                for evaluation, folder, fit_partition in [("validation", "training", "train"),
                                                          ("test", "development", "development")]:
                    start = perf_counter()
                    model = fit_exported_model(name, seed, directory / folder, fit_partition,
                                               X, y, selection, config["top_k"])
                    target_indices = indices[evaluation]
                    probability = disease_probability(model, X.loc[target_indices])
                    rows.append(result_row(seed, name, evaluation, y.loc[target_indices], probability,
                                           perf_counter() - start, selection))
                    predictions.append(result_predictions(seed, name, evaluation, X.loc[target_indices],
                                                          y.loc[target_indices], probability, selection))
                    if evaluation == "test":
                        path = ROOT / "models/holdout" / f"seed_{seed}" / f"{selection}_{name}.pkl"
                        dump_pickle(model, path)
                        np.testing.assert_allclose(disease_probability(load_pickle(path), X.loc[target_indices]), probability)
        candidates = pd.DataFrame([r for r in rows if r["seed"] == seed and r["evaluation"] == "validation"
                                   and r["model"] != "dummy_prior"])
        best = candidates.sort_values(["roc_auc", "accuracy", "fnr", "model", "selection"],
                                      ascending=[False, False, True, True, True]).iloc[0]
        selected.append(dict(seed=seed, model=best.model, selection=best.selection))
        print(f"Holdout seed {seed} complete", flush=True)
    directory = ROOT / "results" / "holdout"
    metrics, summary = write_reports(rows, predictions, directory)
    pd.DataFrame(selected).to_csv(directory / "validation_winners.csv", index=False)
    (ROOT / "models/holdout" / "metadata.json").write_text(json.dumps(config, indent=2) + "\n")
    return metrics, summary


def run_lopo_workflow(seed=42, selection="rfe", models=WORKFLOW_MODELS):
    """One row per person is assumed; pool held-out predictions before scoring."""
    X, y, config = require_exports()
    probability = {name: np.full(len(y), np.nan) for name in models}
    seconds = {name: 0.0 for name in models}
    seen = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(LeaveOneOut().split(X)):
        directory = DATASET_DIR / "lopo" / f"seed_{seed}" / selection / f"fold_{fold:04d}"
        export_partition(X, y, X.index[train], {"train": X.index[train], "test": X.index[test]},
                         directory, seed, selection, config["top_k"])
        for name in models:
            start = perf_counter()
            model = fit_exported_model(name, seed, directory, "train", X, y, selection, config["top_k"])
            probability[name][test] = disease_probability(model, X.iloc[test])
            seconds[name] += perf_counter() - start
        seen[test] += 1
        if fold == 0 or (fold + 1) % 40 == 0 or fold + 1 == len(y):
            print(f"LOPO {fold + 1}/{len(y)}", flush=True)
    if not np.all(seen == 1):
        raise RuntimeError("Each patient row must be held out exactly once.")
    rows, predictions = [], []
    for name in models:
        rows.append(result_row(seed, name, "lopo", y, probability[name], seconds[name], selection))
        predictions.append(result_predictions(seed, name, "lopo", X, y, probability[name], selection))
    metrics, summary = write_reports(rows, predictions, ROOT / "results" / "lopo")
    full = export_partition(X, y, X.index, {"full": X.index}, DATASET_DIR / "lopo" / "final_fit",
                            seed, selection, config["top_k"])
    for name in models:
        model = fit_exported_model(name, seed, full, "full", X, y, selection, config["top_k"])
        path = ROOT / "models/lopo" / f"{name}.pkl"
        dump_pickle(model, path)
        np.testing.assert_allclose(disease_probability(load_pickle(path), X.iloc[:10]),
                                   disease_probability(model, X.iloc[:10]))
    metadata = dict(seed=seed, selection=selection, top_k=config["top_k"], signature=config["signature"],
                    rows=len(X), positive_class="1 = gallstones present (source code 0)",
                    model_usage="Final full-data refits; metrics come from held-out fold models, not these refits.")
    (ROOT / "models/lopo" / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metrics, summary


def run_tuning_workflow(selection="rfe"):
    """Training-only inner CV, validation family selection, then one test per seed."""
    from sklearn.model_selection import GridSearchCV
    X, y, config = require_exports()
    grids = {
        "logistic": {"model__C": [0.1, 1.0, 10.0]},
        "random_forest": {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [2, 4]},
        "xgboost": {"model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.05]},
        "lightgbm": {"model__num_leaves": [7, 15], "model__learning_rate": [0.03, 0.05]},
    }
    rows, predictions, choices = [], [], []
    directory = ROOT / "results" / "tuning"
    directory.mkdir(parents=True, exist_ok=True)
    for seed in config["seeds"]:
        indices = split_indices(X, y, seed)
        candidates = []
        searches = {}
        for name, grid in grids.items():
            start = perf_counter()
            search = GridSearchCV(make_evaluation_model(name, X, seed, selection, config["top_k"]),
                                  grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                  n_jobs=1, error_score="raise", refit=True)
            search.fit(X.loc[indices["train"]], y.loc[indices["train"]])
            prob = disease_probability(search.best_estimator_, X.loc[indices["validation"]])
            row = result_row(seed, name, "validation", y.loc[indices["validation"]], prob,
                             perf_counter() - start, selection)
            candidates.append(row)
            rows.append(row)
            predictions.append(result_predictions(seed, name, "validation", X.loc[indices["validation"]],
                                                  y.loc[indices["validation"]], prob, selection))
            searches[name] = search
            pd.DataFrame(search.cv_results_).to_csv(directory / f"seed_{seed}_{name}_cv.csv", index=False)
        best = pd.DataFrame(candidates).sort_values(["roc_auc", "accuracy", "fnr", "model"],
                                                   ascending=[False, False, True, True]).iloc[0]
        parameters = {k.removeprefix("model__"): v for k, v in searches[best.model].best_params_.items()}
        prepared = DATASET_DIR / "holdout" / f"seed_{seed}" / selection / "development"
        start = perf_counter()
        final = fit_exported_model(best.model, seed, prepared, "development", X, y,
                                   selection, config["top_k"], parameters)
        prob = disease_probability(final, X.loc[indices["test"]])
        rows.append(result_row(seed, best.model, "test", y.loc[indices["test"]], prob,
                               perf_counter() - start, selection))
        predictions.append(result_predictions(seed, best.model, "test", X.loc[indices["test"]],
                                               y.loc[indices["test"]], prob, selection))
        path = ROOT / "models/tuned" / f"seed_{seed}" / f"{best.model}.pkl"
        dump_pickle(final, path)
        np.testing.assert_allclose(disease_probability(load_pickle(path), X.loc[indices["test"]]), prob)
        choices.append(dict(seed=seed, model=best.model, selection=selection, parameters=parameters,
                            inner_cv_auc=float(searches[best.model].best_score_)))
        print(f"Tuning seed {seed}: {best.model}", flush=True)
    metrics, summary = write_reports(rows, predictions, directory)
    test_summary = metrics.loc[metrics.evaluation == "test", list(REPORT_METRICS)].agg(["mean", "var", "std"]).T
    test_summary.to_csv(directory / "selected_procedure_summary.csv", index_label="metric")
    (ROOT / "models/tuned" / "metadata.json").write_text(json.dumps(dict(config=config, choices=choices), indent=2) + "\n")
    return metrics, test_summary


def plot_evaluation(predictions, title, output):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (seed, name), rows in predictions.groupby(["seed", "model"]):
        RocCurveDisplay.from_predictions(rows.actual_present, rows.probability_present,
                                         name=f"{name}, seed {seed}", ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    ConfusionMatrixDisplay.from_predictions(predictions.actual_present, predictions.predicted_present,
                                            display_labels=["Absent", "Present"], labels=[0, 1],
                                            ax=axes[1], colorbar=False)
    axes[1].set_title("Evaluation observations (splits may overlap)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()


def error_analysis_tables():
    X, y, config = require_exports()
    frames = []
    for method in ("holdout", "lopo", "tuning"):
        path = ROOT / "results" / method / "predictions.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame.loc[frame.evaluation.isin(["test", "lopo"])].copy()
        frame["method"] = method
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("Run holdout_method.ipynb or leave_one_out_method.ipynb first.")
    predictions = pd.concat(frames, ignore_index=True)
    errors = predictions.loc[predictions.error_type != "Correct"].copy()
    errors["explanation"] = np.where(errors.error_type == "FN",
        "Gallstones present (source 0), predicted absent: missed positive case.",
        "Gallstones absent (source 1), predicted present: false alarm.")
    errors["wrong_class_probability"] = np.where(errors.actual_present == 1,
                                                 1 - errors.probability_present, errors.probability_present)
    errors["threshold_distance"] = (errors.probability_present - 0.5).abs()
    errors = errors.join(X, on="source_row").sort_values("wrong_class_probability", ascending=False)
    grouped = predictions.assign(actual_positive=predictions.actual_present,
                                 FN=predictions.error_type.eq("FN"), FP=predictions.error_type.eq("FP"))
    summary = grouped.groupby(["method", "model", "selection"]).agg(
        evaluation_rows=("source_row", "size"), unique_rows=("source_row", "nunique"),
        positives=("actual_positive", "sum"), FN=("FN", "sum"), FP=("FP", "sum"))
    summary["fnr"] = summary.FN / summary.positives
    output = ROOT / "results" / "error_analysis"
    output.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output / "error_review.csv", index=False)
    summary.to_csv(output / "error_summary.csv")
    return predictions, errors, summary
