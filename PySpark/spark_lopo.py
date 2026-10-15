"""Optimized Spark LOPO, with optional CUDA XGBoost/Optuna hybrid candidate."""
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from time import perf_counter
from pyspark.sql import functions as F, Window
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from spark_workflow import MODEL_NAMES, prediction_frame, pooled_metrics
from spark_iterations import make_selected_pipeline
VARIANTS = {'rfe20_standard': ('rfe', 20), 'rfe10_standard': ('rfe', 10)}
from spark_feature_selection import _MODEL_CACHE
from spark_lopo_acceleration import PreparedFold, native_candidate, gpu_candidate, check_cuda
SUPPORTED_MODELS = (*MODEL_NAMES, "xgboost_gpu")


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def run_lopo(spark, frame, numeric, categorical, *, data_path, output_dir=None,
             variants=('rfe20_standard',), models=MODEL_NAMES, tune=True,
             inner_folds=3, seed=42, max_folds=None, gpu_trials=20, use_gpu=True):
    """One row is one patient, matching the research dataset's LeaveOneOut protocol.

    No patient identifier exists in this dataset. Repeated observations from one
    patient require a different, group-based splitter. A limited run never writes
    final metrics; rerun with max_folds=None to resume the same configuration.
    """
    variants, models = tuple(variants), tuple(models)
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('Choose distinct supported variants')
    if not models or len(set(models)) != len(models) or any(m not in SUPPORTED_MODELS for m in models):
        raise ValueError('Choose distinct Spark models or xgboost_gpu')
    if gpu_trials < 1 or inner_folds < 2 or (max_folds is not None and max_folds < 1):
        raise ValueError('Require at least two inner folds and a positive fold limit')
    records = frame.select('source_row', 'label').orderBy('source_row').collect()
    if len(records) < 3 or len({r.source_row for r in records}) != len(records):
        raise ValueError('Require unique source_row values and at least three observations')
    from collections import Counter
    counts = Counter(r.label for r in records)
    if set(counts) != {0.0, 1.0} or min(counts.values()) - 1 < (inner_folds if tune else 1):
        raise ValueError('Too few records per class after leaving a patient out')
    for variant in variants:
        k = VARIANTS[variant][1]
        if k is not None and k > len(numeric) + len(categorical):
            raise ValueError('Selected feature count exceeds predictor width')
    dependencies = ['spark_lopo.py', 'spark_workflow.py', 'spark_iterations.py', 'spark_feature_selection.py', 'spark_lopo_acceleration.py']
    code_hash = hashlib.sha256(b''.join(Path(__file__).with_name(f).read_bytes() for f in dependencies)).hexdigest()
    config = dict(protocol_version=3, protocol='leave one patient record out',
        data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(), code_sha256=code_hash,
        spark_version=spark.version, python_version=platform.python_version(),
        rows=len(records), models=list(models), variants=list(variants), tune=tune,
        inner_folds=inner_folds, seed=seed, threshold=0.5,
        gpu_trials=gpu_trials if 'xgboost_gpu' in models else None,
        xgboost_device=('cuda' if use_gpu else 'cpu') if 'xgboost_gpu' in models else None,
        preparation_cache='one fit per inner training fold and outer training partition, shared across candidates',
        timing_note='fit_predict_seconds excludes shared inner-fold preparation; shared_cv_preparation_seconds records that cost separately',
        scaling='training sample standard deviation, no centering; ordinal codes unchanged',
        positive_class='1 = gallstones present (source code 0)')
    if 'xgboost_gpu' in models:
        from importlib.metadata import version
        config['gpu_libraries'] = {name:version(name) for name in ['xgboost','optuna','numpy','scikit-learn']}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out = Path(output_dir or Path(__file__).parent / 'results' / 'spark_lopo') / signature
    out.mkdir(parents=True, exist_ok=True)
    cp = out / 'run_config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Run configuration mismatch')
    write_json(cp, config)
    limit = min(max_folds, len(records)) if max_folds is not None else len(records)
    if limit < len(records) and (out / 'progress.json').exists():
        if json.loads((out / 'progress.json').read_text()).get('complete'):
            raise ValueError('Complete results already exist; use max_folds=None')
    evidence = []
    gpu_checked = False
    for fold, record in enumerate(records[:limit]):
        held_out = int(record.source_row)
        pending = {}
        # Read checkpoints before creating any Spark training partition.
        for variant in variants:
            for name in models:
                dest = out / 'folds' / f'{fold:04d}' / variant / name
                checkpoint = dest/'prediction.json'
                if checkpoint.exists():
                    row = json.loads(checkpoint.read_text())
                    if (row['source_row'] != held_out or row['label'] != record.label
                        or row['model'] != name or row['variant'] != variant
                        or row['fold'] != fold or row['seed'] != seed
                        or row['train_rows'] != len(records)-1
                        or row['retained_features'] != VARIANTS[variant][1]
                        or not math.isfinite(row['probability_present'])
                        or not 0 <= row['probability_present'] <= 1
                        or row['prediction'] != float(row['probability_present'] >= .5)):
                        raise ValueError(f'Invalid checkpoint: {checkpoint}')
                    evidence.append(row)
                else:
                    pending[(variant, name)] = dest
        if pending:
            if use_gpu and any(name == 'xgboost_gpu' for _,name in pending) and not gpu_checked:
                print('Checking CUDA:', check_cuda(), flush=True)
                gpu_checked = True
            train = frame.filter(F.col('source_row') != held_out).cache()
            test = frame.filter(F.col('source_row') == held_out)
            try:
                for variant in variants:
                    work = [(name,dest) for (v,name),dest in pending.items() if v == variant]
                    if not work:
                        continue
                    prepared = None
                    try:
                        prepare_started = perf_counter()
                        prepared = PreparedFold(train, test, numeric, categorical, seed,
                            VARIANTS[variant][1], inner_folds, tune, make_selected_pipeline)
                        shared_prepare_seconds = perf_counter()-prepare_started
                        for name,dest in work:
                            dest.mkdir(parents=True,exist_ok=True)
                            print(f'LOPO {fold+1}/{len(records)}: {variant} {name}',flush=True)
                            started = perf_counter()
                            if name == 'xgboost_gpu':
                                probability,score,trials,backend = gpu_candidate(prepared,tune=tune,
                                    trials=gpu_trials,directory=dest,use_gpu=use_gpu)
                            else:
                                probability,score,trials,backend = native_candidate(prepared,name,tune)
                            if not math.isfinite(probability) or not 0 <= probability <= 1:
                                raise ValueError('Invalid held-out probability')
                            fields = prepared.fields()
                            row = dict(source_row=held_out,label=float(record.label),
                                probability_present=probability,prediction=float(probability >= .5),
                                model=name,seed=seed,fold=fold,variant=variant,inner_cv_auc=score,
                                retained_features=len(fields),train_rows=len(records)-1,
                                fit_predict_seconds=perf_counter()-started,backend=backend,
                                shared_cv_preparation_seconds=shared_prepare_seconds)
                            write_json(dest/'cv_search.json',trials)
                            write_json(dest/'selected_features.json',fields)
                            write_json(dest/'prediction.json',row)
                            evidence.append(row)
                    finally:
                        if prepared is not None:
                            prepared.close()
                        _MODEL_CACHE.clear()
            finally:
                train.unpersist()
        write_json(out/'progress.json',dict(complete=False,completed_folds=fold+1,
            total_folds=len(records),evaluations=len(evidence)))
    with (out / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence[0]))
        writer.writeheader(); writer.writerows(evidence)
    if limit != len(records):
        print(f'Partial check: {limit}/{len(records)} folds; final metrics not produced. Resume with max_folds=None.')
        return None, out
    metrics = []
    for variant in variants:
        selected = [r for r in evidence if r['variant'] == variant]
        for name in models:
            ids = [r['source_row'] for r in selected if r['model'] == name]
            if len(ids) != len(records) or len(set(ids)) != len(records):
                raise ValueError('Incomplete or duplicate held-out predictions')
        predictions = spark.createDataFrame([
            (r['source_row'], float(r['label']), r['probability_present'], float(r['prediction']), r['model'])
            for r in selected], 'source_row long, label double, probability_present double, prediction double, model string')
        for metric in pooled_metrics(predictions):
            metric.update(variant=variant, evaluation='pooled LOPO out-of-fold predictions',
                auc_benchmark_met=metric['roc_auc'] >= .8,
                sensitivity_benchmark_met=metric['sensitivity'] >= .8)
            metrics.append(metric)
    write_json(out / 'metrics.json', metrics)
    with (out / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    write_json(out / 'progress.json', dict(complete=True, completed_folds=len(records),
        total_folds=len(records), evaluations=len(evidence)))
    return metrics, out


def run_repeated_lopo(spark, frame, numeric, categorical, *, data_path, output_dir,
                      seeds=(42, 7, 21, 84, 123), variant='rfe20_standard',
                      models=MODEL_NAMES, tune=True, inner_folds=3, max_folds=None,
                      gpu_trials=20, use_gpu=True):
    """Mean/sample SD across complete seeded LOPO runs, not single-record folds.

    Every seed uses all patients as held-out cases. Seeds change model randomness
    and inner-CV allocation, not outer membership. Runs are correlated; their SD
    describes algorithm variability rather than independent-population uncertainty.
    """
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Use at least two distinct seeds to calculate sample SD')
    completed, directories = [], []
    cached_here = not frame.is_cached
    if cached_here:
        frame = frame.cache()
    try:
        for seed in seeds:
            metrics, dest = run_lopo(spark, frame, numeric, categorical,
                data_path=data_path, output_dir=output_dir, variants=[variant],
                models=models, tune=tune, inner_folds=inner_folds,
                seed=seed, max_folds=max_folds, gpu_trials=gpu_trials, use_gpu=use_gpu)
            directories.append(dest)
            if metrics is not None:
                completed.extend(dict(row, seed=seed, recall=row['sensitivity']) for row in metrics)
    finally:
        if cached_here:
            frame.unpersist()
    if len(completed) != len(seeds) * len(models):
        print('Partial repeated LOPO check; no mean/SD report produced.')
        return None, None, directories
    table = spark.read.json(spark.sparkContext.parallelize([json.dumps(row) for row in completed]))
    summary = table.groupBy('variant', 'model').agg(
        F.count('*').alias('completed_seed_runs'),
        *[expr for metric in ['accuracy', 'precision', 'recall'] for expr in
          [F.mean(metric).alias(metric+'_mean'), F.stddev_samp(metric).alias(metric+'_std'),
           F.count(metric).alias(metric+'_valid_runs')]])
    identity = dict(run_directories=[str(p) for p in directories], seeds=list(seeds),
                    aggregation='mean and sample standard deviation of pooled LOPO metrics across seeds')
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    dest = Path(output_dir)/'summaries'/signature
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest/'summary_config.json', identity)
    write_json(dest/'metrics_per_seed.json', completed)
    summary_rows = [row.asDict() for row in summary.orderBy('model').collect()]
    write_json(dest/'summary.json', summary_rows)
    with (dest/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    with (dest/'metrics_per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
        writer.writeheader(); writer.writerows(completed)
    print('Mean / sample SD saved to:', dest)
    return table, summary, directories
