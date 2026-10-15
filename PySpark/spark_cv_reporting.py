"""Serialize and print Spark training-CV evidence, separately from test metrics."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def cv_evidence(search, grid, *, seed, model, variant='all_standard'):
    scores = [float(value) for value in search.avgMetrics]
    if len(scores) != len(grid) or not scores or not all(math.isfinite(v) for v in scores):
        raise ValueError('Expected finite CV scores for every parameter configuration')
    deviations = getattr(search, 'stdMetrics', None)
    if deviations is None or len(deviations) != len(scores):
        deviations = [None]*len(scores)
    best = max(range(len(scores)), key=scores.__getitem__)
    return [dict(seed=seed, model=model, variant=variant, candidate=i+1,
        parameters={p.name:value for p,value in parameters.items()},
        mean_cv_auc=score, cv_fold_std=float(deviations[i]) if deviations[i] is not None else None,
        selected=i==best) for i,(parameters,score) in enumerate(zip(grid,scores))]


def print_cv_results(rows, *, output_dir=None):
    if not rows:
        print('Training CV disabled: fixed estimator settings; no CV scores reported.',flush=True)
        return None
    for row in rows:
        sd = f" | Spark fold SD={row['cv_fold_std']:.6f}" if row.get('cv_fold_std') is not None else ''
        print(f"[TRAINING CV] seed={row['seed']} {row['variant']} {row['model']} "
              f"candidate={row['candidate']} AUC={row['mean_cv_auc']:.6f}{sd} "
              f"parameters={json.dumps(row['parameters'],sort_keys=True)}"
              f"{' [SELECTED]' if row['selected'] else ''}",flush=True)
    selected = next(row for row in rows if row['selected'])
    if output_dir is not None:
        dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
        (dest/'cv_search.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        (dest/'best_parameters.json').write_text(json.dumps(selected,indent=2,allow_nan=False))
    return selected


def save_cv_summary(rows, output_dir):
    """One row per grid candidate; mean/sample SD of selected scores across seeds."""
    if not rows:
        return
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'training_cv_results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    fields=['seed','variant','model','candidate','parameters','mean_cv_auc','cv_fold_std','selected']
    with (out/'training_cv_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows(dict(row,parameters=json.dumps(row['parameters'],sort_keys=True)) for row in rows)
    groups={}
    for row in rows:
        if row['selected']:
            groups.setdefault((row['variant'],row['model']),[]).append(row['mean_cv_auc'])
    summary=[]
    for (variant,model),values in sorted(groups.items()):
        row=dict(variant=variant,model=model,seed_runs=len(values),
                 selected_cv_auc_mean=mean(values),selected_cv_auc_sample_sd=stdev(values) if len(values)>1 else None)
        summary.append(row)
        sd=f"{row['selected_cv_auc_sample_sd']:.6f}" if len(values)>1 else 'unavailable (one seed)'
        print(f"[TRAINING CV SUMMARY] {variant} {model}: mean selected AUC={mean(values):.6f}, "
              f"sample SD across seeds={sd}, seeds={len(values)}",flush=True)
    (out/'training_cv_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (out/'training_cv_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
