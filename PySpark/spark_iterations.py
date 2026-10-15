"""Matched native Spark holdout: all/CART20/RFE20/RFE10, standardized inputs."""
import argparse, csv, hashlib, json, platform
from pathlib import Path
from time import perf_counter
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F, Window
from sklearn.model_selection import train_test_split
from spark_workflow import create_spark, load_dataset, make_pipeline, pooled_metrics, prediction_frame, MODEL_NAMES
from spark_feature_selection import SparkFeatureSelector
from spark_cv_reporting import cv_evidence, print_cv_results, save_cv_summary
VARIANTS = {'all_standard': ('all', None), 'cart20_standard': ('cart', 20),
            'rfe20_standard': ('rfe', 20), 'rfe10_standard': ('rfe', 10)}
SEEDS = (42, 7, 21, 84, 123)

def make_selected_pipeline(name, numeric, categorical, seed, method, k):
    stages=[]
    numeric_imputed=[f'{c}__imputed' for c in numeric]
    category_imputed=[f'{c}__code' for c in categorical]
    if numeric:stages.append(Imputer(strategy='median', inputCols=numeric, outputCols=numeric_imputed))
    if categorical:stages.append(Imputer(strategy='mode', inputCols=categorical, outputCols=category_imputed))
    stages.append(SparkFeatureSelector(inputCols=numeric_imputed+category_imputed,
        continuousCols=numeric_imputed, method=method, k=k or len(numeric)+len(categorical), seed=seed))
    old, grid = make_pipeline(name, numeric, categorical, seed)
    stages.append(old.getStages()[-1])
    return Pipeline(stages=stages), grid

def run(spark, frame, numeric, categorical, *, data_path, output_dir=None,
        seeds=SEEDS, variants=tuple(VARIANTS), models=MODEL_NAMES, tune=True):
    if len(seeds)!=5 or len(set(seeds))!=5:raise ValueError('Five distinct seeds are required')
    if not variants or any(v not in VARIANTS for v in variants):raise ValueError('Unknown preparation variant')
    if not models or any(m not in MODEL_NAMES for m in models):raise ValueError('Unknown Spark model')
    source_hash=hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    code_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('spark_feature_selection.py').read_bytes()+Path(__file__).with_name('spark_workflow.py').read_bytes()+Path(__file__).with_name('spark_cv_reporting.py').read_bytes()).hexdigest()
    config=dict(protocol_version=1,source_hash=source_hash,code_hash=code_hash,spark_version=spark.version,
        python_version=platform.python_version(),seeds=list(seeds),variants=list(variants),models=list(models),
        test_fraction=.3,inner_folds=3,tune=tune,threshold=.5,scaling='continuous training sample std, no centering; ordinal codes unchanged')
    run_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    out=Path(output_dir) if output_dir else Path(__file__).resolve().parent/'results/spark_iterations'/run_id
    out.mkdir(parents=True,exist_ok=True);cp=out/'run_config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Explicit directory has incompatible configuration; choose a new directory')
    cp.write_text(json.dumps(config,indent=2))
    records=frame.select('source_row','label').orderBy('source_row').collect()
    all_metrics=[];all_cv=[]
    for seed in seeds:
        train_ids,test_ids=train_test_split([r.source_row for r in records],test_size=.3,
            stratify=[r.label for r in records],random_state=seed)
        assert not set(train_ids)&set(test_ids)
        train=frame.filter(F.col('source_row').isin([int(i) for i in train_ids])).cache()
        test=frame.filter(F.col('source_row').isin([int(i) for i in test_ids])).cache()
        train_n,test_n=train.count(),test.count()
        sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
        with (sd/'split_membership.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['source_row','partition'])
            w.writerows((i,'train') for i in train_ids);w.writerows((i,'test') for i in test_ids)
        if tune:
            window=Window.partitionBy('label').orderBy(F.rand(seed),F.col('source_row'))
            inner=train.withColumn('cv_fold',((F.row_number().over(window)-1)%3).cast('int')).cache()
            inner.count()
        for variant in variants:
            method,k=VARIANTS[variant]
            for name in models:
                dest=sd/variant/name;dest.mkdir(parents=True,exist_ok=True);checkpoint=dest/'metrics.json'
                if checkpoint.exists():
                    metric=json.loads(checkpoint.read_text())
                    if not (dest/'model').exists() or not (dest/'predictions').exists():raise ValueError('Incomplete model checkpoint')
                else:
                    print(f'Spark seed {seed}: {variant} {name}',flush=True);started=perf_counter()
                    pipeline,grid=make_selected_pipeline(name,numeric,categorical,seed,method,k)
                    if tune:
                        cv=CrossValidator(estimator=pipeline,estimatorParamMaps=grid,
                            evaluator=BinaryClassificationEvaluator(metricName='areaUnderROC'),
                            numFolds=3,foldCol='cv_fold',seed=seed,parallelism=1)
                        search=cv.fit(inner);fitted=search.bestModel
                        evidence=cv_evidence(search,grid,seed=seed,model=name,variant=variant)
                        best_row=next(row for row in evidence if row['selected'])
                        cv_score=best_row['mean_cv_auc']
                    else:
                        fitted=pipeline.fit(train);evidence=[];cv_score=None
                    predictions=prediction_frame(fitted,test,name,seed).cache()
                    metric=pooled_metrics(predictions)[0]
                    selector=next(stage for stage in fitted.stages if hasattr(stage,'scales'))
                    fields=selector.getOrDefault(selector.inputCols)
                    assert len(fields)==(k or len(numeric)+len(categorical))
                    metric.update(seed=seed,variant=variant,train_rows=train_n,test_rows=test_n,
                        retained_features=len(fields),inner_cv_auc=cv_score,fit_predict_seconds=perf_counter()-started)
                    metric['auc_benchmark_met']=metric['roc_auc']>=.8
                    metric['sensitivity_benchmark_met']=metric['sensitivity']>=.8
                    fitted.write().overwrite().save(str(dest/'model'))
                    predictions.write.mode('overwrite').option('header',True).csv(str(dest/'predictions'))
                    (dest/'selected_features.json').write_text(json.dumps([c.removesuffix('__imputed').removesuffix('__code') for c in fields],indent=2))
                    (dest/'cv_search.json').write_text(json.dumps(evidence,indent=2))
                    checkpoint.write_text(json.dumps(metric,indent=2,allow_nan=False));predictions.unpersist()
                if tune:
                    evidence=json.loads((dest/'cv_search.json').read_text())
                    if not all('selected' in row and 'seed' in row for row in evidence):
                        raise ValueError('Old checkpoint lacks full CV evidence; use a new run directory')
                    print_cv_results(evidence,output_dir=dest)
                    all_cv.extend(evidence)
                    save_cv_summary(all_cv,out)
                else:
                    print_cv_results([])
                if metric['rows']!=test_n:raise ValueError('Checkpoint test size mismatch')
                all_metrics.append(metric)
                (out/'progress.json').write_text(json.dumps(dict(complete=False,evaluations=len(all_metrics),expected=len(seeds)*len(variants)*len(models))))
        if tune:inner.unpersist()
        train.unpersist();test.unpersist()
    (out/'metrics.json').write_text(json.dumps(all_metrics,indent=2))
    table=spark.read.json(spark.sparkContext.parallelize([json.dumps(row) for row in all_metrics]))
    table.coalesce(1).write.mode('overwrite').option('header',True).csv(str(out/'metrics_csv'))
    summary=table.groupBy('variant','model').agg(*[a for metric in ['roc_auc','accuracy','sensitivity','specificity','f1']
        for a in [F.mean(metric).alias(metric+'_mean'),F.stddev_samp(metric).alias(metric+'_sd')]])
    summary.coalesce(1).write.mode('overwrite').option('header',True).csv(str(out/'summary_csv'))
    (out/'progress.json').write_text(json.dumps(dict(complete=True,evaluations=len(all_metrics),expected=len(seeds)*len(variants)*len(models))))
    return table,summary,out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    local_data=Path(__file__).resolve().parent/'data/raw/dataset_gallstone'
    default_data=local_data if local_data.exists() else Path(__file__).resolve().parent.parent/'iteration4_BDAS/data/raw/dataset_gallstone'
    p.add_argument('--data',type=Path,default=default_data);p.add_argument('--output-dir',type=Path)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS));p.add_argument('--fixed',action='store_true')
    args=p.parse_args();spark=create_spark();frame,numeric,categorical=load_dataset(spark,args.data)
    metrics,summary,out=run(spark,frame,numeric,categorical,data_path=args.data,
        output_dir=args.output_dir,variants=args.variants,tune=not args.fixed)
    summary.show(truncate=False);print('Results:',out)
