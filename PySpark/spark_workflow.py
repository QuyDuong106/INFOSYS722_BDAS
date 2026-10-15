"""Native Spark SQL / ML workflow. Supplementary native Spark holdout implementation."""
import csv
import hashlib
import json
from spark_cv_reporting import cv_evidence, print_cv_results, save_cv_summary
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

CODED_FIELDS = {"gender", "comorbidity", "cad", "hypothyroidism", "hyperlipidemia", "dm", "hfa"}
MODEL_NAMES = ("logistic", "random_forest", "gbt")
PROTOCOL_VERSION = 3


def create_spark(master="local[*]", app_name="Iteration4-BDAS-Colab"):
    """Colab runs local Spark; pass a cluster master for a configured cluster."""
    session = (SparkSession.builder.master(master).appName(app_name)
               .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    session.sparkContext.setLogLevel("WARN")
    return session


def load_dataset(spark, path):
    """Read this single-line CSV using Spark, preserving zero-based source rows.

    The input's leading unnamed exported index is excluded. Quoted multiline
    CSV fields are not supported by this source-row-preserving reader.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = spark.sparkContext.textFile(str(path))
    header = next(csv.reader([lines.first()]))
    keep = [i for i, name in enumerate(header)
            if name.strip() and not name.startswith("Unnamed:") and name != "record_id"]
    names = [header[i] for i in keep]
    if "gallstone_status" not in names or len(names) != len(set(names)):
        raise ValueError("Expected unique predictor names and gallstone_status")

    def parse(item):
        line, index = item
        values = next(csv.reader([line]))
        if len(values) != len(header):
            raise ValueError(f"Malformed CSV source row {index - 1}")
        return (index - 1, *[values[i] for i in keep])

    schema = T.StructType([T.StructField("source_row", T.LongType(), False)] +
                          [T.StructField(c, T.StringType(), True) for c in names])
    raw = spark.createDataFrame(lines.zipWithIndex().filter(lambda x: x[1] > 0).map(parse), schema)
    source_label = F.col("gallstone_status").cast("double")
    if raw.filter(source_label.isNull() | ~source_label.isin(0.0, 1.0)).limit(1).count():
        raise ValueError("Source gallstone_status must contain only 0 and 1")
    features = [c for c in names if c != "gallstone_status"]
    categorical = [c for c in features if c in CODED_FIELDS]
    numeric = [c for c in features if c not in categorical]
    expressions = [F.col("source_row"), (1.0 - source_label).alias("label")]
    # Preserve supplied ordinal and binary codes without reindexing or one-hot expansion.
    for c in [*categorical, *numeric]:
        observed = ~F.trim(F.col(c)).isin("", "?", "NA", "NaN", "nan")
        if raw.filter(observed & F.col(c).cast("double").isNull()).limit(1).count():
            raise ValueError(f"Expected pre-encoded numeric values in {c}")
        value = F.col(c).cast("double")
        expressions.append(F.when(value.isNull() | F.isnan(value) |
                                  (F.abs(value) == F.lit(float("inf"))),
                                  F.lit(None).cast("double")).otherwise(value).alias(c))
    frame = raw.select(*expressions)
    if frame.select("label").distinct().count() != 2:
        raise ValueError("Both outcome classes are required")
    if frame.drop("source_row").dropDuplicates().count() != frame.count():
        raise ValueError("Review duplicate observations before record-level evaluation")
    return frame, numeric, categorical


def make_pipeline(name, numeric, categorical, seed=42):
    """All learned preparation stages are fitted inside the training partition.

    Existing ordinal/binary codes receive training-only mode imputation without
    re-encoding. Continuous values receive training-only median imputation and
    scaling. All source predictors are retained.
    """
    stages, assembled = [], []
    if numeric:
        imputed = [f"{c}__imputed" for c in numeric]
        stages.append(Imputer(strategy="median", inputCols=numeric, outputCols=imputed))
        stages.append(VectorAssembler(inputCols=imputed, outputCol="continuous_features"))
        stages.append(StandardScaler(inputCol="continuous_features", outputCol="scaled_continuous",
                                     withMean=False, withStd=True))
        assembled.append("scaled_continuous")
    if categorical:
        imputed_codes = [f"{c}__code" for c in categorical]
        stages.append(Imputer(strategy="mode", inputCols=categorical, outputCols=imputed_codes))
        assembled.extend(imputed_codes)
    stages.append(VectorAssembler(inputCols=assembled, outputCol="features"))
    if name == "logistic":
        estimator = LogisticRegression(maxIter=300, regParam=0.01, elasticNetParam=0.0)
        grid = ParamGridBuilder().addGrid(estimator.regParam, [0.001, 0.01, 0.1]).build()
    elif name == "random_forest":
        estimator = RandomForestClassifier(numTrees=60, maxDepth=5, minInstancesPerNode=4, seed=seed)
        grid = (ParamGridBuilder().addGrid(estimator.maxDepth, [3, 5])
                .addGrid(estimator.numTrees, [40, 60]).build())
    elif name == "gbt":
        estimator = GBTClassifier(maxIter=60, maxDepth=2, stepSize=0.05,
                                  minInstancesPerNode=4, seed=seed)
        grid = (ParamGridBuilder().addGrid(estimator.maxDepth, [2, 3])
                .addGrid(estimator.stepSize, [0.05, 0.1]).build())
    else:
        raise ValueError(f"Unsupported Spark model: {name}")
    return Pipeline(stages=[*stages, estimator]), grid


def fit_model(train, name, numeric, categorical, *, seed=42, tune=False, inner_folds=3,
              return_cv=False, cv_output_dir=None):
    pipeline, grid = make_pipeline(name, numeric, categorical, seed)
    if not tune:
        fitted = pipeline.fit(train)
        print_cv_results([])
        return (fitted, [], None) if return_cv else fitted
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least two")
    if min(r['count'] for r in train.groupBy("label").count().collect()) < inner_folds:
        raise ValueError("Too few records per class for stratified inner folds")
    window = Window.partitionBy("label").orderBy(F.rand(seed), F.col("source_row"))
    inner = train.withColumn("cv_fold", ((F.row_number().over(window) - 1) % inner_folds).cast("int"))
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                        evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC"),
                        numFolds=inner_folds, foldCol="cv_fold", seed=seed, parallelism=1)
    inner=inner.cache()
    try:
        search=cv.fit(inner)
        rows=cv_evidence(search,grid,seed=seed,model=name)
        best=print_cv_results(rows,output_dir=cv_output_dir)
        return (search.bestModel, rows, best) if return_cv else search.bestModel
    finally:
        inner.unpersist()


def prediction_frame(fitted, test, model, fold):
    scored = fitted.transform(test).withColumn("probability_present", vector_to_array("probability")[1])
    return scored.select("source_row", "label", "probability_present",
                         (F.col("probability_present") >= 0.5).cast("double").alias("prediction"),
                         F.lit(model).alias("model"), F.lit(fold).alias("seed"))


def pooled_metrics(predictions):
    """Compute pooled counts and discrimination in Spark, never via pandas."""
    output = []
    for name in sorted(r['model'] for r in predictions.select("model").distinct().collect()):
        data = predictions.filter(F.col("model") == name)
        counts = {(r['label'], r['prediction']): r['count']
                  for r in data.groupBy("label", "prediction").count().collect()}
        tn, fp, fn, tp = (counts.get(pair, 0) for pair in [(0., 0.), (0., 1.), (1., 0.), (1., 1.)])
        n = tn + fp + fn + tp
        ratio = lambda a, b: float(a / b) if b else None
        sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
        auc = BinaryClassificationEvaluator(rawPredictionCol="probability_present", metricName="areaUnderROC").evaluate(data)
        output.append(dict(model=name, rows=n, TN=tn, FP=fp, FN=fn, TP=tp,
                           accuracy=ratio(tp + tn, n), roc_auc=auc,
                           sensitivity=sensitivity, specificity=specificity,
                           precision=ratio(tp, tp + fp), f1=ratio(2 * tp, 2 * tp + fp + fn),
                           fnr=ratio(fn, tp + fn), error_rate=ratio(fp + fn, n)))
    return output


def run_holdout(spark, frame, numeric, categorical, *, output_dir, seeds=(42,7,21,84,123), tune=True):
    """Exact stratified 70/30 splits; Spark CV is fitted on training only."""
    import math
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    all_metrics=[];all_cv=[]
    for seed in seeds:
        # Match the canonical main workflow's exact membership, without pandas.
        from sklearn.model_selection import train_test_split
        records=frame.select("source_row","label").orderBy("source_row").collect()
        train_ids,test_ids=train_test_split([row.source_row for row in records],test_size=.3,stratify=[row.label for row in records],random_state=seed)
        train=frame.filter(F.col("source_row").isin([int(i) for i in train_ids]))
        test=frame.filter(F.col("source_row").isin([int(i) for i in test_ids]))
        predictions=None
        for name in MODEL_NAMES:
            model_dir=out/f"seed_{seed}"/name
            fitted,cv_rows,best=fit_model(train,name,numeric,categorical,seed=seed,tune=tune,
                inner_folds=3,return_cv=True,cv_output_dir=model_dir)
            all_cv.extend(cv_rows)
            scored=prediction_frame(fitted,test,name,seed)
            predictions=scored if predictions is None else predictions.unionByName(scored)
            fitted.write().overwrite().save(str(out/f"seed_{seed}"/name/"model"))
        predictions.write.mode("overwrite").option("header",True).csv(str(out/f"seed_{seed}"/"predictions"))
        for metric in pooled_metrics(predictions):
            metric['seed']=seed
            selected=next((row for row in all_cv if row['seed']==seed and row['model']==metric['model'] and row['selected']),None)
            metric['inner_cv_auc']=selected['mean_cv_auc'] if selected else None
            all_metrics.append(metric)
        save_cv_summary(all_cv,out)
    (out/"metrics.json").write_text(json.dumps(all_metrics,indent=2))
    if tune:print("Training-CV evidence:",out/"training_cv_results.csv",flush=True)
    return all_metrics
