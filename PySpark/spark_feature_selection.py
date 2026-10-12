"""Persistable Spark CART/RFE estimator, fitted inside every training CV fold."""
import math,hashlib,json
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql import functions as F

_MODEL_CACHE = {}  # Reuse only identical fold data and selector settings.

class _SelectionParams(Params):
    inputCols = Param(Params._dummy(), 'inputCols', 'Imputed predictor columns', TypeConverters.toListString)
    continuousCols = Param(Params._dummy(), 'continuousCols', 'Columns requiring scaling', TypeConverters.toListString)
    outputCol = Param(Params._dummy(), 'outputCol', 'Classifier vector', TypeConverters.toString)
    method = Param(Params._dummy(), 'method', 'all, cart or rfe', TypeConverters.toString)
    k = Param(Params._dummy(), 'k', 'Retained field count', TypeConverters.toInt)
    seed = Param(Params._dummy(), 'seed', 'Tree seed', TypeConverters.toInt)
    scales = Param(Params._dummy(), 'scales', 'Training sample standard deviations', TypeConverters.toListFloat)

class SparkFeatureSelector(Estimator, _SelectionParams, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCols=None, continuousCols=None, outputCol='features', method='rfe', k=20, seed=42):
        super().__init__()
        self._setDefault(method='rfe', k=20, seed=42, outputCol='features', continuousCols=[])
        self._set(inputCols=inputCols or [], continuousCols=continuousCols or [], outputCol=outputCol,
                  method=method, k=k, seed=seed)

    def _fit(self, dataset):
        active = list(self.getOrDefault(self.inputCols))
        continuous = set(self.getOrDefault(self.continuousCols))
        method, k = self.getOrDefault(self.method), self.getOrDefault(self.k)
        if method not in {'all', 'cart', 'rfe'} or not 1 <= k <= len(active):
            raise ValueError('Require all/cart/rfe and a field count within the input width')
        # Classifier grid candidates share the same selector settings/folds.
        # Fingerprint imputed training values and labels using two bounded Spark
        # aggregates; never collect the full training table or key on test data.
        hash_columns=active+['label']+(['source_row'] if 'source_row' in dataset.columns else [])
        fingerprint=dataset.agg(F.count('*').alias('n'),
            F.sum(F.xxhash64(*hash_columns).cast('decimal(38,0)')).alias('h1'),
            F.sum(F.xxhash64(F.lit('selector-fold-v1'),*[F.col(c) for c in hash_columns]).cast('decimal(38,0)')).alias('h2')).first()
        cache_key=hashlib.sha256(json.dumps([fingerprint.n,str(fingerprint.h1),str(fingerprint.h2),
            active,sorted(continuous),method,k,self.getOrDefault(self.seed),self.getOrDefault(self.outputCol)],sort_keys=True).encode()).hexdigest()
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key].copy()
        # Supervised ranking sees only the dataset passed to this pipeline fit.
        if method != 'all':
            while len(active) > k:
                vector = VectorAssembler(inputCols=active, outputCol='__ranking_features').transform(dataset)
                tree = DecisionTreeClassifier(featuresCol='__ranking_features', labelCol='label',
                    maxDepth=5, minInstancesPerNode=4, seed=self.getOrDefault(self.seed)).fit(vector)
                importance = list(tree.featureImportances)
                if method == 'cart':
                    keep = sorted(range(len(active)), key=lambda i: (-importance[i], active[i]))[:k]
                    active = [active[i] for i in sorted(keep)]
                    break
                # RFE removes one field, refitting the native Spark CART each time.
                removed = min(range(len(active)), key=lambda i: (importance[i], active[i]))
                active.pop(removed)
        else:
            k = len(active)
        numeric = [c for c in active if c in continuous]
        summary = dataset.agg(*[F.stddev_samp(c).alias(c) for c in numeric]).first() if numeric else None
        scales = []
        for c in active:
            std = summary[c] if c in continuous else 1.0
            scales.append(float(std) if std is not None and math.isfinite(std) and std > 0 else (0.0 if c in continuous else 1.0))
        fitted = SparkFeatureSelectionModel()
        fitted._set(inputCols=active, continuousCols=[c for c in active if c in continuous],
                    outputCol=self.getOrDefault(self.outputCol), method=method, k=k,
                    seed=self.getOrDefault(self.seed), scales=scales)
        _MODEL_CACHE[cache_key]=fitted
        return fitted.copy()

class SparkFeatureSelectionModel(Model, _SelectionParams, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self):
        super().__init__()

    def _transform(self, dataset):
        active = self.getOrDefault(self.inputCols)
        scales = self.getOrDefault(self.scales)
        continuous = set(self.getOrDefault(self.continuousCols))
        transformed = dataset
        columns = []
        for i, (c, scale) in enumerate(zip(active, scales)):
            out = f'__selected_value_{i}'
            # Match native Spark baseline scaling: divide by training sample std,
            # withMean=False; coded categories are passed through unchanged.
            expression = (F.col(c) / F.lit(scale) if scale > 0 else F.lit(0.0)) if c in continuous else F.col(c)
            transformed = transformed.withColumn(out, expression)
            columns.append(out)
        return VectorAssembler(inputCols=columns, outputCol=self.getOrDefault(self.outputCol)).transform(transformed)
