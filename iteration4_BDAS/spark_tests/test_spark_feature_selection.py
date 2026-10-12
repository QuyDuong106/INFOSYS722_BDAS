"""Run in a Spark/Java runtime before full CART20/RFE20/RFE10 experiments."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
module_root=Path(__file__).resolve().parents[1]
if not (module_root/'spark_feature_selection.py').exists():module_root=module_root.parent/'PySpark'
sys.path.insert(0,str(module_root))
AVAILABLE=importlib.util.find_spec('pyspark') is not None
if AVAILABLE:
    from pyspark.sql import SparkSession,functions as F
    from pyspark.ml.tuning import CrossValidator
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from spark_feature_selection import SparkFeatureSelector,SparkFeatureSelectionModel
    from spark_iterations import make_selected_pipeline

@unittest.skipUnless(AVAILABLE,'PySpark is not installed in this environment')
class SparkSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark=(SparkSession.builder.master('local[2]').appName('Spark-feature-selection-checks')
                   .config('spark.sql.shuffle.partitions','2').getOrCreate())
        cls.spark.sparkContext.setLogLevel('ERROR')
        cls.numeric=[f'x{i}' for i in range(10)]
        cols=['source_row','label',*cls.numeric,'c0','c1']
        rows=[(i,float(i%2),*[float((i*(j+3))%17) for j in range(10)],float(1+4*(i%2)),float(i%3)) for i in range(40)]
        cls.data=cls.spark.createDataFrame(rows,cols)
    @classmethod
    def tearDownClass(cls):cls.spark.stop()

    def test_rfe_count_codes_and_model_roundtrip(self):
        train=self.data.filter(F.col('source_row')<32)
        model=SparkFeatureSelector(inputCols=self.numeric+['c0','c1'],continuousCols=self.numeric,k=10,method='rfe').fit(train)
        fields=model.getOrDefault(model.inputCols);self.assertEqual(len(fields),10);self.assertIn('c0',fields)
        row=model.transform(self.data.filter(F.col('source_row')==33)).first()
        self.assertEqual(row.features[fields.index('c0')],row.c0)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'selector');model.write().overwrite().save(path)
            loaded=SparkFeatureSelectionModel.load(path)
            actual=loaded.transform(self.data.filter(F.col('source_row')==33)).first().features
            self.assertEqual(list(row.features),list(actual))

    def test_selector_cache_separates_changed_training_values(self):
        train=self.data.filter(F.col('source_row')<32)
        selector=SparkFeatureSelector(inputCols=self.numeric+['c0','c1'],continuousCols=self.numeric,k=12,method='all')
        before=selector.fit(train)
        changed=train.withColumn('x0',F.when(F.col('source_row')==0,F.col('x0')+100).otherwise(F.col('x0')))
        after=selector.fit(changed)
        self.assertNotEqual(before.getOrDefault(before.scales)[0],after.getOrDefault(after.scales)[0])
        repeated=selector.fit(train)
        self.assertEqual(before.getOrDefault(before.scales),repeated.getOrDefault(repeated.scales))

    def test_cv_selection_never_receives_holdout_rows(self):
        train=self.data.filter(F.col('source_row')<32).withColumn('cv_fold',(F.floor(F.col('source_row')/2)%3).cast('int'))
        pipeline,grid=make_selected_pipeline('logistic',self.numeric,['c0','c1'],42,'cart',10)
        seen=[];original=SparkFeatureSelector._fit
        def observed(selector,data):
            seen.append({r.source_row for r in data.select('source_row').collect()})
            return original(selector,data)
        with patch.object(SparkFeatureSelector,'_fit',observed):
            fitted=CrossValidator(estimator=pipeline,estimatorParamMaps=grid[:1],
                evaluator=BinaryClassificationEvaluator(metricName='areaUnderROC'),
                numFolds=3,foldCol='cv_fold',parallelism=1,seed=42).fit(train).bestModel
        self.assertEqual(len(seen),4)
        self.assertTrue(all(ids<=set(range(32)) for ids in seen))
        self.assertEqual(sum(len(ids)==32 for ids in seen),1)
        self.assertTrue(all(len(ids)<32 for ids in seen[:3]))
        self.assertEqual(fitted.transform(self.data.filter(F.col('source_row')>=32)).count(),8)

if __name__=='__main__':unittest.main()
