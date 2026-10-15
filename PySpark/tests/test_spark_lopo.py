"""Colab integration check: exclusion, partial-run handling, pooling and resume."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('pyspark'), 'PySpark runtime required')
class SparkLOPOTest(unittest.TestCase):
    def test_exclusion_partial_pooling_and_resume(self):
        import spark_lopo
        from spark_workflow import create_spark
        spark = create_spark(master='local[2]', app_name='LOPO-integration-check')
        frame = spark.createDataFrame([(i, float(i % 2), *[float(i % 2) + i/20+j/10 for j in range(10)])
            for i in range(8)], 'source_row long, label double, ' + ', '.join(f'x{j} double' for j in range(10)))
        import spark_lopo_acceleration
        original = spark_lopo_acceleration._fit_preparation
        fitted_ids = []
        def checked_fit(pipeline, train):
            fitted_ids.append({r.source_row for r in train.select('source_row').collect()})
            return original(pipeline, train)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.csv'; source.write_text('synthetic integration test')
            options = dict(data_path=source, output_dir=Path(tmp)/'outputs',
                variants=['rfe10_standard'], models=['logistic'], tune=False)
            with patch.object(spark_lopo_acceleration, '_fit_preparation', side_effect=checked_fit):
                metrics, out = spark_lopo.run_lopo(spark, frame, [f'x{j}' for j in range(10)], [], max_folds=2, **options)
                self.assertIsNone(metrics)
                self.assertFalse((out/'metrics.json').exists())
                self.assertEqual(fitted_ids, [set(range(8))-{0}, set(range(8))-{1}])
                metrics, resumed = spark_lopo.run_lopo(spark, frame, [f'x{j}' for j in range(10)], [], **options)
                self.assertEqual(out, resumed)
                self.assertEqual(len(fitted_ids), 8)  # Two checkpoints reused.
                for held_out, seen in enumerate(fitted_ids):
                    self.assertEqual(seen, set(range(8))-{held_out})
                self.assertEqual(metrics[0]['rows'], 8)
                self.assertEqual(sum(metrics[0][c] for c in ['TN','FP','FN','TP']), 8)
                import json
                self.assertTrue(json.loads((out/'progress.json').read_text())['complete'])
                spark_lopo.run_lopo(spark, frame, [f'x{j}' for j in range(10)], [], **options)
                self.assertEqual(len(fitted_ids), 8)  # Completed resume does not fit.

    def test_seed_summary_mean_and_sample_sd(self):
        import math
        import spark_lopo
        from spark_workflow import create_spark
        spark = create_spark(master='local[2]', app_name='LOPO-summary-check')
        frame = spark.createDataFrame([(0, 0.0)], 'source_row long, label double')
        with tempfile.TemporaryDirectory() as tmp:
            def completed_run(*args, seed, **kwargs):
                value = .6 if seed == 42 else .8
                return [dict(model='logistic', variant='rfe10_standard',
                    accuracy=value, precision=value, sensitivity=value)], Path(tmp)/str(seed)
            with patch.object(spark_lopo, 'run_lopo', side_effect=completed_run):
                table, summary, dirs = spark_lopo.run_repeated_lopo(spark, frame, [], [],
                    data_path=Path(tmp)/'unused.csv', output_dir=Path(tmp)/'results',
                    seeds=[42,7], variant='rfe10_standard', models=['logistic'])
                row = summary.first()
                for metric in ['accuracy','precision','recall']:
                    self.assertAlmostEqual(row[metric+'_mean'], .7)
                    self.assertAlmostEqual(row[metric+'_std'], math.sqrt(.02))
                    self.assertEqual(row[metric+'_valid_runs'], 2)
                self.assertEqual(table.count(), 2)
            with patch.object(spark_lopo, 'run_lopo', return_value=(None,Path(tmp)/'partial')):
                table, summary, dirs = spark_lopo.run_repeated_lopo(spark, frame, [], [],
                    data_path=Path(tmp)/'unused.csv', output_dir=Path(tmp)/'partial_results',
                    seeds=[42,7], variant='rfe10_standard', models=['logistic'], max_folds=2)
                self.assertIsNone(table)
                self.assertIsNone(summary)
                self.assertFalse((Path(tmp)/'partial_results/summaries').exists())

    def test_preparation_reused_and_outer_patient_excluded(self):
        import spark_lopo_acceleration as acceleration
        from spark_workflow import create_spark
        from spark_iterations import make_selected_pipeline
        spark = create_spark(master='local[2]', app_name='LOPO-fold-cache-check')
        frame = spark.createDataFrame([(i, float(i%2), *[float(i+j) for j in range(10)])
            for i in range(12)], 'source_row long, label double, '+', '.join(f'x{j} double' for j in range(10)))
        from pyspark.sql import functions as F
        train=frame.filter(F.col('source_row')!=11);test=frame.filter(F.col('source_row')==11)
        original=acceleration._fit_preparation
        seen=[]
        def checked(pipeline, rows):
            seen.append({r.source_row for r in rows.select('source_row').collect()})
            return original(pipeline,rows)
        with patch.object(acceleration,'_fit_preparation',side_effect=checked):
            prepared=acceleration.PreparedFold(train,test,[f'x{j}' for j in range(10)],[],
                42,10,3,True,make_selected_pipeline)
            try:
                self.assertEqual(len(seen),3)
                self.assertTrue(all(11 not in ids and len(ids)<11 for ids in seen))
                prepared.final();prepared.final();prepared.fields()
                self.assertEqual(len(seen),4) # Outer preparation refitted just once.
                self.assertEqual(seen[-1],set(range(11)))
                a=prepared.matrix(prepared.inner[0][0],'shared')
                b=prepared.matrix(prepared.inner[0][0],'shared')
                self.assertIs(a,b) # One driver collection for all GPU candidates.
            finally:
                prepared.close()
