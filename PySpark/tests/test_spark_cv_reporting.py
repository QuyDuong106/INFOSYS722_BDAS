"""Check CV parameter/score association, selected settings and saved summaries."""
import contextlib,csv,io,json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from spark_cv_reporting import cv_evidence, print_cv_results, save_cv_summary

class CVReportingTests(unittest.TestCase):
    def evidence(self,seed=42,scores=(.72,.83,.80)):
        class Parameter:
            name='regParam'
        parameter=Parameter()
        grid=[{parameter:value} for value in [.001,.01,.1]]
        return cv_evidence(SimpleNamespace(avgMetrics=scores,stdMetrics=[.02,.03,.04]),
                           grid,seed=seed,model='logistic')
    def test_parameter_association_selection_and_artifacts(self):
        rows=self.evidence()
        self.assertEqual(rows[1]['parameters'],{'regParam':.01})
        self.assertEqual([r['selected'] for r in rows],[False,True,False])
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()) as captured:
            best=print_cv_results(rows,output_dir=tmp)
            self.assertEqual(best['mean_cv_auc'],.83)
            self.assertEqual(captured.getvalue().count('[SELECTED]'),1)
            self.assertIn('[TRAINING CV]',captured.getvalue())
            self.assertEqual(json.loads((Path(tmp)/'cv_search.json').read_text()),rows)
            self.assertEqual(json.loads((Path(tmp)/'best_parameters.json').read_text()),best)
            save_cv_summary(rows+self.evidence(seed=7,scores=(.74,.85,.82)),tmp)
            summary=json.loads((Path(tmp)/'training_cv_summary.json').read_text())[0]
            self.assertAlmostEqual(summary['selected_cv_auc_mean'],.84)
            self.assertAlmostEqual(summary['selected_cv_auc_sample_sd'],math.sqrt(.0002))
            self.assertEqual(summary['seed_runs'],2)
            with (Path(tmp)/'training_cv_results.csv').open() as stream:
                saved=list(csv.DictReader(stream))
            self.assertEqual(len(saved),6)
            self.assertEqual(sum(r['selected']=='True' for r in saved),2)
    def test_invalid_scores_and_disabled_cv(self):
        with self.assertRaises(ValueError):self.evidence(scores=(.7,float('nan'),.8))
        with self.assertRaises(ValueError):self.evidence(scores=(.7,))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(print_cv_results([]))
        self.assertIn('disabled',output.getvalue())
    def test_exact_ties_choose_first_candidate(self):
        rows=self.evidence(scores=(.83,.83,.8))
        self.assertTrue(rows[0]['selected']);self.assertFalse(rows[1]['selected'])

if __name__=='__main__':unittest.main()
