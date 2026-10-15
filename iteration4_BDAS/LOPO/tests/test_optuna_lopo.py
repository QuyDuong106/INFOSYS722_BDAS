"""Nested preparation isolation, tuning scope, resumability and scoring checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import optuna_lopo as workflow


class RecordingTransformer(TransformerMixin, BaseEstimator):
    fits = []

    def fit(self, X, y=None):
        type(self).fits.append(list(X.index))
        self.center_ = X.mean()
        return self

    def transform(self, X):
        return X - self.center_


class OptunaLeaveOnePatientOutTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(24, 4)),
                              index=np.arange(100, 124), columns=list('abcd'))
        self.y = pd.Series(np.tile([0, 1], 12), index=self.X.index)

    def test_preparation_refits_in_each_inner_training_fold(self):
        RecordingTransformer.fits = []
        template = Pipeline([('prepare', RecordingTransformer()),
                             ('model', LogisticRegression())])
        # Outer test record 100 never enters the tuning function.
        X_train, y_train = self.X.iloc[1:], self.y.iloc[1:]
        with patch.object(workflow, 'make_evaluation_model', return_value=template):
            fitted, params, score, evidence = workflow.tune_training_fold(
                'logistic', X_train, y_train, trials=2, inner_folds=3, top_k=4)
        fits = RecordingTransformer.fits
        self.assertEqual(len(fits), 7)  # two trials * three inner folds, then refit
        self.assertTrue(all(100 not in rows for rows in fits))
        self.assertTrue(all(len(rows) < len(X_train) for rows in fits[:-1]))
        self.assertEqual(fits[-1], list(X_train.index))
        self.assertEqual(params['model__C'], fitted.named_steps['model'].C)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(len(evidence), 2)
        for trial in evidence.user_attrs_inner_auc:
            self.assertEqual(len(trial), 3)

    def test_resume_excludes_outer_patient_and_scores_only_complete_run(self):
        training_sets = []

        def fake_tuning(name, X, y, **kwargs):
            training_sets.append(list(X.index))
            return DummyClassifier(strategy='prior').fit(X, y), {}, .5, pd.DataFrame()

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            kwargs = dict(models=['logistic'], trials=1, inner_folds=2, top_k=4,
                          output_dir=out)
            with patch.object(workflow, 'workflow_data', return_value=(self.X, self.y)), \
                    patch.object(workflow, 'tune_training_fold', side_effect=fake_tuning):
                metrics, summary = workflow.run_optuna_lopo(max_folds=2, **kwargs)
                self.assertIsNone(metrics)
                self.assertFalse((out / 'metrics.csv').exists())
                self.assertFalse(json.loads((out / 'progress.json').read_text())['complete'])
                self.assertEqual(training_sets[0], list(self.X.index[1:]))
                self.assertNotIn(101, training_sets[1])
                metrics, summary = workflow.run_optuna_lopo(**kwargs)
                # First two folds were reused rather than retuned.
                self.assertEqual(len(training_sets), len(self.X))
                predictions = pd.read_csv(out / 'predictions.csv')
                self.assertEqual(len(predictions), len(self.X))
                self.assertTrue(predictions.source_row.is_unique)
                self.assertEqual(int(metrics.iloc[0]['rows']), len(self.X))
                self.assertTrue(json.loads((out / 'progress.json').read_text())['complete'])
                with self.assertRaisesRegex(ValueError, 'complete evaluation'):
                    workflow.run_optuna_lopo(max_folds=1, **kwargs)
                with self.assertRaisesRegex(ValueError, 'different configuration'):
                    workflow.run_optuna_lopo(**{**kwargs, 'trials':2})

    def test_full_data_refit_is_separate_from_outer_predictions(self):
        sizes = []

        def fake_tuning(name, X, y, **kwargs):
            sizes.append(len(X))
            return DummyClassifier(strategy='prior').fit(X, y), {}, .5, pd.DataFrame()

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(workflow, 'workflow_data', return_value=(self.X, self.y)), \
                    patch.object(workflow, 'tune_training_fold', side_effect=fake_tuning):
                workflow.run_optuna_lopo(models=['logistic'], trials=1, inner_folds=2,
                    top_k=4, output_dir=temporary, refit_final=True)
            self.assertEqual(sizes[:-1], [23] * 24)
            self.assertEqual(sizes[-1], 24)
            out = Path(temporary)
            self.assertTrue((out / 'final_models/logistic.pkl').exists())
            self.assertEqual(len(pd.read_csv(out / 'predictions.csv')), 24)


if __name__ == '__main__':
    unittest.main()
