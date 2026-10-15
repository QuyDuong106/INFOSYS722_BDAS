"""Reusable fold-safe Spark preparation and optional driver-side CUDA XGBoost.

Spark preparation and pooled metrics remain native. XGBoost is an explicitly
hybrid estimator: a bounded prepared matrix is collected to the Colab driver.
Native Spark LR/RF/GBT remain CPU estimators. No test label enters tuning.
"""
import json
from pathlib import Path
import numpy as np


def check_cuda():
    """Fail early if XGBoost falls back to CPU; never label CPU work as GPU."""
    import xgboost as xgb
    data = xgb.DMatrix(np.array([[0], [1], [2], [3]], dtype=np.float32), label=[0, 0, 1, 1])
    model = xgb.train(dict(objective='binary:logistic', device='cuda', tree_method='hist',
                          max_depth=1, nthread=1), data, num_boost_round=1)
    device = json.loads(model.save_config())['learner']['generic_param']['device']
    if not device.startswith('cuda'):
        raise RuntimeError('XGBoost did not use CUDA. Select a Colab GPU runtime; CPU fallback is disabled.')
    return device


def _fit_preparation(pipeline, train):
    from pyspark.ml import Pipeline
    return Pipeline(stages=pipeline.getStages()[:-1]).fit(train)


class PreparedFold:
    """Fit preparation once per inner fold and once for the outer training set."""
    def __init__(self, train, test, numeric, categorical, seed, k, inner_folds, tune, factory):
        from pyspark.sql import functions as F, Window
        self.cached, self.inner, self.arrays = [], [], {}
        self.final_preparation = None
        self.factory, self.numeric, self.categorical = factory, numeric, categorical
        self.seed, self.k = seed, k
        self.outer_train, self.outer_test = train, test
        if tune:
            order = Window.partitionBy('label').orderBy(F.rand(seed), F.col('source_row'))
            allocation = train.withColumn('cv_fold', ((F.row_number().over(order)-1) % inner_folds).cast('int')).cache()
            self.cached.append(allocation)
            try:
                for fold in range(inner_folds):
                    fit_rows = allocation.filter(F.col('cv_fold') != fold)
                    valid_rows = allocation.filter(F.col('cv_fold') == fold)
                    pipeline, _ = factory('logistic', numeric, categorical, seed, 'rfe', k)
                    prep = _fit_preparation(pipeline, fit_rows)
                    self.inner.append((self._cache(prep.transform(fit_rows)),
                                       self._cache(prep.transform(valid_rows))))
            except BaseException:
                self.close()
                raise

    def _cache(self, data):
        result = data.select('source_row', 'label', 'features').cache()
        self.cached.append(result)
        result.count()
        return result

    def final(self):
        if self.final_preparation is None:
            pipeline, _ = self.factory('logistic', self.numeric, self.categorical, self.seed, 'rfe', self.k)
            self.final_preparation = _fit_preparation(pipeline, self.outer_train)
            self.final_train = self._cache(self.final_preparation.transform(self.outer_train))
            self.final_test = self._cache(self.final_preparation.transform(self.outer_test))
        return self.final_train, self.final_test

    def fields(self):
        self.final()
        selector = self.final_preparation.stages[-1]
        fields = selector.getOrDefault(selector.inputCols)
        if len(fields) != self.k:
            raise ValueError('Wrong retained feature count')
        return [c.removesuffix('__imputed').removesuffix('__code') for c in fields]

    def matrix(self, data, key, max_rows=20000):
        # One collection per prepared partition, reused across all GPU trials.
        if key not in self.arrays:
            rows = data.select('features', 'label').limit(max_rows+1).collect()
            if len(rows) > max_rows:
                raise ValueError('Driver-side GPU bridge is limited to 20,000 rows per partition')
            self.arrays[key] = (np.asarray([r.features.toArray() for r in rows], dtype=np.float32),
                                np.asarray([r.label for r in rows], dtype=np.float32))
        return self.arrays[key]

    def close(self):
        for data in reversed(self.cached):
            data.unpersist()
        self.cached.clear()
        self.arrays.clear()


def native_candidate(prepared, name, tune):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
    pipeline, grid = prepared.factory(name, prepared.numeric, prepared.categorical,
                                      prepared.seed, 'rfe', prepared.k)
    estimator = pipeline.getStages()[-1]
    trials = []
    best_score, best_params = None, {}
    if tune:
        for parameters in grid:
            scores = []
            for training, validation in prepared.inner:
                fitted = estimator.copy(parameters).fit(training)
                scores.append(float(evaluator.evaluate(fitted.transform(validation))))
            mean = float(np.mean(scores))
            trials.append(dict(parameters={p.name:v for p,v in parameters.items()},
                               mean_cv_auc=mean, fold_auc=scores))
            if best_score is None or mean > best_score:
                best_score, best_params = mean, parameters
    train, test = prepared.final()
    model = estimator.copy(best_params).fit(train)
    probability = float(model.transform(test).select(vector_to_array('probability')[1].alias('p')).first().p)
    return probability, best_score, trials, 'native Spark ML CPU'


def tune_xgboost(matrices, *, seed, trials, directory, use_gpu=True):
    """Optuna objective accepts inner-fold arrays only; outer test is absent."""
    import xgboost as xgb
    import optuna
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = 'cuda' if use_gpu else 'cpu'
    pairs = [(xgb.DMatrix(x, label=y), xgb.DMatrix(v), vy)
             for (x, y), (v, vy) in matrices]
    base = dict(objective='binary:logistic', eval_metric='auc', device=device,
                tree_method='hist', nthread=1, seed=seed)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    # Completed trial JSON is written atomically to Drive after every trial.
    # No SQLite database is placed on the mounted Drive filesystem.
    checkpoint = directory/'optuna_trials.json'
    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.NopPruner())
    if checkpoint.exists():
        for row in json.loads(checkpoint.read_text()):
            distributions = {k:optuna.distributions.json_to_distribution(v)
                             for k,v in row['distributions'].items()}
            study.add_trial(optuna.trial.create_trial(params=row['parameters'],
                distributions=distributions, value=row['mean_cv_auc'], user_attrs={'fold_auc':row['fold_auc']}))

    def objective(trial):
        parameters = dict(n_estimators=trial.suggest_int('n_estimators',40,160,step=20),
            max_depth=trial.suggest_int('max_depth',1,4),
            learning_rate=trial.suggest_float('learning_rate',.01,.15,log=True),
            subsample=trial.suggest_float('subsample',.6,1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.0),
            reg_lambda=trial.suggest_float('reg_lambda',.1,10,log=True))
        rounds = parameters.pop('n_estimators')
        parameters['eta'] = parameters.pop('learning_rate')
        parameters['lambda'] = parameters.pop('reg_lambda')
        scores = []
        for train, valid, actual in pairs:
            booster = xgb.train({**base, **parameters}, train, num_boost_round=rounds)
            actual_device = json.loads(booster.save_config())['learner']['generic_param']['device']
            if use_gpu and not actual_device.startswith('cuda'):
                raise RuntimeError('CUDA training silently fell back to CPU; stopping')
            scores.append(float(roc_auc_score(actual, booster.predict(valid))))
        trial.set_user_attr('fold_auc', scores)
        return float(np.mean(scores))

    def persist(study, trial):
        rows = [dict(parameters=t.params, mean_cv_auc=t.value, fold_auc=t.user_attrs['fold_auc'],
            distributions={k:optuna.distributions.distribution_to_json(v) for k,v in t.distributions.items()})
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        temp = checkpoint.with_suffix('.tmp')
        temp.write_text(json.dumps(rows,indent=2,allow_nan=False)); temp.replace(checkpoint)
    remaining = max(0, trials-len(study.trials))
    if remaining:
        study.optimize(objective,n_trials=remaining,n_jobs=1,callbacks=[persist])
    return study.best_params, float(study.best_value), json.loads(checkpoint.read_text()), base


def gpu_candidate(prepared, *, tune, trials, directory, use_gpu=True):
    import xgboost as xgb
    if tune:
        matrices = [(prepared.matrix(train, f'train_{i}'), prepared.matrix(valid, f'valid_{i}'))
                    for i,(train,valid) in enumerate(prepared.inner)]
        params, score, evidence, base = tune_xgboost(matrices, seed=prepared.seed,
            trials=trials, directory=directory, use_gpu=use_gpu)
    else:
        params, score, evidence = dict(n_estimators=60,max_depth=2,learning_rate=.05), None, []
        base = dict(objective='binary:logistic',device='cuda' if use_gpu else 'cpu',
                    tree_method='hist',nthread=1,seed=prepared.seed)
    train, test = prepared.final()
    x,y = prepared.matrix(train, 'outer_train')
    # Labels from the outer test are not collected by this bridge.
    test_rows = test.select('features').collect()
    if len(test_rows) != 1:
        raise ValueError('Expected one held-out patient')
    xt = np.asarray([test_rows[0].features.toArray()],dtype=np.float32)
    params = dict(params)
    rounds = params.pop('n_estimators')
    params['eta'] = params.pop('learning_rate')
    if 'reg_lambda' in params: params['lambda'] = params.pop('reg_lambda')
    model = xgb.train({**base,**params}, xgb.DMatrix(x,label=y),num_boost_round=rounds)
    if use_gpu and not json.loads(model.save_config())['learner']['generic_param']['device'].startswith('cuda'):
        raise RuntimeError('Final fit fell back to CPU')
    return float(model.predict(xgb.DMatrix(xt))[0]), score, evidence, 'Spark preparation + driver CUDA XGBoost' if use_gpu else 'Spark preparation + driver CPU XGBoost'
