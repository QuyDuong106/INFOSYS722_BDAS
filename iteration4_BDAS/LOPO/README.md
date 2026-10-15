# LOPO tuning method with non-native Spark

This is a reproducible copy of the **non-Spark Python** nested Optuna Leave-One-Patient-Out method and completed evidence from `INFOSYS722/Research_Paper`. It does not import PySpark or claim GPU execution. It is separate from the native/hybrid notebooks in the repository's `PySpark` folder.

## Saved results

All five completed models use seed 42, 319 outer patient exclusions per model, 20 Optuna trials per outer fold, and three-fold inner training CV. `results/metrics_combined.csv` holds pooled test metrics; `predictions_combined.csv` has 1,595 held-out predictions. Each `results/optuna_<model>` directory preserves its original configuration, per-fold probabilities, trial CSVs, best parameters, complete flag and model-specific metrics/summary. Partial validation runs and fixed-model LOPO results were not included. `provenance.json` records source hashes and validation.

Preparation is fitted inside each training fold: median/mode imputation, CART-based RFE selecting **20 source predictors**, continuous Yeo–Johnson followed by standardization, and **one-hot encoding** of selected categorical fields. Twenty refers to source fields, not the expanded feature-vector width. This historical encoding differs from the current ordinal-preserving holdout workflow and is retained to match the measured research output.

Standalone LR/RF/XGBoost/LightGBM use the search spaces in `optuna_lopo.py`. The saved stacking run tunes **only logistic meta-learner C**, with its base learners fixed. It is not the later holdout experiment that jointly tunes boosting base learners. A single complete seed supplies pooled metrics, **not mean ± SD across independent or repeated seed runs**. Do not average single-patient precision/recall or calculate per-patient ROC-AUC.

## Run

From this folder, use an existing compatible environment. No dependency reinstall is required if the packages in `requirements.txt` are already present. The recorded package versions are in each saved `run_config.json`.

```bash
python run_lopo.py --models logistic random_forest xgboost lightgbm stacking --seed 42 --trials 20 --inner-folds 3 --top-k 20 --output-dir results/new_runs/all_models
```

A short check evaluates the first two outer patients while still training on all remaining patients:

```bash
python run_lopo.py --models logistic --trials 2 --max-folds 2 --output-dir results/new_runs/check
```

Resume a check or interrupted run with the **same settings and output directory**, omitting `--max-folds` to finish all patients. Partial runs do not write full-evaluation metrics. Configuration/data/package changes require a new output directory. The imported completed per-model output directories remain the evidence snapshot; new experiments default here to a separate `results/new_runs` location.

The existing original data and three source modules were copied without changing their modelling logic. The small launcher, this documentation, provenance record and combined metric/prediction CSVs are new. The report's Section 8.5 discusses the saved results; they do not replace its five-seed holdout benchmark.


## Observed Research Paper results only

Each outer fit trains on 318 records, using three-fold training CV and 20 Optuna trials. The evaluation pools all **319 held-out patient predictions** per model. Only training seed 42 is complete in this archive, so no across-seed mean or variance is claimed. No bootstrap estimates are included.

```bash
python report_results.py
```

This verifies each original `metrics.csv` against `predictions.csv` and its completed configuration, copies the observed values into `results/metrics_combined.csv`, records source hashes in `results/report_validation.json`, and regenerates `../Report/generated/research_optuna_lopo.tex`. It performs no tuning, retraining or resampling. The canonical working folder is `INFOSYS722_BDAS/iteration4_BDAS`.
