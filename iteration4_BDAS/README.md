# INFOSYS722 BDAS — Iteration 4 with PySpark

This gallstone-prediction project follows **CRISP-DM**. Development and report writing cover **21 September–16 October 2026**. The primary submission workflow now uses native Spark; the earlier five-family Python/Optuna experiments remain separately reproducible.

## Open in Google Colab

| Notebook | Purpose |
| --- | --- |
| [Primary Spark iterations](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/run_spark_iterations_colab.ipynb) | All-field standardization, CART20, RFE20 and RFE10; five holdouts and training-only Spark CV |
| [Saved Spark execution](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/run_spark_workflow.ipynb) | Author's completed all-field run and saved native Spark result table |
| [Saved Spark visualization](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/spark_visualization_colab.ipynb) | All 31 continuous predictors: distributions, outcome comparisons and correlations |
| [Retained Python prototype](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/iteration4_BDAS/run_python_workflow.ipynb) | Five-family fixed benchmarks and Optuna tuning |

[Report PDF](Report/main.pdf) · [PySpark implementation](../PySpark) · [Iteration 4 files](.)

## 1. Business understanding

Investigate whether routine demographic, comorbidity, laboratory and body-composition data can support early gallstone risk assessment. This is an academic decision-support prototype rather than a deployed clinical system. Project benchmarks are ROC–AUC and gallstone sensitivity both at least 0.80, alongside specificity, false negatives, calibration-related loss, stability and interpretability.

## 2. Data understanding

The [raw dataset](data/raw/dataset_gallstone) has 319 records (161 cases and 158 controls) and 38 predictors. Source outcome 0 means gallstones present; model class 1 means present. The loaders validate numeric fields and outcome codes, check duplicates and remove exported index/identifier fields. Existing ordinal/binary predictor codes are preserved. The Spark visualization notebook supports exploratory understanding; full-cohort plots are not held-out model-selection evidence. Dataset provenance is documented in the report bibliography.

The latest saved Colab visualization plots **all 31 continuous predictors** in two 8 × 4 grids: overall histograms and boxplots grouped by gallstone outcome. Spark computes histogram counts and grouped approximate quartiles; Matplotlib renders the figures. Boxplot whiskers show observed minimum and maximum values. Both figures are included in the [updated report](Report/main.pdf), with the existing analysis retained. The [executed notebook snapshot and provenance](results/spark_visualization_reference) preserve the outputs used in the report (15 October 2026).

## 3. Data preparation

Native Spark prepares four matched versions: all source fields, CART importance selection with 20 fields, CART-based recursive feature elimination with 20 fields, and RFE with 10 fields. Continuous missing values use training medians; coded fields use training modes. Continuous values are divided by their training sample standard deviation without centering, matching the saved Spark baseline; category codes are passed through unchanged.

[spark_feature_selection.py](../PySpark/spark_feature_selection.py) implements persistable native Spark selectors. RFE removes one least-important field at a time and refits a Spark CART. Selection and scaling are fitted again in every inner CV fit, excluding validation/test outcomes. Saved Spark models retain the actual selected source fields.

The retained Python prototype separately compares CART/RFE20 with Yeo–Johnson plus standardization or standardization alone, and now includes a completed RFE10/standardization iteration. Python scores are not labelled as Spark results.

## 4. Modelling

The primary holdout workflows use five stratified 70/30 holdouts with seeds 42, 7, 21, 84 and 123: 223 training and 96 test records per split. Section 8.5 separately reports the Research Paper nested Optuna LOPO experiment, with three-fold CV within each outer training set. Native Spark model families are logistic regression, random forest and gradient-boosted trees. Tuning uses three stratified CV folds exclusively inside training, with full pipeline preparation refitted per fold. The classifier threshold is fixed at 0.5.

Native grids compare logistic regularisation 0.001/0.01/0.1; forest depth 3/5 and tree count 40/60; and GBT depth 2/3 and step size 0.05/0.1. The selector uses depth 5/minimum leaf size 4; controlled preparation versions compare feature-selection method and 20/10-field width. Spark extension outputs retain every grid candidate and CV score.

The Python prototype retains logistic regression, random forest, XGBoost, LightGBM and stacking. Its primary RFE20/standardization version has 20 Optuna trials per predictive family/seed. Stacking tunes only layer-1 XGBoost/LightGBM while its two logistic components stay at C=1 and its forest remains fixed.

## 5. Evaluation and iteration evidence

Completed evidence is clearly separated by implementation:

- [Python baseline/tuning results](results/holdout): 150 model evaluations and 500 Optuna trials; metrics, selected fields, parameters, models and predictions.
- [New Python RFE10 results](results/holdout_rfe10): 30 matched fixed-model evaluations; exactly ten fields in every fitted selector/stacking base. The report compares RFE10 against the fixed RFE20 reference rather than assuming fewer fields improve prediction.
- [Measured Spark holdout results](results/spark_holdout_drive): full-precision saved outputs for 15 native model/seed evaluations used in the report. The earlier [notebook-display reference](results/spark_colab_reference) remains archived with its six-decimal precision documented.
- [Observed Research Paper LOPO results](LOPO/README.md): five model families, each evaluated across all 319 held-out patients, using three-fold training CV for Optuna. These are observed pooled predictions, separate from the five-seed holdout results.
- New native Spark CART20/RFE20/RFE10 code is implemented and includes runnable integration checks. Its scores enter the report only after completing a Spark extension run; the current report marks those additional native results as pending.

The report includes AUC, sensitivity, specificity, accuracy, confusion matrices and further prototype metrics, with highest execution-table means bold. Five overlapping holdouts describe partition sensitivity rather than independent clinical replication. Test data never supplies the tuning objective.

## 6. Deployment, reproducibility and future work

The distribution mechanism is GitHub plus Google Colab notebooks. The report appendix links to the repository and current notebooks. Future work includes completing native feature-selection runs, broader training-only searches, calibration/threshold procedures, external validation and preservation of full runtime/tuning provenance.

### Primary Spark execution

Open the primary Colab notebook and run setup, integration checks and experiments in order. Setup checks Java and installs only missing Spark dependencies in Colab. Existing local environments are not reinstalled. An up-to-date repository checkout is used or cloned automatically. New native output is written to a configuration-specific `results/spark_iterations/` directory with data/code signatures; identical configurations resume completed models.

From a Spark-capable local checkout:

```bash
cd iteration4_BDAS
python run_workflow.py
# Or run selected preparation versions:
python run_workflow.py --variants rfe20_standard rfe10_standard
```

The main local notebook `iteration4_BDAS/run_workflow.ipynb` invokes the same native extension. `PySpark/run_spark_iterations_colab.ipynb` is the repository's dedicated Colab entrypoint. Java and dependencies in [PySpark/requirements-spark.txt](../PySpark/requirements-spark.txt) must be available for local Spark execution. The existing author environment has no local PySpark/Java; the extension has not been executed there.

### Retained Python prototype

Use an existing environment with [requirements.txt](requirements.txt):

```bash
cd iteration4_BDAS
python run_python_workflow.py --trials 20
python holdout_workflow.py --variants rfe10_standard --output-dir results/holdout_rfe10
```

The original local `.venv` link is machine-specific and is excluded from Git. Python default runs separate incompatible package/data configurations automatically. The completed RFE10 run used the existing environment without package installation.

### Rebuild report evidence

```bash
cd iteration4_BDAS
python scripts/build_holdout_report.py
python scripts/build_extension_report.py
# After completing new native Spark runs:
python scripts/build_extension_report.py --spark-results /path/to/completed/spark/run
cd Report
latexmk -pdf -interaction=nonstopmode main.tex
```

The Spark evidence importer requires a completed run. LaTeX compilation needs latexmk, biber and a TeX distribution. Both project-plan diagrams use two-day date labels and guide lines.

### Validation

```bash
cd iteration4_BDAS
python -m unittest discover -s tests -v
# In a Spark/Java runtime:
python -m unittest discover -s spark_tests -v
```

The normal checks cover holdout separation, exact split counts, matching resumes and configuration separation. New native integration checks cover RFE width, unchanged category codes, model persistence and training-CV/holdout separation. They are skipped locally without PySpark, and are included before full execution in the Colab notebook. All 30 new Python pipelines reproduced stored probabilities, with ten selected fields verified separately for each stacking base.

## Layout

```text
PySpark/
  run_spark_iterations_colab.ipynb  Primary native Spark extension
  run_spark_workflow.ipynb          Saved all-field Colab execution
  spark_visualization_colab.ipynb   Saved exploratory figures
  spark_iterations.py              Matched feature-selection iterations
  spark_feature_selection.py       Persistable native CART/RFE estimator
  spark_workflow.py                Native baseline and shared helpers
  tests/                          Spark-runtime integration checks
iteration4_BDAS/
  run_workflow.ipynb, run_workflow.py   Primary Spark entrypoints
  run_python_workflow.ipynb/.py        Retained Python entrypoints
  holdout_workflow.py, optuna_holdout.py
  data/raw/
  results/holdout/, holdout_rfe10/, spark_colab_reference/
  scripts/, tests/, spark_tests/
  Report/                         Sources, figures and main.pdf
```

Caches, virtual environments, archives and temporary LaTeX build files are excluded. Research Paper work remains separate.


The single local working copy is `/Users/quyduong106/projects/INFOSYS722_BDAS/iteration4_BDAS`. The redundant `INFOSYS722/iteration4_BDAS` copy has been removed; that old path is now a compatibility symlink to this folder so the active Colab notebook can continue autosaving without creating another physical copy. Native Spark modules remain in `../PySpark`; `.venv` links to the existing `../../INFOSYS722/Experimental/.venv` environment without reinstalling packages.
