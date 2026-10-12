# INFOSYS722 BDAS — Iteration 4

A reproducible gallstone-prediction project organised using **CRISP-DM (Cross-Industry Standard Process for Data Mining)**. The Iteration 4 development and report-writing period is **21 September–16 October 2026**.

## Start here

| Workflow | Notebook | Models and preparation |
| --- | --- | --- |
| Main report experiments | [run_workflow.ipynb](run_workflow.ipynb) | Logistic regression, random forest, XGBoost, LightGBM and stacking; CART/RFE20; four preprocessing benchmarks and training-only Optuna tuning |
| Native PySpark in Colab | [run_spark_workflow.ipynb](run_spark_workflow.ipynb) | Spark logistic regression, random forest and GBT; all source predictors, continuous standardization and training-only Spark CV |

[Open the native Spark notebook in Colab](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/iteration4_BDAS/run_spark_workflow.ipynb) · [Open the main notebook in Colab](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/iteration4_BDAS/run_workflow.ipynb) · [Read the report](Report/main.pdf)

The two workflows are separate implementations. The committed report metrics come from the main five-family Python workflow. Native Spark training has not been executed for the newly created standalone notebook; its cells and notebook structure have been validated.

## 1. Business understanding

Investigate whether routinely available demographic, clinical, laboratory and body-composition measurements can support early gallstone-risk assessment. This is an academic proof of concept for prioritising further clinical assessment. It is not a clinical deployment or a replacement for diagnostic imaging.

The project benchmarks are ROC–AUC **at least 0.80** and gallstone sensitivity **at least 0.80**, alongside specificity, calibration-related loss, false negatives, stability and interpretability. These are project targets rather than clinical thresholds.

## 2. Data understanding

The source cohort contains **319 records: 161 gallstone cases and 158 controls**, with 38 source predictors. Raw files are retained in [data/raw](data/raw). Dataset provenance and exploratory analysis are documented in the report and its bibliography.

The source outcome uses `0 = gallstones present` and `1 = absent`; modelling recodes it to `1 = present` and `0 = absent`. Loading validates numeric predictors, binary outcomes and duplicate records, and excludes the exported row index and patient identifier from predictors. Original ordinal/binary category codes are preserved.

## 3. Data preparation

The main workflow compares four controlled preparation variants:

| Variant | Feature selection | Continuous transformations |
| --- | --- | --- |
| `cart_yeo_standard` | CART importance, 20 source fields | Yeo–Johnson then standardization |
| `cart_standard` | CART importance, 20 source fields | Standardization |
| `rfe_yeo_standard` | CART-based RFE, 20 source fields | Yeo–Johnson then standardization |
| `rfe_standard` | CART-based RFE, 20 source fields | Standardization |

Continuous missing values use training medians; coded categories use training modes. Category codes are passed through without one-hot expansion. Imputation, selection and transformation are fitted again inside every training CV fold. Stacking also uses complete base-learner preparation pipelines inside its own out-of-fold construction.

The supplementary native Spark implementation uses all source predictors with median/mode imputation and continuous standardization; it does not implement these RFE20 comparisons.

## 4. Modelling

All experiments use **five stratified 70/30 holdouts**, with seeds **42, 7, 21, 84 and 123**. Each split contains **223 training and 96 test records**. Iteration 4 uses holdout evaluation only; leave-one-out evaluation is outside this submission.

For the main workflow, all five predictive families and a training-prior dummy receive fixed-parameter benchmarks in each preparation variant. The predeclared `rfe_standard` variant additionally receives **20 Optuna TPE trials per predictive family per seed**, using mean ROC–AUC from **three-fold CV within training only**. No held-out test record enters tuning, preprocessing or model selection.

The searches cover logistic regularisation; forest tree count, depth, leaf size and feature sampling; and boosting tree count, learning rate, sampling/leaf controls and L2 regularisation. Stacking jointly searches twelve layer-1 XGBoost/LightGBM parameters. Both stacking logistic components remain fixed at `C=1`, as does its forest. Standalone logistic regression is tuned separately. Detailed ranges appear in the report and [optuna_holdout.py](optuna_holdout.py).

The native Spark notebook uses Spark CrossValidator with three training-only folds for its logistic regression, forest and GBT parameter grids. It does not use Optuna, XGBoost, LightGBM or stacking.

## 5. Evaluation

Evaluation fixes the classification threshold at **0.5**, with disease-positive class 1. Main outputs include ROC–AUC, average precision, accuracy, balanced accuracy, precision, sensitivity, specificity, F1, false-negative rate, log loss, confusion counts and execution time.

[results/holdout](results/holdout) contains the completed main run:

- **150 model evaluations**, **14,400 test predictions** and **500 Optuna trials**.
- Per-seed split membership and per-model fitted pipelines, selected features, predictions, trial evidence and parameters.
- Aggregate [benchmark metrics](results/holdout/benchmark_metrics.csv), [tuned metrics](results/holdout/tuned_metrics.csv), [mean/SD/min/max summaries](results/holdout/summary.csv) and [matched tuning changes](results/holdout/tuning_comparison.csv).
- Training-CV-selected families and their corresponding test outcomes, without selecting families from test results.

The report separates training-CV algorithm-selection evidence, held-out results and controlled iteration comparisons. Highest column means are bold in its execution table. Averaged confusion matrices are generated from the five matched test splits. Tuning does not improve every family: measured changes and sensitivity/AUC trade-offs are retained.

Five holdout test sets overlap; their variation measures sensitivity to partition choice rather than independent external validation. The small single-source cohort, finite search budget and fixed threshold limit generalisation.

## 6. Deployment and reproducibility

Deployment in this iteration means packaging reproducible research code, notebooks, evidence and the report. No prediction service or clinical system is deployed. External validation, calibration and clinically justified threshold selection remain future work.

### Run locally

Clone the repository and enter the project:

```bash
git clone https://github.com/QuyDuong106/INFOSYS722_BDAS.git
cd INFOSYS722_BDAS/iteration4_BDAS
python run_workflow.py --trials 20
```

Use a Python interpreter with the dependencies in [requirements.txt](requirements.txt). An existing suitable environment can be reused; no installation is performed automatically. The author's local `.venv` link is machine-specific and is excluded from Git. Native Spark additionally requires the dependencies in [requirements-spark.txt](requirements-spark.txt) and Java.

The committed configuration records exact package versions and an input-data signature. If the default saved run differs from the current configuration, the main workflow automatically chooses a stable directory under `results/holdout_runs/<configuration-id>/`. Identical configurations resume completed evaluations. An explicitly supplied incompatible directory is rejected with the differing fields listed. This prevents mixing local and Colab checkpoints.

```python
from holdout_workflow import run
metrics = run(trials=20)
print(metrics.attrs['output_dir'])
```

Generate figures and tables for the corresponding results directory, then rebuild the report:

```bash
python scripts/build_holdout_report.py --results-dir results/holdout
cd Report
latexmk -pdf -interaction=nonstopmode main.tex
```

For an automatically separated run, replace `results/holdout` with the printed results directory. LaTeX compilation requires a TeX distribution, latexmk and biber.

### Run in Google Colab

Open the relevant notebook using the links above. The native Spark notebook embeds its implementation: upload [dataset_gallstone](data/raw/dataset_gallstone) when prompted and run the cells in order. PySpark, scikit-learn and Java must already be available. Results are saved under the printed `results/spark_holdout/<configuration-id>/` folder; re-running its training cell retrains the native Spark models.

For the main notebook, clone this repository in a Colab cell:

```python
!git clone https://github.com/QuyDuong106/INFOSYS722_BDAS.git /content/INFOSYS722_BDAS
```

Set `PROJECT_ROOT = Path('/content/INFOSYS722_BDAS/iteration4_BDAS')` in the notebook's setup cell. Use a runtime with the required packages, run the cells in order and save results before the session ends. Neither notebook mounts Google Drive or installs packages automatically.

### Validation

```bash
python -m unittest discover -s tests -v
```

Checks cover training/test separation, exact split sizes, five seeds, matching-config resume, automatic environment separation and rejection of explicit mismatched folders. All 150 committed pipelines reproduced their stored probabilities in the author's existing environment. See [VALIDATION.md](VALIDATION.md).

Only load the committed model pickles as trusted project artifacts, using the recorded compatible environment.

## Repository layout

```text
iteration4_BDAS/
  run_workflow.ipynb          Main five-family notebook
  run_spark_workflow.ipynb    Standalone native Spark notebook
  run_workflow.py             Command-line entrypoint
  holdout_workflow.py         Holdout benchmarks, resume and aggregation
  optuna_holdout.py           Training-only tuning
  modeling.py, utils.py       Estimators and preparation helpers
  spark_workflow.py           Native Spark implementation
  data/raw/                  Source dataset
  results/holdout/            Completed main evidence and pipelines
  scripts/                   Report table/figure generation
  tests/                     Workflow checks
  Report/                    LaTeX source, figures and main.pdf
```

Virtual environments, caches, generated ZIPs, retired iterations and temporary LaTeX build files are excluded. This repository contains Iteration 4; Research Paper work remains separate.
