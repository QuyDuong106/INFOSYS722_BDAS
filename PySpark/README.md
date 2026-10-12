# PySpark: saved modelling and visualization notebooks

This is the canonical folder for the latest native Spark notebooks and supporting code on `main`. Both notebooks retain the author's saved Colab results. The old duplicate Spark notebook in Iteration 4 and the root-level model notebook have been replaced by this layout; older versions remain in Git history.

| Notebook | Purpose | Open in Colab |
| --- | --- | --- |
| [run_spark_workflow.ipynb](run_spark_workflow.ipynb) | Saved five-seed Spark holdout run: logistic regression, random forest and GBT; three-fold training-only CV | [Run modelling](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/run_spark_workflow.ipynb) |
| [spark_visualization_colab.ipynb](spark_visualization_colab.ipynb) | Saved Spark exploratory summaries and seven visualization figures | [Run visualization](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/spark_visualization_colab.ipynb) |

## Modelling

Run the cells in order. The notebook uses `spark_workflow.py` from a current repository checkout, cloning the repository into Colab if needed. It installs only missing PySpark/scikit-learn/pandas packages and checks Java. Google Drive is optional. Data remains at `../iteration4_BDAS/data/raw/dataset_gallstone`. Fresh models, predictions and metrics are written to `PySpark/results/spark_holdout/` in the checkout; the retained table represents the earlier Colab run until you rerun the execution and display cells.

The Spark models use all source predictors, training-only imputation, continuous standardization and unchanged ordinal codes. They do not implement the five-family Optuna/RFE20 experiments, which remain in [Iteration 4's main notebook](../iteration4_BDAS/run_workflow.ipynb). The saved table uses pandas only to render the already-computed Spark metrics.

## Visualization

Run `spark_visualization_colab.ipynb` from top to bottom. It fetches the raw dataset if absent and installs only missing Python packages when enabled. Spark computes counts, missingness, binned histograms, outcome-group quartiles, categorical proportions, Pearson correlations and BMI/glucose bins. Matplotlib renders the collected summaries; patient data is not converted to pandas.

The saved figures remain embedded in the notebook. Fresh PNG/PDF figures and summaries are written to `pyspark_visualizations/` in the runtime working directory. Download individual files before ending the Colab session.

This supports CRISP-DM data understanding. Whole-cohort exploration is separate from training-only model selection and held-out evaluation. Local checks cover notebook structure, code syntax, paths and preservation of saved model/plot outputs. No local dependencies were installed or Spark experiments rerun during this reorganization.
