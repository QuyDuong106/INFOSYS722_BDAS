# Native PySpark workflows

**Primary new workflow:** [run_spark_iterations_colab.ipynb](run_spark_iterations_colab.ipynb), or [open in Colab](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/run_spark_iterations_colab.ipynb). It compares standardized all-field, CART20, RFE20 and RFE10 versions across three native Spark families and five holdouts, tuning only inside three training folds.

The `spark_feature_selection.py` estimator fits supervised CART/RFE ranking inside the pipeline and persists selected fields/scales. `spark_iterations.py` retains per-candidate CV evidence, split membership, fitted pipelines and predictions, and resumes compatible checkpoints. Runtime integration tests in `tests/` validate width, ordinal preservation, model persistence and CV separation before the notebook runs full experiments. New selected-field Spark results remain pending until the notebook completes; the report does not invent them.

**Saved completed evidence:** [run_spark_workflow.ipynb](run_spark_workflow.ipynb) retains the author's fifteen all-field Colab model/seed results. [spark_visualization_colab.ipynb](spark_visualization_colab.ipynb) retains the author's Spark-computed exploratory figures. Both saved notebooks and their outputs are preserved.

Raw data stays under `../iteration4_BDAS/data/raw`. The main report, saved Spark metrics with provenance, and retained Python prototype are under `../iteration4_BDAS`. Whole-cohort exploratory figures do not supply held-out model-selection evidence. Java and Spark are needed for native execution; setup installs only missing Python packages in Colab and does not reinstall the author's local environment.


### Separate research RFE/LOPO notebook

Copy `run_spark_lopo_colab.ipynb`, `spark_lopo.py` and `spark_lopo_acceleration.py` to `/content/drive/MyDrive/INFOSYS_722_Notebook/iteration4_BDAS`, keeping the existing Spark modules and dataset. Select a Colab GPU runtime. Native Spark LR/RF/GBT and fold-safe RFE/scaling remain CPU-based; the additional `xgboost_gpu` candidate uses CUDA XGBoost with training-only Optuna (`GPU_TRIALS=20`). The GPU bridge collects only small prepared partitions to the driver (20,000-row limit) and is explicitly hybrid rather than a native Spark GPU classifier. CUDA verification rejects silent CPU fallback. No packages are installed locally.

Preparation is fitted once per inner training partition and reused across classifier/grid candidates. Fully checkpointed outer folds skip training actions, and GPU trials resume from Drive JSON. Trial execution is sequential on one GPU. Each complete seed run pools all held-out patients before calculating metrics; five runs report mean/sample SD of accuracy, precision and recall. `MAX_FOLDS=2` checks a prefix; `None` completes the run. Changed code/backend settings get separate signed output folders. The report's holdout workflow is unchanged. This small cohort may not benefit from GPU overhead; the implementation does not promise faster end-to-end execution.


### Spark holdout CV output

Use the updated `spark_workflow.py`, `spark_iterations.py` and `spark_cv_reporting.py` together. Both the original all-field `run_holdout(..., tune=True)` and the feature-selected `spark_iterations.run(..., tune=True)` print every candidate mean training-CV AUC, Spark's fold standard deviation when available, parameter values, and `[SELECTED]`. Per-model `cv_search.json` and `best_parameters.json` retain that evidence. Run-level `training_cv_results.csv` contains every grid candidate; `training_cv_summary.csv` reports mean and sample SD of the selected CV scores across seeds. These are training-only selection scores, separate from held-out test metrics. Existing historical test-only results require a new training run to obtain CV evidence; their CV scores are not reconstructed from test AUC.
