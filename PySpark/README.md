# Native PySpark workflows

**Primary new workflow:** [run_spark_iterations_colab.ipynb](run_spark_iterations_colab.ipynb), or [open in Colab](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/run_spark_iterations_colab.ipynb). It compares standardized all-field, CART20, RFE20 and RFE10 versions across three native Spark families and five holdouts, tuning only inside three training folds.

The `spark_feature_selection.py` estimator fits supervised CART/RFE ranking inside the pipeline and persists selected fields/scales. `spark_iterations.py` retains per-candidate CV evidence, split membership, fitted pipelines and predictions, and resumes compatible checkpoints. Runtime integration tests in `tests/` validate width, ordinal preservation, model persistence and CV separation before the notebook runs full experiments. New selected-field Spark results remain pending until the notebook completes; the report does not invent them.

**Saved completed evidence:** [run_spark_workflow.ipynb](run_spark_workflow.ipynb) retains the author's fifteen all-field Colab model/seed results. [spark_visualization_colab.ipynb](spark_visualization_colab.ipynb) retains the author's Spark-computed exploratory figures. Both saved notebooks and their outputs are preserved.

Raw data stays under `../iteration4_BDAS/data/raw`. The main report, saved Spark metrics with provenance, and retained Python prototype are under `../iteration4_BDAS`. Whole-cohort exploratory figures do not supply held-out model-selection evidence. Java and Spark are needed for native execution; setup installs only missing Python packages in Colab and does not reinstall the author's local environment.
