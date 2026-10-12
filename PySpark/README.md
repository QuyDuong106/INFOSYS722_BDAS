# PySpark exploratory visualization

[Open the notebook in Google Colab](https://colab.research.google.com/github/QuyDuong106/INFOSYS722_BDAS/blob/main/PySpark/spark_visualization_colab.ipynb)

Run `spark_visualization_colab.ipynb` from top to bottom. The notebook fetches the raw dataset from this repository if it is not present locally. It checks Java and installs only missing Python packages when `INSTALL_MISSING=True`. It does not mount Drive or require a ZIP.

Spark computes dataset validation, grouped counts, missingness, equal-width histograms, grouped quartiles, categorical proportions, a continuous-predictor Pearson correlation matrix and two-dimensional BMI/glucose bins. Matplotlib renders the bounded summaries collected from Spark; the notebook does not use pandas or `toPandas()`.

Outputs include seven visualization figures (PNG/PDF), a data-quality JSON, a correlation CSV and a Spark CSV directory of binned summaries, when the data support each figure. They are saved to `pyspark_visualizations/` in the runtime working directory. Download individual files through the Colab sidebar before ending the session.

This is the CRISP-DM **Data Understanding** stage. Full-cohort exploratory figures are not model-selection or held-out evaluation evidence. Outcome codes, binning rules, approximate quartiles, minimum cell sizes and correlation complete-case counts are explained in the notebook. Native Spark modelling remains in `iteration4_BDAS/run_spark_workflow.ipynb`; the five-family Optuna report workflow remains in `iteration4_BDAS/run_workflow.ipynb`.

Local validation checked every Python cell and notebook schema. Spark execution was not performed locally because the existing environment has neither PySpark nor a working Java runtime; no local dependencies were installed.
