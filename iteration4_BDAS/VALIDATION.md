# Iteration 4 validation

Completed 5 seeds × 30 evaluations = 150 metric rows, 14,400 test predictions and 500 Optuna trials (20 × 5 predictive families × 5 seeds). Every split has 223 training/96 test rows. Every confusion matrix sums to 96; FNR equals 1 − recall. Saved model predictions were reproduced against stored probabilities. Tests verify tuning receives training-only rows, exact split sizes, identical resume, and rejection of incompatible trial settings. Stacking exposes 12 boosting parameters and fixes both logistic C values at 1. Notebook schema and active source compilation pass. The native Spark supplementary implementation was updated but not executed.

Existing `.venv` links to `../Experimental/.venv`; no packages were installed. Package versions and input signature are in `results/holdout/run_config.json`. Report evidence is generated from this completed run; `Report/main.pdf` was rebuilt with latexmk.
