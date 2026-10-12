# Validation — 13 October 2026

The original 150 Python evaluations and 500 Optuna trials remain separate from native Spark evidence. The new RFE10 fixed-model experiment completed 30 evaluations across five seeds; every non-dummy selector and stacking base retains exactly ten fields, and all 30 persisted pipelines reproduced stored test probabilities. The original workflow/configuration tests pass.

Fifteen native Spark baseline model/seed metrics were extracted from the author's saved Colab display with source-commit provenance. Counts sum to 96 for every evaluation; values have saved-display precision of six decimals. This is measured author-run evidence, not a new local execution.

The new native Spark CART20/RFE20/RFE10 pipeline and primary notebook pass syntax/schema checks. Runtime tests cover selected width, ordinal preservation, model persistence and training-CV/holdout separation; they were skipped locally because the existing environment has no PySpark/Java. The Colab notebook runs those tests before full fitting. New native feature-selection performance remains explicitly pending until completed output is supplied.

Report main.pdf rebuilt after adding future work, GitHub appendix, native baseline/feature-count evidence and two-day project-plan guides. No local requirements were reinstalled.
