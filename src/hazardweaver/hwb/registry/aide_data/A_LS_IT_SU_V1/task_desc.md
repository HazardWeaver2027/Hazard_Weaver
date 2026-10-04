# HWB tabular baseline task protocol (fixed split — do not reshuffle)

Use the CSV files in this directory exactly as provided:

- `train.csv` — training only
- `val.csv` — validation / model selection / hyperparameter tuning
- `test.csv` — held-out test; report final metric here only once

Target column: `target`
Features: slope_aver, tcurv_aver, twi_stddev, CLYPPT_ave, SNDPPT_ave, BLDFIE_ave
Evaluation metric: accuracy

Rules (binding for fair HWB comparison):
1. Do NOT merge splits or re-split the data.
2. Fit only on train.csv; tune on val.csv; final score on test.csv.
3. Print the validation metric during development and the test metric at the end.
4. Match metric definition: MAE = mean absolute error; accuracy = classification accuracy.
