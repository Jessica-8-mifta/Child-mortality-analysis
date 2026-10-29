# Child mortality: 2019 cross-country analysis

181 complete economies of 217 eligible; 144 training, 36 holdout.
Bangladesh, when available, is an additional excluded case.

| Model | CV RMSE | CV MAE | Status |
|---|---:|---:|---|
| Mean baseline | 34.370 | 24.532 | ok |
| OLS | 19.987 | 10.268 | ok |
| OLS + Box-Cox | 19.478 | 9.521 | ok |
| Huber | 18.813 | 9.083 | ok |
| Random Forest | 20.285 | 9.700 | ok |

Selected by training CV RMSE: **Huber**. Locked holdout RMSE: **13.022**;
MAE: **8.587**; R²: **0.691**.
RMSE and MAE are in deaths per 1,000 live births.

Bangladesh case: {"actual": 32.3, "prediction": 31.33009615182528, "training": "training partition only; Bangladesh and holdout excluded"}.

![Model comparison](model_comparison.png)
![OLS diagnostics](ols_diagnostics.png)

## Interpretation limits
Country/economy associations cannot establish individual risk or causal effects.
Random country splits do not measure future-year performance or regional transfer.
Complete-case selection can bias the sample. Economies receive equal weight;
this is not a birth-weighted global estimate. Predictors are contemporaneous.
Normality concerns residuals, not the marginal outcome; non-normal errors alone
do not bias OLS under exogeneity. HC3 errors address heteroskedasticity under
appropriate assumptions, not confounding or dependence between countries.
Box-Cox inverse predictions may be undefined; any failed fold disqualifies the
model, without removing countries from scores. Finite negative predictions are
counted and scored without clipping. Transformed-target inverse predictions do
not automatically correct retransformation bias. No hyperparameter search was
performed. CV-selected performance is optimistic; the single untouched holdout
is small and uncertain. Test results must not guide subsequent model selection.

## Reproducibility
See summary.json for versions, seed, sample sizes and input SHA-256;
splits.csv for partitions, fold_results.csv for warnings, and data_audit.csv
for exclusions. Model artifact is fitted to the training partition only.
