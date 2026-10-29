"""Run: python -m child_mortality --help."""
import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from .data import download, validate, features, CODES
from .models import models, metrics

def diagnostics(X, y, out):
    design = sm.add_constant(X, has_constant='add')
    fit = sm.OLS(y, design).fit()
    (out / 'ols_summary_hc3.txt').write_text(fit.get_robustcov_results(cov_type='HC3').summary().as_text())
    values = {'outcome_skewness': float(stats.skew(y)),
              'residual_skewness': float(stats.skew(fit.resid)),
              'shapiro_p': float(stats.shapiro(fit.resid).pvalue),
              'breusch_pagan_p': float(het_breuschpagan(fit.resid, design)[1])}
    (out / 'diagnostics.json').write_text(json.dumps(values, indent=2))
    pd.DataFrame({'term': design.columns, 'vif': [variance_inflation_factor(design.values, i)
                  for i in range(design.shape[1])]}).to_csv(out / 'vif.csv', index=False)
    influence = fit.get_influence()
    pd.DataFrame({'iso3': X.index, 'cooks_distance': influence.cooks_distance[0],
                  'leverage': influence.hat_matrix_diag}).to_csv(out / 'influence.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sm.qqplot(fit.resid, line='s', ax=axes[0])
    axes[0].set_title('Training OLS residuals')
    axes[1].scatter(fit.fittedvalues, fit.resid, s=14, alpha=.7)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set(xlabel='Fitted mortality rate', ylabel='Residual', title='Residuals vs fitted')
    fig.tight_layout(); fig.savefig(out / 'ols_diagnostics.png', dpi=160); plt.close(fig)

def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty; use a new directory to avoid mixing runs')
    source = Path(args.data) if args.data else download(args.year, args.cache)
    df, audit, missing = validate(pd.read_csv(source), args.year)
    audit.to_csv(out / 'data_audit.csv', index=False)
    missing.to_csv(out / 'missingness.csv')
    df.to_csv(out / 'analysis_data.csv')
    X, y = features(df), df['u5_mort']
    # Bangladesh stays outside selection and all training/validation folds.
    candidates = df.index[df.index != 'BGD']
    train_ids, test_ids = train_test_split(candidates, test_size=.2, random_state=args.seed)
    split = pd.Series('train', index=df.index, name='split')
    split.loc[test_ids] = 'test'
    if 'BGD' in split.index:
        split.loc['BGD'] = 'bangladesh_case'
    split.to_csv(out / 'splits.csv')
    diagnostics(X.loc[train_ids], y.loc[train_ids], out)
    X.loc[train_ids].corr().to_csv(out / 'predictor_correlations.csv')
    folds = list(KFold(5, shuffle=True, random_state=args.seed).split(train_ids))
    records, fold_records, oof_records = [], [], []
    for name, template in models(args.seed).items():
        prediction = np.full(len(train_ids), np.nan)
        errors = []
        for fold, (tr, va) in enumerate(folds, 1):
            estimator = clone(template)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    estimator.fit(X.loc[train_ids[tr]], y.loc[train_ids[tr]])
                    p = estimator.predict(X.loc[train_ids[va]])
                score = metrics(y.loc[train_ids[va]], p)
                prediction[va] = p
                fold_records.append({'model': name, 'fold': fold, 'status': 'ok', **score,
                                     'warnings': '; '.join(str(w.message) for w in caught)})
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                errors.append(f'fold {fold}: {exc}')
                fold_records.append({'model': name, 'fold': fold, 'status': 'failed', 'error': str(exc)})
        if errors:
            records.append({'model': name, 'status': 'failed', 'error': '; '.join(errors)})
        else:
            records.append({'model': name, 'status': 'ok', **metrics(y.loc[train_ids], prediction)})
        oof_records.extend({'iso3': iso, 'model': name, 'actual': float(y.loc[iso]),
                            'prediction': float(p)} for iso, p in zip(train_ids, prediction))
    cv = pd.DataFrame(records)
    cv.to_csv(out / 'cv_results.csv', index=False)
    pd.DataFrame(fold_records).to_csv(out / 'fold_results.csv', index=False)
    pd.DataFrame(oof_records).to_csv(out / 'oof_predictions.csv', index=False)
    eligible = cv.loc[cv['status'].eq('ok')].sort_values('rmse')
    if eligible.empty:
        raise RuntimeError('No model completed all folds')
    winner = eligible.iloc[0]['model']
    fitted = models(args.seed)[winner].fit(X.loc[train_ids], y.loc[train_ids])
    test_p = fitted.predict(X.loc[test_ids])
    test_metrics = metrics(y.loc[test_ids], test_p)
    pd.DataFrame({'iso3': test_ids, 'actual': y.loc[test_ids].values,
                  'prediction': test_p}).to_csv(out / 'test_predictions.csv', index=False)
    case = None
    if 'BGD' in df.index:
        p = float(fitted.predict(X.loc[['BGD']])[0])
        case = {'actual': float(y.loc['BGD']), 'prediction': p if np.isfinite(p) else None,
                'training': 'training partition only; Bangladesh and holdout excluded'}
    joblib.dump({'model': fitted, 'features': list(X.columns), 'log_gdp': True,
                 'year': args.year, 'seed': args.seed, 'model_name': winner}, out / 'model.joblib')
    summary = {'year': args.year, 'seed': args.seed, 'n_eligible': len(audit), 'n_complete': len(df),
               'n_train': len(train_ids), 'n_test': len(test_ids), 'selected_model': winner,
               'holdout': test_metrics, 'bangladesh': case,
               'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'python': platform.python_version(),
               'versions': {p: importlib.metadata.version(p) for p in
                            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'matplotlib', 'joblib']}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(eligible['model'], eligible['rmse'], color='#167d9a')
    ax.set(xlabel='Training 5-fold out-of-fold RMSE (deaths per 1,000 live births)',
           title='Model comparison — lower is better')
    fig.tight_layout(); fig.savefig(out / 'model_comparison.png', dpi=160); plt.close(fig)
    rows = '\n'.join(f"| {r['model']} | {r.get('rmse', float('nan')):.3f} | {r.get('mae', float('nan')):.3f} | {r['status']} |" for r in records)
    report = f'''# Child mortality: {args.year} cross-country analysis

{len(df)} complete economies of {len(audit)} eligible; {len(train_ids)} training, {len(test_ids)} holdout.
Bangladesh, when available, is an additional excluded case.

| Model | CV RMSE | CV MAE | Status |
|---|---:|---:|---|
{rows}

Selected by training CV RMSE: **{winner}**. Locked holdout RMSE: **{test_metrics['rmse']:.3f}**;
MAE: **{test_metrics['mae']:.3f}**; R²: **{test_metrics['r2']:.3f}**.
RMSE and MAE are in deaths per 1,000 live births.

Bangladesh case: {json.dumps(case)}.

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
'''
    (out / 'report.md').write_text(report)
    print(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, default=2019)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', help='Previously downloaded economy-level CSV; no network needed')
    parser.add_argument('--cache', default='data/raw')
    parser.add_argument('--output', default='outputs/run')
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
