# How to explain this project

## Problem
Compare prediction methods for country-level under-five mortality using health
coverage and income indicators, with transparent diagnostics and evaluation.

## Walkthrough
1. Read `data.py`: downloads and validates the five indicators. Explain why
   regional aggregates must be removed and why missingness changes the sample.
2. Read `models.py`: explain OLS, Huber's robust loss, Box–Cox target
   transformation, Random Forest averaging, and the mean baseline.
3. Read `pipeline.py`: trace the Bangladesh exclusion, train/test split,
   training-only cross-validation and final holdout evaluation.
4. Inspect `outputs/reference/report.md`: report the actual results instead of
   assuming a more complex method must win.
5. Run `predict.py`: show that a saved model can process a new predictor CSV.

## Questions to be ready for

- Why log GDP? Income is highly skewed; log GDP changes the assumed functional
  relation and reduces the influence of very large GDP values.
- Why scale Huber? Its optimization and regularization depend on predictor scale.
- Why not reject OLS when the outcome is skewed? Its assumptions concern the
  conditional error and conditional mean, not the marginal outcome distribution.
- Why fit Box–Cox within each fold? Its estimated transformation must not learn
  from validation outcomes.
- Why is dropping invalid predictions unfair? It scores a model on an easier
  or different subset and hides failures.
- Why no individual risk claims? The observations are economies, not children.
- Why no causal claims? Confounding, reverse causation, and contemporaneous
  measurement have not been resolved.
- Why no hyperparameter tuning? This first version compares prespecified
  methods. Any later tuning belongs inside training CV or nested CV.
- Why might results change? World Bank revises estimates; the saved snapshot
  and recorded dependencies pin this run more closely.

## Next research improvements

A future study could predefine regional holdouts, assess missing-data
sensitivity, add repeated nested CV for tuning, or extend to panel data with
appropriate country/time splits. These are future analyses, not claims about
what this version has already validated.
