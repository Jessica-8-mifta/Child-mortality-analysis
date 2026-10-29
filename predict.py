"""Batch prediction using a trusted, locally generated model artifact."""
import argparse
import joblib
import numpy as np
import pandas as pd
from child_mortality.data import features, FEATURES

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True, help='CSV with raw GDP and percentage predictors')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    raw = data[FEATURES].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(raw).all().all() or (raw['gdp_pc'] <= 0).any():
        raise ValueError('Predictors must be finite; GDP must be positive')
    if not raw[['measles_imm', 'water', 'sanit']].apply(lambda s: s.between(0, 100)).all().all():
        raise ValueError('Percentage predictors must lie in [0, 100]')
    bundle = joblib.load(args.model)  # Load only artifacts you trust.
    prediction = bundle['model'].predict(features(raw))
    if not np.isfinite(prediction).all():
        raise ValueError('Model generated nonfinite predictions')
    data['predicted_u5_mort'] = prediction
    data['negative_prediction'] = prediction < 0
    data.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
