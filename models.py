import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def models(seed=42):
    return {
        'Mean baseline': DummyRegressor(strategy='mean'),
        'OLS': LinearRegression(),
        'OLS + Box-Cox': TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=PowerTransformer(method='box-cox', standardize=False)),
        'Huber': make_pipeline(StandardScaler(), HuberRegressor(max_iter=5000)),
        'Random Forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=seed, n_jobs=-1),
    }

def metrics(y, prediction):
    prediction = np.asarray(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions; model cannot be scored on a subset')
    return {'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
            'mae': float(mean_absolute_error(y, prediction)),
            'r2': float(r2_score(y, prediction)),
            'negative_predictions': int((prediction < 0).sum())}
