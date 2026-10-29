"""
Project 2: Applying a thesis-style model-selection workflow to real child-health data.
Outcome: under-5 mortality rate (skewed, positive). Unit: country, year 2019 (World Bank open data).
Workflow: OLS + diagnostics -> Box-Cox -> robust regression -> Random Forest -> compare by 5-fold CV.
Country-level (ecological) analysis: associations only, not child-level or causal.
"""
import numpy as np, pandas as pd, requests
import scipy.stats as st
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

YEAR = 2019
CODES = {'u5_mort': 'SH.DYN.MORT',      # under-5 mortality per 1,000 live births
         'measles_imm': 'SH.IMM.MEAS',  # measles immunization, % of children aged 12-23 months
         'gdp_pc': 'NY.GDP.PCAP.CD',    # GDP per capita, current US$
         'water': 'SH.H2O.BASW.ZS',     # % using at least basic drinking water
         'sanit': 'SH.STA.BASS.ZS'}     # % using at least basic sanitation


def wb(code):
    url = f'https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000&date={YEAR}'
    d = requests.get(url, timeout=60).json()[1]
    return pd.Series({r['countryiso3code']: r['value'] for r in d
                      if r['countryiso3code'] and r['value'] is not None}, name=code)


# ---------------------------------------------------------------- data
cl = requests.get('https://api.worldbank.org/v2/country?format=json&per_page=400', timeout=60).json()[1]
countries = {c['id'] for c in cl if c['region']['id'] != 'NA'}          # drops regional aggregates
raw = pd.DataFrame({k: wb(v) for k, v in CODES.items()})
print('rows before dropna:', raw.shape, '\nmissing share:\n', raw.isna().mean().round(2))
df = raw[raw.index.isin(countries)].dropna()
print('countries analysed:', len(df))

X = df[['measles_imm', 'gdp_pc', 'water', 'sanit']].copy()
X['gdp_pc'] = np.log(X['gdp_pc'])                                       # skewed predictor
y = df['u5_mort']
print('\npredictor correlations:\n', X.corr().round(2))

# ---------------------------------------------------------------- step 1: OLS + diagnostics
ols = sm.OLS(y, sm.add_constant(X)).fit()
print(ols.summary().tables[1])
print('outcome skewness:', round(st.skew(y), 2),
      '| residual skewness:', round(st.skew(ols.resid), 2),
      '| Shapiro p:', round(st.shapiro(ols.resid).pvalue, 4),
      '| Breusch-Pagan p:', round(sm.stats.diagnostic.het_breuschpagan(ols.resid, ols.model.exog)[1], 4))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sm.qqplot(ols.resid, line='s', ax=ax[0]); ax[0].set_title('OLS residuals: Q-Q plot')
ax[1].scatter(ols.fittedvalues, ols.resid, s=12); ax[1].axhline(0, color='k')
ax[1].set_title('Residuals vs fitted'); ax[1].set_xlabel('fitted')
plt.savefig('ols_diagnostics.png', dpi=150, bbox_inches='tight')

# ---------------------------------------------------------------- steps 2-4: compare by 5-fold CV
models = {
    'OLS': LinearRegression(),
    'OLS + Box-Cox': TransformedTargetRegressor(LinearRegression(),
                                                transformer=PowerTransformer('box-cox', standardize=False)),
    'Robust (Huber)': make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)),
    'Random Forest': RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
}
cv = KFold(5, shuffle=True, random_state=0)
out = []
for name, m in models.items():
    p = cross_val_predict(m, X, y, cv=cv)
    ok = np.isfinite(p)                                                 # Box-Cox inverse can fail outside its range
    out.append((name, np.sqrt(np.mean((y[ok] - p[ok]) ** 2)), np.mean(np.abs(y[ok] - p[ok])), int((~ok).sum())))
cvres = pd.DataFrame(out, columns=['model', 'CV_RMSE', 'CV_MAE', 'n_nonfinite_preds']).round(2)
print('\n5-fold CV results (under-5 deaths per 1,000):\n', cvres.to_string(index=False))
cvres.to_csv('cv_results.csv', index=False)

# ---------------------------------------------------------------- Bangladesh: predicted vs actual
if 'BGD' in df.index:
    tr = df.index != 'BGD'
    print('\nBangladesh actual:', round(y['BGD'], 1))
    for name, m in models.items():
        print(f'  {name} prediction from other countries:', round(float(m.fit(X[tr], y[tr]).predict(X.loc[['BGD']])[0]), 1))
else:
    print('\nBangladesh dropped for missing data; skip the Bangladesh comparison.')
