######################################
#### MODULES AND PATHS ####
######################################

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 
from scipy import stats
import statsmodels.api as sm 
from pathlib import Path
from math import pi
from math import log
import random
import pyblp
from linearmodels.iv import IV2SLS

MAIN = Path(__file__).resolve().parents[1]

DATA = MAIN / "data"
CODE = MAIN / "src"

for directory in (DATA, CODE):
    directory.mkdir(parents=True, exist_ok=True)

%cd '/Users/prateekmittal/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/IO HW/HW 1'

df = pd.read_excel("Problem Set Data PhD IO 2026.xlsx")

######################################
#### Question 1 ####
######################################

# renaming columns that have reserved names in pyblp
df = df.rename(columns={'Market_ID': 'market_ids', 'Plan_ID': 'product_ids', 'Share': 'shares', 'Premium': 'prices'})

###################
#### 1.a ####
###################

df_ols = df

# Outside Share
df_ols['TotalShare'] = df.groupby(["market_ids"])['shares'].transform("sum")
df_ols['Outside_Share'] = 1 - df_ols['TotalShare']

# Log of Shares
df_ols['log_os'] = np.log(df_ols['Outside_Share'])
df_ols['log_s'] = np.log(df_ols['shares'])

# Difference 
df_ols['ShareDiff'] = df_ols['log_s'] - df_ols['log_os']

# Naive OLS
y = df_ols['ShareDiff']
x = df_ols[['HMO', 'Network', 'Satisfaction', 'prices']]
X = sm.add_constant(x)

naive_ols = sm.OLS(y, X).fit()
print(naive_ols.summary())

# Let's now get the naive OLS estimate using PyBLP
# OLS treats prices as exogenous whereas PyBLP treats prices as endogenous. We will use the reserved name demand_instruments0 to instrument for prices with prices. This will let us match the OLS results above.  
df['demand_instruments0'] = df['prices']

#ols_problem = pyblp.Problem(pyblp.Formulation('1 + HMO + Network + Satisfaction + prices'), df)

#ols_results = ols_problem.solve(method='1s')

## Instruments for Price 

# BLP Instruments - Sum of characteristics of rival goods in the same market

for j in ['HMO', 'Satisfaction', 'Network']: 
        df[f'{j}_instrument'] = df.groupby(['market_ids'])[j].transform("sum") - df[j]

# Hausman
df['prod_id_len'] = df.groupby(['product_ids'])['product_ids'].transform("size") - 1
df['hausman_instrument'] = (df.groupby(['product_ids'])['prices'].transform("sum") - df['prices'])/ df['prod_id_len']



# 2 SLS

# Regress Price on instruments

y = df['prices']
x = df[['HMO_instrument', 'Satisfaction_instrument', 'Network_instrument']]
X = sm.add_constant(x)

first_stage = sm.OLS(y, X).fit()
print(first_stage.summary())
## HORRIBLE INSTRUMENTS

## Try Hausman

x = df[['HMO', 'Satisfaction', 'Network', 'hausman_instrument']]
X = sm.add_constant(x)
first_stage_hausman = sm.OLS(df['prices'], X).fit()
print(first_stage_hausman.summary())

df['fitted_vals'] = first_stage_hausman.fittedvalues

## Second Stage

Y = df['ShareDiff']
x = df[['HMO', 'Network', 'Satisfaction', 'fitted_vals']]
X = sm.add_constant(x)
Z = df['hausman_instrument']

sls_ols = sm.OLS(Y, X).fit()
print(sls_ols.summary())

iv2sls = IV2SLS.from_formula('ShareDiff ~ 1 + HMO + Satisfaction + Network + [prices ~ hausman_instrument]', data =df).fit(cov_type='robust')
print(iv2sls.summary)



# Nested Logit

df['nest_share'] = df.groupby(['market_ids', 'HMO'])['shares'].transform('sum')
df['cond_share'] = df['shares'] / df['nest_share']
df['lg_cond_share'] = np.log(df['cond_share'])


# Naive OLS of cond_shares

y = df_ols['ShareDiff']
x = df_ols[['HMO', 'Network', 'Satisfaction', 'prices', 'lg_cond_share']]
X = sm.add_constant(x)


naive_ols = sm.OLS(y, X).fit(cov_type = 'cluster', cov_kwds = {'groups': df['market_ids']})
print(naive_ols.summary())


# Instrument for conditional shares - Number of products per nest

df['n_prod_market'] = df.groupby(['market_ids', 'HMO'])['product_ids'].transform("count")

y = df['lg_cond_share']
x = df[['HMO', 'Network', 'Satisfaction', 'hausman_instrument', 'n_prod_market']]
X = sm.add_constant(x)

r = sm.OLS(y, X).fit()
print(r.summary())

iv2sls = IV2SLS.from_formula(
    'ShareDiff ~ 1 + HMO + Satisfaction + Network + [prices + lg_cond_share ~ hausman_instrument + n_prod_market]', 
    data=df
).fit(cov_type='robust')
print(iv2sls.summary)

