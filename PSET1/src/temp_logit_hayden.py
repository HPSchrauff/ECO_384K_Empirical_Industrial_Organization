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
from statsmodels.sandbox.regression.gmm import IV2SLS

MAIN = Path(__file__).resolve().parents[1]

DATA = MAIN / "data"
CODE = MAIN / "src"

for directory in (DATA, CODE):
    directory.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA / 'io_pset_1_data.xlsx')

######################################
#### Question 1 ####
######################################

# renaming columns that have reserved names in pyblp
df = df.rename(columns={'Market_ID': 'market_ids', 'Plan_ID': 'product_ids', 'Share': 'shares', 'Premium': 'prices'})

#################################
#### 1.a ####
#################################

###################
#### BLP-style instruments ####
###################

###################
#### Manual code ####
###################
df_ols = df.copy()

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
p = df_ols['prices']
x = df_ols[['HMO', 'Network', 'Satisfaction', 'prices']]
X = sm.add_constant(x)

# let's first naively run OLS - assuming prices are exogenous
naive_ols = sm.OLS(y, X).fit()
print(naive_ols.summary())

###################
#### BLP-style instruments ####
###################

# Of course, we believe that prices are correlated with unobservable product characteristics. So, we will need to instrument for price. We will construct BLP-style sum instruments. 

for i in x.columns:
    df_ols[i+"_instruments"] = df_ols.groupby(['market_ids'])[i].transform("sum") - df_ols[i]

blp_instruments = df_ols[['HMO', 'Network', 'Satisfaction', 'HMO_instruments', 'Network_instruments', 'Satisfaction_instruments']]

blp_instruments = sm.add_constant(blp_instruments)

ols_1_blp = sm.OLS(p, blp_instruments).fit()
print(ols_1_blp.summary()) # we didn't add HMO, Network, or Satisfaction originally. However, even after adding them the instruments are still bad. 

pred_p_blp = pd.DataFrame(ols_1_blp.predict())

x_tsls_blp = pd.concat([df_ols[['HMO', 'Network', 'Satisfaction']], pred_p_blp], axis=1)

x_tsls_blp = sm.add_constant(x_tsls_blp)

x_tsls_blp = x_tsls_blp.rename(columns = {0: 'instrumented_price'})

ols_2_blp = sm.OLS(y, x_tsls_blp).fit()
print(ols_2_blp.summary())

# The standard errors are incorrect, so we will use the built-in package to compute them. 
tsls_model_blp = IV2SLS(y, X, blp_instruments).fit()
print(tsls_model_blp.summary()) # we get a positive price coefficient - not good

###################
#### Hausman-style instruments ####
###################

# let's try Hausman-style instruments
df_ols['prod_leng'] = df_ols.groupby(['product_ids'])['product_ids'].transform("size") - 1
hausman_instruments = (df_ols.groupby(['product_ids'])['prices'].transform("sum") - df_ols['prices']) / df_ols['prod_leng']

hausman_instruments_x = pd.concat([hausman_instruments, df_ols[['HMO', 'Network', 'Satisfaction']]], axis=1)

hausman_instruments_x = sm.add_constant(hausman_instruments_x)

ols_1_hausman = sm.OLS(p, hausman_instruments_x).fit()
print(ols_1_hausman.summary()) # this does seem to have a first-stage effect

# get the predicted prices
pred_p_hausman = pd.DataFrame(ols_1_hausman.predict())

x_tsls_hausman = pd.concat([df_ols[['HMO', 'Network', 'Satisfaction']], pred_p_hausman], axis=1)

x_tsls_hausman = sm.add_constant(x_tsls_hausman)

x_tsls_hausman = x_tsls_hausman.rename(columns = {0: 'instrumented_price'})

ols_2_hausman = sm.OLS(y, x_tsls_hausman).fit()
print(ols_2_hausman.summary())

# The standard errors are incorrect, so we will use the built-in package to compute them. 
tsls_model_hausman = IV2SLS(y, X, hausman_instruments_x).fit()
print(tsls_model_hausman.summary())

###################
#### PyBLP ####
###################

# Let's now get the naive OLS estimate using PyBLP
# OLS treats prices as exogenous whereas PyBLP treats prices as endogenous. We will use the reserved name demand_instruments0 to instrument for prices with prices. This will let us match the OLS results above.  
df['demand_instruments0'] = df['prices']

ols_problem = pyblp.Problem(pyblp.Formulation('1 + HMO + Network + Satisfaction + prices'), df)

ols_results = ols_problem.solve(method='1s')

# Now, let's build BLP-style instruments. Since there is no firm ID, we will follow PyBLP's documentation suggestion and add a constant firm ID column and retain the first half of the instrument columns

df_pyblp = df.copy()
df_pyblp = df_pyblp.drop(["demand_instruments0"], axis=1)
df_pyblp['firm_ids'] = 1
demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('1 + HMO + Network + Satisfaction'), df_pyblp)
demand_instruments_fixed = demand_instruments[:, :3]

for i, column in enumerate(demand_instruments_fixed.T):
    df_pyblp[f'demand_instruments{i}'] = column

pyblp_logit = pyblp.Problem(pyblp.Formulation('1 + HMO + Network + Satisfaction + prices'), df_pyblp)

pyblp_logit_results = pyblp_logit.solve()
pyblp_logit_results

#################################
#### 1.b ####
#################################
# Now, we wil perform nested logit. To do so, we need to create the within-nest share of product j and take the log of that. We already have the share of product j within a market. So, all we need now is the share of each HMO group within a market

df_ols['hmo_share'] = df_ols.groupby(["market_ids", "HMO"])['shares'].transform("sum")
df_ols['within_hmo_share'] = df_ols['shares'] / df_ols['hmo_share']
df_ols['log_within_hmo_share'] = np.log(df_ols['within_hmo_share'])

# let's do naive OLS again
x = df_ols[['HMO', 'Network', 'Satisfaction', 'prices', 'log_within_hmo_share']]
X = sm.add_constant(x)

# let's first naively run OLS - assuming prices are exogenous
naive_ols = sm.OLS(y, X).fit()
print(naive_ols.summary())

# we need to create an additional instrument for this. PyBLP's documentation suggests the number of products within a nest
df_ols['hmo_instrument'] = df_ols['hmo_share'] = df_ols.groupby(["market_ids", "HMO"])['product_ids'].transform("nunique")

hausman_group_instruments = pd.concat([hausman_instruments, df_ols[['HMO', 'Network', 'Satisfaction', 'hmo_instrument']]], axis=1)

hausman_group_instruments = hausman_group_instruments.rename(columns = {0: 'hausman_instruments'})

hausman_group_instruments = sm.add_constant(hausman_group_instruments)

# let's check whether our instruments pass the first stage
ols_1_hausman_group_share = sm.OLS(df_ols["log_within_hmo_share"], hausman_group_instruments).fit()
print(ols_1_hausman_group_share.summary()) 
pred_share_hausman_group = pd.DataFrame(ols_1_hausman_group_share.predict()) #


ols_1_hausman_group_price = sm.OLS(df_ols["prices"], hausman_group_instruments).fit()
print(ols_1_hausman_group_price.summary()) 
pred_share_hausman_price = pd.DataFrame(ols_1_hausman_group_price.predict())


# both pass first stage test. let's now use the built-in package
tsls_model_hausman = IV2SLS(y, X, hausman_group_instruments).fit()
print(tsls_model_hausman.summary())
