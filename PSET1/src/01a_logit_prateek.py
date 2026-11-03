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
import time
from linearmodels.iv import IV2SLS

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

## Coefficients
coefficients_logit = iv2sls.params
price_logit = coefficients_logit['prices']
HMO_logit = coefficients_logit['HMO']
Satisfaction_logit = coefficients_logit['Satisfaction']
Network_logit = coefficients_logit['Network']


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

# Coefficients
coefficients_nested_logit = iv2sls.params
price_nested_logit = coefficients_nested_logit['prices']
HMO_nested_logit = coefficients_nested_logit['HMO']
Satisfaction_nested_logit = coefficients_nested_logit['Satisfaction']
Network_nested_logit = coefficients_nested_logit['Network']
sigma_nested_logit = coefficients_nested_logit['lg_cond_share']


## BLP

np.random.seed(5)

# market_ids = df_ols['market_ids'].unique() # Each unique market id
# markets, draws, char = len(market_ids), 500, 3
# nu_draws = np.random.randn(markets, draws, char) # 75 markets, 500 draws, 3 characteristics

# def gmm(sigma): 
# ## Program to get Simulated Shares

#     def sim_shares(sigma, delta):
#         s_hat = []
#         start = 0
#         # idx goes from 0-74 and t goes from 1-75 (Actual market ids) 
#         for idx, i in enumerate(market_ids):
#             # For market i, get the product characterstics
#             x = df_ols.loc[df_ols['market_ids'] == i, ['HMO', 'Satisfaction', 'Network']].to_numpy()
#             J = x.shape[0] # No of products in each market

#             # Mu for each consumer across each product in market i
#             mu = x @ (sigma[:, None] * nu_draws[idx].T)

#             # Delta
#             d = delta[start: start + J] # Uses number of products in each market to get the initial value of delta
#             start += J # Start at next market in next loop

#             # Simulates shares
#             util = d[:, None] + mu
#             num = np.exp(util)
#             denom = 1 + num.sum(axis = 0, keepdims = True)

#             prob = num/denom
#             s_hat.append(prob.mean(axis = 1))

#         return np.concatenate(s_hat)

#     # delta0 = df_ols['ShareDiff'].to_numpy()
#     # sigma = np.array([3, 3, 3])
#     # sim_share = sim_shares(sigma, delta0)

#     ## Contraction Mapping

#     def contraction(sigma, delta, tol = 1e-12, maxit = 1000):
#         # S_obs = df_ols['shares'].to_numpy()
#         for i in range(maxit):
#             delta_new = delta + np.log(S_obs) - np.log(sim_shares(sigma, delta))
#             if np.max(np.abs(delta_new - delta)) < tol:
#                     return delta_new
#             delta = delta_new
#         print("Warning: Did not converge")
#         return delta

#     # delt = contraction(sigma, delta0)


#     ## Now use IV to get \psi as residual

#     # ## First Stage

#     # y = df_ols['prices']
#     # x = df_ols[['hausman_instrument', 'HMO_instrument', 'Satisfaction_instrument', 'Network_instrument',
#     #         'HMO', 'Satisfaction', 'Network']]

#     # X = sm.add_constant(x)

#     # frt_stage = sm.OLS(y, X).fit(cov_type = 'HC1')
#     # print(frt_stage.summary())


#     # exog = ['HMO', 'Satisfaction', 'Network']
#     # excl = ['hausman_instrument', 'HMO_instrument', 'Satisfaction_instrument', 'Network_instrument']
#     # d = sm.OLS(df_ols['ShareDiff'], sm.add_constant(df_ols[exog + excl[1:]])).fit(cov_type='HC1')
#     # print(d.summary())

#     # ## Bad Instruments 

#     ## Create Gandhi Houde Instruments 

#     def gh_instruments(var):
#         quad = []

#         for t in df_ols['market_ids'].unique():
#             x = df_ols[df_ols['market_ids'] == t][var].to_numpy()
#             for j in range(len(x)):
#                 total = 0
#                 for k in range(len(x)):
#                     if k != j:
#                             total = total + (x[k] - x[j])**2
#                 quad.append(total)       
#         return quad

#     df_ols['Network_quad'] = gh_instruments('Network')
#     df_ols['Satisfaction_quad'] = gh_instruments('Satisfaction')      


#     # y = df['prices']
#     # x = df[['hausman_instrument', 'HMO_instrument', 'Satisfaction_quad', 'Network_quad',
#     #         'HMO', 'Satisfaction', 'Network']]

#     # X = sm.add_constant(x)

#     # frt_stage = sm.OLS(y, X).fit(cov_type = 'HC1')
#     # print(frt_stage.summary())


#     # exog = ['HMO', 'Satisfaction', 'Network']
#     # excl = ['hausman_instrument', 'HMO_instrument', 'Satisfaction_quad', 'Network_quad']
#     # d = sm.OLS(df_ols['ShareDiff'], sm.add_constant(df_ols[exog + excl[1:]])).fit(cov_type='HC1')
#     # print(d.summary())

#     # Delta = alpha * price + beta * observed_characterstics + \psi

#     df_ols['delta_calc'] = contraction(sigma, delta0)

#     iv2sls = IV2SLS.from_formula('delta_calc ~ 1 + HMO + Satisfaction + Network + [prices ~ hausman_instrument + ' \
#     'HMO_instrument + Satisfaction_quad + Network_quad]', data =df_ols).fit(cov_type='robust')
#     # print(iv2sls.summary)

#     ## Ksi term
#     xi = iv2sls.resids.to_numpy()

#     ## GMM 
#     df_ols['const'] = 1.0
#     z = ['const', 'HMO', 'Satisfaction', 'Network', 'hausman_instrument', 'HMO_instrument', 'Satisfaction_quad', 'Network_quad']

#     Z = df_ols[z].to_numpy()
#     N = len(xi)

#     g = Z.T @ xi / N
#     W = np.linalg.inv(Z.T @ Z / N)

#     return g @ W @ g

# def objective(free_sigma):
#     # free_sigma has 2 entries: [sigma_Satisfaction, sigma_Network]
#     sigma = np.array([free_sigma[0], free_sigma[1], free_sigma[2]])   # HMO fixed at 0
#     return gmm(sigma)

# result = minimize(objective, x0=np.array([0.5, 0.5, 0.5]), method='Nelder-Mead',
#                   options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 200})

# print(result.x)     # estimated sigmas
# print(result.fun)   # objective value at the optimum
# print(result.success, result.message)

# for start in [[0.1, 0.1], [0.5, 0.5], [1.0, 0.3], [0.3, 1.0]]:
#     r = minimize(objective, x0=np.array(start), method='Nelder-Mead')
#     print(start, '->', np.abs(r.x).round(3), round(r.fun, 6))


market_ids = df_ols['market_ids'].unique() # Each unique market id
markets, draws, char = len(market_ids), 500, 3
nu_draws = np.random.randn(markets, draws, char) # 75 markets, 500 draws, 3 characteristics


S_obs = df_ols['shares'].to_numpy() # Observed Shares
delta0 = df_ols['ShareDiff'].to_numpy() # Intial Shares (Delta_Int)

## Create Gandhi Houde Instruments 
def gh_instruments(var):
    quad = []

    for t in df_ols['market_ids'].unique():
        x = df_ols[df_ols['market_ids'] == t][var].to_numpy()
        for j in range(len(x)):
            total = 0
            for k in range(len(x)):
                if k != j:
                        total = total + (x[k] - x[j])**2
            quad.append(total)       
    return quad

df_ols['Network_quad'] = gh_instruments('Network')
df_ols['Satisfaction_quad'] = gh_instruments('Satisfaction')     
df_ols['const'] = 1.0

## GMM Estiimation Elements
z = ['const', 'HMO', 'Satisfaction', 'Network', 'hausman_instrument', 'HMO_instrument', 'Satisfaction_quad', 'Network_quad']
Z = df_ols[z].to_numpy()
N = len(df_ols)
W = np.linalg.inv(Z.T @ Z / N)

x_list = [df_ols.loc[df_ols['market_ids'] == i, ['HMO', 'Satisfaction', 'Network']].to_numpy()
          for i in market_ids]  

def sim_shares(sigma, delta):
    s_hat = []
    start = 0
    # idx goes from 0-74 and t goes from 1-75 (Actual market ids) 
    for idx, x in enumerate(x_list):
        # For market i, get the product characterstics
        J = x.shape[0] # No of products in each market

        # Mu for each consumer across each product in market i
        mu = x @ (sigma[:, None] * nu_draws[idx].T)

        # Delta
        d = delta[start: start + J] # Uses number of products in each market to get the initial value of delta
        start += J # Start at next market in next loop

        # Simulates shares
        util = d[:, None] + mu
        num = np.exp(util)
        denom = 1 + num.sum(axis = 0, keepdims = True)

        prob = num/denom
        s_hat.append(prob.mean(axis = 1))

    return np.concatenate(s_hat)

# Contraction mapping
def contraction(sigma, delta, tol = 1e-12, maxit = 1000):
    for i in range(maxit):
        delta_new = delta + np.log(S_obs) - np.log(sim_shares(sigma, delta))
        if np.max(np.abs(delta_new - delta)) < tol:
                return delta_new
        delta = delta_new
    print("Warning: Did not converge")
    return delta

last = {}

def gmm(sigma):
     df_ols['delta_calc'] = contraction(sigma, delta0)
     
     iv2sls = IV2SLS.from_formula('delta_calc ~ 1 + HMO + Satisfaction + Network + [prices ~ hausman_instrument + ' \
    'HMO_instrument + Satisfaction_quad + Network_quad]', data =df_ols).fit(cov_type='robust')
     
     last['iv'] = iv2sls

    # ## Ksi term
     xi = iv2sls.resids.to_numpy()
     
     g = Z.T @ xi / N

     return g @ W @ g


def objective(free_sigma):
    return gmm(np.array(free_sigma))                 # all 3 sigmas free, as in your version

# First, time a single call
t0 = time.time()
print(objective([1, 1, 1]), time.time() - t0, "seconds")

result = minimize(objective, x0=np.array([1, 1, 1]), method='Nelder-Mead',
                  options={'xatol': 1e-4, 'fatol': 1e-12, 'maxiter': 200})

gmm(result.x)                                        # leaves the 2SLS result in last['iv']
print("sigma (abs):", np.abs(result.x))
print("Q:", result.fun)
print(last['iv'].params)                             # intercept, HMO, Satisfaction, Network, prices
print(last['iv'].summary)

# for start in [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.7, 1.0, 0.3], [0.5, 0.3, 1.0]]:

#     result = minimize(objective, x0=np.array(start), method='Nelder-Mead',
#                   options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 200})

#     # ---------- CHANGED (B): recover all parameters at the optimum ----------
#     gmm(result.x)                                        # leaves the 2SLS result in last['iv']
#     print("sigma (abs):", np.abs(result.x))
#     print("Q:", result.fun)
#     print(last['iv'].params)                             # intercept, HMO, Satisfaction, Network, prices
#     print(last['iv'].summary)

## BLP Coefficients
coefficients_blp = last['iv'].params
price_blp = coefficients_blp['prices']
HMO_blp = coefficients_blp['HMO']
Network_blp = coefficients_blp['Network']
Satisfaction_blp = coefficients_blp['Satisfaction']
sigma_blp = np.abs(result.x)


#####################################################################################################

# Own Price Elasticty

# Logit 
own_elasticity = price_logit * df['prices'] * (1 - df['shares'])
logit_mean_elasticity = own_elasticity.mean()
logit_markup = np.abs(1/logit_mean_elasticity)

# Nested Logit
own_elasticity = price_nested_logit* df['prices'] * ( 1/(1 - sigma_nested_logit) - (sigma_nested_logit / (1 - sigma_nested_logit))* df['cond_share'] - df['shares'])
nested_logit_mean_elasticity = own_elasticity.mean()
nested_logit_markup = np.abs(1/nested_logit_mean_elasticity)


# BLP

