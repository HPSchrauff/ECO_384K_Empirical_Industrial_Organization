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

np.random.seed(5)

######################################
#### Question 1 ####
######################################

# renaming columns that have reserved names in pyblp
df = df.rename(columns={'Market_ID': 'market_ids', 'Plan_ID': 'product_ids', 'Share': 'shares', 'Premium': 'prices'})

# Outside Share
df['TotalShare'] = df.groupby(["market_ids"])['shares'].transform("sum")
df['Outside_Share'] = 1 - df['TotalShare']

# Log of Shares
df['log_os'] = np.log(df['Outside_Share'])
df['log_s'] = np.log(df['shares'])

# Difference 
df['ShareDiff'] = df['log_s'] - df['log_os']

######################################
#### 1. Simulate market shares ####
######################################

# Given the RE utility specification does not imply an analytic form for market shares, we need to simulate market shares. 
# That requires us to simulate, for each market, 500 agents with taste parameters for 3 product characteristics drawn from N(0,1). We do not have a random coefficient on price

# number of markets
M = df['market_ids'].nunique()
# number of draws per market
draws = 500
# number of product characteristics 
K = 3
# demand characteristics with random coefficients
X = df[['market_ids', 'HMO', 'Satisfaction', 'Network']]

# demand characteristics and constant
X1 = df[['prices', 'HMO', 'Satisfaction', 'Network']].to_numpy()
X1 = sm.add_constant(X1)

# We found for logit that BLP instruments didn't seem great. So, we'll use Hausman instruments, which are the average prices of the same good in another market. 

df['prod_length'] = df.groupby(['product_ids'])['product_ids'].transform("size") - 1
hausman_instruments = (df.groupby(['product_ids'])['prices'].transform("sum") - df['prices']) / df['prod_length']
hausman_instruments = pd.concat([hausman_instruments, df[['HMO', 'Satisfaction', 'Network']]], axis=1)
hausman_instruments = sm.add_constant(hausman_instruments)
hausman_instruments = hausman_instruments.rename(columns = {0: 'instrumented_price'})

# Note that there are 8 parameters to estimate and Hausman + product characteristics instrumenting for themselves provide only 5. So, we need three more. 
for i, row in df.iterrows(): 
    market = row['market_ids']
    x_j = row['Network']

    x_k = df.loc[df['market_ids'] == market, 'Network']

    df.loc[i, "network_gh"] = np.sum((x_j - x_k) ** 2)

for i, row in df.iterrows(): 
    market = row['market_ids']
    x_j = row['Satisfaction']

    x_k = df.loc[df['market_ids'] == market, 'Satisfaction']

    df.loc[i, "satisfaction_gh"] = np.sum((x_j - x_k) ** 2)

df['hmo_blp'] = df.groupby(['market_ids'])['HMO'].transform("sum") - df['HMO']

Z = pd.concat([hausman_instruments, df[['network_gh', 'satisfaction_gh', 'hmo_blp']]], axis=1)
Z = Z.to_numpy()

nu = np.random.standard_normal(size = (M, draws, K))

# We need to specify all sub functions that will be called in a minimize function

def calc_shares(delta, sigma, X, nu): 
    """
    
    """
    # We need to loop through each market to match observed with computed shares for a candidate set of parameters

    shares = []

    for i, m in enumerate(df["market_ids"].unique()): 
        delta_m = delta[df['market_ids'] == m]
        delta_shaped = np.tile(delta_m, (draws, 1)).T
        nu_m = nu[i]
        X_m = X[X['market_ids'] == m]
        X_m = X_m.drop(columns=["market_ids"])
        mu_m = X_m @ sigma @ nu_m.T
    
        f_jm = (np.exp(delta_shaped + mu_m)) / (1 + np.sum(np.exp(delta_shaped + mu_m), axis=0))

        s_jm = np.average(f_jm, axis=1)

        shares.append(s_jm)

    shares_flat = np.concatenate(shares)
    
    return shares_flat


# We need to loop through different delta_j values until we find one such that observable shares are sufficiently close to model-implied shares
def contraction_mapping(delta_init, sigma, X, nu, tol=1e-12, max_iter=1000):

    error = 5
    iteration = 0

    delta = delta_init

    while error > tol and iteration < max_iter: 
        delta_old = delta

        delta = delta_old + df['log_s'] - np.log(calc_shares(delta_old, sigma, X, nu))

        error = np.linalg.norm(delta - delta_old, ord=np.inf)

        iteration += 1

        print(f"Iteration{iteration}")

    return delta

def candidate_gmm_obj(Z, X, delta): 

    W = np.linalg.inv((Z.T @ Z) / len(df))

    iv_gmm = np.linalg.inv(X.T @ Z @ W @ Z.T @ X) @ X.T @ Z @ W @ Z.T @ delta

    xi = delta - X1 @ iv_gmm

    g = (Z.T @ xi) / len(df)

    obj = g.T @ W @ g

    return obj

def objective(sigma_init): 
    sigma = np.diag(sigma_init)

    delta_hat = contraction_mapping(
    delta_init=df['ShareDiff'],
    sigma=sigma,
    X=X,
    nu=nu
    )

    return candidate_gmm_obj(Z, X1, delta_hat)


gmm = minimize(fun=objective, x0=np.array([1, 1, 1]), bounds=((0, None), (0, None), (0, None)))

# This gives us the random coefficient sigma parameters. We can now recover the parameters on price and product characteristics
sigma_hat = np.diag(gmm.x)
delta_hat = contraction_mapping(df['ShareDiff'], sigma_hat, X, nu)
W = np.linalg.inv((Z.T @ Z) / len(df))
beta_hat = np.linalg.inv(X1.T @ Z @ W @ Z.T @ X1) @ X1.T @ Z @ W @ Z.T @ delta_hat
print(beta_hat)

# [ 1.36912644 -2.39191968 -0.730913    2.90925006  2.86487495]
#array([[0.72518878, 0.        , 0.        ],
#       [0.        , 1.32609945, 0.        ],
#       [0.        , 0.        , 1.58038596]])