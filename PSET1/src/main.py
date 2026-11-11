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
from statsmodels.iolib.summary2 import summary_col
import optimagic as om
import openpyxl
from scipy.linalg import hadamard
from scipy.optimize import root
from scipy.stats import norm

MAIN = Path(__file__).resolve().parents[1]

DATA = MAIN / "data"
CODE = MAIN / "src"
OUTPUT = MAIN / "output"

for directory in (DATA, CODE, OUTPUT):
    directory.mkdir(parents=True, exist_ok=True)

np.random.seed(5)

######################################
#### Generate shares and clean ####
######################################

df = pd.read_excel(DATA / 'io_pset_1_data.xlsx')

df1 = df.copy()

df1 = df1.rename(columns={'Market_ID': 'market_ids', 'Plan_ID': 'product_ids', 'Share': 'shares', 'Premium': 'prices'})


# Outside Share
df1['TotalShare'] = df1.groupby(["market_ids"])['shares'].transform("sum")
df1['Outside_Share'] = 1 - df1['TotalShare']

# Log of Shares
df1['log_os'] = np.log(df1['Outside_Share'])
df1['log_s'] = np.log(df1['shares'])

# Shares by HMO status and within market (for nested logit)
df1['hmo_share'] = df1.groupby(
    ["market_ids", "HMO"]
)['shares'].transform("sum")

df1['within_hmo_share'] = df1['shares'] / df1['hmo_share']
df1['log_within_hmo_share'] = np.log(df1['within_hmo_share'])

# Difference 
df1['ShareDiff'] = df1['log_s'] - df1['log_os']

df1['firm_ids'] = df1['product_ids']

######################################
# Subset data #
######################################

Y = df1['ShareDiff'] # outcomes
P = df1['prices'] # endogenous price
X = df1[['prices', 'HMO', 'Satisfaction', 'Network']]
X = sm.add_constant(X) # constant + product characteristics 

X_id = df1[['market_ids', 'HMO', 'Satisfaction', 'Network']] # market id + exogenous characteristics 
X_id.insert(1, 'const', 1.0) # after talking with Tony, we need to add a random coefficient on the intercept
######################################
# BLP settings #
######################################

# number of markets
M = df1['market_ids'].nunique()
# number of draws per market
draws = 500
# number of product characteristics 
K = 4

# we simulate 500 agents' random coefficients for each product characteristic
nu = np.random.standard_normal(size = (M, draws, K))

######################################
# Plain logit #
######################################

# Let's first assume that prices are exogenous. Then we can use standard OLS
naive_ols = sm.OLS(
    Y, 
    X
).fit()
print(naive_ols.summary())

ols_table = summary_col(
    [naive_ols],
    model_names=["OLS"],
    stars=True,
    float_format="%.3f"
)

latex = ols_table.as_latex()
latex = latex.split(r"\begin{tabular}")[1].split(r"\end{tabular}")[0]
latex = r"\begin{tabular}" + latex + r"\end{tabular}"

with open(OUTPUT / "logit_ols.tex", "w", encoding="utf-8") as f:
    f.write(latex)

######################################
# Create the instruments #
######################################

# Of course, we believe that prices are correlated with unobservable product characteristics. So, we will need to instrument for price. We will construct BLP-style sum instruments. 
for i in ["HMO", "Satisfaction", "Network"]:
    df1[i + "_blp"] = (
        df1.groupby("market_ids")[i].transform("sum") - df1[i]
    )

df1["prod_length"] = (
    df1.groupby("product_ids")["product_ids"].transform("size") - 1
)

df1["hausman"] = (
    df1.groupby("product_ids")["prices"].transform("sum")
    - df1["prices"]
) / df1["prod_length"]

df1['hmo_instrument'] = (
    df1.groupby(["market_ids", "HMO"])['product_ids'].transform("nunique")
)

for i, row in df1.iterrows(): 
    market = row['market_ids']
    x_j = row['Network']

    x_k = df1.loc[df1['market_ids'] == market, 'Network']

    df1.loc[i, "network_gh"] = np.sum((x_j - x_k) ** 2)

for i, row in df1.iterrows(): 
    market = row['market_ids']
    x_j = row['Satisfaction']

    x_k = df1.loc[df1['market_ids'] == market, 'Satisfaction']

    df1.loc[i, "satisfaction_gh"] = np.sum((x_j - x_k) ** 2)

blp_instruments = sm.add_constant(
    df1[[
        "HMO",
        "Satisfaction",
        "Network",
        "HMO_blp",
        "Satisfaction_blp",
        "Network_blp"
    ]]
)

hausman_instruments = sm.add_constant(
    df1[[
        "HMO",
        "Satisfaction",
        "Network",
        "hausman"
    ]]
)

hausman_group_instruments = pd.concat([hausman_instruments, df1[['hmo_instrument']]], axis=1)


df1["n_products"] = df1.groupby("market_ids")["product_ids"].transform("size")

Z = sm.add_constant(df1[[
    "HMO", "Satisfaction", "Network",
    "hausman", "network_gh", "satisfaction_gh",
    "HMO_blp", "Satisfaction_blp", "Network_blp",
    "hmo_instrument", "n_products"
]])

######################################
# Plain Logit - BLP Instruments #
######################################

ols_1_blp = sm.OLS(P, blp_instruments).fit()
print(ols_1_blp.summary()) # there is a very weak first-stage effect 

# Predicted prices from the first stage
pred_p_blp = pd.Series(
    ols_1_blp.predict(),
    index=df1.index,
    name="instrumented_price"
)

# Second-stage regressors
x_tsls_blp = sm.add_constant(
    pd.concat([
        pred_p_blp,
        df1[["HMO", "Satisfaction", "Network"]]
    ], axis=1)
)

# Second-stage OLS
ols_2_blp = sm.OLS(Y, x_tsls_blp).fit()
print(ols_2_blp.summary())

# The standard errors are incorrect, so we will use the built-in package to compute them. 
tsls_model_blp = iv_blp = IV2SLS(
    endog=Y,
    exog=X,
    instrument=blp_instruments
).fit()
print(tsls_model_blp.summary()) # we get a positive price coefficient - not good

######################################
# Plain Logit - Hausman Instruments #
######################################

# The standard errors are incorrect, so we will use the built-in package to compute them. 
tsls_model_hausman = IV2SLS(
    endog=Y,
    exog=X,
    instrument=hausman_instruments
).fit()
print(tsls_model_hausman.summary())

hausman_table = summary_col(
    [tsls_model_hausman],
    model_names=["IV Hausman"],
    stars=True,
    float_format="%.3f"
)

latex = hausman_table.as_latex()
latex = latex.split(r"\begin{tabular}")[1].split(r"\end{tabular}")[0]
latex = r"\begin{tabular}" + latex + r"\end{tabular}"

with open(OUTPUT / "logit_hausman.tex", "w", encoding="utf-8") as f:
    f.write(latex)

logit_results = tsls_model_hausman

alpha_logit = -logit_results.params["prices"]
beta_const_logit = logit_results.params["const"]
beta_hmo_logit = logit_results.params["HMO"]
beta_satisfaction_logit = logit_results.params["Satisfaction"]
beta_network_logit = logit_results.params["Network"]

# Store parameters
params_logit = pd.Series({
    "alpha_logit": alpha_logit,
    "beta_const_logit": beta_const_logit,
    "beta_hmo_logit": beta_hmo_logit,
    "beta_satisfaction_logit": beta_satisfaction_logit,
    "beta_network_logit": beta_network_logit
})

# Export to CSV
params_logit.to_csv(OUTPUT / "params_logit.csv", header=["estimate"])

######################################
# Nested Logit - Gandhi Instruments #
######################################

# let's do naive OLS again
X_nested = X.assign(
    rho=df1["log_within_hmo_share"]
)

# let's first naively run OLS - assuming prices are exogenous
naive_ols = sm.OLS(Y, X).fit()
print(naive_ols.summary())

# let's check whether our instruments pass the first stage
ols_1_hausman_group_share = sm.OLS(
    df1["log_within_hmo_share"], 
    hausman_group_instruments).fit()
print(ols_1_hausman_group_share.summary()) 
pred_share_hausman_group = pd.DataFrame(ols_1_hausman_group_share.predict()) 

ols_1_hausman_group_price = sm.OLS(
    df1["prices"], 
    hausman_group_instruments).fit()
print(ols_1_hausman_group_price.summary()) 
pred_share_hausman_price = pd.DataFrame(ols_1_hausman_group_price.predict())

# both pass first stage test. let's now use the built-in package
tsls_model_hausman_nested = IV2SLS(
    endog=Y, 
    exog=X_nested, 
    instrument=hausman_group_instruments).fit()
print(tsls_model_hausman_nested.summary())

hausman_nested_table = summary_col(
    [tsls_model_hausman_nested],
    model_names=["IV Hausman"],
    stars=True,
    float_format="%.3f"
)

latex = hausman_nested_table.as_latex()
latex = latex.split(r"\begin{tabular}")[1].split(r"\end{tabular}")[0]
latex = r"\begin{tabular}" + latex + r"\end{tabular}"

with open(OUTPUT / "nested_logit_hausman.tex", "w", encoding="utf-8") as f:
    f.write(latex)

nested_logit_results = tsls_model_hausman_nested

alpha_nested_logit = -nested_logit_results.params["prices"]
beta_const_nested_logit = nested_logit_results.params["const"]
beta_hmo_nested_logit = nested_logit_results.params["HMO"]
beta_satisfaction_nested_logit = nested_logit_results.params["Satisfaction"]
beta_network_nested_logit = nested_logit_results.params["Network"]
rho_nested_logit = nested_logit_results.params["rho"]

# Store parameters
params_nested_logit = pd.Series({
    "alpha_nested_logit": alpha_nested_logit,
    "beta_const_nested_logit": beta_const_nested_logit,
    "beta_hmo_nested_logit": beta_hmo_nested_logit,
    "beta_satisfaction_nested_logit": beta_satisfaction_nested_logit,
    "beta_network_nested_logit": beta_network_nested_logit,
    "rho_nested_logit": rho_nested_logit
})

# Export parameters to CSV
params_nested_logit.to_csv(
    OUTPUT / "params_nested_logit.csv",
    header=["estimate"]
)

######################################
# BLP #
######################################


def calc_shares(delta, sigma, X, nu): 
    # We need to loop through each market to match observed with computed shares for a candidate set of parameters

    shares = []

    for i, m in enumerate(df1["market_ids"].unique()): 
        delta_m = delta[df1['market_ids'] == m]
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

    delta = np.asarray(delta_init).copy()
    log_s = df1["log_s"].to_numpy()

    while error > tol and iteration < max_iter: 
        delta_old = delta

        delta = delta_old + df1['log_s'] - np.log(calc_shares(delta_old, sigma, X, nu))

        error = np.linalg.norm(delta - delta_old, ord=np.inf)

        iteration += 1

        print(f"Iteration{iteration}")

    return delta

def candidate_gmm_obj(Z, X, delta, W=None): 

    Z = np.asarray(Z)
    X = np.asarray(X)
    delta = np.asarray(delta)

    if W is None:
        W = np.linalg.inv((Z.T @ Z) / len(df1))
    iv_gmm = (
        np.linalg.inv(X.T @ Z @ W @ Z.T @ X) 
        @ X.T @ Z @ W @ Z.T @ delta
    )
    xi = delta - X @ iv_gmm
    g = (Z.T @ xi) / len(df1)
    obj = g.T @ W @ g

    return obj

def objective(sigma_init): 
    sigma = np.diag(sigma_init)

    delta_hat = contraction_mapping(
    delta_init=df1['ShareDiff'],
    sigma=sigma,
    X=X_id,
    nu=nu
    )

    return candidate_gmm_obj(Z, X, delta_hat, W)

W = None
gmm = minimize(
    fun=objective, 
    x0=np.array([1, 1, 1, 1]), 
    bounds=((0, None), (0, None), (0, None), (0, None))
)

# This gives us the random coefficient sigma parameters. We can now recover the parameters on price and product characteristics
sigma_hat = np.diag(gmm.x)
X_np = X.to_numpy()
Z_np = Z.to_numpy()
delta_hat = contraction_mapping(
    df1["ShareDiff"],
    sigma_hat,
    X_id,
    nu
)

W = np.linalg.inv((Z_np.T @ Z_np) / len(df1))

beta_hat = (
    np.linalg.inv(X_np.T @ Z_np @ W @ Z_np.T @ X_np)
    @ X_np.T @ Z_np @ W @ Z_np.T @ delta_hat
)

xi = np.asarray(delta_hat) - X_np @ beta_hat

W = np.linalg.inv((Z_np * xi[:, None]).T @ (Z_np * xi[:, None]) / len(df1))

gmm = minimize(
    fun=objective, 
    x0=gmm.x, 
    bounds=((0, None), (0, None), (0, None), (0, None))
)

sigma_hat = np.diag(gmm.x)
delta_hat = contraction_mapping(
    df1["ShareDiff"],
    sigma_hat,
    X_id,
    nu
)

beta_hat = (
    np.linalg.inv(X_np.T @ Z_np @ W @ Z_np.T @ X_np)
    @ X_np.T @ Z_np @ W @ Z_np.T @ delta_hat
)

# We need the standard errors (G'WG)^-1
def blp_cov(eps):

    delta_grad_unstacked = []
    for i in range(sigma_hat.shape[0]):
        sigma_hat_plus = sigma_hat.copy()
        sigma_hat_minus = sigma_hat.copy()
        sigma_hat_plus[i, i] += eps
        sigma_hat_minus[i, i] -= eps

        delta_hat_plus = contraction_mapping(
            df1["ShareDiff"],
            sigma_hat_plus,
            X_id,
            nu
        )
        delta_hat_minus = contraction_mapping(
                    df1["ShareDiff"],
                    sigma_hat_minus,
                    X_id,
                    nu
        )

        delta_hat_grad = (delta_hat_plus - delta_hat_minus) / (2 * eps)

        delta_grad_unstacked.append(delta_hat_grad)

    delta_grad = np.column_stack(delta_grad_unstacked) 

    dxi_dtheta = np.column_stack([-X_np, delta_grad])  

    G = (1 / len(df1)) * (Z_np.T @ dxi_dtheta)

    se = np.linalg.inv(G.T @ W @ G)

    return se
        
se_hat = np.sqrt(np.diag(blp_cov(1e-6)) / len(df1))

pd.DataFrame({"delta_hat": delta_hat}).to_csv(
    OUTPUT / "delta_hat.csv", index=False
)

# We need to get standard errors for this estimate. We need to calculate the gradient 

print(beta_hat)
print(sigma_hat)

# Linear parameters
alpha_blp = -beta_hat[1]
beta_const_blp = beta_hat[0]
beta_hmo_blp = beta_hat[2]
beta_satisfaction_blp = beta_hat[3]
beta_network_blp = beta_hat[4]

# Random coefficient standard deviations
sigma_intercept = sigma_hat[0, 0]
sigma_hmo_blp = sigma_hat[1, 1]
sigma_satisfaction_blp = sigma_hat[2, 2]
sigma_network_blp = sigma_hat[3, 3]

# Store parameters
params_blp = pd.Series({
    "alpha_blp": alpha_blp,
    "beta_const_blp": beta_const_blp,
    "beta_hmo_blp": beta_hmo_blp,
    "beta_satisfaction_blp": beta_satisfaction_blp,
    "beta_network_blp": beta_network_blp,
    "sigma_intercept_blp": sigma_intercept,
    "sigma_hmo_blp": sigma_hmo_blp,
    "sigma_satisfaction_blp": sigma_satisfaction_blp,
    "sigma_network_blp": sigma_network_blp
})

# Export to CSV
params_blp.to_csv(
    OUTPUT / "params_blp.csv",
    header=["estimate"]
)

blp_estimates = np.concatenate([np.ravel(beta_hat), np.diag(sigma_hat)])
se = np.ravel(se_hat)
pvals = 2 * norm.sf(np.abs(blp_estimates / se))

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

blp_labels = [
    "Constant", "Price", "HMO", "Satisfaction", "Network",
    r"$\sigma_{0}$", r"$\sigma_{HMO}$", r"$\sigma_{Sat}$", r"$\sigma_{Net}$",
]

rows = []
for i, (lab, est, s, p) in enumerate(zip(blp_labels, blp_estimates, se, pvals)):
    if i == 5:
        rows.append(r"\hline")  # separate linear parameters from sigmas
    rows.append(f"{lab} & {est:.3f}{stars(p)} \\\\")
    rows.append(f" & ({s:.3f}) \\\\")

latex = "\n".join([
    r"\begin{tabular}{ll}",
    r"\hline",
    r" & BLP \\",
    r"\hline",
    *rows,
    r"\hline",
    r"\end{tabular}",
])

with open(OUTPUT / "blp.tex", "w", encoding="utf-8") as f:
    f.write(latex)

######################################
# Logit mean own price elasticity #
######################################

# The own-price elasticity is given by -alpha * p_jt * (1 - s_jt)
logit_params = pd.read_csv(
    OUTPUT / "params_logit.csv", header=0, names=['params', 'estimates']
)

logit_alpha = logit_params.loc[
    logit_params["params"] == "alpha_logit", "estimates"
].to_numpy()

logit_elast = (
    -logit_alpha 
    * P 
    * (1 - df1["shares"]))

mean_logit_elast = np.mean(
    logit_elast, 
    axis=0
)

logit_markup = -1 / mean_logit_elast

######################################
# Nested logit mean own price elasticity #
######################################

nested_logit_params = pd.read_csv(
    OUTPUT / "params_nested_logit.csv", header=0, names=['params', 'estimates']
)

nested_logit_alpha = nested_logit_params.loc[
    nested_logit_params["params"] == "alpha_nested_logit", "estimates"
].to_numpy()

nested_logit_rho = nested_logit_params.loc[
    nested_logit_params["params"] == "rho_nested_logit", "estimates"
].to_numpy()

nested_logit_elast = (
    - nested_logit_alpha * P * (
        1 / (1 - nested_logit_rho) 
        - (nested_logit_rho / (1 - nested_logit_rho))
        * df1['within_hmo_share']
        - df1['shares']
    )
)

mean_nested_logit_elast = np.mean(nested_logit_elast)

nested_logit_markup = -1 / mean_nested_logit_elast

######################################
# BLP mean own price elasticity #
######################################

blp_params = pd.read_csv(
    OUTPUT / "params_blp.csv", header=0, names=['params', 'estimates']
)

blp_alpha = blp_params.loc[
    blp_params["params"] == "alpha_blp", "estimates"
].to_numpy()

blp_beta_satisfaction = blp_params.loc[
    blp_params["params"] == "beta_satisfaction_blp", "estimates"
].to_numpy()

blp_sigma_hmo = blp_params.loc[
    blp_params["params"] == "sigma_hmo_blp", "estimates"
].to_numpy()

blp_sigma_intercept = blp_params.loc[
    blp_params["params"] == "sigma_intercept_blp", "estimates"
].to_numpy()

blp_sigma_satisfaction = blp_params.loc[
    blp_params["params"] == "sigma_satisfaction_blp", "estimates"
].to_numpy()

blp_sigma_network = blp_params.loc[
    blp_params["params"] == "sigma_network_blp", "estimates"
].to_numpy()

sigma_hat = np.diag([
    blp_sigma_intercept.item(),
    blp_sigma_hmo.item(),
    blp_sigma_satisfaction.item(),
    blp_sigma_network.item()
])

delta_hat = pd.read_csv(
    OUTPUT / "delta_hat.csv"
)["delta_hat"]
delta_hat = pd.DataFrame(delta_hat)

df1_delta = pd.concat([df1, delta_hat],  axis=1)

def calc_shares_delta(delta, sigma, X, nu, dataframe): 
    # We need to loop through each market to match observed with computed shares for a candidate set of parameters

    shares = []
    delta = np.asarray(delta)

    for i, m in enumerate(dataframe["market_ids"].unique()): 
        delta_m = delta[dataframe['market_ids'].to_numpy() == m]
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


def share_jac(delta, alpha, eps):

    jacobian = {}
    delta = np.asarray(delta)

    for i, m in enumerate(df1_delta['market_ids'].unique()):

        dfm = df1[df1["market_ids"] == m].copy()
        dfm['delta'] = delta[df1["market_ids"].to_numpy() == m]

        X_id_m = X_id[X_id['market_ids'] == m]

        nu_m = nu[i:i+1]

        n_products_m = len(np.unique(dfm['product_ids']))

        jac_m = np.zeros((n_products_m, n_products_m))

        for k in range(n_products_m):

                dfm['delta_plus'] = dfm["delta"].copy()
                dfm['delta_minus'] = dfm["delta"].copy()

                dfm.iloc[k, dfm.columns.get_loc('delta_plus')] -= eps * alpha.item()
                dfm.iloc[k, dfm.columns.get_loc('delta_minus')] += eps * alpha.item()

                shares_plus = calc_shares_delta(
                    dfm['delta_plus'], 
                    sigma_hat, 
                    X_id_m, 
                    nu_m, 
                    dfm
                )
                shares_mins = calc_shares_delta(
                    dfm['delta_minus'], 
                    sigma_hat, 
                    X_id_m, 
                    nu_m, 
                    dfm
                    )

                jac_m[:, k] = (shares_plus - shares_mins) / (2 * eps)

        jacobian[m] = jac_m

    return jacobian

share_price_jac = share_jac(delta=df1_delta["delta_hat"], alpha=blp_alpha, eps=1e-6)

share_price_jac_np = np.array(
    list(share_price_jac.values()),
    dtype=object
)

for i, m in enumerate(df1["market_ids"].unique()):
    own_elast_m =   np.diag(np.diag(share_price_jac_np[i])) @ (
                         (df1.loc[df1['market_ids'] == m, 'prices'].to_numpy()) 
                         / 
                         (df1.loc[df1['market_ids'] == m, 'shares'].to_numpy())
                    )

    df1.loc[df1['market_ids'] == m, 'own_elast_blp'] = own_elast_m

mean_blp_elast = df1["own_elast_blp"].mean()

for m in df1["market_ids"].unique():
    mask = df1["market_ids"] == m

    p_m = df1.loc[mask, "prices"].to_numpy()
    s_m = df1.loc[mask, "shares"].to_numpy()
    jac_m = share_price_jac[m]

    Delta_m = -np.eye(len(p_m)) * jac_m      
    markup_m = np.linalg.inv(Delta_m) @ s_m

    df1.loc[mask, "blp_markup"] = markup_m / p_m

blp_markup = df1["blp_markup"].mean()

######################################
# BLP MC and characteristic relation #
######################################

# Let's calculate marginal costs for products
mc = P * (1 - df1["blp_markup"])
log_mc = np.log(mc)

mc_ols = sm.OLS(
    log_mc, 
    X.loc[:, X.columns != 'prices']
).fit()
print(mc_ols.summary())

mc_table = summary_col(
    [mc_ols],
    model_names=["log(MC)"],
    stars=True,
    float_format="%.3f"
)

latex = mc_table.as_latex()
latex = latex.split(r"\begin{tabular}")[1].split(r"\end{tabular}")[0]
latex = r"\begin{tabular}" + latex + r"\end{tabular}"

with open(OUTPUT / "mc_ols.tex", "w", encoding="utf-8") as f:
    f.write(latex)

######################################
# MC with ownership matrix #
######################################

def get_mc(ownership_matrix):

    mc = {}
    delta_p = {}

    # We compute marginal costs within each market
    for m in df1['market_ids'].unique():

        ownership_matrix_m = ownership_matrix[m]
        share_price_jac_m = share_price_jac[m]
        delta_pm = - np.multiply(ownership_matrix_m, share_price_jac_m)

        c_m = df1.loc[df1["market_ids"] == m, "prices"].to_numpy() - np.linalg.inv(delta_pm) @ df1.loc[df1["market_ids"] == m, "shares"].to_numpy()

        mc[m] = c_m
        delta_p[m] = delta_pm

    return mc, delta_p


def ownership(mergers=None):

    # mergers maps product_ids to new firm_ids, e.g. {2: 1} means firm 1 acquires product 2
    if mergers is None:
        mergers = {}

    ownership_matrix = {}

    for m in df1['market_ids'].unique():

        firms_m = df1.loc[df1["market_ids"] == m, "product_ids"].map(
            lambda j: mergers.get(j, df1.loc[df1["product_ids"] == j, "firm_ids"].iloc[0])
        ).to_numpy()

        # O_jk = 1 if products j and k are owned by the same firm
        ownership_matrix[m] = (firms_m[:, None] == firms_m[None, :]).astype(float)

    return ownership_matrix 

H = ownership()
mc, delta_p = get_mc(H)

######################################
# New price from subsidy #
######################################

def foc_residual(p, subsidy, ownership_matrix, mc_flat, p_obs, delta_hat):

    # Consumers pay p - subsidy, which shifts mean utility
    delta_p = delta_hat - blp_alpha.item() * (p - p_obs)

    shares_p = calc_shares_delta(delta_p, sigma_hat, X_id, nu, df1)
    jac_p = share_jac(delta=delta_p, alpha=blp_alpha, eps=1e-6)

    # We compute FOC residuals within each market
    residuals = []

    for m in df1['market_ids'].unique():

        delta_pm = - np.multiply(ownership_matrix[m], jac_p[m])
        p_m = p[df1["market_ids"].to_numpy() == m]
        c_m = mc_flat[df1["market_ids"].to_numpy() == m]
        s_m = shares_p[df1["market_ids"].to_numpy() == m]

        residuals.append(delta_pm @ (p_m + subsidy - c_m) - s_m)

    return np.concatenate(residuals)


def price_solver(subsidy, ownership_matrix, mc):

    p_obs = df1["prices"].to_numpy()
    delta_hat = df1_delta["delta_hat"].to_numpy()
    mc_flat = np.concatenate([mc[m] for m in df1['market_ids'].unique()])

    sol = root(
        fun=foc_residual,
        x0=p_obs,
        args=(subsidy, ownership_matrix, mc_flat, p_obs, delta_hat),
        method="hybr"
    )

    print(sol.message)

    p = sol.x
    delta_p = delta_hat - blp_alpha.item() * (p - subsidy - p_obs)
    shares_p = calc_shares_delta(delta_p, sigma_hat, X_id, nu, df1)

    return p, shares_p


mc, delta_p = get_mc(ownership())
p_subsidy, shares_subsidy = price_solver(subsidy=0.25, ownership_matrix=ownership(), mc=mc)

df1["prices_subsidy"] = p_subsidy
df1["shares_subsidy"] = shares_subsidy
df1['outside_share_subsidy'] = 1 - df1.groupby(["market_ids"])['shares_subsidy'].transform("sum")

df1[['prices', 'prices_subsidy', 'shares', 'shares_subsidy', 'Outside_Share', 'outside_share_subsidy']].describe()

summary_vars = {
    "prices": "Premium (baseline)",
    "prices_subsidy": "Premium (subsidy)",
    "shares": "Share (baseline)",
    "shares_subsidy": "Share (subsidy)",
    "Outside_Share": "Outside share (baseline)",
    "outside_share_subsidy": "Outside share (subsidy)",
}

stats = df1[list(summary_vars)].describe().T
stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
stats.index = [summary_vars[v] for v in stats.index]

rows = [
    f"{lab} & " + " & ".join(f"{r[c]:.3f}" for c in stats.columns) + r" \\"
    for lab, r in stats.iterrows()
]

latex = "\n".join([
    r"\begin{tabular}{lccccccc}",
    r"\hline",
    r" & Mean & SD & Min & P25 & Median & P75 & Max \\",
    r"\hline",
    *rows,
    r"\hline",
    r"\end{tabular}",
])

with open(OUTPUT / "subsidy_summary.tex", "w", encoding="utf-8") as f:
    f.write(latex)

df1.to_csv(OUTPUT / "df1_subsidy.csv", index=False)

######################################
# New price from merger #
######################################

df2 = pd.read_csv(OUTPUT / "df1_subsidy.csv")

H_pre  = ownership()
H_post = ownership({2: 4})

mc_pre, delta_p = get_mc(H_pre)          
p_merger, shares_merger = price_solver(subsidy=0, ownership_matrix=H_post, mc=mc_pre)

df2["prices_merger"] = p_merger
df2["shares_merger"] = shares_merger

df2['outside_share_merger'] = 1 - df2.groupby(["market_ids"])['shares_merger'].transform("sum")

df2[['prices', 'prices_merger', 'shares', 'shares_merger', 'Outside_Share', 'outside_share_merger']].describe()