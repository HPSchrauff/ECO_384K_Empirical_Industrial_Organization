import pandas as pd
import numpy as np

df = pd.read_excel(DATA / 'io_pset_1_data.xlsx')

df1 = df.copy()

df1 = df.rename(columns={'Market_ID': 'market_ids', 'Plan_ID': 'product_ids', 'Share': 'shares', 'Premium': 'prices'})


# Outside Share
df1['TotalShare'] = df1.groupby(["market_ids"])['shares'].transform("sum")
df1['Outside_Share'] = 1 - df1['TotalShare']

# Log of Shares
df1['log_os'] = np.log(df1['Outside_Share'])
df1['log_s'] = np.log(df1['shares'])

# Difference 
df1['ShareDiff'] = df1['log_s'] - df1['log_os']

