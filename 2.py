import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data/lr-Real-estate.csv')

cols = {
    'X1 transaction date': 'date',
    'X2 house age': 'age',
    'X3 distance to the nearest MRT station': 'mrt_dist',
    'X4 number of convenience stores': 'stores',
    'X5 latitude': 'lat',
    'X6 longitude': 'long',
    'Y house price of unit area': 'price'
}
df.rename(columns=cols, inplace=True)

print(f"Missing values before cleaning:\n{df.isnull().sum()}")

df['age'] = df['age'].fillna(df['age'].median())
df['mrt_dist'] = df['mrt_dist'].fillna(df['mrt_dist'].median())
df.dropna(subset=['price'], inplace=True)

z_scores = np.abs(stats.zscore(df[['mrt_dist', 'price']]))
df = df[(z_scores < 3).all(axis=1)]

df['transaction_year'] = df['date'].astype(int)
