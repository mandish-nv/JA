import pandas as pd

df = pd.read_csv('data/lr-Real-estate.csv')

df_encoded = pd.get_dummies(df, columns=['X4 number of convenience stores'], prefix='stores')

print("Columns after One-Hot Encoding:")
print(df_encoded.columns)