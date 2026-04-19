import pandas as pd
import numpy as np

try:
    df = pd.read_csv('data/lr-Real-estate.csv')
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: File not found.")

print("\nBasic Information & Data Types:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values Count:")
missing_vals = df.isnull().sum()
print(missing_vals[missing_vals > 0])

duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

print("\nStatistical Summary (Numerical):")
print(df.describe())


print("\nCategorical Column Inspection:")
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f"{col}: {df[col].nunique()} unique values")
    