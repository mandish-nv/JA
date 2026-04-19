import pandas as pd 
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

try:
    df = pd.read_csv('data/lr-Real-estate.csv')
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: File not found.")

X = df.drop(['Y house price of unit area'], axis=1)
Y = df['Y house price of unit area']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'linear_regression_model.pkl')
joblib.dump(lr_model, model_path)

print(f"Model saved successfully to {model_path}")


predictions = lr_model.predict(X_test)
comparison_df = pd.DataFrame({'GT': Y_test, 'Predictions': predictions})

print("\nModel Comparison (First 15 Rows):")
print(comparison_df.head(15))
