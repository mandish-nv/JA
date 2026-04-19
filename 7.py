import pandas as pd 
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


try:
    df = pd.read_csv('data/lr-Real-estate.csv')
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: File not found.")

X = df.drop(['Y house price of unit area'], axis=1)
Y = df['Y house price of unit area']

model_path = 'models/linear_regression_model.pkl'
try:
    lr_model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please train the model first.")

predictions = lr_model.predict(X)

mae = mean_absolute_error(Y, predictions)
mse = mean_squared_error(Y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y, predictions)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE):    {mae:.2f}")
print(f"Mean Squared Error (MSE):     {mse:.2f}")
print(f"Root Mean Squared Error(RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score):         {r2:.2f}")

comparison_df = pd.DataFrame({'GT': Y, 'Predictions': predictions})
print("\nComparison (First 10 Rows):")
print(comparison_df.head(10))

