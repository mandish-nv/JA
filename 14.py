import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/lr-Real-estate.csv')
X = df.drop(['No', 'Y house price of unit area'], axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "KNN (k=7)": KNeighborsRegressor(n_neighbors=7)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    # Predict
    preds = model.predict(X_test_scaled)
    # Score
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append({"Model": name, "RMSE": rmse, "R2 Score": r2})

# 5. Display Comparison
comparison_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print("Model Performance Comparison:")
print(comparison_df)

best_model = comparison_df.iloc[0]['Model']
print(f"\nBest Selection for this dataset: {best_model}")
