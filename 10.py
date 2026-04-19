import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

df = pd.read_csv('data/lr-Real-estate.csv')
X = df.drop(['No', 'Y house price of unit area'], axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning using Grid Search
param_grid = {
    'n_neighbors': np.arange(1, 25),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

# Evaluate the Best Model
best_knn = grid_search.best_estimator_
predictions = best_knn.predict(X_test_scaled)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Optimized R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")

model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

knn_model_path = os.path.join(model_dir, 'best_knn_model.pkl')
joblib.dump(best_knn, knn_model_path)

scaler_path = os.path.join(model_dir, 'knn_scaler.pkl')
joblib.dump(scaler, scaler_path)

print(f"Model saved to: {knn_model_path}")
print(f"Scaler saved to: {scaler_path}")
