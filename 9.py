import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('data/lr-Real-estate.csv')
X = df.drop(['No', 'Y house price of unit area'], axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model WITHOUT Scaling
lr_unscaled = LinearRegression()
lr_unscaled.fit(X_train, y_train)
unscaled_preds = lr_unscaled.predict(X_test)

# Model WITH Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LinearRegression()
lr_scaled.fit(X_train_scaled, y_train)
scaled_preds = lr_scaled.predict(X_test_scaled)

# 4. Compare Performance
results = pd.DataFrame({
    'Metric': ['MAE', 'R2 Score'],
    'Unscaled': [
        mean_absolute_error(y_test, unscaled_preds),
        r2_score(y_test, unscaled_preds)
    ],
    'Scaled': [
        mean_absolute_error(y_test, scaled_preds),
        r2_score(y_test, scaled_preds)
    ]
})

print("Performance Comparison:")
print(results)

print("\nCoefficients Analysis (Scaled):")
coef_df = pd.DataFrame({'Feature': X.columns, 'Weight': lr_scaled.coef_})
print(coef_df.sort_values(by='Weight', ascending=False))

# The model explains roughly 68% of the price variation. The "Unscaled" version gave the same accuracy, but only the "Scaled" version allowed to see that MRT Distance is the main factor of house pricing in the area.
