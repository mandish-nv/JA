import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/lr-Real-estate.csv')
X = df.drop(['No', 'Y house price of unit area'], axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Analyze Overfitting vs. Depth
depths = range(1, 21)
train_rmse = []
test_rmse = []

for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # Calculate RMSE for both sets
    train_pred = dt.predict(X_train)
    test_pred = dt.predict(X_test)
    
    train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, test_pred)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_rmse, label='Train RMSE', color='blue', marker='o')
plt.plot(depths, test_rmse, label='Test RMSE', color='red', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('RMSE (Lower is better)')
plt.title('Decision Tree: Identifying Overfitting')
plt.legend()
plt.grid(True)
plt.savefig('result/decision_tree.png')
plt.show()

