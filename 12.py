import joblib
import numpy as np
import pandas as pd

new_house = np.array([[2013.5, 15.0, 250.0, 7, 24.978, 121.535]])

feature_names = [
                  'X1 transaction date', 'X2 house age', 
                  'X3 distance to the nearest MRT station', 
                  'X4 number of convenience stores', 'X5 latitude', 'X6 longitude'
                ]

features_df = pd.DataFrame(new_house, columns=feature_names)

loaded_scaler = joblib.load('models/knn_scaler.pkl')
loaded_model = joblib.load('models/best_knn_model.pkl')

scaled_features = loaded_scaler.transform(features_df)
prediction = loaded_model.predict(scaled_features)

print(f"Predicted House Price per Unit Area: {prediction[0]:.2f}")
