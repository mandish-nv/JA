import joblib
import os

loaded_scaler = joblib.load('models/knn_scaler.pkl')
loaded_model = joblib.load('models/best_knn_model.pkl')

if loaded_model:
  print("Model loaded successfully!")
  
if loaded_scaler:
  print("Scaler loaded successfully!")
