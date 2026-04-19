import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    df = pd.read_csv('data/lr-Real-estate.csv')
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: File not found.")

X = df.drop(['Y house price of unit area'], axis = 1) # independent variable
Y = df['Y house price of unit area'] # dependent variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42) 

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
