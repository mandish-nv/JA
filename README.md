# Real Estate Price Prediction Project

This project contains a series of Python scripts for analyzing and predicting real estate prices using various machine learning techniques.

## Files Overview

### 1.py
**Description:** Loads the real estate dataset, displays basic information including data types, first 5 rows, missing values count, duplicate rows, statistical summary for numerical columns, and inspection of categorical columns.  
**Outputs:** 
   ```bash
Dataset Loaded Successfully.

Basic Information & Data Types:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 414 entries, 0 to 413
Data columns (total 8 columns):
 #   Column                                  Non-Null Count  Dtype
---  ------                                  --------------  -----
 0   No                                      414 non-null    int64
 1   X1 transaction date                     414 non-null    float64
 2   X2 house age                            414 non-null    float64
 3   X3 distance to the nearest MRT station  414 non-null    float64
 4   X4 number of convenience stores         414 non-null    int64
 5   X5 latitude                             414 non-null    float64
 6   X6 longitude                            414 non-null    float64
 7   Y house price of unit area              414 non-null    float64
dtypes: float64(6), int64(2)
memory usage: 26.0 KB
None

First 5 Rows:
   No  X1 transaction date  X2 house age  ...  X5 latitude  X6 longitude  Y house price of unit area
0   1             2012.917          32.0  ...     24.98298     121.54024                        37.9
1   2             2012.917          19.5  ...     24.98034     121.53951                        42.2
2   3             2013.583          13.3  ...     24.98746     121.54391                        47.3
3   4             2013.500          13.3  ...     24.98746     121.54391                        54.8
4   5             2012.833           5.0  ...     24.97937     121.54245                        43.1

[5 rows x 8 columns]

Missing Values Count:
Series([], dtype: int64)

Duplicate Rows: 0

Statistical Summary (Numerical):
               No  X1 transaction date  X2 house age  ...  X5 latitude  X6 longitude  Y house price of unit area
count  414.000000           414.000000    414.000000  ...   414.000000    414.000000                  414.000000
mean   207.500000          2013.148971     17.712560  ...    24.969030    121.533361                   37.980193
std    119.655756             0.281967     11.392485  ...     0.012410      0.015347                   13.606488
min      1.000000          2012.667000      0.000000  ...    24.932070    121.473530                    7.600000
25%    104.250000          2012.917000      9.025000  ...    24.963000    121.528085                   27.700000
50%    207.500000          2013.167000     16.100000  ...    24.971100    121.538630                   38.450000
75%    310.750000          2013.417000     28.150000  ...    24.977455    121.543305                   46.600000
max    414.000000          2013.583000     43.800000  ...    25.014590    121.566270                  117.500000

[8 rows x 8 columns]
   ```

### 2.py
**Description:** Renames columns for clarity, handles missing values by filling with medians or dropping rows, removes outliers using Z-score, and adds a transaction year column.  
**Outputs:**  
   ```bash
Missing values before cleaning:
No          0
date        0
age         0
mrt_dist    0
stores      1
lat         0
long        0
price       0
dtype: int64
   ```

### 3.py
**Description:** Generates visualizations including a correlation heatmap, regression plot of MRT distance vs price, boxplot of price by number of stores, and geospatial scatter plot of prices.  
**Outputs:** correlation_heatmap.png, price_vs_mrt.png, price_vs_stores.png, geospatial_price.png (saved in result/ folder)

### 4.py
**Description:** Performs one-hot encoding on the 'number of convenience stores' column to prepare categorical data for modeling.  
**Outputs:**  
   ```bash
Columns after One-Hot Encoding:
Index(['No', 'X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station', 'X5 latitude', 'X6 longitude',
       'Y house price of unit area', 'stores_0', 'stores_1', 'stores_2',
       'stores_3', 'stores_4', 'stores_5', 'stores_6', 'stores_7', 'stores_8',
       'stores_9', 'stores_10'],
      dtype='object')
   ```

### 5.py
**Description:** Splits the dataset into training and testing sets (80/20 split) for model evaluation.  
**Outputs:**  
   ```bash
Dataset Loaded Successfully.
(331, 7) (331,) (83, 7) (83,)
   ```

### 6.py
**Description:** Trains a linear regression model on the training data, saves the model, and displays predictions vs ground truth for the first 15 test samples.  
**Outputs:** linear_regression_model.pkl (saved in models/ folder)

### 7.py
**Description:** Loads the trained linear regression model, evaluates it on the entire dataset using MAE, MSE, RMSE, and R2 score, and shows a comparison of predictions vs actual values.  
**Outputs:**  
   ```bash
Dataset Loaded Successfully.
Model loaded successfully from models/linear_regression_model.pkl

Model Evaluation Metrics:
Mean Absolute Error (MAE):    6.10
Mean Squared Error (MSE):     77.05
Root Mean Squared Error(RMSE): 8.78
R-squared (R2 Score):         0.58

Comparison (First 10 Rows):
     GT  Predictions
0  37.9    48.201071
1  42.2    48.819724
2  47.3    49.998518
3  54.8    49.544084
4  43.1    47.195371
5  32.1    32.186374
6  40.3    39.598944
7  46.7    48.058136
8  18.8     9.961038
9  22.1    36.106944
   ```

### 8.py
**Description:** Analyzes decision tree overfitting by plotting RMSE for training and test sets across different tree depths.  
**Outputs:** decision_tree.png (saved in result/ folder)

### 9.py
**Description:** Compares linear regression performance with and without feature scaling, displays evaluation metrics, and analyzes feature coefficients.  
**Interpretation:** The model explains roughly 68% of the price variation. The "Unscaled" version gave the same accuracy, but only the "Scaled" version allowed to see that MRT Distance is the main factor of house pricing in the area.
**Outputs:**  
   ```bash
Performance Comparison:
     Metric  Unscaled    Scaled
0       MAE  5.305356  5.305356
1  R2 Score  0.681058  0.681058

Coefficients Analysis (Scaled):
                                  Feature    Weight
3         X4 number of convenience stores  3.218873
4                             X5 latitude  2.855108
0                     X1 transaction date  1.529631
5                            X6 longitude -0.441009
1                            X2 house age -3.062694
2  X3 distance to the nearest MRT station -5.786926
   ```

### 10.py
**Description:** Performs hyperparameter tuning for KNN regressor using grid search, evaluates the best model, and saves the model and scaler.  
**Outputs:** best_knn_model.pkl, knn_scaler.pkl (saved in models/ folder)

### 11.py
**Description:** Loads the saved KNN model and scaler to verify successful loading.  
**Outputs:**  
   ```bash
Model loaded successfully!
Scaler loaded successfully!
   ```

### 12.py
**Description:** Uses the loaded KNN model to predict house price for a new sample data point.  
**Outputs:**  
   ```bash
Predicted House Price per Unit Area: 47.96
   ```

### 13.py
**Description:** Applies K-means clustering to segment the market into 3 clusters based on features, displays cluster profiles, and visualizes clusters by MRT distance and price.  
**Outputs:** KMeans_Clustering.png (saved in result/ folder)

### 14.py
**Description:** Compares the performance of Linear Regression, Decision Tree, and KNN models on the test set using RMSE and R2 score.  
**Outputs:**  
   ```bash
Model Performance Comparison:
               Model      RMSE  R2 Score
1      Decision Tree  5.933597  0.790131
2          KNN (k=7)  6.686034  0.733529
0  Linear Regression  7.314754  0.681058

Best Selection for this dataset: Decision Tree
   ```

### 15.md
**Description:** Provides notes on choosing models for stock market prediction, discussing tree-based and ensemble models with their pros and cons.  
