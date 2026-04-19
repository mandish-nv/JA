import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/lr-Real-estate.csv')
cols = {
    'X1 transaction date': 'date',
    'X2 house age': 'age',
    'X3 distance to the nearest MRT station': 'mrt_dist',
    'X4 number of convenience stores': 'stores',
    'X5 latitude': 'lat',
    'X6 longitude': 'long',
    'Y house price of unit area': 'price'
}
df.rename(columns=cols, inplace=True)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 8))
correlation_matrix = df.drop(columns=['No']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('result/correlation_heatmap.png')

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='mrt_dist', y='price', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Impact of MRT Proximity on House Price')
plt.xlabel('Distance to Nearest MRT Station (m)')
plt.ylabel('House Price per Unit Area')
plt.savefig('result/price_vs_mrt.png')

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='stores', y='price', palette='viridis')
plt.title('House Price Distribution by Nearby Convenience Stores')
plt.xlabel('Number of Convenience Stores')
plt.ylabel('House Price per Unit Area')
plt.savefig('result/price_vs_stores.png')

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['long'], df['lat'], c=df['price'], cmap='magma', alpha=0.6)
plt.colorbar(scatter, label='House Price per Unit Area')
plt.title('Geospatial Price Distribution (Latitude/Longitude)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('result/geospatial_price.png')