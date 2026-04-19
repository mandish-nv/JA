import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/lr-Real-estate.csv')
features = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 
            'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
# 3 clusters -> 'Premium', 'Standard', and 'Entry-level' markets
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_profile = df.groupby('Cluster').mean().drop('No', axis=1)
print("Cluster Profiles (Average Values):")
print(cluster_profile)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X3 distance to the nearest MRT station', 
                y='Y house price of unit area', hue='Cluster', palette='Set1')
plt.title('K-Means Clustering: Market Segments by MRT Distance and Price')
plt.savefig('result/KMeans_Clustering.png')
plt.show()
