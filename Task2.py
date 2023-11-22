import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.0)

# Convert the NumPy array to a DataFrame for better handling
customer_data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# Display the first few rows of the dataset
print(customer_data.head())

# Visualize the synthetic dataset
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Feature1', y='Feature2', data=customer_data, palette='viridis')
plt.title('Synthetic Customer Dataset')
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data)

# Analyze the characteristics of each cluster
cluster_summary = customer_data.groupby('Cluster').mean()

# Display the summary statistics for each cluster
print(cluster_summary)

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Feature1', y='Feature2', data=customer_data, hue='Cluster', palette='viridis')
plt.title('Customer Segmentation with K-Means Clustering')
plt.show()