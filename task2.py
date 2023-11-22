import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the customer dataset
customer_data = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(customer_data.head())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Use the Elbow Method to find the optimal number of clusters (K)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Choose the optimal K based on the Elbow Method
optimal_k = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Analyze the characteristics of each cluster
cluster_summary = customer_data.groupby('Cluster').mean()

# Display the summary statistics for each cluster
print(cluster_summary)

# Create a pair plot to visualize the clusters
sns.pairplot(customer_data, hue='Cluster', palette='viridis')
plt.show()
