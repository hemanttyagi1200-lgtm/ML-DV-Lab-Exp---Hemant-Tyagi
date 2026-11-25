# Experiment 9: K-Means Clustering

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)

# Get cluster labels
labels = model.labels_

# Display results
print("Cluster labels:")
print(labels)

print("Cluster centers:")
print(model.cluster_centers_)

# Plot the first two features and color points by cluster
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.title("K-Means Clustering on Iris Dataset")
plt.show()
