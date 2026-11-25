# Experiment 5: Dimensionality Reduction using PCA

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Sample data with 3 features
X = np.array([
    [2, 4, 6],
    [3, 5, 7],
    [4, 6, 8],
    [5, 7, 9],
    [6, 8, 10]
])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce 3 features to 2 principal components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Display results
print("Original data shape:", X.shape)
print("Reduced data shape:", X_reduced.shape)
print("Reduced data:")
print(X_reduced)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
