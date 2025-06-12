import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

#Load the Iris dataset
iris = load_iris()

# Extract the features and Labels
X= iris.data
y= iris.target

# Instantiate a StandardScaler object
scaler = StandardScaler()

# Fit and transform the data using the scaler object
X_standardized = scaler.fit_transform(X)

#Instantiate PCA model
pca= PCA(n_components=4 )

# Fit and Transform the data
X_transformed = pca.fit_transform(X_standardized)

# Get eigenvectors and eigenvalues
eigenvectors = pca.components_
eigenvalues= pca.explained_variance_

#print eigenvectors and eigenvalues
print("Eigenvectors:\n", eigenvectors)
print("Eigenvalues:\n",eigenvalues)

#plot eigenvalues
plt.bar(range(len(eigenvalues)), eigenvalues)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

#Plot the data
plt.scatter(X_transformed[:,0], X_transformed[:,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
V= pca.components_
plt.axis('equal')
plt.show()

for component_num, component in enumerate(eigenvectors[:4], start =1):
    print(f"\nContributions to Principal Component {component_num}:")
    for feature_num, contribution in enumerate(component):
        feature_name = iris.feature_names[feature_num]
        print(f"{feature_name}: {contribution:.3f}")