import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Definr mean and covariance matrix for the distribution
mean=[0,0]
cov = [[2,1],[1,2]]

#Generate a random dataset with the given mean and covariance matrix
data= np.random.multivariate_normal(mean, cov, 1000)

#Randomly rotate the dataset
theta = np.deg2rad(30)
rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]])
data = data @ rot_mat

#plot the dataset
plt.scatter(data[:,0], data[:,1], alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

theta = np.deg2rad(30)
print(theta)
print(rot_mat)

data2= np.random.multivariate_normal(mean, cov, 1000)
print(data2)

#plot the dataset
plt.scatter(data2[:,0], data2[:,1], alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()


#use the data generated above
X= data

#Instantiate the PCA model
pca= PCA(n_components=2)

# Fit and Transform the data
X_transformed = pca.fit_transform(X)

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
plt.scatter(X[:,0], X[:,1], alpha=0.5)
V= pca.components_
plt.quiver([0,0],[0,0],V[:,0],V[:,1], color=['r','g'], scale=4)
plt.show()