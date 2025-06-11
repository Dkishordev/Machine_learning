import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv(r"../Machine_learning/Data/iris.csv")


print(df.info())
print(df.head())
print(df.tail())
print(df.shape)


df1= df[['sepal_length' , 'sepal_width']]

kmeans= KMeans(n_clusters=3)
kmeans.fit(df1)

labels=kmeans.labels_
centroids=kmeans.cluster_centers_

plt.scatter(df1['sepal_length'], df1['sepal_width'],c=labels, cmap='brg')
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=200, c='k')
plt.xlabel('sepal_length (cm)')
plt.ylabel('sepal_width (cm)')
plt.title('k-means clustering of Iris dataset for k=3')
plt.show()


df2= df[['petal_length' , 'petal_width']]

kmeans= KMeans(n_clusters=3)
kmeans.fit(df2)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_

plt.scatter(df2['petal_length'], df2['petal_width'],c=labels, cmap='brg')
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=200, c='k')
plt.xlabel('petal_length (cm)')
plt.ylabel('petal_width (cm)')
plt.title('k-means clustering of Iris dataset for k=3')
plt.show()
