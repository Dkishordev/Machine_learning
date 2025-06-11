from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv(r"../Machine_learning/Data/OnlineRetailByCustomer.csv")
print(df)

df= df[['TotalAmount' , 'NumberOfTransactions']]

print(df.head())

df=(df-df.mean())/df.std()

print(df.head())

kmeans= KMeans(n_clusters=3)
kmeans.fit(df)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_
print(centroids)

plt.scatter(df['TotalAmount'], df['NumberOfTransactions'],c=labels, cmap='brg')
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=200, c='k')
plt.xlabel('Standarized amount spent')
plt.ylabel('Standarized number ot transactions')
plt.title('k-means clustering of Online Retail By Customer dataset for k=3')
plt.show()