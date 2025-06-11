#import libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df= pd.read_csv(r"../Machine_learning/Data/UNdata_Export_20250316_162701697.csv")

print(df.head())
print(df.tail())

print(df.info())

#removing unnecessary columns
del df['Quantity Name']
del df['Quantity']

#checking for null / NA values
print(df.isnull().sum())

#removing null values
df.dropna(inplace=True)
print(df.isnull().sum())

df.rename(columns={'Country or Area': 'Country'}, inplace=True)
print(df['Flow'].unique())

#labeling of data 
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
df['Flow'] = label_encoder.fit_transform(df['Flow'])
df['Commodity'] = label_encoder.fit_transform(df['Commodity'])
df['Country'] = label_encoder.fit_transform(df['Country'])

print(df['Flow'].unique())

#use only two attributes to perform K - means
X = df[['Trade (USD)', 'Weight (kg)']]

#finding effective value of K
inertia = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Sum of squared distances /Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

kmeans= KMeans(n_clusters=4)
df['Cluster']=kmeans.fit(X)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_

plt.scatter( df['Trade (USD)'], df['Weight (kg)'], c=labels, cmap='brg')
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, c='k')
plt.xlabel('Weight in Kilograms')
plt.ylabel('Trade value in USD')
plt.title('k-means clustering of Trading data for k = 4')
plt.show()
