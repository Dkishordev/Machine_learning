import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv(r"../Machine_learning/Data/iris.csv")
X=df.iloc[:, :-1,].values

method='ward'
Z= linkage(X,method)

plt.figure(figsize=(10,5))
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering dendogram')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.show()