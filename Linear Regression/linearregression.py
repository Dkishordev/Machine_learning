import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


warnings.filterwarnings('ignore')

df = pd.read_csv(r"../Machine_learning/Data/advertising.csv")
print(df.head())


sns.heatmap(df.corr(), cmap="YlGnBu",  annot= True)
plt.show()

sns.pairplot(df, x_vars=['TV', 'Newspaper','Radio'], y_vars ='Sales', height=4, aspect=1, kind='scatter')
plt.show()

x=df[['Radio']]
y=df['Sales']


x_train, x_test, y_train, y_test =train_test_split(x, y, train_size = 0.70, test_size=0.30,random_state=100)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train_sm =sm.add_constant(x_train)
lr=sm.OLS(y_train, x_train_sm).fit()

print(lr.params)

print(lr.summary())

x_test_sm =sm.add_constant(x_test)
y_pred =lr.predict(x_test_sm)


print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared: ", r2_score(y_test,y_pred))

plt.scatter(x_train, y_train)
plt.plot(x_train, 4.42 + 0.12*x_train,'r')
plt.xlabel('Features')
plt.ylabel('Target (y_train)')
plt.title('Scatter Plot of Features vs. Target')
plt.show()