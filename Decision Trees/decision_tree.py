# Import libraries for loading and splitting data
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Import libraries to run CART decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Import libraries to visualize the decision tree and the boundary regions in the data space
import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
import os

#select data directory to load
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
file_path = os.path.join(BASE_DIR,  'Data', 'loan_default_risk_dataset.csv')

# Load the data
df = pd.read_csv(file_path)
print(df.head())

#checking the dtypes and null values
print(df.info())

#checking the null values in the dataset
print(df.isnull().sum())

#dropping null values
df = df.dropna()

#re-checking the null values again to check if all the na values are removed
print(df.isnull().sum())

#checking the unique values in the string columns before converting it to numeric
df['Loan_Type'].unique()

 #checking the unique values in the string columns before converting it to numeric
print(df['Loan_Type'].unique())
print(df['Customer_Segment'].unique())

label_encoder = preprocessing.LabelEncoder()
df['Loan_Type']= label_encoder.fit_transform(df['Loan_Type'])
df['Customer_Segment']= label_encoder.fit_transform(df['Customer_Segment'])

cat_mapping = dict(enumerate(label_encoder.classes_))
print(cat_mapping)

#alternative method-1
#df[['Loan_Type','Customer_Segment']] = df[['Loan_Type','Customer_Segment']].apply(label_encoder.fit_transform)

#alternative method-2
# categorical_columns = ['Loan_Type','Customer_Segment']
# for col in categorical_columns:
#     df[col] = label_encoder.fit_transform(df[col])

#checking the values of the column to ensure if the values are transformed
print(df['Loan_Type'].unique())
print(df['Customer_Segment'].unique())

#Setting the X and y values. Y is the target column and X is the remaining columns
X = df.drop(['Loan_Default_Risk'], axis=1)
y = df['Loan_Default_Risk']

 #Splitting in train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
#checking the shape of the data
print(X_train.shape, X_test.shape)

# instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# fit the model
clf_gini.fit(X_train, y_train)

#Predicting the data
y_pred_gini = clf_gini.predict(X_test)

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

#Printing the predicted values
y_pred_train_gini = clf_gini.predict(X_train)
print(y_pred_train_gini)

 #calculating the accuracy of the training dataset
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))

 #Plotting the decision tree using matplot library (this will be black & white image)
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train))

#Plotting the decision tree using graphviz library (This will be more vibrant and visually enhanced)
dot_data= tree.export_graphviz(
    clf_gini,
    out_file=None,
    feature_names= X_train.columns,
    class_names=[str(c) for c in y_train.unique()],
    filled =True,
    rounded= True,
    special_characters=True 
)
graph=graphviz.Source(dot_data)
graph