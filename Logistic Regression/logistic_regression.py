import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
file_path = os.path.join(BASE_DIR,  'Data', 'diabetes.csv')

df = pd.read_csv(file_path)

print(df.head())
print(df.tail())

print(df.info())

print(df.columns)

feature_cols=['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
x=df[feature_cols]
y=df.Outcome

x_train, x_test, y_train, y_test =train_test_split(x, y, train_size = 0.75, test_size=0.25,random_state=16)

logreg= LogisticRegression(random_state=16, max_iter=1000)
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

cnf_matrix=metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cnf_matrix, cmap="Blues",  annot= True, fmt='g', cbar =False)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.text(0.5, 0.1, 'TN', ha='center', va='center', color ='white')
plt.text(0.5, 1.1, 'FP', ha='center', va='center')
plt.text(1.5, 0.1, 'FN', ha='center', va='center')
plt.text(1.5, 1.1, 'TP', ha='center', va='center')

plt.show()


target_names=['without_diabetes','with_diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))


logreg.fit(x_train, y_train)

y_prob = logreg.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc= auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area =%0.2f)' %roc_auc)
plt.plot([0,1],[0,1], color='navy',lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


def sigmoid(x):
    return 1/(1 + np.exp(-x))

feature_name='BMI'
x_feature= df[feature_name].values.reshape(-1, 1)

logreg.fit(x_feature, y)

x_values= np.linspace(np.min(x_feature), np.max(x_feature), 100).reshape(-1,1)
y_values =sigmoid(logreg.coef_ * x_values + logreg.intercept_).ravel()

plt.figure(figsize=(8,6))
plt.plot(x_values, y_values, color='blue', label ='Sigmoid Curve')

plt.scatter(x_feature, y, color ='red', label='Data Points')

plt.xlabel(feature_name)
plt.ylabel('Predicted Probability')
plt.title('Logistic Regression Sigmoid Function with Data Points for ' + feature_name)
plt.legend()
plt.grid(True)
plt.show()