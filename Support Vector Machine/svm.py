import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn import metrics
from sklearn.svm import SVC
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
file_path = os.path.join(BASE_DIR,  'Data', 'diabetes.csv')

df = pd.read_csv(file_path)

print(df.head())
print(df.tail())

print(df.info())

print(df.columns)

class_counts = df['Outcome'].value_counts()
print(class_counts)

plt.figure(figsize=(6,4))
plt.pie(class_counts, labels= class_counts.index, autopct='%1.1f%%',startangle =140)
plt.axis('equal')
plt.title('Distribution of Classes')
plt.show()

feature_cols=['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
x=df[feature_cols]
y=df.Outcome


sc=StandardScaler()
x= sc.fit_transform(x)


x_train, x_test, y_train, y_test =train_test_split(x, y, train_size = 0.70, test_size=0.30,random_state=25)


smote = SMOTE(sampling_strategy=1.0, random_state =42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print(y_train_resampled.value_counts())


print("Using Linear Kernal")
svm_model = SVC(kernel ='linear', random_state = 42)
svm_model.fit(x_train_resampled, y_train_resampled)
y_pred = svm_model.predict(x_test)
print(classification_report(y_test, y_pred))

print("Using rbf Kernal")
svm_model = SVC(kernel ='rbf', random_state = 42)
svm_model.fit(x_train_resampled, y_train_resampled)
y_pred = svm_model.predict(x_test)
print(classification_report(y_test, y_pred))

print("Using poly Kernal")
svm_model = SVC(kernel ='poly', random_state = 42)
svm_model.fit(x_train_resampled, y_train_resampled)
y_pred = svm_model.predict(x_test)
print(classification_report(y_test, y_pred))

print("Using sigmoid Kernal")
svm_model = SVC(kernel ='sigmoid', random_state = 42)
svm_model.fit(x_train_resampled, y_train_resampled)
y_pred = svm_model.predict(x_test)
print(classification_report(y_test, y_pred))


cnf_matrix=metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4,3))
sns.heatmap(cnf_matrix, cmap="Blues",  annot= True, fmt='g', cbar =False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



svm_model.fit(x_train_resampled, y_train_resampled)
y_prob = svm_model.decision_function(x_test)
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