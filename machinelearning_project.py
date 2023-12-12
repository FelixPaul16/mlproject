# -*- coding: utf-8 -*-
"""MachineLearning_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gCd_MJk3V0DqLXKEbjfzSZge_fWi0XEy

***Heart Failure Dataset***

**About dataset**

**Heart failure** is a common event caused by Cardiovascular diseases (CVD's) and this dataset contains 11 features that can be used to predict a possible heart disease.People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

**Importing the Library or Packages**
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")

"""**Listing The Directory Item**"""

import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/datasets/heart_failure_ml_project.csv")
df

"""**Shape of the Dataset**"""

df.shape    #this dataset have 918 rows and 12 columns

from google.colab import drive
drive.mount('/content/drive')

"""**To check the data types**"""

df.dtypes  #it is used to check the data types



"""**To check any missing values**"""

df.isna().sum() #it is used to check any missing values in the columns

"""**Visualization**"""

sns.countplot(x='HeartDisease',data=df)  #to create a count plot based on the 'Heart Diease' from the dataframe df.

sns.countplot(x='HeartDisease',data=df,hue='Sex')
#to create a count plot based on the 'Heart Diease' from the dataframe df, it uses 'sex' column to differ.here it show the heart disease based on both male and female.

"""**LabelEncoder**"""

le=LabelEncoder()     #it is a technique which is used to convert categorical values into fixed number
df['Sex']=le.fit_transform(df['Sex'])
df['ChestPainType']=le.fit_transform(df['ChestPainType'])
df['RestingECG']=le.fit_transform(df['RestingBP'])
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=le.fit_transform(df['ST_Slope'])
df

df.dtypes

"""**Seperate X and Y**"""

#independent variables also known as features
X=df.iloc[:,:-1]    #from this ,it creates a new dataframe X that contain all rows and all columns from original dataframe,except the last column.
X

X.shape

#target(dependent variables)
y=df.iloc[:,-1]     #here the last column is selected.
y

"""**MinMaxScaler**"""

scaler=MinMaxScaler()  #the term "scaler" typically refers to a preprocessing step that is applied to the input data before feeding it into a machine learning algorithm.
X=scaler.fit_transform(X)
X
#The goal of scaling is to standardize or normalize the features of the input data, ensuring that they are on a similar scale.
#This method scales the features to a specific range, usually between 0 and 1.

X.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)
#The dataset is divided into two subsets: one for training the model and another for testing its performance.
# The primary goal is to assess how well the model generalizes to new, unseen data.

X_train.shape

knn=KNeighborsClassifier()
sv=SVC()
gu=GaussianNB()
ds=DecisionTreeClassifier()
rf=RandomForestClassifier(random_state=1)
ad=AdaBoostClassifier(random_state=1)
algo_accuracy=[]

models=[knn,sv,gu,ds,rf,ad]
for model in models:
  print("*********",model,"*******")
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  print(classification_report(y_test,y_pred))
  print('Accuracy score=',accuracy_score(y_test,y_pred))
  algo_accuracy.append(accuracy_score(y_test,y_pred)*100)

print(algo_accuracy)

algo=['KNN','SVC','Gua','Dtree','Rforest','Adboost']
plt.bar(algo,algo_accuracy,width=0.4,color='g')
plt.plot()
plt.xlabel('Classification Algorithm')
plt.ylabel('Accuracy_Score')
plt.title('Classification algorithm vs Accuracy score')
plt.show()

"""Here RandomForest has the best accuracy"""

X_train.shape

y_train.shape

"""**HyperParameter Tuning**

 the process of finding the best set of hyperparameters for a machine learning
model.




"""

# rf1=RandomForestClassifier()
# params={'criterion': ['gini','entropy'],'n_estimators': [50, 100, 120, 150, 200],'random_state':[1,2,3,4,5]}
# clf=GridSearchCV(rf1,params,cv=10,scoring='accuracy')
# clf.fit(X_train,y_train)

# print(clf.best_params_)

# {'criterion': 'gini', 'n_estimators': 150, 'random_state': 1}

rf_new=RandomForestClassifier(criterion='gini',random_state=1,n_estimators=150)
rf_new.fit(X_train,y_train)
y_pred=rf_new.predict(X_test)
y_pred
print(classification_report(y_test,y_pred))
#In Random Forest hyperparameter tuning, criterion, n_estimators, and random_state are key parameters that can significantly impact the performance and behavior of the model.
#n_estimators specifies the number of trees in the Random Forest. Increasing the number of trees generally improves the model's performance up to a certain point.
#The random_state parameter is used to set the seed for the random number generator.

y.value_counts()

"""**Oversampling**"""

oversample=SMOTE(random_state=1)      #Oversampling is a technique used in the context of imbalanced datasets.
X_os,y_os=oversample.fit_resample(X,y)           #Oversampling involves increasing the number of instances in the minority class to balance the class distribution.

y_os.value_counts()

X_train_os,X_test_os,y_train_os,y_test_os=train_test_split(X_os,y_os,random_state=3,test_size=0.3)

scaler=MinMaxScaler()
X_train_os=scaler.fit_transform(X_train_os)
X_test_os=scaler.transform(X_test_os)

knn_os=KNeighborsClassifier()
sv_os=SVC()
nb_os=GaussianNB()
rf_os=RandomForestClassifier()
algo_accuracy=[]
l=[sv_os,knn_os,nb_os,rf_os]
for i in l:
  print('*********************************',i,'****************************************')
  i.fit(X_train_os,y_train_os)
  y_pred_os=i.predict(X_test_os)
  print(classification_report(y_test_os,y_pred_os))
  print('Accuracy=',accuracy_score(y_test_os,y_pred_os))
  algo_accuracy.append(accuracy_score(y_test_os,y_pred_os)*100)

algo=['KNN','SVC','Gua','Rforest']
plt.bar(algo,algo_accuracy,width=0.4,color='g')
plt.plot()
plt.xlabel('Classification Algorithm')
plt.ylabel('Accuracy_Score')
plt.title('Accuracy in Oversampling')
plt.show()

"""Best result= **RandomForestClassifier**"""



"""**Undersampling**"""

Undersampler=RandomUnderSampler(random_state=1)
X_us,y_us=Undersampler.fit_resample(X,y)
#Undersampling involves reducing the number of instances in the majority class to balance the class distribution.

y_us.value_counts()

X_train_us,X_test_us,y_train_us,y_test_us=train_test_split(X_us,y_us,random_state=3,test_size=0.3)

X_train_us=scaler.fit_transform(X_train_us)
X_test_us=scaler.transform(X_test_us)

knn_us=KNeighborsClassifier()
sv_us=SVC()
nb_us=GaussianNB()
rf_us=RandomForestClassifier()
alg_accuracy=[]
l=[sv_us,knn_us,nb_us,rf_us]
for i in l:
  print('*********************************',i,'****************************************')
  i.fit(X_train_us,y_train_us)
  y_pred_us=i.predict(X_test_us)
  print(classification_report(y_test_us,y_pred_us))
  print('Accuracy=',accuracy_score(y_test_us,y_pred_us))
  alg_accuracy.append(accuracy_score(y_test_us,y_pred_us)*100)

algo=['KNN','SVC','Gua','Rforest']
plt.bar(algo,alg_accuracy,width=0.4,color='g')
plt.plot()
plt.xlabel('Classification Algorithm')
plt.ylabel('Accuracy_Score')
plt.title('Accuracy in Undersampling')
plt.show()

"""Best result=**RandomForestClassifier**"""



"""**Prediction**"""

# y_predict=rf_new.predict(scaler.transform([[57,1,0,130,131,0,31,115,1,1.2,1]]))
# y_predict

import pickle

filename = 'model.pkl'
pickle.dump(rf_new,open(filename,'wb'))

scalername = 'minmax.pkl'
pickle.dump(scaler,open(scalername,'wb'))

!pip install streamlit

!wget -q -O - - ipv4.icanhazip.com

!streamlit run app.py & npx localtunnel --port 8501