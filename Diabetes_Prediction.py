#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:33:02 2024

@author: DrVaniV
"""

# importing libraries

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#hyperparameter tuning using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix,accuracy_score
# read the file , print no. of  rows and columns and 5 rows using head
data = pd.read_csv('data/diabetes.csv')
data.shape
data.head()

# check whether there is a null value
data.isnull().values.any()

#check the correlation

corrmat = data.corr()
top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='RdYlGn')

# Incase if the Outcome col is bool then 
#diabetes_map ={True:1,False:0}
#data['Outcome']= data['Outcome'].map(diabetes_map)

data.info()

# Train Test Split
feature_column=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
prediction_class=['Outcome']

X = data[feature_column].values
y = data[prediction_class].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.35, random_state=10)

# check for missing values
print("total number of rows = {0}".format(len(data)))
print("Number of rows missing  pregnancies {0}".format(len(data.loc[data['Pregnancies']==0])))
print("Number of rows missing  Glucose {0}".format(len(data.loc[data['Glucose']==0])))
print("Number of rows missing  BloodPressure{0}".format(len(data.loc[data['BloodPressure']==0])))
print("Number of rows missing  SkinThickness{0}".format(len(data.loc[data['SkinThickness']==0 ])))
print("Number of rows missing  Insulin {0}".format(len(data.loc[data['Insulin']==0])))
print("Number of rows missing  BMI {0}".format(len(data.loc[data['BMI']==0])))
print("Number of rows missing  DiabetesPedigreeFunction {0}".format(len(data.loc[data['DiabetesPedigreeFunction']==0])))
print("Number of rows missing  Age {0}".format(len(data.loc[data['Age']==0])))

      
# replace missing values with mode
fill_values = SimpleImputer(missing_values = 0, strategy='most_frequent')
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)

# Apply Ensemble algorithm - RandomForest 
model = RandomForestClassifier(random_state=10)
model.fit(X_train,y_train.ravel())
y_predict= model.predict(X_test)
print("Accuracy: {0:.3f}".format(metrics.accuracy_score(y_test, y_predict)))


#Hyperparameter tuning
params = {"learning_rate" :[0.05,0.10,0.15,0.20,0.25,0.30],
          "max_depth": [3,4,5,6,8,10,12,15],
          "min_child_weight" : [1,3,5,7],
          "gamma":[0.0,0.1,0.2,0.3,0.4],
          "colsample_bytree" : [0.3,0.4,0.5,0.7]}
classifier = XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train, y_train)
random_search.best_estimator_
classifier=XGBClassifier(
              colsample_bytree=0.4, 
              enable_categorical=False, 
              gamma=0.3, learning_rate=0.1, max_depth=12, 
              min_child_weight=7, 
              n_estimators=100)
classifier.fit(X_train,y_train)
score = cross_val_score(classifier, X_train,y_train.ravel(),cv=10)
score.mean()
y_predict = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print("Confusion Matrix: {0}".format(cm))
print("Accuracy {0}".format(acc))
