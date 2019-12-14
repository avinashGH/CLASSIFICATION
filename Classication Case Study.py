# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:08:08 2019

@author: Microsoft
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix
import os
os.getcwd()
cd C:\Users\Microsoft\Downloads
os.getcwd()
data_income = pd.read_csv('income(1).csv')
data= data_income.copy()
data.head()
#check the variable data type
print(data.info())
#check missing values
data.isnull()
print('Data columns with null values:\n',data.isnull().sum())
#Summary of numerical variables
summary_num=data.describe()
print(summary_num)
#Summary of categorical variables
summary_cate=data.describe(include='O')
print(summary_cate)
#Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()
#Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
#There exists ? instead  of nan.
data=pd.read_csv('income(1).csv',na_values=[" ?"])
#Data Preprocessing
data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
data2=data.dropna(axis=0)
#Relationship between variables
correlation=data2.corr()
#There is no correlation between numerical variables
#Extracting the column names
data2.columns
#Gender proportion table
gender=pd.crosstab(index=data2["gender"],columns="count",normalize=True)
print(gender)
#Gender vs salary status
gender_salstat=pd.crosstab(index=data2["gender"],columns=data2["SalStat"],margins=True,normalize='index')
print(gender_salstat)
#Frequncy distribution of salary status
SalStat=sns.countplot(data2["SalStat"])
#Histogram of age
sns.boxplot("SalStat","age",data=data2)
data2.groupby("SalStat")["age"].median

##  Exporatory data analysis
#JobType vs SalStat
#Education vs SalStat
#occupation vs Salstat
#Capital gain
#Capital loss
#Hours per week vs SalStat

##  Case study on Classification
#Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
new_data=pd.get_dummies(data2,drop_first=True)
#Storing the column names
column_list=list(new_data.columns)
print(column_list)
#Seprating the input names from the data
features=list(set(column_list)-set(['SalStat']))
print(features)
#Storing the output values in y 
y=new_data['SalStat'].values
print(y)
#Storing the values from input features
x=new_data[features].values
print(x)
#Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#Make an instance of model
logistic=LogisticRegression()
#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_
#Prediction from the data
prediction=logistic.predict(test_x)
print(prediction)
#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)
#Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
#Printing the misclassifeid values from the prediction
print('Misclassified samples:%d'%(test_y!=prediction).sum())

#Logistic Regression- Removing Insignificant Variables
#Reindexing the salary status names to 0,1
#Already indexed above
print(data2['SalStat'])
cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(new_data,drop_first=True)
#Storing the column names 
column_list=list(new_data.columns)
print(column_list)
#Separatinf the input names from data
features=list(set(column_list)-set(['SalStat']))
print(features)
#set the output values in y
y=new_data['SalStat'].values
print(y)
#storing the values from input features
x=new_data[features].values
print(x)
#splitting the data into train and text
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#Make an instance of model 
logistic=LogisticRegression()
#Fitting the values from x and y
logistic.fit(train_x,train_y)
#prediction from test data
prediction=logistic.predict(test_x)
print(prediction)
#Calculate the accuracy
accuracy_score= accuracy_score(test_y,prediction)
print(accuracy_score)
#printing the misclassified values from prediction 
print('Misclassified samples: %d'%(test_y!=prediction).sum())
#KNN 
#importing the library of knn
from sklearn.neighbors import KNeighborsClassifier
#importing library  for plotting
import matplotlib.pyplot as plt
#storing the knn classifier
knn_classifier=KNeighborsClassifier(n_neighbors=5)
#fitting the values for x and y
knn_classifier.fit(train_x,train_y)
#predicting the test values with model
prediction = knn_classifier.predict(test_x)
print(prediction)
#Performance matrix check 
confusion_matrix=confusion_matrix(test_y,prediction)
print("\t","Predicted Values")
print("Original values","\n",confusion_matrix)
#Calculate the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print('Misclassified samples:%d'%(test_y!=prediction).sum())
#Effect of k values on classifier
Misclassified_sample=[]
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y!=pred_i).sum())
    
print(Misclassified_sample)
