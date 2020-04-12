# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:58:02 2020

@author: ZenithVIIV
"""


import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  


data_set= pd.read_csv('C:/Users/Prajwal Gowda/Documents/Carpurchased.csv')  
data1=data_set.copy()
x= data1.drop["CarPurchased"]
print(x)
y= data_set["CarPurchased"]  
print(y)

# Splitting the dataset into training and test set.  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0) 
print(x_train, x_test, y_train, y_test) 

#Fitting the Simple Linear Regression model to the training dataset  
regressor= LinearRegression()  
print(regressor.fit(x_train, y_train)  )

#Prediction of Test and Training set result  
y_pred= regressor.predict(x_test)  
print(y_pred)
x_pred= regressor.predict(x_train)  
print(x_pred)
