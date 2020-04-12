# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:47:42 2020

@author: ZenithVIIV
"""

# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd 

#importing datasets  
data_set= pd.read_csv('Carpurchased.csv')  
print(data_set)

#Extracting Independent and dependent Variable  
x= data_set.iloc[:, :-1].values  
print(x)
y= data_set.iloc[:, 4].values 
print(y)

#Catgorical data  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
labelencoder_x= LabelEncoder()  
x[:, 1]= labelencoder_x.fit_transform(x[:,1])  
onehotencoder= OneHotEncoder(categorical_features= [1])    
x= onehotencoder.fit_transform(x).toarray()  

#avoiding the dummy variable trap:  
#x = x[:, 1:] 
#print(x)

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0) 
print(x_train, x_test, y_train, y_test)

#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
print(regressor.fit(x_train, y_train)  )

#Predicting the Test set result;  
y_pred= regressor.predict(x_test) 
print(y_pred)