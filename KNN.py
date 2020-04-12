# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:51:20 2020

@author: ZenithVIIV
"""

 
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
  
#importing datasets  
data_set= pd.read_csv('C:/Users/Prajwal Gowda/Documents/Carpurchased.csv')  
  
#Extracting Independent and dependent Variable  
x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  
  
# Splitting the dataset into training and test set.  

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
print(x_train, x_test, y_train, y_test)
  
#feature Scaling  
  
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
print(x_train)
x_test= st_x.transform(x_test)  
print(x_test)

#Fitting K-NN classifier to the training set  
  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
print(classifier.fit(x_train, y_train)  )

#Predicting the test set result  
y_pred= classifier.predict(x_test) 
print(y_pred)
