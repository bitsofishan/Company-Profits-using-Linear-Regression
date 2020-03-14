# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:20:40 2020

@author: Ishan
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
companies=pd.read_csv(r'C:\Users\Ishan\Documents\Python Scripts\Datasets\1000 Companies- Linear Regression.csv')
X=companies.iloc[:,:-1].values
y=companies.iloc[:,4].values
print(companies.head)
sns.heatmap(companies.corr())

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
linmodel=LinearRegression()
linmodel.fit(X_train,y_train)

y_pred=linmodel.predict(X_test)

print(linmodel.coef_)
print(linmodel.intercept_)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)*100)