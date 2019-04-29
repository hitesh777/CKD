# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 00:45:55 2019

@author: TITANS
"""
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv(r"C:\Users\TITANS\Downloads\session_244_AU1154_1.2.csv")

x=np.array(dataset["Total Population Slum"]).reshape(36,1) # independent variable dat

y=np.array(dataset["Slum Reported Towns"]).reshape(36,1) #dependent variable data
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)# gives data for training and testing


regressor=LinearRegression()#making object of regression model
regressor.fit(x_train,y_train)#model training
y_predict=regressor.predict(x_test)

m=regressor.coef_
c=regressor.intercept_

 
from sklearn import metrics as mt
y_predict=regressor.predict(pd.DataFrame(x_test))

ms=ms=mt.mean_squared_error(y_test,y_predict)
mas=mt.mean_absolute_error(y_test,y_predict)
rms=math.sqrt(mt.mean_absolute_error(y_test,y_predict))
print("MEAN SQUARE ERROR:",ms)
print("MEAN ABSOLUTE ERROR:",mas)
print("ROOT MEAN SQUARE ERROR:",rms)

 # now plotting

xx=[min(x),max(x)]
yy=[float(m*min(x)+c),float(m*max(x)+c)]

#plotting
plt.scatter(x,y)#plots actual values
plt.plot(xx,yy)#plots predicted value
plt.show()

#NOW QUERRY
nx=int(input("ENTER VALUE OF X"))
op=float((m*nx)+c)

print("QUERRY INPUT:",nx)
print("BEST OUTPUT :",op)









