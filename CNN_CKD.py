# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:12:18 2019

@author: TITANS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics 
from sklearn import metrics


dataset=pd.read_csv(r"C:\Users\TITANS\Desktop\IN PROGRESS\kidney_disease.csv" )

# data preprocessing
dataset.replace(['nan','normal','abnormal','present','not present','yes','no','good','poor','ckd','notckd'],[0,0,1,1,0,1,0,1,0,1,0],regex=True,inplace=True)

dataset.replace(np.nan, 0, inplace=True)# to replace nan with zero over entire dataframe
del(dataset["id"])
dataset["pcv"] = pd.to_numeric(dataset["pcv"],errors='coerce')#for converting object typoe to float
dataset["wc"] = pd.to_numeric(dataset["wc"],errors='coerce')
dataset["rc"] = pd.to_numeric(dataset["rc"],errors='coerce')
dataset.fillna(dataset.mean())
#x=dataset["age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"]
pd.DataFrame(dataset).replace(np.nan, 0, inplace=True)
X = (dataset.iloc[:, :24])# selecting 24 features

#processed data
Y=np.array(dataset["classification"])
X=np.asarray(X)

x_train, x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=2)# gives data for training and testing

from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(random_state=None)
clf.fit(x_train,y_train)

y_pred= clf.predict(x_test)

print(clf.predict(x_test))
accu=(metrics.accuracy_score(y_test,y_pred))*100
confusion_mat=metrics.confusion_matrix(y_test,y_pred)#confusion matrix

from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,y_pred))

# saving a trained model
#from sklearn.externals import joblib
#joblib.dump(clf, 'CKD_CNN_MOD.pkl')# model save at 91.6666% accuracy
"""
# Load the model from the file 
knn_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
knn_from_joblib.predict(X_test)"""















