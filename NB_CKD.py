# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:31:54 2019

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



from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(x_train, y_train)
#GaussianNB(priors=None, var_smoothing=1e-09)
y_predict_gnb=clf_gnb.predict(x_test)
print(clf_gnb.predict(x_test))
accu_gnb=(metrics.accuracy_score(y_test,y_predict_gnb))*100
confusion_mat_gnb=metrics.confusion_matrix(y_test,y_predict_gnb)#confusion matrix

from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,y_predict_gnb))

# saving a trained model
from sklearn.externals import joblib
joblib.dump(clf_gnb, 'CKD_GAUSSIAN_NB.pkl')# model save at 98.3333% accuracy
"""
# Load the model from the file 
knn_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
knn_from_joblib.predict(X_test)"""



#using bernoulli NB

from sklearn.naive_bayes import BernoulliNB
clf_bnb = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
clf_bnb.fit(x_train,y_train)
y_predict_bnb=clf_bnb.predict(x_test)
print(y_predict_bnb)
accu_bnb=(metrics.accuracy_score(y_test,y_predict_bnb))*100
confusion_mat_bnb=metrics.confusion_matrix(y_test,y_predict_bnb)#confusion matrix

from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,y_predict_bnb))

from sklearn.externals import joblib
joblib.dump(clf_bnb, 'CKD_BERNOULLI_NB.pkl')# model save at 95.83333% accuracy
"""
# Load the model from the file 
knn_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
knn_from_joblib.predict(X_test)"""



from sklearn.naive_bayes import MultinomialNB
clf_mnb = MultinomialNB()
clf_mnb.fit(x_train, y_train)
#GaussianNB(priors=None, var_smoothing=1e-09)
y_predict_mnb=clf_mnb.predict(x_test)
print(clf_mnb.predict(x_test))

accu_mnb=(metrics.accuracy_score(y_test,y_predict_mnb))*100
    
confusion_mat_mnb=metrics.confusion_matrix(y_test,y_predict_bnb)#confusion matrix
from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,y_predict_mnb))

from sklearn.externals import joblib
joblib.dump(clf_mnb, 'CKD_MULTINOMIAL_NB.pkl')# model save at 98.3333% accuracy
"""
# Load the model from the file 
knn_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
knn_from_joblib.predict(X_test)"""
























