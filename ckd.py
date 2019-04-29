# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:30:47 2019

@author: TITANS
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt
y=[]
x=[]



    
for i in range (-5,5):
    x.append(i)
    i=i*-1
    y.append(1/(1+(m.exp(i))))
    
       
plt.plot(x,y)
plt.scatter(x,y)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics 


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

logreg = LogisticRegression( )
logreg.fit(x_train,y_train)
y_pred_Logreg=logreg.predict(x_test)

from sklearn import metrics
confusion_mat_Logreg=metrics.confusion_matrix(y_test,y_pred_Logreg)#confusion matrix
accu_Logreg=(metrics.accuracy_score(y_test,y_pred_Logreg))*100#  accuracy of model
print("accuracy with LOGISTIC REGRESSION :",accu_Logreg,"%")

from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,y_pred_Logreg))

from sklearn.externals import joblib
joblib.dump(logreg, 'CKD_LOGREG_MOD.pkl')# model save at 97.5% accuracy
"""
# Load the model from the file 
knn_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
knn_from_joblib.predict(X_test)"""

##for saving confusion matrix


for matrix in confusion_mat_Logreg:
    
    fig = plt.figure()
    plt.matshow(confusion_mat_Logreg)
    plt.title('Problem 1: Confusion Matrix CKD PATIENTS')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    fig.savefig('confusion_matrix_LOGREG_CKD.jpg')
    plt.show()




##using knn classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

x=[]
y=[]
case_list={}
for i in range (3,30,2):
    
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred_knn=knn.predict(x_test)
    confusion_mat_knn=metrics.confusion_matrix(y_test,y_pred_knn)#confusion matrix
    
    accu_knn=(metrics.accuracy_score(y_test,y_pred_knn))*100
    #print("accuracy with KNN :",accu_knn,"%")
    print(i,accu_knn)
    x.append(i)
    y.append(accu_knn)
    #case={i:accu_knn}
    #case_list.update(case)
from sklearn.externals import joblib
joblib.dump(knn, 'knn_simp9_mod.pkl')# model save at 98.3333% accuracy
"""
# Load the model from the file 
knn_from_joblib = joblib.load('filename.pkl')  
  
# Use the loaded model to make predictions 
knn_from_joblib.predict(X_test)"""
    

'''
best k using grid search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn= KNeighborsClassifier()

from sklearn.grid_search import GridSearchCV
k_range = list(range(2, 31))
print(k_range)
param_grid = dict(n_neighbors=k_range)
print(param_grid)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(x_train,y_train)
grid.grid_scores_

grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)

plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

knn= KNeighborsClassifier( n_neighbors=11)
x_train, x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=2)# gives data for training and testing
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

from sklearn import metrics
confusion_mat_knn=metrics.confusion_matrix(y_test,y_pred)#confusion matrix
acc = accuracy_score(y_test,y_pred) * 100

'''

    
y.sort()
print(max(y))    
knn_accuracy=max(y)   
plt.plot(x,y)
plt.scatter(x,y)


##################
#K fold KNN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

names = ['x', 'y', 'class']

# loading training data

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
#df = pd.read_csv(r"C:\Users\TITANS\Desktop\IN PROGRESS\kidney_disease.csv" )
#print(df.head())

# create design matrix X and target vector y
X = np.array(dataset.iloc[:, 0:24]) # end index is exclusive
y = np.array(dataset['classification']) # showing you two ways of indexing a pandas df


# ### Simple Cross Validation 

# In[26]:
#k=11 , 74%

# split the data set into train and test
X_1, X_test, y_1, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.3)
a=[]
for i in range(1,30,2):
    # instantiate learning model (k = 30)
    knn = KNeighborsClassifier(n_neighbors=i)

    # fitting the model on crossvalidation train
    knn.fit(X_tr, y_tr)

    # predict the response on the crossvalidation train
    pred = knn.predict(X_cv)

    # evaluate CV accuracy
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    print('\nCV accuracy for k = %d is %d%%' % (i, acc))
    a.append(acc)
k_simple=a.index(max(a))+(a.index(max(a))+1)
    
knn = KNeighborsClassifier(n_neighbors=k_simple)
knn.fit(X_tr,y_tr)
pred = knn.predict(X_test)
acc = accuracy_score(y_test, pred, normalize=True) * float(100)
print('\n****Test accuracy  is %d%%' % (acc),'for k=',k_simple)
from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,pred))




for i in range(0,24):
    plt.scatter(X[:,i],y)
    plt.show()
    





plt.scatter(X,y)
plt.show()


# ### 10 fold cross validation 

# In[21]:




# creating odd list of K for KNN
myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors[1:]:
    knn = KNeighborsClassifier(n_neighbors=k)
    for i in range(2,11):
        scores = cross_val_score(knn, X_tr, y_tr, cv=i, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(scores.mean(),i,k)
        
print(max(cv_scores))

# changing to misclassification error
MSE=[1 - x for x in cv_scores]
print(MSE)
print(len(MSE))

# determining best k
print(min(MSE))
print(MSE.index(min(MSE)))
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

# plot misclassification error vs k 
plt.plot(neighbors[1:], MSE)

for xy in zip(neighbors, np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

print("the misclassification error for each k value is : ", np.round(MSE,3))


# In[22]:


# ============================== KNN with k = optimal_k ===============================================
# instantiate learning model k = optimal_k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn_optimal.fit(X_tr, y_tr)

# predict the response
pred = knn_optimal.predict(X_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, acc))
from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,pred))

#k=9, 70%   














#querry

a1= float(input("Enter age in years :\n"))
a3=float(input("Enter blood pressure in mm/Hg:\n"))
a4=float(input("Enter specific gravity :\n"))
a5=float(input("Enter albumin :\n"))
a6=float(input("Enter sugar :\n"))
a7=float(input("Enter red blood cells normal or abnormal :\n"))
a8=float(input("Enter pus cell normal or abnormal :\n"))
a9=float(input("Enter pus cell clumps present or notpresent :\n"))
a10=float(input("Enter bacteria present or notpresent :\n"))
a11=float(input("Enter blood glucose random in mgs/dl :\n"))
a12=float(input("Enter blood urea in mgs/dl :\n"))

a13= float(input("Enter serum creatinine in mgs/dl :\n"))
a14=float(input("Enter sodium  in mEq/L:\n"))
a15=float(input("Enter potassium in mEq/L :\n"))
a16=float(input("Enter hemoglobin in gms:\n"))
a17=float(input("Enter packed cell volume :\n"))
a18=float(input("Enter white blood cell count in cells/cumm :\n"))
a19=float(input("Enter red blood cell count in millions/cmm :\n"))
a20=float(input("Enter hypertension-yes or no :\n"))
a21=float(input("Enter diabetes mellitus yes or no :\n"))
a22=float(input("Enter coronary artery disease yes or no :\n"))
a23=float(input("Enter appetite good or poor :\n"))
a24=float(input("Enter pedal edema yes or no:\n"))
a25=float(input("Enter anemia yes or no :\n"))

querry_case=np.array([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25],])

querry_pred_Logreg=logreg.predict(querry_case)
print(querry_pred_Logreg)




