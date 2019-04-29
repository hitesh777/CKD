# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:46:44 2019

@author: TITANS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
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

x_train, x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=None)# gives data for training and testing



from sklearn.svm import LinearSVC
clf_svc =LinearSVC()
clf_svc.fit(x_train, y_train)
#GaussianNB(priors=None, var_smoothing=1e-09)
y_predict_svc=clf_svc.predict(x_test)
print(clf_svc.predict(x_test))
accu_svc=(metrics.accuracy_score(y_test,y_predict_svc))*100
confusion_mat_svc=metrics.confusion_matrix(y_test,y_predict_svc)#confusion matrix
print("ACCURACY FOR SVM IS ",accu_svc,"%")

from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,y_predict_svc))

from pycm import *
cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_predict_svc) # Create CM From Data
cm.classes
cm.table
cm.Overall_ACC
cm.Kappa 
cm.FPR
cm.ACC
cm.FNR














from sklearn.svm import SVC
#clf = NuSVC((nu=0.5, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None))

clf=SVC(cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
          max_iter=-1, nu=0.5, probability=False, random_state=None,
          shrinking=True, tol=0.001, verbose=False)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test) 

print(clf.predict(x_test))
accu=(metrics.accuracy_score(y_test,y_pred))*100
confusion_mat=metrics.confusion_matrix(y_test,y_pred)#confusion matrix

from sklearn.metrics import classification_report   #for calculating precision recall f1-score abd support
print(classification_report(y_test,y_pred))





"""
from sklearn.grid_search import GridSearchCV
svm = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
clf = GridSearchCV(svm, parameters)
clf.fit(x_train,y_train)
print("accuracy:"+str(np.average(cross_val_score(clf, x_train, y_train, scoring='accuracy'))))

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(x_train,y_train)
grid.grid_scores_

grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
"""
"""
Overall Statistics : 

95% CI                                                           (0.30439,0.86228)
Bennett_S                                                        0.375
Chi-Squared                                                      6.6
Chi-Squared DF                                                   4
Conditional Entropy                                              0.95915
Cramer_V                                                         0.5244
Cross Entropy                                                    1.59352
Gwet_AC1                                                         0.38931
Joint Entropy                                                    2.45915
KL Divergence                                                    0.09352
Kappa                                                            0.35484
Kappa 95% CI                                                     (-0.07708,0.78675)
Kappa No Prevalence                                              0.16667
Kappa Standard Error                                             0.22036
Kappa Unbiased                                                   0.34426
Lambda A                                                         0.16667
Lambda B                                                         0.42857
Mutual Information                                               0.52421
Overall_ACC                                                      0.58333
Overall_RACC                                                     0.35417
Overall_RACCU                                                    0.36458
PPV_Macro                                                        0.56667
PPV_Micro                                                        0.58333
Phi-Squared                                                      0.55
Reference Entropy                                                1.5
Response Entropy                                                 1.48336
Scott_PI                                                         0.34426
Standard Error                                                   0.14232
Strength_Of_Agreement(Altman)                                    Fair
Strength_Of_Agreement(Cicchetti)                                 Poor
Strength_Of_Agreement(Fleiss)                                    Poor
Strength_Of_Agreement(Landis and Koch)                           Fair
TPR_Macro                                                        0.61111
TPR_Micro                                    



"""




"""
Class statics 
ACC(Accuracy)                                                    0.83333                 0.75                    0.58333                 
BM(Informedness or bookmaker informedness)                       0.77778                 0.22222                 0.16667                 
DOR(Diagnostic odds ratio)                                       None                    4.0                     2.0                     
ERR(Error rate)                                                  0.16667                 0.25                    0.41667                 
F0.5(F0.5 score)                                                 0.65217                 0.45455                 0.57692                 
F1(F1 score - harmonic mean of precision and sensitivity)        0.75                    0.4                     0.54545                 
F2(F2 score)                                                     0.88235                 0.35714                 0.51724                 
FDR(False discovery rate)                                        0.4                     0.5                     0.4                     
FN(False negative/miss/type 2 error)                             0                       2                       3                       
FNR(Miss rate or false negative rate)                            0.0                     0.66667                 0.5                     
FOR(False omission rate)                                         0.0                     0.2                     0.42857                 
FP(False positive/type 1 error/false alarm)                      2                       1                       2                       
FPR(Fall-out or false positive rate)                             0.22222                 0.11111                 0.33333                 
G(G-measure geometric mean of precision and sensitivity)         0.7746                  0.40825                 0.54772                 
LR+(Positive likelihood ratio)                                   4.5                     3.0                     1.5                     
LR-(Negative likelihood ratio)                                   0.0                     0.75                    0.75                    
MCC(Matthews correlation coefficient)                            0.68313                 0.2582                  0.16903                 
MK(Markedness)                                                   0.6                     0.3                     0.17143                 
N(Condition negative)                                            9                       9                       6                       
NPV(Negative predictive value)                                   1.0                     0.8                     0.57143                 
P(Condition positive)                                            3                       3                       6                       
POP(Population)                                                  12                      12                      12                      
PPV(Precision or positive predictive value)                      0.6                     0.5                     0.6                     
PRE(Prevalence)                                                  0.25                    0.25                    0.5                     
RACC(Random accuracy)                                            0.10417                 0.04167                 0.20833                 
RACCU(Random accuracy unbiased)                                  0.11111                 0.0434                  0.21007                 
TN(True negative/correct rejection)                              7                       8                       4                       
TNR(Specificity or true negative rate)                           0.77778                 0.88889                 0.66667                 
TON(Test outcome negative)                                       7                       10                      7                       
TOP(Test outcome positive)                                       5                       2                       5                       
TP(True positive/hit)                                            3                       1                       3                       
TPR(Sensitivity, recall, hit rate, or true positive rate) 
"""  









