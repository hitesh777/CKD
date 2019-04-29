

#VISUALISATION
k=np.array([a1])
plt.scatter(X[:,0:1],Y)
plt.scatter(k,predicted,edgecolors='r')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()



##with knn classifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,test_size=0.3,random_state=2)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


for i in range (3,26,2):    
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred_knn=knn.predict(x_test)
    accu_knn=(metrics.accuracy_score(y_test,y_pred_knn))*100
    print(i,accu_knn)
confusion_mat_knn=metrics.confusion_matrix(y_test,y_pred_knn)#confusion matrix


print("accuracy with KNN :",accu_knn)










#classification using LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_logreg=logreg.predict(x_test)
confusion_mat_logreg=metrics.confusion_matrix(y_test,y_pred_logreg)#confusion matrix
accu_logreg=(metrics.accuracy_score(y_test,y_pred_logreg))*100
print("accuracy with LOGISTIC REGRESSION :",accu_logreg)



a1=float(input("0 ENTER sepal length in cm:\n"))
a2=float(input("1 ENTER sepal width in cm :\n"))
a3=float(input("2 ENTER petal hlengt in cm :\n"))
a4=float(input("3 ENTER petal width cm :\n"))



querry_case=np.array([[a1,a2,a3,a4],])
predicted = logreg.predict(querry_case)
print("Expected Output is: ",predicted)

if(predicted==0):
    print("CLASS :Iris-Setosa")
elif(predicted==1):
    print("CLASS : Iris-Versicolour")
elif(predicted==2):
    print("CLASS:Iris-Virginica ")










