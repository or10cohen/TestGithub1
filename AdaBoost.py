import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import math


data_set = datasets.load_iris()
print(data_set.keys())
X = data_set.data
y = data_set.target
print(type(X))
##-----------------------------------AdaBoost DecisionTreeClassifier----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train_before = X_train.copy
abc = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',
                         random_state=None)
model_abc_dt = abc.fit(X_train, y_train)
y_pred_abc_dt = model_abc_dt.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_abc_dt))
##-----------------------------------AdaBoost SVC----------------------------------------------------
svc = SVC(probability=True, kernel='linear')
abc = AdaBoostClassifier(base_estimator=svc, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',
                         random_state=None)
model_abc_SVC = abc.fit(X_train, y_train)
y_pred_abc_SVC = model_abc_SVC.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_abc_SVC))
X_train = np.array(X_train)
print(type(X_train))
print('X_train_before', X_train_before)
print('X_train', X_train)
# print(np.subtract((X_train, X_train_before)))
# print(X_train - X_train_before)





