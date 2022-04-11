# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:36:23 2022

@author: asus
"""

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import GridSearchCV 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#data_train = pd.read_csv('train-project.data',sep=",",header=None)
data_test = pd.read_csv('test-project.data',sep=",",header=None)
#unlabeled data
data_unl = pd.read_csv('unlabeled-project.data',sep=",",header=None)

#print(data_train)
print(data_test)
print(data_unl)
#print(data_train.values)
print(data_test.values)
print(data_unl.values)

#print(data_train.shape)
#print(data_train.isnull().sum())
print(data_test.shape)
#print(data_test.isnull().sum())
print(data_unl.shape)
#print(data_unl.isnull().sum())


#x_train = data_train.iloc[:, :-1]
#y_train = data_train.iloc[:, 15:]
x_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, 15:]
x_unl = data_unl

#drop columns that are irrelevant
#x_train = x_train.drop(columns = [3,5,6,9])
x_test = x_test.drop(columns = [3,5,6,9])
x_unl = data_unl.drop(columns = [3,5,6,9])

#fill NaN values with mean of its column
#x_train[0].fillna(x_train[0].median(), inplace =True)
#print(x_train)
x_test[0].fillna(x_test[0].median(), inplace =True)
#print(x_train)
x_unl[0].fillna(x_unl[0].median(), inplace =True)
#print(x_unl)

#Encode for attributes with 2 categories
LE = LabelEncoder()
#LE.fit(x_train[10])
#x_train[10]= LE.transform(x_train[10])
#print(x_train[10])
LE.fit(x_test[10])
x_test[10]= LE.transform(x_test[10])
LE.fit(x_unl[10])
x_unl[10]= LE.transform(x_unl[10])

#OneHotEncoding for 3 or more categories
X_unl = pd.get_dummies(x_unl,columns=[2], prefix=[2])
X1_unl = pd.get_dummies(X_unl,columns=[4], prefix=[4])
X2_unl = pd.get_dummies(X1_unl,columns=[7], prefix=[7])
X3_unl = pd.get_dummies(X2_unl,columns=[8], prefix=[8])
X_test = pd.get_dummies(x_test,columns=[2], prefix=[2])
X1_test = pd.get_dummies(X_test,columns=[4], prefix=[4])
X2_test = pd.get_dummies(X1_test,columns=[7], prefix=[7])
X3_test = pd.get_dummies(X2_test,columns=[8], prefix=[8])
X4_test = pd.get_dummies(X3_test,columns=[14], prefix=[14])
#X_unl = pd.get_dummies(x_tunl,columns=[2], prefix=[2])
#X1_unl = pd.get_dummies(X_unl,columns=[4], prefix=[4])
#X2_unl = pd.get_dummies(X1_unl,columns=[7], prefix=[7])
#X3_unl = pd.get_dummies(X2_unl,columns=[8], prefix=[8])
#X4_unl = pd.get_dummies(X3_unl,columns=[14], prefix=[14])
print(X3_unl)
print(X4_test)

#Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()  
X3_unl = sc.fit_transform(X3_unl)
X4_test = sc.fit_transform(X4_test)
#X4_unl = sc.fit_transform(X4_unl)

#Define classifiers

#GridSearh for best parameters in DT and NN
#Dcision Tree
parameter_space_dt = {
    'max_depth':[3,5,7,9,10],
#    'criterion':['gini','entropy']
}
dt =  tree.DecisionTreeClassifier()
clfDT = GridSearchCV(dt, parameter_space_dt, n_jobs=-1, cv=5)
#NeuralNetwork
parameter_space_mlp = {
    'hidden_layer_sizes': [(5,), (8,),  (5,3)],
#    'learning_rate': ['constant','adaptive']
}
mlp = MLPClassifier(max_iter=500,batch_size=50)
clfNN = GridSearchCV(mlp, parameter_space_mlp, n_jobs=-1, cv=5)
#SVM
#parameter_space_svm = {
#    'C': [0.1, 1, 10, 100, 1000],  
#    'kernel': ['poly'],
#    'degree':[1,2,3,4],
#}
#svm.SVC()
#svm = svm.SVC( probability=True)
#clfSVM = GridSearchCV(svm, parameter_space_svm, n_jobs=-1, cv=5, scoring='f1_macro')
clfSVM= svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, random_state=None, shrinking=True,
    tol=0.001, verbose=False, probability=True)
#NaiveBayes
clfNB = GaussianNB()
#RandomForest
clfRF = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=0)

#train the classifiers
clfDT.fit(X3_unl)
clfNN.fit(X3_unl)
clfSVM.fit(X3_unl)
clfNB.fit(X3_unl)
clfRF.fit(X3_unl)

#predictions and evaluation
y_test_pred_DT=clfDT.predict(X4_test)
y_test_pred_NN=clfNN.predict(X4_test)
y_test_pred_SVM=clfSVM.predict(X4_test)
y_test_pred_NB=clfNB.predict(X4_test) 
y_test_pred_RF=clfRF.predict(X4_test)

print('\nDecision Tree Best parameters found:\n', clfDT.best_params_)
best_DT=clfDT.best_estimator_;
print ('\nDecision Tree Best estimator found:\n', clfDT.best_estimator_)
print('\nNeural Net Best parameters found:\n', clfNN.best_params_)
best_DT=clfDT.best_estimator_;
print ('\nNeural Net Best estimator found:\n', clfNN.best_estimator_)
#print('\n\nBest parameters found (Ploynomial kernel):', clfSVM.best_params_)
print ('Support Vector: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_SVM, average='macro'))
print ('Support Vector: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_SVM, average='micro'))
print ('Random Forest: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RF, average='macro'))
print ('\n')

print('Report for Decision Trees:')
print(classification_report(y_test, y_test_pred_DT))
print('Report for Neural Nets:')
print(classification_report(y_test, y_test_pred_NN))
print('Report for Suport Vector Machine:')
print(classification_report(y_test, y_test_pred_SVM))
print('Report for Naive Bayes:')
print(classification_report(y_test, y_test_pred_NB))
print('Report for Random Forest:')
print(classification_report(y_test, y_test_pred_RF))

#ROC curve and proba
pr_y_test_pred_DT=clfDT.predict_proba(X4_test)
roc_auc = roc_auc_score(y_test, pr_y_test_pred_DT[:,1])
print('Logistic ROC AUC DT %.3f' % roc_auc)
metrics.plot_roc_curve(clfDT, X4_test, y_test)
plt.title('Receiver Operating Curve for Decision Trees')
plt.show()

pr_y_test_pred_NN=clfNN.predict_proba(X4_test)
roc_auc = roc_auc_score(y_test, pr_y_test_pred_NN[:,1])
print('Logistic ROC AUC NN %.3f' % roc_auc)
metrics.plot_roc_curve(clfNN, X4_test, y_test)
plt.title('Receiver Operating Curve for Neural Nets')
plt.show()

pr_y_test_pred_SVM=clfSVM.predict_proba(X4_test)
roc_auc = roc_auc_score(y_test, pr_y_test_pred_SVM[:,1])
print('Logistic ROC AUC SVM %.3f' % roc_auc)
metrics.plot_roc_curve(clfSVM, X4_test, y_test)
plt.title('Receiver Operating Curve for Support Vector Machines')
plt.show()

pr_y_test_pred_NB=clfNB.predict_proba(X4_test)
roc_auc = roc_auc_score(y_test, pr_y_test_pred_NB[:,1])
print('Logistic ROC AUC NB %.3f' % roc_auc)
metrics.plot_roc_curve(clfNB, X4_test, y_test)
plt.title('Receiver Operating Curve for Naive Bayes')
plt.show()

pr_y_test_pred_RF=clfRF.predict_proba(X4_test)
roc_auc = roc_auc_score(y_test, pr_y_test_pred_RF[:,1])
print('Logistic ROC AUC RF%.3f' % roc_auc)
metrics.plot_roc_curve(clfRF, X4_test, y_test)
plt.title('Receiver Operating Curve for Random Frest')
plt.show()