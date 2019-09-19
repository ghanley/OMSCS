from sklearn import svm
import numpy as np
import pandas as pd
from time import clock
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split, validation_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import roc_curve, auc
import util
import scikitplot as skplt
from collections import defaultdict


def run_ulti(X, y, X_train, X_test, y_train, y_test, title):
  values = defaultdict(dict)
  scaler = StandardScaler()
  X_trainStandard = scaler.fit_transform(X_train)
  X_testStandard = scaler.transform(X_test)
  values['train']['dt'] = 0
  values['test']['dt'] =0
  values['train']['ada'] = 0
  values['test']['ada'] = 0
  values['train']['svm'] = 0
  values['test']['svm'] = 0
  values['train']['svm normalized'] = 0
  values['test']['svm normalized'] = 0
  values['train']['ann normalized'] = 0
  values['test']['ann normalized'] = 0
  values['train']['ann'] = 0
  values['test']['ann'] = 0
  values['train']['knn'] = 0
  values['test']['knn'] = 0



  for i in range(1,10):

    dt = tree.DecisionTreeClassifier(max_depth=10)
    start = clock()
    dt.fit(X_train, y_train)
    values['train']['dt'] += clock() - start

    start = clock()
    y_pred = dt.predict(X_test)
    values['test']['dt']+= clock()-start
    values['results']['dt'] = accuracy_score(y_test, y_pred)


    ada = ada = AdaBoostClassifier(dt)
    start = clock()
    ada.fit(X_train, y_train)
    values['train']['ada'] += clock() - start

    start = clock()
    y_pred = ada.predict(X_test)
    values['test']['ada']+= clock()-start
    values['results']['ada'] = accuracy_score(y_test, y_pred)

    svm = SVC(gamma='auto', kernel='linear')
    start = clock()
    svm.fit(X_train, y_train)
    values['train']['svm'] += clock() - start

    start = clock()
    y_pred = svm.predict(X_test)
    values['test']['svm']+= clock()-start
    values['results']['svm'] = accuracy_score(y_test, y_pred)

    svm = SVC(gamma='auto', kernel='linear')
    start = clock()
    svm.fit(X_trainStandard, y_train)
    values['train']['svm normalized'] += clock() - start

    start = clock()
    y_pred = svm.predict(X_testStandard)
    values['test']['svm normalized']+= clock()-start
    values['results']['svm normalized'] = accuracy_score(y_test, y_pred)

    ann = MLPClassifier(solver='adam')
    start = clock()
    ann.fit(X_train, y_train)
    values['train']['ann'] += clock() - start

    start = clock()
    y_pred = ann.predict(X_test)
    values['test']['ann']+= clock()-start
    values['results']['ann'] = accuracy_score(y_test, y_pred)

    ann = MLPClassifier(solver='adam')
    start = clock()
    ann.fit(X_trainStandard, y_train)
    values['train']['ann normalized'] += clock() - start

    start = clock()
    y_pred = ann.predict(X_testStandard)
    values['test']['ann normalized']+= clock()-start
    values['results']['ann normalized'] = accuracy_score(y_test, y_pred)

    knn = KNeighborsClassifier()
    start = clock()
    knn.fit(X_train, y_train)
    values['train']['knn'] += clock() - start

    start = clock()
    y_pred = knn.predict(X_test)
    values['test']['knn']+= clock()-start
    values['results']['knn'] = accuracy_score(y_test, y_pred)

    values = pd.DataFrame(values)
    values.to_csv('./output/{}_timing.csv'.format(title))



if __name__ == '__main__':
  # data = load_iris()

  # X = data.data
  # y = data.target

  # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  # run_svm(X, y, X_train, X_test, y_train, y_test, "SVM iris linear Timing")
  data = pd.read_csv('breast.csv')
  y = data['Class']
  X = data.drop(['Class'], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2)
  run_ulti(X, y, X_train, X_test, y_train, y_test, "Breast Cancer")

  data = load_digits()

  X = data.data
  y = data.target

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2)

  run_ulti(X, y, X_train, X_test, y_train, y_test, "Digits")
