from sklearn import svm
import numpy as np
import pandas as pd
import time
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
## My Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

# iris
# best score 0.9714285714285714
# best params {'C': 1, 'kernel': 'linear'}
# CV = 10

#BC
# best params {'C': 0.01, 'kernel': 'linear'}


# Digits
# best score 0.9801113762927606
# best params {'C': 1, 'kernel': 'linear'}


def run_svm(X, y, X_train, X_test, y_train, y_test, title):

  scaler = StandardScaler()
  X_trainStandard = scaler.fit_transform(X_train)
  X_testStandard = scaler.transform(X_test)

  # ############################################ Support Vector Machine ###################################################
  #  Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
  #  Create a SVC classifier and train it.

  svc_model = SVC(gamma='auto', kernel='linear')
  util.plot_learning_curve(svc_model, title + " SVM LC", X, y, cv=5, n_jobs=-1)
  # svc_model.fit(X_trainStandard, y_train)

  # y_predSVCTrain = svc_model.predict(X_trainStandard)
  # y_predSVCTest = svc_model.predict(X_testStandard)

  # Test its accuracy on the training set using the accuracy_score method.
  #: Test its accuracy on the test set using the accuracy_score method.


  #:Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
  #  Print the best params, using .best_params_, and print the best score, using .best_score_.
  #

  # training_svc = accuracy_score(y_train, y_predSVCTrain)
  # testing_svc = accuracy_score(y_test, y_predSVCTest)

  # param_range = [.0001,.001,.01,.1, 1, 10, 100]
  # param_name = "C"
  param_range = [1, 10, 100, 1000, 10000, 100000, 1000000]
  param_name = "max_iter"
  train_scores, test_scores = validation_curve(svc_model, X, y, param_name, param_range, cv=5, scoring="accuracy", n_jobs=-1 )


  # Calculate mean and standard deviation for training set scores
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)

  # Calculate mean and standard deviation for test set scores
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)

  plt.figure()
  # Plot mean accuracy scores for training and test sets
  lw = 2
  plt.semilogx(param_range, train_mean, label="Training score",
                color="darkorange", lw=lw)
  # plt.plot(param_range, train_mean, 'o-', label="Training score", color="g")
  plt.semilogx(param_range, test_mean, label="Cross-validation score",
                color="navy", lw=lw)
  # plt.plot(param_range, test_mean, 'o-', label="Cross-validation score", color="r")

  # Plot accurancy bands for training and test sets
  # plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,alpha=.1, color="r")
  # plt.fill_between(param_range, test_mean - test_std, test_mean + test_std,alpha=.1, color="g")

  # Create plot
  plt.title(title)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy Score")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(title+'timing.png')
  plt.figure()





  # grid = GridSearchCV(cv=10, estimator = SVC(gamma='auto'), param_grid = {'kernel':['linear', 'rbf'], 'C': [0.001,0.01,0.1,1,100]})
  # grid = grid.fit(X_trainStandard, y_train)


  # # print("SVM Training Acc", training_svc)
  # # print("SVM Testing Acc", testing_svc)

  # # print(grid.cv_results_)
  # # #
  # # y_predict = grid.predict(X_testStandard)
  # # print(y_predict)
  # # examine the best model
  # print("best score",grid.best_score_)
  # print("best params", grid.best_params_)
  # print("best estimator", grid.best_estimator_)
  # print(classification_report(y_test, y_predict))
  # util.plot_roc(y_test, y_predict)
  # y_predict.reshape(len(y_predict),1)
  # print(y_predict.shape)

  # skplt.metrics.plot_roc_curve(y_test,y_predict)
  # plt.show()


if __name__ == '__main__':
  # data = load_iris()

  # X = data.data
  # y = data.target

  # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  # run_svm(X, y, X_train, X_test, y_train, y_test, "SVM iris linear Timing")
  data = pd.read_csv('breast.csv')
  y = data['Class']
  X = data.drop(['Class'], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)
  run_svm(X, y, X_train, X_test, y_train, y_test, "SVM Breast Cancer Linear")
  data = load_digits()

  X = data.data
  y = data.target

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  run_svm(X, y, X_train, X_test, y_train, y_test, "SVM digits Linear ")
