import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split, validation_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.datasets import load_iris, load_digits
from sklearn import tree
import matplotlib.pyplot as plt
import util
import warnings
warnings.filterwarnings('ignore')


def run_dt(X, y, X_train, X_test, y_train, y_test, title):

  # clf = RandomForestClassifier(n_estimators=100, max_depth = 100, random_state = 100)
  # clf = tree.DecisionTreeClassifier()
  # clf.fit(X_train, y_train)
  # y_predTreeTrain = clf.predict(X_train)
  # y_predTreeTest = clf.predict(X_test)
  dt_learning = tree.DecisionTreeClassifier(max_depth=10)

  util.plot_learning_curve(dt_learning, title + "Dt Learning Curve", X, y, cv=5, n_jobs=-1)


  param_range = range(1,50)
  depth_scores = []

  dt = tree.DecisionTreeClassifier()
  # param_name = "max_depth"

  param_name = "min_samples_leaf"

  train_scores, test_scores = validation_curve(dt, X, y, param_name, param_range, cv=5, scoring="accuracy", n_jobs =-1 )


  # Calculate mean and standard deviation for training set scores
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)

  # Calculate mean and standard deviation for test set scores
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)

  plt.figure()
  # Plot mean accuracy scores for training and test sets
  plt.plot(param_range, train_mean, 'o-', label="Training score", color="g")
  plt.plot(param_range, test_mean, 'o-', label="Cross-validation score", color="r")

  # Plot accurancy bands for training and test sets
  # plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,alpha=.1, color="r")
  # plt.fill_between(param_range, test_mean - test_std, test_mean + test_std,alpha=.1, color="g")

  # Create plot
  plt.title("DT N Samples Leaf Validation Curve "+title)
  plt.xlabel("min_samples_leaf")
  plt.ylabel("Accuracy Score")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(title+'LeafDTvalidation.png')
  plt.figure()





  # param_grid = dict(max_depth=depth_range)
  # grid = GridSearchCV(dt, param_grid, cv=10, scoring='accuracy')

  # grid.fit(X,y)
  # pd.DataFrame(grid.cv_results_)
  # grid_mean_scores = grid.cv_results_['mean_test_score']
  # plt.figure()
  # plt.plot(depth_range, grid_mean_scores)
  # plt.xlabel('Tree Depth')
  # plt.ylabel('Accuracy')
  # plt.savefig(title + 'scoreByDepth.png')



  # training_for = accuracy_score(y_train, y_predTreeTrain)
  # testing_for = accuracy_score(y_test, y_predTreeTest)
  # print("RF Training ACC:", training_for)
  # print("RF Testing ACC:", testing_for)
  # grid = GridSearchCV(cv=10, estimator = RandomForestClassifier(n_estimators=100, max_depth = 100, random_state = 100), param_grid = {'n_estimators':[10,50,100], 'max_depth': [10,50,100]})
  # grid = grid.fit(X_train, y_train)
  # # print(grid.cv_results_)
  # # print(grid.best_estimator_)


  # print("Random Forest features:", clf.feature_importances_)
  # print ("sorted", clf.feature_importances_.argsort()[::-1])



if __name__ == '__main__':
  # data = load_iris()

  # X = data.data
  # y = data.target



  # run_dt(X, y, X_train, X_test, y_train, y_test, "iris")
  data = pd.read_csv('breast.csv')
  y = data['Class']
  X = data.drop(['Class'], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)
  run_dt(X, y, X_train, X_test, y_train, y_test, "BreastCancer")

  data = load_digits()

  X = data.data
  y = data.target

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  run_dt(X, y, X_train, X_test, y_train, y_test, "digits")
