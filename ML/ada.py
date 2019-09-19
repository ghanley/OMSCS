import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split, validation_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.datasets import load_iris, load_digits
from sklearn import tree
import matplotlib.pyplot as plt
import util
import warnings
warnings.filterwarnings('ignore')

#class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)#
def run_ada(X, y, X_train, X_test, y_train, y_test, title):
  # param_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  # param_range = [1,5,10,15,20,25,30,35,40,45,50]
  # param_name = 'n_estimators'

  param_range = [.0001, .001, .01, .1, 1, 10]
  param_name = 'learning_rate'

  # data = load_digits()

  # X = data.data
  # y = data.target

  dt = tree.DecisionTreeClassifier(max_depth =10)
  # class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)[source]¶
  ada = AdaBoostClassifier(dt)


  util.plot_learning_curve(ada, title + " ADA LC", X, y, cv=5, n_jobs=-1)


  train_scores, test_scores = validation_curve(ada, X, y, param_name, param_range, cv=5, scoring="accuracy", n_jobs =-1 )


  # Calculate mean and standard deviation for training set scores
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)

  # Calculate mean and standard deviation for test set scores
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)

  plt.figure()
  # Plot mean accuracy scores for training and test sets
  # plt.plot(param_range, train_mean, 'o-', label="Training score", color="g")
  # plt.plot(param_range, test_mean, 'o-', label="Cross-validation score", color="r")
  lw = 2
  plt.semilogx(param_range, train_mean, label="Training score",
                  color="darkorange", lw=lw)
  # plt.plot(param_range, train_mean, 'o-', label="Training score", color="g")
  plt.semilogx(param_range, test_mean, label="Cross-validation score",
                  color="navy", lw=lw)
  # plt.plot(param_range, test_mean, 'o-', label="Cross-validation score", color="r")



  # Create plot
  plt.title("Ada Boost "+title+ " Validation")
  plt.xlabel(param_name)
  plt.ylabel("Accuracy Score")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(title+'ADAvalidation.png')
  plt.figure()





# X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=100)

# clf = AdaBoostClassifier(n_estimators=75, random_state=100)

# clf.fit(X_train, y_train)

# y_predTrain = clf.predict(X_train)
# y_predTest = clf.predict(X_test)

# training_ada = accuracy_score(y_train, y_predTrain)
# testing_ada = accuracy_score(y_test, y_predTest)

# print("Ada Training Acc", training_ada)
# print("Ada Testing Acc", testing_ada)


if __name__ == '__main__':
  # data = load_iris()

  # X = data.data
  # y = data.target

  # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  # run_knn(X, y, X_train, X_test, y_train, y_test, "iris", 13)
  data = pd.read_csv('breast.csv')
  y = data['Class']
  X = data.drop(['Class'], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)
  run_ada(X, y, X_train, X_test, y_train, y_test, "BreastCancer LR")

  data = load_digits()

  X = data.data
  y = data.target

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  run_ada(X, y, X_train, X_test, y_train, y_test, "digits LR")
