from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, validation_curve
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util
import warnings
warnings.filterwarnings('ignore')
import scikitplot as skplt

# plt.figure()
# skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
# plt.show()


def run_knn(X, y, X_train, X_test, y_train, y_test, title, k):
  knn_learning = KNeighborsClassifier(n_neighbors=k)

  util.plot_learning_curve(knn_learning, title + " KNN LC", X, y, cv=10, n_jobs=-1)


  #search for an optimal value of K for KNN
  #credit https://www.youtube.com/watch?v=6dbrR-WymjI
  param_range = range(1,31)
  k_scores = []

  # for k in k_range:
  #   knn = KNeighborsClassifier(n_neighbors=k)
  #   scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
  #   # print(k, scores.mean())
  #   k_scores.append(scores.mean())
  # print(k_scores)
  knn = KNeighborsClassifier()
  param_name = "n_neighbors"
  train_scores, test_scores = validation_curve(knn, X, y, param_name, param_range, cv=5, scoring="accuracy", n_jobs =-1 )


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
  plt.title("KNN Validation Curve "+title)
  plt.xlabel("Number of K Neighbors")
  plt.ylabel("Accuracy Score")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(title+'KNNvalidation.png')
  plt.figure()


  # util.plot_validation_curve(knn_learning, title + 'KNN by K', X, y, "n_neighbors", k_range)

  # param_grid = dict(n_neighbors=k_range)
  # grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

  # grid.fit(X,y)
  # pd.DataFrame(grid.cv_results_)
  # grid_mean_scores = grid.cv_results_['mean_test_score']
  # plt.figure()
  # plt.plot(k_range, grid_mean_scores)
  # plt.xlabel('Value of K for KNN')
  # plt.ylabel('Accuracy')
  # plt.savefig(title + 'scorebyK.png')



  #grid_score contains mean accuracy of CV, along with the SD of that score i.e. how wide the range was
  # y_predict = grid.predict(X_test)
  # print(y)
  # # examine the best model
  # print(grid.best_score_)
  # print(grid.best_params_)
  # print(grid.best_estimator_)
  # print(classification_report(y_test, y_predict))

  # grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]



  # plt.plot(k_range, k_scores)
  # plt.xlabel('Value of K for KNN')
  # plt.ylabel('Cross-Validated Accuracy')
  # plt.show()
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
  run_knn(X, y, X_train, X_test, y_train, y_test, "BreastCancer", 3)

  data = load_digits()

  X = data.data
  y = data.target

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  run_knn(X, y, X_train, X_test, y_train, y_test, "digits", 3)
