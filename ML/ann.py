from sklearn import svm
import numpy as np
import pandas as pd
import time
from sklearn.neural_network import MLPClassifier
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

# digits adam
# best score 0.9801113762927606
# best params {'alpha': 0.1}


# adam
# best score 0.9428571428571428
# best params {'alpha': 0.0001}
def run_knn(X, y, X_train, X_test, y_train, y_test, title):

  scaler = StandardScaler()
  X_trainStandard = scaler.fit_transform(X_train)
  X_testStandard = scaler.transform(X_test)

  solvers = ['lbfgs', 'sgd', 'adam']
  # The solver for weight optimization.

  #there is also MLPRegressor
  param_range = [0.000001,0.00001, 0.0001, .001,.01,.1]
  param_name = 'alpha'

  param_range = [(100,), (100,1), (100,2), (100,3), (100,4), (100,5)]
  param_name = 'hidden_layer_sizes'
  # max_iter = [200, 1000, 5000]


  clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
  util.plot_learning_curve(clf, title + " ANN LC", X, y, cv=10, n_jobs=-1)

  # grid = GridSearchCV(cv=10, estimator = clf, param_grid = {'alpha': [0.000001,0.00001, 0.0001, .001,.01,.1, 1]})
  # grid = grid.fit(X_trainStandard, y_train)

  # print("best score",grid.best_score_)
  # print("best params", grid.best_params_)

  # param_range = [1, 10, 100, 1000, 10000]
  # param_name = "max_iter"
  hidden_layers = [1,2,3,4,5,6]

  train_scores, test_scores = validation_curve(clf, X, y, param_name, param_range, cv=5, scoring="accuracy", n_jobs=-1 )


    # Calculate mean and standard deviation for training set scores
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)

  plt.figure()
    # Plot mean accuracy scores for training and test sets
  lw = 2
  # plt.semilogx(hidden_layers, train_mean, label="Training score",
  #                 color="darkorange", lw=lw)
  plt.plot(hidden_layers, train_mean, 'o-', label="Training score", color="g")
  # plt.semilogx(hidden_layers, test_mean, label="Cross-validation score",
  #                 color="navy", lw=lw)
  plt.plot(hidden_layers, test_mean, 'o-', label="Cross-validation score", color="r")

    # Plot accurancy bands for training and test sets
    # plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,alpha=.1, color="r")
    # plt.fill_between(param_range, test_mean - test_std, test_mean + test_std,alpha=.1, color="g")

    # Create plot
  plt.title(title+"ANN Number Hidden Layers Validation Curve")
  plt.xlabel("Hidden Layers")
  plt.ylabel("Accuracy Score")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(title+'ANNhiddenVC.png')
  plt.figure()












  # clf.fit(X_train, y_train)

  # y_predTrain = clf.predict(X_trainStandard)
  # y_predTest = clf.predict(X_testStandard)

  # training_ann = accuracy_score(y_train, y_predTrain)
  # testing_ann = accuracy_score(y_test, y_predTest)

  # print("ANN Training Acc", training_ann)
  # print("ANN Testing Acc", testing_ann)


  # N_TRAIN_SAMPLES = X_train.shape[0]
  # N_EPOCHS = 25
  # N_BATCH = 128
  # N_CLASSES = np.unique(y_train)

  # scores_train = []
  # scores_test = []

  # # EPOCH
  # epoch = 0
  # while epoch < N_EPOCHS:
  #     print('epoch: ', epoch)
  #     # SHUFFLING
  #     random_perm = np.random.permutation(X_train.shape[0])
  #     mini_batch_index = 0
  #     while True:
  #         # MINI-BATCH
  #         indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
  #         mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
  #         mini_batch_index += N_BATCH

  #         if mini_batch_index >= N_TRAIN_SAMPLES:
  #             break

  #     # SCORE TRAIN
  #     scores_train.append(mlp.score(X_train, y_train))

  #     # SCORE TEST
  #     scores_test.append(mlp.score(X_test, y_test))

  #     epoch += 1

  # """ Plot """
  # fig, ax = plt.subplots(2, sharex=True, sharey=True)
  # ax[0].plot(scores_train)
  # ax[0].set_title('Train')
  # ax[1].plot(scores_test)
  # ax[1].set_title('Test')
  # fig.suptitle("Accuracy over epochs", fontsize=14)
  # plt.show()




  # training_st = accuracy_score(y_train, y_predTrain)
  # testing_ast = accuracy_score(y_test, y_predTest)

  # print("ANN ST Training Acc", training_st)
  # print("ANN ST Testing Acc", testing_st)
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
  run_knn(X, y, X_train, X_test, y_train, y_test, "BreastCancer")

  data = load_digits()

  X = data.data
  y = data.target

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3)

  run_knn(X, y, X_train, X_test, y_train, y_test, "digits")
