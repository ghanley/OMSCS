import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib
matplotlib.use('Agg')  # call .use() before importing .pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from util import import_data

X1, Y1, X2, Y2 = import_data()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 100)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def testBothParams(X_train, y_train, X_test, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(1, 21):
        for j in range(1,21):
            clf_gini = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=j)
            clf_gini.fit(X_train, y_train)
            y_pred = clf_gini.predict(X_test)
            # ax.plot_wireframe(i, j, accuracy_score(y_test,y_pred)*100, rstride=0, cstride=10)
            ax.scatter(i, j, accuracy_score(y_test,y_pred)*100, c='b', marker='.', s=1)

    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Min Samples Leaf')
    ax.set_zlabel('Accuracy')

    plt.tight_layout()
    plt.show()


def testMaxDepth(X_train, y_train, X_test, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    accuracy_test = []
    accuracy_train = []
    for i in range(1, 23):
            clf_gini = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=1)
            clf_gini.fit(X_train, y_train)
            y_pred = clf_gini.predict(X_test)
            y_pred_train = clf_gini.predict(X_train)
            accuracy_test.append(accuracy_score(y_test, y_pred) * 100)
            accuracy_train.append(accuracy_score(y_train, y_pred_train) * 100)

    ax.plot(range(1, 23), accuracy_test )
    ax.plot(range(1, 23), accuracy_train)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Max Depth')

    plt.tight_layout()
    plt.show()

def getParametersFromGridSearchCV(rangeEnd, X_train, y_train):

    param_grid = {'min_samples_leaf': np.arange(1, rangeEnd), 'max_depth' : np.arange(1, 20)}

    tree = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid=param_grid, cv= 10)
    tree.fit(X_train, y_train)
    print tree.best_params_
    return tree.best_params_['max_depth'], tree.best_params_['min_samples_leaf']

def draw_learning_curve_1():
    title = "Learning Curve for Phishing Dataset (Decision Tree)"
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
    max_depth, min_samples_leaf = getParametersFromGridSearchCV(30, X_train, y_train)
    estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=100, min_samples_leaf=min_samples_leaf)
    plot_learning_curve(estimator, title, X1, Y1, ylim=None, cv=cv)

    plt.show()

def draw_learning_curve_2():
    title = "Learning Curve for Optical Digit Recognition Dataset (Decision Tree)"
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.3)
    max_depth, min_samples_leaf = getParametersFromGridSearchCV(64, X_train, y_train)
    estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=100, min_samples_leaf=min_samples_leaf)
    plot_learning_curve(estimator, title, X2, Y2, ylim=None, cv=cv)

    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.3)
draw_learning_curve_2()
# testMaxDepth(X_train, y_train, X_test, y_test)
