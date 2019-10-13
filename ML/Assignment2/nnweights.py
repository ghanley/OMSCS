import mlrose
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from time import clock
from sklearn.preprocessing import StandardScaler, normalize
from collections import defaultdict
import matplotlib.pyplot as plt


#Adapted from https://towardsdatascience.com/fitting-a-neural-network-using-randomized-optimization-in-python-71595de4ad2d
# Load the Digits dataset
# data = load_digits()
data = pd.read_csv('breast.csv')
y = data['Class']
X = data.drop(['Class'], axis=1)


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2, random_state = 3)

# scaler = MinMaxScaler()
scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

one_hot = OneHotEncoder()

# y_train_hot = one_hot.fit_transform(y_train)
# y_test_hot = one_hot.transform(y_test)


y_train_hot = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.values.reshape(-1, 1)).todense()


# y_train_hot = y_train
# y_test_hot = y_test

algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']

RHC_iterations = []
SA_iterations = []
GA_iterations = []
MIMIC_iterations = []

timings = defaultdict(list)
timings['random_hill_climb'] = []
timings['simulated_annealing'] = []
timings['genetic_alg'] = []
timings['gradient_descent'] = []

iterations = [1, 1000, 5000, 10000, 20000, 30000]

train_score = defaultdict(list)
train_score['random_hill_climb'] = []
train_score['simulated_annealing'] = []
train_score['genetic_alg'] = []
train_score['gradient_descent'] = []

test_score = defaultdict(list)
test_score['random_hill_climb'] = []
test_score['simulated_annealing'] = []
test_score['genetic_alg'] = []
test_score['gradient_descent'] = []








for algo in algorithms:
  timings_name = algo+' timings'
  score_name = algo+' scores'
  for iter in iterations:
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [100], activation = 'relu',
                                    algorithm = algo, max_iters = iter,
                                    bias = True, is_classifier = True, learning_rate = 0.00001, restarts=5,
                                    early_stopping = True, clip_max = 5, max_attempts = 10)
    start = clock()
    nn_model1.fit(X_train_scaled, y_train_hot)
    timings[algo].append(clock() - start)



    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    train_score[algo].append(y_train_accuracy)
    print(algo + str(iter) +'Training accuracy: ', y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    test_score[algo].append(y_test_accuracy)

    # print('Test accuracy: ', y_test_accuracy)
  plt.figure()
  ticks = range(len(iterations))

  plt.plot(ticks, train_score[algo], 'o-', label="Train Score", color="g" )
  plt.plot(ticks, test_score[algo], '.-', label="Test Score",color="r")
  plt.xticks(ticks, iterations)
  plt.title(score_name + "SS")
  plt.xlabel("Iterations")
  plt.ylabel("Accuracy")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(timings_name +"SS.png")

plt.figure()
colors = ["r", "b", "y", "g"]
count = 0
for algo in algorithms:
  plt.plot(ticks, timings[algo], 'o-', label=algo, color=colors[count])
  count += 1
plt.xticks(ticks, iterations)

plt.title("Timings Curve SS")
plt.xlabel("Iterations")
plt.ylabel("Time")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("timingsCurveSS.png")




print(timings)
print(train_score)
