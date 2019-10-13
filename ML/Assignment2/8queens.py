import random
import mlrose
import numpy as np
import matplotlib.pyplot as plt
from time import clock

#https://mlrose.readthedocs.io/en/stable/source/fitness.html
fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness,
                             maximize = False, max_val = 8)



# Define decay schedule
schedule = mlrose.ExpDecay()

# Define initial state
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
RHC_iterations = []
SA_iterations = []
GA_iterations = []
MIMIC_iterations = []
iterations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

RHC_timings = []
SA_timings = []
GA_timings = []
MIMIC_timings = []


for iter in iterations:

  # Randomized Hill Climbing
  start = clock()
  best_RHC_state, best_RHC_fitness = mlrose.random_hill_climb(problem, max_attempts=10, max_iters=iter, restarts=5, init_state=None, curve=False, random_state=None)
  RHC_timings.append(clock() - start)

  # Solve problem using simulated annealing
  start = clock()
  best_SA_state, best_SA_fitness, curve = mlrose.simulated_annealing(problem, max_attempts = 100, max_iters = iter, init_state = None,curve=True, random_state = 1)
  SA_timings.append(clock() - start)


  # genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=inf, curve=False, random_state=None)
  start = clock()
  best_GA_state, best_GA_fitness = mlrose.genetic_alg(problem, mutation_prob = 0.2, max_attempts = 10, max_iters = iter,
                                                random_state = 1)
  GA_timings.append(clock() - start)

  # MIMIC
  start = clock()
  best_Mimic_state, best_Mimic_fitness = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=iter, curve=False, random_state=None)
  MIMIC_timings.append(clock() - start)
  print("m fit:", best_Mimic_fitness)
  print("M state", best_Mimic_state)

  RHC_iterations.append(best_RHC_fitness)
  SA_iterations.append(best_SA_fitness)
  GA_iterations.append(best_GA_fitness)
  MIMIC_iterations.append(best_Mimic_fitness)

RHC_iterations = np.asarray(RHC_iterations)
SA_iterations = np.asarray(SA_iterations)
GA_iterations = np.asarray(GA_iterations)
MIMIC_iterations = np.asarray(MIMIC_iterations)

# RHC_iterations += .1
# SA_iterations += .1
# GA_iterations += .1
# MIMIC_iterations += .1


RHC_iterations = 1 / RHC_iterations
SA_iterations = 1 / SA_iterations
GA_iterations = 1 / GA_iterations
MIMIC_iterations = 1 / MIMIC_iterations


plt.figure()
ticks = range(len(iterations))
print(GA_iterations)
print(MIMIC_iterations)
print(best_Mimic_state)
print(best_SA_state)
    # Plot mean accuracy scores for training and test sets
lw = 2
  # plt.semilogx(hidden_layers, train_mean, label="Training score",
  #                 color="darkorange", lw=lw)
plt.plot(ticks, RHC_iterations, 'o-', label="RHC", color="g")
  # plt.semilogx(hidden_layers, test_mean, label="Cross-validation score",
  #                 color="navy", lw=lw)
plt.plot(ticks, SA_iterations, 'o-', label="SA", color="r")
plt.plot(ticks, GA_iterations, '.-', label="GA", color="c")
plt.plot(ticks, MIMIC_iterations, 'o-', label="MIMIC", color="y")
plt.xticks(ticks, iterations) #set the ticks to be a
# plt.xaxis.set_ticklabels(iterations) # change the ticks' names to x




plt.title("8Queens Iteration Curve")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig('8QueensIterations.png')
plt.figure()




plt.figure()
ticks = range(len(iterations))

    # Plot mean accuracy scores for training and test sets
lw = 2

plt.plot(ticks, RHC_timings, 'o-', label="RHC", color="g")
plt.plot(ticks, SA_timings, 'o-', label="SA", color="r")
plt.plot(ticks, GA_timings, '.', label="GA", color="c")
plt.plot(ticks, MIMIC_timings, 'o-', label="MIMIC", color="y")
plt.xticks(ticks, iterations) #set the ticks to be a
# plt.xaxis.set_ticklabels(iterations) # change the ticks' names to x

plt.title("8 Queens Timings Curve")
plt.xlabel("Iterations")
plt.ylabel("Time")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig('8QueensTimings.png')
plt.figure()
