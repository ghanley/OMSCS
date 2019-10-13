import random
import mlrose
import numpy as np
import matplotlib.pyplot as plt
from time import clock
#https://mlrose.readthedocs.io/en/stable/source/fitness.html
## GA / Mimic are scoring the highest, RHC the worst


fitness = mlrose.SixPeaks(t_pct=0.15)
leng = 128

problem = mlrose.DiscreteOpt(length = leng, fitness_fn = fitness, maximize=True)

# Define decay schedule
schedule = mlrose.ExpDecay()


# Define initial state
init_state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])

RHC_iterations = []
SA_iterations = []
GA_iterations = []
MIMIC_iterations = []
iterations = [512,1024, 2048, 4098, 8196]

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

  RHC_iterations.append(best_RHC_fitness)
  SA_iterations.append(best_SA_fitness)
  GA_iterations.append(best_GA_fitness)
  MIMIC_iterations.append(best_Mimic_fitness)


plt.figure()
ticks = range(len(iterations))
print(GA_iterations)
print(MIMIC_iterations)
print(best_Mimic_state)
    # Plot mean accuracy scores for training and test sets
lw = 2
  # plt.semilogx(hidden_layers, train_mean, label="Training score",
  #                 color="darkorange", lw=lw)
plt.plot(ticks, RHC_iterations, 'o-', label="RHC", color="g")
  # plt.semilogx(hidden_layers, test_mean, label="Cross-validation score",
  #                 color="navy", lw=lw)
plt.plot(ticks, SA_iterations, 'o-', label="SA", color="r")
plt.plot(ticks, GA_iterations, '.-', label="GA", color="b")
# plt.plot(ticks, MIMIC_iterations, 'o-', label="MIMIC", color="y")
plt.xticks(ticks, iterations) #set the ticks to be a
# plt.xaxis.set_ticklabels(iterations) # change the ticks' names to x




plt.title(str(leng) +"Length 6Peaks Iteration Curve")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig(str(leng) + ' length6peaksIterations.png')
plt.figure()


plt.figure()
ticks = range(len(iterations))

    # Plot mean accuracy scores for training and test sets
lw = 2

plt.plot(ticks, RHC_timings, 'o-', label="RHC", color="g")
plt.plot(ticks, SA_timings, 'o-', label="SA", color="r")
plt.plot(ticks, GA_timings, '.-', label="GA", color="c")
# plt.plot(ticks, MIMIC_timings, 'o-', label="MIMIC", color="y")
plt.xticks(ticks, iterations) #set the ticks to be a
# plt.xaxis.set_ticklabels(iterations) # change the ticks' names to x

plt.title(str(leng) + "Size 6Peaks Timings Curve")
plt.xlabel("Iterations")
plt.ylabel("Time")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig(str(leng) +'problem6peaksTimings.png')
plt.figure()




