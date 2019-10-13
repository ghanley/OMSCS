import random
import mlrose
import numpy as np


fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness,
                             maximize = False, max_val = 8)



# Define decay schedule
schedule = mlrose.ExpDecay()

# Define initial state
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# Solve problem using simulated annealing
best_SA_state, best_SA_fitness, curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state,curve=True, random_state = 1)

best_RHC_state, best_RHC_fitness = mlrose.random_hill_climb(problem, max_attempts=10, max_iters=1000, restarts=0, init_state=None, curve=False, random_state=None)


# genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=inf, curve=False, random_state=None)
best_GAstate, best_GAfitness = mlrose.genetic_alg(problem, mutation_prob = 0.2, max_attempts = 100, max_iters = 1000,
                                              random_state = 1)

best_Mimic_state, best_Mimic_fitness = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=1000, curve=False, random_state=None)

print('The best SA state found is: ', best_SA_state)
print('The SA fitness at the best state is: ', best_SA_fitness)
# print('curve:', curve)
# print('iterations:', len(curve))

print('The best state found is: ', best_RHC_state)
print('The fitness at the best state is: ', best_RHC_fitness)
# print('curve:', curve)
# print('iterations:', len(curve))



print('The best GAstate found is: ', best_GAstate)
print('The GAfitness at the best state is: ', best_GAfitness)
# print('curve:', curve)
# print('iterations:', len(curve))

print('The best Mimicstate found is: ', best_Mimic_state)
print('The Mimicfitness at the best state is: ', best_Mimic_fitness)
# print('curve:', curve)
# print('iterations:', len(curve))

