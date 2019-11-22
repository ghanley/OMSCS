import numpy as np
import pprint
import sys
import gym
import time
# borrowed stole from https://github.com/dennybritz/reinforcement-learning/tree/master/DP
# if "../" not in sys.path:
#   sys.path.append("../")
from gridworld import GridworldEnv

# pp = pprint.PrettyPrinter(indent=2)
# env = GridworldEnv(shape=[20,20])
#danny britz file

def value_iteration(env, theta=0.0001, discount_factor=.9):
    iterations = 0
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):

        """
        Helper function to calculate the value for all action in a given state.

        Args:

                    state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
        # Check if we can stop
        iterations += 1
        if delta < theta:
            print('Value converged at iteration:', iterations)
            break


    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V


sizes = [5,10,20,30,50]
for size in sizes:
    print("Running VI Size: ", size)
    env = GridworldEnv(shape=[size,size])

    tic = time.time()
    policy, v = value_iteration(env)
    toc = time.time()
    elapsed_time = (toc - tic) * 1000
    print (f"Time to converge: {elapsed_time: 0.3} ms")




# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")

# print("Value Function:")
# print(v)
# print("")

# print("Reshaped Grid Value Function:")
# print(v.reshape(env.shape))
# print("")
