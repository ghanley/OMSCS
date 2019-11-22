from gym import wrappers
from time import time # just to have timestamps in the files


env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
