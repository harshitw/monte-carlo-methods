# state value prediction
# given a policy how might the agent will estimate the value function 
# different policiy will for evaluating and interacting with the environment. Called the off policy method.
# we are working with episodic tasks
# RECHECK
# the agent needs to play blackjack around 500 times to find  good estimate of the value function which is too long for the evaluation of policy

from collections import defaultdict
import numpy as np
import sys

# generate episode returns episode of interaction of agent with the environment.
# monte carlo prediction of state value function
def mc_prediction_v(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionary of list returns = defaultdict(list)
    returns = defaultdict(list)
    
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # first-visit MC prediction 
        # estimating the value of a particular state 
        episode = generate_episode(env)
        states, actions, rewards = zip(*episode)
        # preparing for discounts 
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        # storing the return value for each state 
        for i, state in enumerate(states):
            returns[state].append(sum(rewards[i:]*discounts[:-(i+1)]))
    # calculate the state value function estimate 
    print(state)
    print(discounts)
    print(returns)
    V = {k: np.mean(v) for k, v in returns.items()}
    return V

# predicting action value function from state value function
def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        episode = generate_episode(env)
        states, actions, rewards = zip(*episode)
        # discounts
        discounts = np.array([gamma**i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(i+1)])
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = returns_sum[state][actions[i]]/N[state][actions[i]]
    return Q