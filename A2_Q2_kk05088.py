#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25

def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

def figure_3_2(value):
    
    it = 0
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-2:
            break
        value = new_value
        it += 1

    
    return value

def policy_improve(value, policy):
    
    max_list = []
    action_list = []
    
    for i in policy.keys():
        for action in ACTIONS:
            next_state, reward = step(i, action)
            # print(i, next_state)
            max_list.append(reward)
            action_list.append(action)


        max_idx = np.argwhere(max_list == np.amax(max_list)).flatten()


        for x in max_idx:
            policy[i].append(action_list[x])
            


    return policy

def main():
    values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    policy = {}
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            policy[i,j] = []

    value = figure_3_2(values)
    opt_policy = policy_improve(value, policy)

    return opt_policy

print(main())