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
        # input("Press Enter to continue...")
        # np.set_printoptions(precision=2)
    #     print(value)
    #     print()
    # print("Converges in {} iterations".format(it))
    
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
        # print(max_idx)
        

        
        # print(max_list)
        # for j in range(len(max_list)-1):
        #     # print(j,"bruh")
        #     if max_list[j] > max_num:
        #         # print(max_list[j])
        #         max_num = max_list[j]
        #         max_idx.append(j)
        

        
        # for x in range(len(max_idx)-1):
        #     # print(action_list,"<==")
        #     if x == 0:
        #         # final_action.append(action_list[x])
        #         policy[i] = [action_list[max_idx[x]]]
        #     if x > 0:
        #         policy[i].append(action_list[max_idx[x]])

        for x in max_idx:
            policy[i].append(action_list[x])
            
        # policy[i] = action_list[np.argmax(max_list)]

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
    # for i in range(WORLD_SIZE):
    #     for j in range(WORLD_SIZE):
    #         policy[i,j] = np.argmax(values)
# def optimal_policy(value):
#     policy = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=int)
#     change = False
#     for i in range(WORLD_SIZE):
#         for j in range(WORLD_SIZE):
#             temp=policy[i,j]
#             values = []
#             for action_idx, action in enumerate(ACTIONS):
#                 (next_i, next_j), reward = step([i, j], action)
#                 # calculate the expected value of the action
#                 expected_value = reward + DISCOUNT * value[next_i, next_j]
#                 values.append(expected_value)
#             # select the action with the maximum expected value
#             policy[i, j] = np.argmax(values)

#             if temp != policy[i,j]:
#                 change = True

#     return policy, change

# def value_iter(value):
#     theta = 0.9
#     delta = 1e-10
#     diff  = 0
#     while diff >= delta:

# states=[]
# probs = {}
# for i in WORLD_SIZE:
#     for j in WORLD_SIZE:
#         states.append([i,j])
#         probs[i,j] = 0.5

# value = np.zeros((WORLD_SIZE, WORLD_SIZE))
# def value_iter1(value):
#     gamma = 0.9
#     delta = 1e-10
#     diff  = 0
#     # Start value iteration
#     while diff >= delta:
#         max_diff = 0  # Initialize max difference
        
#         for i in states:
#                 max_val = 0
#                 for a in ACTIONS:
#                     # Compute state value
#                     next_state, val = step(i, a)  # Get direct reward
#                     for s_next in states:
#                         val += probs[i][s_next][a] * (
#                             gamma * value[s_next]
#                         )  # Add discounted downstream values

#                     # Store value best action so far
#                     max_val = max(max_val, val)

#                     # Update best policy
#                     if V[s] < val:
#                         pi[s] = ACTIONS[a]  # Store action with highest value

#                 V_new[s] = max_val  # Update value with highest value

#                 # Update maximum difference
#                 max_diff = max(max_diff, abs(V[s] - V_new[s]))

#             # Update value functions
#             V = V_new

#             # If diff smaller than threshold delta for all states, algorithm terminates
#             if max_diff < delta:
#                 break