# #######################################################################
# # Copyright (C)                                                       #
# # 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# # 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# # Permission given to modify the code as long as you keep this        #
# # declaration at the top                                              #
# #######################################################################

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

def figure_3_5(value):
    
    it = 0
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-2:
            break
        value = new_value
        it += 1
        # input("Press Enter to continue...")
        np.set_printoptions(precision=2)
        print(value)
        print()
    print("Converges in {} iterations".format(it))
    return value



def policy_improve(value, policy):
    change = False
    for i in policy.keys():
        temp = policy[i]
        # print(temp)
        max_list = []
        action_list = []

        for action in ACTIONS:
            next_state, reward = step(i, action)
            max_list.append(reward)
            action_list.append(action)
            
        max_idx = max_list.index(max(max_list))

        policy[i] = [action_list[max_idx]]
        print(policy[i])
        # return
        if temp != policy[i]:
            change = True
        
    return policy, change


def policy_iter():
    values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    # print(values)
    policy = {}
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            policy[i,j] = []
    

    stable = False
    while stable == False:
        opt_value = figure_3_5(values)
        policy, change = policy_improve(opt_value, policy)

        if change == False:
            stable = True
    return policy

result = policy_iter()

print(result)

#================================================#
'''
The code below was obtained through AI resources.
It was found as the error in my code above was indentified
'''
# def policy_improve(value):
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