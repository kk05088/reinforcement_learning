import random
import numpy as np

WIDTH = 5
HEIGHT = 5
NUM_STATES = WIDTH * HEIGHT
TERMINAL_STATES = [WIDTH * HEIGHT - 1]
DISCOUNT = 0.9
LEFT_COLUMN = list(range(0, WIDTH * HEIGHT, WIDTH))
RIGHT_COLUMN = list(range(WIDTH - 1, WIDTH * HEIGHT, WIDTH))
ACTIONS = [-1, 1, -1 * HEIGHT, HEIGHT] # left, right, up, down
action_dict = {-1: 'left', 1: 'right', -1 * HEIGHT: 'up', HEIGHT: 'down'}
grid = np.zeros(HEIGHT * WIDTH)
gridCounts = np.zeros(HEIGHT * WIDTH)
avgCounts = np.zeros(HEIGHT * WIDTH)
returns = value = {i:[] for i in range(NUM_STATES)}
# value = np.zeros((WIDTH, HEIGHT))

# print(returns)

def nextState(state, action):
    if state == 1:
        return 21, 10
    if state == 3:
        return 13, 5
    if (state in LEFT_COLUMN) and (action == ACTIONS[0]):
        return state, -1
    if (state in RIGHT_COLUMN) and (action == ACTIONS[1]):
        return state, -1
    next_state = state + action
    if next_state < 0 or next_state >= NUM_STATES:
        return state, -1
    if next_state in TERMINAL_STATES:
        return -1, 0
    return next_state, 0

def startingState():
    state = random.choice(range(NUM_STATES))
    if state not in TERMINAL_STATES:
        return state
    else:
        while state in TERMINAL_STATES:
            state = random.choice(range(NUM_STATES))
    return state

'''
The runEpisode(state) function returns the list of states and
the list of rewards for the episode
'''

def runEpisode(state):
    episode = [state]
    episode_action = []
    episodeReward = [0]
    while True:
        action = random.choice(ACTIONS)
        next_state, reward = nextState(state, action)
#         print(1, next_state, reward)
        if next_state == -1:
            return episode, episodeReward
        else:
            episode.append(next_state)
            episodeReward.append(reward)
        state = next_state
    
    return episode, episodeReward


def valueEpisode(state, e, eR):
    visited_states = []
    first_occurrence_idx = None
    

    for x in e:
        total_returns = 0
        power = 0
        if x in visited_states:
            pass
        
        else:
            visited_states.append(x)
            # if first_occurrence_idx == None:
            first_occurrence_idx = e.index(x)

            print(first_occurrence_idx,"brother pls")

            #calculate returns for current state 
            for j in range(first_occurrence_idx,len(eR)):
                if j == first_occurrence_idx:
                    total_returns += eR[j]
                elif j > 0:
                    total_returns += eR[j] * (DISCOUNT**power)
                power += 1
        
        print(total_returns)
        returns[x].append(total_returns)
        # total_returns = 0
        # power = 0
    
    print("Returns:\n",returns)
    
    # update the value function
    for k in visited_states:
        value[k] = returns[k]

    return value


def main():
    state = startingState()
    e, eR = runEpisode(state)
    # state = 23
    # e = [23, 22, 23, 18, 23, 18, 23, 23]
    # eR = [0, 0, 0, 0, 0, 0, 0, -1]
    value = valueEpisode(state, e, eR)
    print("Episodes:", "\n", e, "\n\n", "Rewards:", "\n",
           eR)
    return value

print(main())