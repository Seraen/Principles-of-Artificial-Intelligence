from maze_env import Maze
import numpy as np
import pandas as pd
import random

UNIT = 40
MAZE_H = 6
MAZE_W = 6

class Agent:
    ### START CODE HERE ###
    ### a random agent ###

    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 1

    def choose_action(self, observation):
        action = np.random.choice(self.actions)
        return action

    ### END CODE HERE ###

class myAgent:
    ### and agent of mine using Dyna-Q###

    def __init__(self,actions):
        self.actions = actions #actions
        self.epsilon = 0.1  #initialize for epsilon greedy
        self.decay = 0.99 #a decay factor on epsilon
        self.gamma = 0.8 #choose a key for gamma
        self.alpha = 0.6 #choose a key for alpha
        self.q_dict = {} #remember Q(s,a)
        self.model_dict = {} #remember Model(s,a)
        self.Dyna_N = 10 #repeat N times
        self.epsilon_dict = {}#remember the next step it can take when in this state
        self.has_been_to_this_state = {}#whether has been to this state?
        self.greedy_random_actions = [] #"random" actions for e-greedy
        self.cnt_of_state = {}# remember how many times has come to this state
        self.cnt_of_state_all_actions = {}
        self.new_state = (20.0,20.0)
        self.all_arrived = False
        self.k = 0.3
        self.epoch_num = 0 # same with episode

        for is_treasure_be_found in [False,True]:
            for i in range(MAZE_W):
                for j in range(MAZE_H):
                    '''
                    initialize state and some dictionaries and stuff
                    state(origin for x,origin for y,destination for x,destination for y,a bool)
                    every item in q_state is [a,b,c,d],which corresponds to rewards of four directions
                    '''
                    state = (float(i*UNIT+5),float(j*UNIT+5),float((i+1)*UNIT-5),float((j+1)*UNIT-5),is_treasure_be_found)
                    self.q_dict[state] = [0,0,0,0]
                    self.epsilon_dict[state] = [False,False,False,False]
                    self.has_been_to_this_state[float((state[0]+state[2])/2),float((state[1]+state[3])/2)] = False
                    self.cnt_of_state[state] = [0,0,0,0]
                    self.cnt_of_state_all_actions[state] = 1


    def update_random_actions_for_greedy(self,s):
        state = tuple(s)
        self.all_arrived = False
        self.has_been_to_this_state[float((state[0] + state[2]) / 2), float((state[1] + state[3]) / 2)] = True
        expected_states = []
        actions_to_get_close = []
        present_state = ((state[0]+state[2])/2,(state[1]+state[3])/2)

        #has not been to this state before
        for s1 in self.has_been_to_this_state:
            if self.has_been_to_this_state[s1] == False:
                expected_states.append(s1)

        if not expected_states:
            self.all_arrived = True
            return []

        if present_state == self.new_state:
            new_state = random.sample(expected_states, 1)
            new_state = new_state[0]
        else:
            new_state = self.new_state
        if new_state[0] > present_state[0]:
            actions_to_get_close.append(2)
        if new_state[0] < present_state[0]:
            actions_to_get_close.append(3)
        if new_state[1] < present_state[1]:
            actions_to_get_close.append(0)
        if new_state[1] > present_state[1]:
            actions_to_get_close.append(1)
        for a2 in actions_to_get_close:
            if self.q_dict[state][a2] == -1:
                actions_to_get_close.remove(a2)
        self.new_state = new_state

        return actions_to_get_close

    def epsilon_decay(self,s):
        '''
        epsilon is large at first,then gradually decays
        :return:
        '''
        epsilon = self.epsilon
        if self.all_arrived == False:
            epsilon = 0.1
        if self.all_arrived == True:
            epsilon = 0
        return epsilon

    def update_cnt_of_state(self,s):
        state = tuple(s)
        total = 0
        for l in range(4):
            total += self.cnt_of_state[state][l]
        self.cnt_of_state_all_actions[state] = total

    def choose_action(self,s):
        '''
        choose an action,use epsilon greedy
        :param s: present state,a list
        :return: an action(x0,y0,x1,y1)
        '''
        actions_to_get_close2 = self.update_random_actions_for_greedy(s)
        state = tuple(s)  # transform list s into a tuple
        self.has_been_to_this_state[float((state[0] + state[2]) / 2), float((state[1] + state[3]) / 2)] = True
        self.epsilon = self.epsilon_decay(state)
        # use epsilon greedy
        if random.random() < self.epsilon:
            action = np.random.choice(self.actions) # a random action
        else:
            max_actions=[]
            for i in range(4):
                if self.q_dict[state][i] == max(self.q_dict[state]):
                    max_actions.append(i)
            action = random.choice(max_actions) #choose action of maximum value

        if (action == 0 and state[1] == 5) or (action == 1 and state[3] == 235) or (action == 2 and state[2] == 235) or \
                (action == 3 and state[0] == 5):
            self.q_dict[state][action] = -100
            self.epsilon_dict[state][action] = True
            self.update_cnt_of_state(s)
            return self.choose_action(list(s))
        else:
            return action

    def update(self,s,a,s_,r):
        '''
        update Q value
        :param s: previous state
        :param a: action
        :param s_: new state
        :param r: reward
        :return: q_dict
        '''
        state = tuple(s)
        state_ = tuple(s_)
        if self.all_arrived:
            self.k = 0


        for j in range(4):
           self.cnt_of_state[state_][j] = self.cnt_of_state[state_][j] * 0.92

        self.cnt_of_state[state][a] += 1
        self.update_cnt_of_state(s)
        self.has_been_to_this_state[float((state[0] + state[2]) / 2), float((state[1] + state[3]) / 2)] = True
        self.q_dict[state][a] += self.alpha*(r + self.gamma*max(((self.q_dict[state_][i])-self.cnt_of_state[state_][i]*self.k) for i in range(4))-self.q_dict[state][a])
        self.model_dict[(state,a)] = [state_,r]

        N = self.Dyna_N
        for n in range(N): # iteration for N times
            ms,ma = random.choice(list(self.model_dict))
            ms_,mr = self.model_dict[(ms,ma)]
            self.q_dict[ms][ma] += self.alpha * (mr + self.gamma * max(((self.q_dict[ms_][i])-self.cnt_of_state[state_][i]*self.k)for i in range(4)) - self.q_dict[ms][ma])

        return self.q_dict




