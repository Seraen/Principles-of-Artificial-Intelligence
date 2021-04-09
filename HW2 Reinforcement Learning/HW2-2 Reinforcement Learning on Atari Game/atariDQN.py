# -*- coding:utf-8 -*-
# DQN homework.
import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers
from utils import *
import math

# hyper-parameter.  
EPISODES = 5000

class SumTree:
    def __init__(self, capacity):
        # sum tree 能存储的最多优先级个数
        self.capacity = capacity
        # 顺序表存储二叉树
        self.tree = [0] * (2 * capacity - 1)
        # 每个优先级所对应的经验片段
        self.data = [None] * capacity
        self.size = 0
        self.curr_point = 0

    # 添加一个节点数据，默认优先级为当前的最大优先级+1
    def add(self, data):
        self.data[self.curr_point] = data
        #这是计算新进来的经验的优先级吗？这后面什么意思呀没有动
        #噢 好像就是如果是新加入的一个经验把优先级设置到最大以便每一个经验都被抽到一遍呢
        self.update(self.curr_point, max(self.tree[self.capacity - 1:self.capacity + self.size]) + 1)

        self.curr_point += 1
        if self.curr_point >= self.capacity:
            self.curr_point = 0

        if self.size < self.capacity:
            self.size += 1

    # 更新一个节点的优先级权重
    def update(self, point, weight):
        idx = point + self.capacity - 1
        change = weight - self.tree[idx]

        self.tree[idx] = weight

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def get_total(self):
        return self.tree[0]

    # 获取最小的优先级，在计算重要性比率中使用
    def get_min(self):
        return min(self.tree[self.capacity - 1:self.capacity + self.size - 1])

    # 根据一个权重进行抽样
    def sample(self, v):
        idx = 0
        while idx < self.capacity - 1:
            l_idx = idx * 2 + 1
            r_idx = l_idx + 1
            if self.tree[l_idx] >= v:
                idx = l_idx
            else:
                idx = r_idx
                v = v - self.tree[l_idx]

        point = idx - (self.capacity - 1)
        # 返回抽样得到的 位置，transition信息，该样本的概率
        return self.data[point]#point, self.data[point], self.tree[idx] / self.get_total()


class Memory(object):
    def __init__(self, batch_size, max_size, beta):
        self.batch_size = batch_size  # mini batch大小
        self.max_size = 2 ** math.floor(math.log2(max_size))  # 保证 sum tree 为完全二叉树
        self.beta = beta

        self._sum_tree = SumTree(max_size)

    def store_transition(self, s, a, r, s_, done):
        self._sum_tree.add((s, a, r, s_, done))

    def get_mini_batches(self):
        n_sample = self.batch_size if self._sum_tree.size >= self.batch_size else self._sum_tree.size
        total = self._sum_tree.get_total()

        # 生成 n_sample 个区间
        step = total // n_sample
        points_transitions_probs = []
        # 在每个区间中均匀随机取一个数，并去 sum tree 中采样
        for i in range(n_sample):
            v = np.random.uniform(i * step, (i + 1) * step - 1)
            t = self._sum_tree.sample(v)
            points_transitions_probs.append(t)

        #points, transitions, probs = zip(*points_transitions_probs)

        # 计算重要性比率
       # max_impmortance_ratio = (n_sample * self._sum_tree.get_min()) ** -self.beta
        #importance_ratio = [(n_sample * probs[i]) ** -self.beta / max_impmortance_ratio
                           # for i in range(len(probs))]

        return points_transitions_probs
       # return tuple(np.array(e) for e in zip(*transitions))#points, tuple(np.array(e) for e in zip(*transitions)), importance_ratio
       # return points_transitions_probs
    def update(self, points, td_error):
        for i in range(len(points)):
            self._sum_tree.update(points[i], td_error[i])




class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.9
        self.learning_rate = 0.02
        self.epsilon = 1.0
        self.epsilon_min = 0.002
        self.epsilon_decay = (self.epsilon-self.epsilon_min) / 5000
        self.batch_size = 128
        self.train_start = 128
        self.beta=0.1
        # create replay memory using deque
        self.maxlen = 25000
        self.memory = Memory(self.batch_size,self.maxlen,self.beta)#deque(maxlen=self.maxlen)

        # create main model
        self.model_target = self.build_model()
        self.model_eval = self.build_model()

    # approximate Q function using Neural Network
    # you can modify the network to get higher reward.
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model_eval.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
        # epsilon decay.
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if self.memory._sum_tree.size < self.train_start:
            return
        batch_size = min(self.batch_size, self.memory._sum_tree.size)
        mini_batch = self.memory.get_mini_batches()#random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model_eval.predict(update_input)
        target_val = self.model_target.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model_eval.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
    
    def eval2target(self):
        self.model_target.set_weights(self.model_eval.get_weights())



if __name__ == "__main__":
    # load the gym env
    env = gym.make('MsPacman-ram-v0')
    # set  random seeds to get reproduceable result(recommended)
    set_random_seed(0)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # create the agent
    agent = DQNAgent(state_size, action_size)
    # log the training result
    scores, episodes = [], []
    graph_episodes = []
    graph_score = []
    avg_length = 10
    sum_score = 0
    iteration=0

    # train DQN
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        lives = 3
        while not done: 
            dead = False         
            while not dead:
                # render the gym env
                if agent.render:
                    env.render()
                # get action for the current state
                action=agent.get_action(state)
                # take the action in the gym env, obtain the next state
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                # judge if the agent dead
                dead = info['ale.lives']<lives
                lives = info['ale.lives']
                # update score value
                score+=reward
                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state,action,reward,next_state,done)
                # train the evaluation network
                agent.train_model()
                # go to the next state
                state = next_state

            # update the target network after some iterations. 
            iteration += 1
            if iteration >= 10:
                iteration = 0
                agent.eval2target()
        # print info and draw the figure.
        if done:
            scores.append(score)
            sum_score += score
            episodes.append(e)
            # plot the reward each episode
            # pylab.plot(episodes, scores, 'b')
            print("episode:", e, "  score:", score, "  memory length:",
                  agent.memory._sum_tree.curr_point   , "  epsilon:", agent.epsilon)
        if e%avg_length == 0:
            graph_episodes.append(e)
            graph_score.append(sum_score / avg_length)
            sum_score = 0
            # plot the reward each avg_length episodes
            pylab.plot(graph_episodes, graph_score, 'r')
            pylab.savefig("./pacman_avg.png")
        
        # save the network if you want to test it.