from maze_env import Maze
from RL_brain import Agent
from RL_brain import myAgent
import time
import matplotlib as plt
from pylab import *


if __name__ == "__main__":
    ### START CODE HERE ###
    # This is an agent with random policy. You can learn how to interact with the environment through the code below.
    # Then you can delete it and write your own code.
    graph_episode = []
    graph_reward = []
    env = Maze()
    agent = myAgent(actions=list(range(env.n_actions)))
    for episode in range(500):
        s = env.reset()
        episode_reward = 0
        agent.epoch_num += 1
        while True:
            #env.render()                 # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
            a = agent.choose_action(s)
            s_, r, done = env.step(a)
            q_dict = agent.update(s,a,s_,r)
            episode_reward += r
            s = s_
            agent.has_been_to_this_state[float((s[0] + s[2]) / 2), float((s[1] + s[3]) / 2)] = True
            if done:
                #env.render()
                #time.sleep(0.5)
                graph_reward.append(episode_reward)
                graph_episode.append(episode)
                agent.new_state = (20.0,20.0)
                break
        print('episode:', episode, 'episode_reward:', episode_reward)
        #if episode == 199:
    plt.plot(graph_episode, graph_reward)
    plt.savefig('demo.jpg')
    plt.plot(graph_episode, graph_reward)
    plt.show()
    plt.savefig('demo.jpg')

    ### END CODE HERE ###

    print('\ntraining over\n')
