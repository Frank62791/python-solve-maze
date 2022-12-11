from itertools import count
from os import path
import random
from gym import Env
import numpy as np
#from solutions.TrainedResult.reload import qtable

class SARSA():
    def __init__(self, env: Env):
        self.env = env
        
        if path.exists("solutions/TrainedResult/SARSA.py"):
            self.play_game()
  #          self.Q  = np.array(qtable)

        else:
            self.Q = np.zeros((self.env.get_states_length(), self.env.action_space.n))      
        self.epsilon = 0.9
        self.total_episodes = 10000
        self.max_steps = 100
        self.alpha = 0.8
        self.gamma = 0.7
        #Initializing the reward
        self.reward=0

        # Exploration parameters
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0  # Exploration probability at start
        self.min_epsilon = 0.01  # Minimum exploration probability
        self.decay_rate = 0.01  # Exponential decay rate for exploration prob
        self.train()
        #Function to choose the next action
    def choose_action(self,state):
        exp_exp_tradeoff = random.uniform(0, 1)
# If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > self.epsilon:
            action = np.argmax(self.Q[state, :])

# Else doing a random choice --> exploration
        else:
            action = self.env.action_space.sample()
        return action

    #Function to learn the self.Q-value
    def update(self,state, state2, reward, action, action2):
               
            predict = self.Q[state, action]
            target = reward + self.gamma * self.Q[state2, action2]
            self.Q[state, action] = self.Q[state, action] + self.alpha * (target - predict)



    # Starting the SARSA learning
    def train(self):
        for episode in range(self.total_episodes):
            t = 0
            state1 = self.env.reset()
            action1 = self.choose_action(state1)

            while t < self.max_steps:
                
#                self.env.render()
                
                #Getting the next state
                info, reward, done, state2 = self.env.step(action1)
                if reward == 1 and self.gamma > 0.2:
                    self.alpha -=   0.002
                    self.gamma -=   0.002

                #Choosing the next action
                action2 = self.choose_action(state2)
                
                #Learning the self.Q-value
                self.update(state1, state2, reward, action1, action2)

                state1 = state2
                action1 = action2
                
                #Updating the respective vaLues
                t += 1
                reward += 1
                
                #If at the end of learning process
                if done:
                    break
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*episode)

            #Evaluating the performance
            print ("Performance : ", reward/self.total_episodes)
            
            #Visualizing the Q-matrix
            print(self.Q)
            f = open("solutions/TrainedResult/SARSA.txt", "w")
            f.write(str(self.Q))
            f.close()
    def play_game(self):
        from solutions.TrainedResult.SARSA import sarsa
        self.state = self.env.reset()
        for x in count():
            self.env.render()
            action = np.argmax(sarsa[self.state])
            print(self.state, action,sarsa[self.state])
            _ , reward, done, self.state = self.env.step(action)
            if done:
                break
        self.env.close()
        exit(0)