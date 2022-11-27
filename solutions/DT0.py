from itertools import count
import numpy as np
import random
from tqdm import tqdm
from gym import Env
from os import path


class DT0():
    def __init__(self, env: Env, current_space: tuple = (0, 0)):
        self.env = env
        self.state = self.env.reset()
#        self.env.render()
        self.action_size = self.env.action_space.n
        self.state_size = self.env.get_states_length()
        self.qtable = np.zeros((self.state_size, self.action_size))
        if path.exists("solutions/TrainedResult/Q_learning_DT0.py"):
                self.play_game()
        total_episodes = 1000  # Total episodes
        total_test_episodes = 1  # Total test episodes
        max_steps = 99  # Max steps per episode

        learning_rate = 0.7  # Learning rate
        gamma = 0.618  # Discounting rate


        # Exploration parameters
        epsilon = 1.0  # Exploration rate
        max_epsilon = 1.0  # Exploration probability at start
        min_epsilon = 0.01  # Minimum exploration probability
        decay_rate = 0.01  # Exponential decay rate for exploration prob

# 2 For life or until learning is stopped
        for episode in tqdm(range(total_episodes)):
            # Reset the environment
            state = self.env.reset()
            step = 0
            done = False

            for step in range(max_steps):
                # 3. Choose an action a in the current world state (s)
                # First we randomize a number
                exp_exp_tradeoff = random.uniform(0, 1)

# If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state, :])

# Else doing a random choice --> exploration
                else:
                    action = self.env.action_space.sample()

# Take the action (a) and observe the outcome state(s') and reward (r)
                info, reward, done, new_state = self.env.step(action)
 #                       self.env.render()

# Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.qtable[state, action] = self.qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

# Our new state is state
                state = new_state

# If done : finish episode
                if done == True:
                    break
# Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-decay_rate*episode)

        self.env.reset()
        rewards = []
        f = open("solutions/TrainedResult/Q_learning_DT0.py", "a")
        f.write(str(self.qtable))
        f.close()

        for episode in range(total_test_episodes):
            state = self.env.reset()
            step = 0
            done = False
            total_rewards = 0
            # print("****************************************************")
            #print("EPISODE ", episode)

            for step in range(max_steps):

                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.qtable[state, :])

                info, reward, done, new_state = self.env.step(action)
       #         self.env.render()

                total_rewards += reward

                if done:
                    rewards.append(total_rewards)
                    #print ("Score", total_rewards)
                    break
                state = new_state

        self.env.close()
        print("Score over time: " + str(sum(rewards)/total_test_episodes))


    def play_game(self):
        from solutions.TrainedResult.Q_learning_DT0 import qtable
        for x in count():
            self.env.render()
            action = np.argmax(qtable[self.state])
            print(self.state, action,qtable[self.state])
            _ , reward, done, self.state = self.env.step(action)
            if done:
                break
        self.env.close()
        exit(0)