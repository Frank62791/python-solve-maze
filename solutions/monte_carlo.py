import random
from time import sleep
from gym import Env
from numba import jit, cuda

class MonteCarloSolution():

    def __init__(self, env: Env,obs = None):
        self.env = env
        self.obs = obs
        policy = self.monte_carlo_e_soft(env,episodes=1000)
        print(self.test_policy(policy, env))
    
    def create_random_policy(self):
        policy = {}
        for key in range(0, self.env.get_states_length()):        
            p = {}
            for action in range(0, self.env.action_space.n):
                p[action] = 1 / self.env.action_space.n
                policy[key] = p
        return policy
    
    def create_state_action_dictionary(self, policy:dict):
        Q = {}
        for key in policy.keys():
            Q[key] = {a: 0.0 for a in range(0, self.env.action_space.n)}
        return Q

#    @jit(target_backend='cuda')	                # use the GPU to speed up the process
    def play_game(self, policy, display=True):
        s = self.env.reset()
        episode = []
        finished = False
        

        while not finished:
            
            timestep = []
            timestep.append(s)
            n = random.uniform(0, sum(policy[s].values()))
            top_range = 0
            action = 3
            for prob in policy[s].items():
                top_range += prob[1]            
                if n < top_range:
                    action = prob[0]
                    break 
            position, reward, finished, info = self.env.step(action)          
            if reward < 0:
                action = random.randint(0,3)

            s = info
            timestep.append(action)
            timestep.append(reward)

            episode.append(timestep)

        return episode


    def test_policy(self,policy, env):
        wins = 0
        r = 10
        for i in range(r):
            w = self.play_game( policy, display=False)[-1][-1]
            if w == 1:
                wins += 1
        return wins / r


    def evaluate_policy_check(self,env, episode, policy, test_policy_freq):
        if episode % test_policy_freq == 0:
            print("Test policy for episode {} wins % = {}"
                .format(episode, self.test_policy(policy, env)))
    

    def monte_carlo_e_soft(self,env, episodes=10, policy=None, epsilon=0.01, test_policy_freq=1000):
        if not policy:
            policy = self.create_random_policy()  
        Q = self.create_state_action_dictionary(policy) 
        returns = {} 
        
        for e in range(episodes): 
            G = 0 
            episode = self.play_game( policy=policy, display=False)
      #      self.evaluate_policy_check(env, e, policy, test_policy_freq)
            
            for i in reversed(range(0, len(episode))):   
                s_t, a_t, r_t = episode[i] 
                state_action = (s_t, a_t)
                G += r_t 

                if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                    if returns.get(state_action):
                        returns[state_action].append(G)
                    else:
                        returns[state_action] = [G]   
                        
                    Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) 
                    
                    Q_list = list(map(lambda x: x[1], Q[s_t].items())) 
                    indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                    max_Q = random.choice(indices)
                    
                    A_star = max_Q 
                    
                    for a in policy[s_t].items(): 
                        if a[0] == A_star:
                            policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                        else:
                            policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values()))) 

        return policy