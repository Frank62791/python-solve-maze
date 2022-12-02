import time
import env.maze_gym as maze_gym
from solutions.dp_solution import DPSolution
from solutions.planning import PlanningSolution
from solutions.monte_carlo import MonteCarloSolution
from solutions.q_learning import QLearning
from solutions.SARSA import SARSA

env = maze_gym.MazeChallenger()   # {0: "Down", 1: "Up", 2: "Right", 3: "Left"}



#   dp_solution:  
obs = env.reset()         
optimal_path = DPSolution(env).final_path
while True:                                                 
    
    for action in optimal_path:                         
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.2)
    if done:
        break
env.close()           


'''  planning_solution:
obs,random_position = env.random_position_reset()                     # env.random_position_reset()  use random position to start
    optimal_path = PlanningSolution(env,random_position).get_final_path()
    while True:                                                 
    
        for action in optimal_path:                         
                obs, reward, done, info = env.step(action)
                env.render()
                time.sleep(0.2)
        if done:
                break
    env.close()                  '''


'''  load Q learning talble and play the game:

        
QLearning(env)
               '''


'''  load SARSA learning talble and play the game:    
    

SARSA(env)

                '''