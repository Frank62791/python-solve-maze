import time
import evn.maze_gym as maze_gym
from solutions.dp_solution import DPSolution
from solutions.planning import PlanningSolution
from solutions.monte_carlo import MonteCarloSolution
from solutions.DT0 import DT0

env = maze_gym.MazeChallenger()   # {0: "Down", 1: "Up", 2: "Right", 3: "Left"}
obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.2)
    if done == True:
        break

env.close()

'''    solution 1:
  obs,random_position = env.random_position_reset()         
        optimal_path = DPSolution(env,random_position).final_path
        while True:                                                 
            
            for action in optimal_path:                         
                obs, reward, done, info = env.step(action)
                env.render()
                time.sleep(0.2)
            if done:
                break
        env.close()            '''


'''  soluition 2:
obs,random_position = env.random_position_reset()         
    optimal_path = PlanningSolution(env,random_position).get_final_path()
    while True:                                                 
    
        for action in optimal_path:                         
                obs, reward, done, info = env.step(action)
                env.render()
                time.sleep(0.2)
        if done:
                break
    env.close()                  '''
