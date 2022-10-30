'''   This maze solver is referenced with some degrees from the following link: 
      https://www.geeksforgeeks.org/python-program-for-rat-in-a-maze-backtracking-2/  
      The original code only solves the 2 directional movements (right and down) and the maze is a square.
      "Frank Gu" imporved the code to allow 4 directional movements (right, down, up, left) .
'''
from asyncio.log import logger
from time import sleep
from gym import Env


class PlanningSolution():

    def __init__(self, env: Env, current_space: tuple = (0, 0)):
        self.env = env
        self.y = int(current_space[0]/100)
        self.x = int(current_space[1]/100)
        self.target_path = []
        self.possible_path = []
        self.maze = [[(0, 0) for j in range(10)] for i in range(10)]
        self.actions = []
        self.format_space()
        self.create_maze_map()

    def find_paths(self, x, y, sol=[[0 for j in range(10)] for i in range(10)], directions: list = [True, True, True, True]):

        if (x, y) == (9, 9):
            sol[x][y] = 1
            self.possible_path = sol
            return True

        if self.is_valid(x, y):
            sol[x][y] = 1

            if directions[0] and self.find_paths(x, y+1, sol, [True, False, True, True]) == True:
                return True

            if directions[2] and self.find_paths(x + 1, y, sol, [True, True, True, False]) == True:
                return True

            if directions[3] and self.find_paths(x-1, y, sol, [True, True, False, True]) == True:
                return True

            if directions[1] and self.find_paths(x, y - 1, sol, [False, True, True, True]) == True:
                return True
            sol[x][y] = 0
            return False

    def format_space(self):
        for space in self.env.path:
            self.target_path.append((space[0]/100, space[1]/100))

    def is_valid(self, x, y):
        if x >= 0 and x < 10 and y >= 0 and y < 10 and self.maze[x][y] == 1:
            return True

        return False

    def get_final_path(self):
        if self.find_paths(self.x, self.y) == False:
            logger.info("No path found")
            return False

        for x in range(10):
            for y in range(10):
                if self.is_valid(self.x, self.y+1) and self.possible_path[self.x][self.y+1] == 1:
                    self.actions.append(2)
                    self.possible_path[self.x][self.y+1] = 0
                    self.y = self.y+1
                    continue
                elif self.is_valid(self.x+1, self.y) and self.possible_path[self.x+1][self.y] == 1:
                    self.actions.append(0)
                    self.possible_path[self.x+1][self.y] = 0
                    self.x = self.x+1
                    continue
                elif self.is_valid(self.x-1, self.y) and self.possible_path[self.x-1][self.y] == 1:
                    self.actions.append(1)
                    self.possible_path[self.x-1][self.y] = 0
                    self.x = self.x-1
                    continue
                elif self.is_valid(self.x, self.y-1) and self.possible_path[self.x][self.y-1] == 1:
                    self.actions.append(3)
                    self.possible_path[self.x][self.y-1] = 0
                    self.y = self.y-1
                    continue
                else:
                    break

        print(self.actions)
        return self.actions

    def create_maze_map(self):
        for x in range(10):
            for y in range(10):
                if (x, y) in self.target_path:
                    self.maze[y][x] = 1
                else:
                    self.maze[y][x] = 0
