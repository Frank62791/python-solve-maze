'''  all the logic implementation for the solution 
    and each line of code are written by "Frank Gu" only
    not someone else. 
'''
from gym import Env


class DPSolution():
    def __init__(self, env: Env, current_space: tuple = (0, 0)):
        self.env = env
        self.current_space = [(current_space[0]/100, current_space[1]/100)]
        self.target_path = []
        self.format_space()
        self.prevent_back_loop()
        self.find_optimal_path()

    def prevent_back_loop(self, directions: int = 4):
        if directions == 0:
            self.next_down = True
            self.next_up = False
            self.next_right = True
            self.next_left = True
        elif directions == 1:
            self.next_down = False
            self.next_up = True
            self.next_right = True
            self.next_left = True
        elif directions == 2:
            self.next_down = True
            self.next_up = True
            self.next_right = True
            self.next_left = False
        elif directions == 3:
            self.next_down = True
            self.next_up = True
            self.next_right = False
            self.next_left = True
        else:
            self.next_down = True
            self.next_up = True
            self.next_right = True
            self.next_left = True

    def find_optimal_path(self, optimal_path: list = []):

        down = (self.current_space[-1][0], self.current_space[-1][1] + 1)
        up = (self.current_space[-1][0], self.current_space[-1][1]-1)
        right = (self.current_space[-1][0]+1, self.current_space[-1][1])
        left = (self.current_space[-1][0]-1, self.current_space[-1][1])

        if self.current_space[-1] == (9, 9):
            return optimal_path
        elif down in self.target_path and self.next_down:
            optimal_path.append(0)
            self.current_space.append(down)
            self.prevent_back_loop(0)

        elif right in self.target_path and self.next_right:
            optimal_path.append(2)
            self.current_space.append(right)
            self.prevent_back_loop(2)

        elif up in self.target_path and self.next_up:
            optimal_path.append(1)
            self.current_space.append(up)
            self.prevent_back_loop(1)

        elif left in self.target_path and self.next_left:
            optimal_path.append(3)
            self.current_space.append(left)
            self.prevent_back_loop(3)

        else:
            optimal_path.pop()
            self.target_path.remove(self.current_space[-1])
            self.current_space.pop()
            self.prevent_back_loop()
        self.find_optimal_path(optimal_path)

    def format_space(self):
        for space in self.env.path:
            self.target_path.append((space[0]/100, space[1]/100))
