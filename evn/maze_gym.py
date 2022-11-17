from time import sleep
import numpy as np
import cv2
import random
from playsound import playsound
from gym import Env, spaces


font = cv2.FONT_HERSHEY_COMPLEX_SMALL


class MazeChallenger(Env):
    def __init__(self):
        super(MazeChallenger, self).__init__()

        self.path = [(0, 0), (0, 100), (0, 200), (0, 300), (0, 400), (0, 500), (600, 300), (600, 200),
                     (0, 600), (0, 700), (0, 800), (0, 900), (200,
                                                              100), (200, 200), (200, 300), (200, 400),
                     (200, 600), (200, 700), (200, 800), (300, 800), (400, 800), (500, 800), (100, 200), (100, 700), (500, 700), (
                         500, 900), (500, 600), (500, 400), (500, 500), (600, 500), (700, 500), (800, 500), (700, 600), (700, 700),
                     (700, 800), (900, 800), (900, 700), (900, 600), (900,
                                                                      500), (900, 400), (900, 300), (900, 200), (300, 100),
                     (400, 100), (500, 100), (600, 100), (700,
                                                          100), (800, 100), (800, 200), (400, 300), (400, 200), (900, 900)
                     ]
        self.check_path = [(0, 100), (0, 200), (0, 300), (0, 400), (0, 500), (0, 600), (0, 700),
                           (200, 100), (200, 200), (200, 700), (200, 800), (300, 800), (400, 800), (500, 800), (
                               100, 200), (100, 700), (500, 700), (500, 600),  (500, 500), (600, 500), (700, 500), (800, 500),
                           (900, 800), (900, 700), (900, 600), (900, 500), (900,
                                                                            400), (900, 300), (900, 200), (300, 100),
                           (400, 100), (500, 100), (600, 100), (700, 100), (800, 100), (800, 200), (900, 900)]

        # Define a 2-D observation space
        self.observation_shape = (1000, 1000, 3)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(
                                                self.observation_shape),
                                            shape=self.observation_shape,
                                            dtype=np.float16)

        # Define an action space ranging from 0 to 3
        self.action_space = spaces.Discrete(4,)

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        # Define repeat path to give negative reward to challenger
        self.repeat_path = {(0, 0)}

        # Permissible area of Challenger to be
        self.y_min = 0
        self.x_min = 0
        self.y_max = self.observation_shape[0]
        self.x_max = self.observation_shape[1]

        self.dynamic_maze()
        self.states_in_maze()
        self.count = 0

    def states_in_maze(self):
        self.states = {}
        state = 0
        for position in self.path:
            self.states.update({position: state})
            state += 1

    def get_states_length(self):
        return len(self.states)

    def get_current_state(self):
        return self.states[self.challenger.get_position()]

    def dynamic_maze(self):
        self.dynamic_path = []
        for x in self.path:
            self.dynamic_path.append((int(x[1]/100), int(x[0]/100)))
        destination_x = 9
        destination_y = 9
        self.maze = np.matrix(np.ones(shape=(10, 10)))
        self.maze *= -1
        for position in self.dynamic_path:
            if position in self.repeat_path:
                self.maze[position] = 0
        self.maze[destination_x, destination_y] = 100

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        for x in range(0, 1000, 100):

            for y in range(0, 1000, 100):
                if x == 900 and y == 900:
                    spawned_win = Win("WIN", self.x_max,
                                      self.x_min, self.y_max, self.y_min)
                    spawned_win.set_position(1000, 1000)
                    self.elements.append(spawned_win)

                if (x, y) in self.path:
                    continue

                else:
                    spawned_block = Block(
                        "block", self.x_max, self.x_min, self.y_max, self.y_min)
                    spawned_block.set_position(x, y)
                    self.elements.append(spawned_block)

        # Draw the heliopter on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y: y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        text = 'Steps: {} | Rewards: {}'.format(
            self.ep_return, self.reward_return)

        # Put the info on canvas
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
                                  0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def add_text(self):
        text = 'Steps: {} | Rewards: {}'.format(
            self.ep_return, self.reward_return)

        # Put the info on canvas
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
                                  0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def reset(self, x=0, y=0):

        # Reset the reward
        self.ep_return = 0
        self.reward_return = 0

        # Intialise the challenger
        self.challenger = Challenger("challenger", self.x_max,
                                     self.x_min, self.y_max, self.y_min)
        self.challenger.set_position(x, y)

        self.current_postion = (x, y)
        # Intialise the elements
        self.elements = [self.challenger]

        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        return self.get_current_state()

    def random_position_reset(self):
        position_index = random.randint(0, len(self.path) - 1)
        position = self.path[position_index]
        self.reset(position[0], position[1])
        return self.get_current_state(), self.current_postion

    def render(self, mode="human"):
        assert mode in [
            "human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":

            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas/100

    def close(self):
        cv2.destroyAllWindows()

    def get_action_meanings(self):
        return {0: "Down", 1: "Up", 2: "Right", 3: "Left"}

    def step(self, action: int):
        # Flag that marks the termination of an episode

        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        reward = 0
        last_position = self.challenger.get_position()
        # apply the action to the challenger  {0: "Down", 1: "Up", 2: "Right", 3: "Left"}
        if action == 0:

            self.current_postion = (
                self.current_postion[0], self.current_postion[1]+100)
            if self.current_postion == (900, 900):
                self.challenger.set_position(
                    self.current_postion[0], self.current_postion[1])
                reward = 3000
                done = True
                self.reward_return += reward
                self.draw_elements_on_canvas()
                self.render()
                playsound('src/win.mp3')
                return self.canvas/100, reward, done, {}
            if self.current_postion in self.repeat_path and self.current_postion in self.path:
                reward = -5
                self.challenger.move(0, 100)

            elif self.current_postion in self.path:
                self.count += 10
                reward = self.count
                self.challenger.move(0, 100)
                self.repeat_path.add(self.current_postion)

        elif action == 1:

            self.current_postion = (
                self.current_postion[0], self.current_postion[1]-100)
            if self.current_postion == (900, 900):
                self.challenger.set_position(
                    self.current_postion[0], self.current_postion[1])
                reward = 3000
                done = True
                self.reward_return += reward
                self.draw_elements_on_canvas()
                self.render()
                playsound('src/win.mp3')
                return self.canvas/100, reward, done, {}
            if self.current_postion in self.repeat_path and self.current_postion in self.path:
                reward = -5
                self.challenger.move(0, -100)
            elif self.current_postion in self.path:
                self.count += 10
                reward = self.count
                self.challenger.move(0, -100)
                self.repeat_path.add(self.current_postion)

        elif action == 2:

            self.current_postion = (
                self.current_postion[0]+100, self.current_postion[1])
            if self.current_postion == (900, 900):
                self.challenger.set_position(
                    self.current_postion[0], self.current_postion[1])
                reward = 3000
                done = True
                self.reward_return += reward
                self.draw_elements_on_canvas()
                self.render()
                playsound('src/win.mp3')
                return self.canvas/100, reward, done, {}
            if self.current_postion in self.repeat_path and self.current_postion in self.path:

                reward = -5
                self.challenger.move(100, 0)

            elif self.current_postion in self.path:
                self.count += 10
                reward = self.count
                self.challenger.move(100, 0)
                self.repeat_path.add(self.current_postion)

        elif action == 3:

            self.current_postion = (
                self.current_postion[0]-100, self.current_postion[1])
            if self.current_postion == (900, 900):
                self.challenger.set_position(
                    self.current_postion[0], self.current_postion[1])
                reward = 3000
                done = True
                self.reward_return += reward
                self.draw_elements_on_canvas()
                self.render()
                playsound('src/win.mp3')
                return self.canvas/100, reward, done, {}
            if self.current_postion in self.repeat_path and self.current_postion in self.path:
                reward = -5
                self.challenger.move(-100, 0)

            elif self.current_postion in self.path:
                self.count += 10
                reward = self.count
                self.challenger.move(-100, 0)
                self.repeat_path.add(self.current_postion)

        self.current_postion = self.challenger.get_position()
 #       if self.current_postion in self.check_path:
  #          reward = 20
#        else:
 #           reward = -20
        if last_position == self.current_postion:
            reward = -10
        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()
        # If out, end the episode.
        if self.reward_return <= -3000:
            done = True
            playsound('src/lose.mp3')

        self.reward_return += reward

        return self.current_postion, reward, done, self.get_current_state()


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Challenger(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Challenger, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("src/robot.png") / 255.0
        self.icon_w = 100
        self.icon_h = 100
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Block(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Block, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("src/block.png") / 255.0
        self.icon_w = 100
        self.icon_h = 100
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Win(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Win, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("src/win.png") / 255.0
        self.icon_w = 100
        self.icon_h = 100
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
