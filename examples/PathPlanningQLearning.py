from learning import QLearning as method

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from ttictoc import tic, toc
import numpy as np
import itertools as it


class PathPlanningQLearning(method.QL):

    def __init__(self, actions, states, learning_rate, discount_factor, target, obstacles):
        super().__init__(actions, states, learning_rate, discount_factor)
        self.logs = []
        self.target = target
        self.obstacles = obstacles
        self.robot = [0, 0]
        self.last_robot = [0, 0]
        self.path = []

    def start(self, epochs, max_steps, done=1, scenario=1):
        self.logs = []
        acc_reward = 0
        self.robot = [0, 0]
        self.path = []
        self.path.append(self.robot)
        success = 0
        failure = 0
        successes = 0
        for epoch in range(epochs):
            for step in range(max_steps):
                tic()
                _state = q_learning.get_state(tuple(self.robot))
                _new_state, _action, _reward = q_learning.step_execution(_state, step)
                acc_reward += _reward
                delay = toc()

                success = (self.robot[0] == self.target[0] and self.robot[1] == self.target[1])
                failure = list(self.robot) in list(self.obstacles)

                self.logs.append([epoch, step, _action, _reward, success, failure, delay])

                if success or failure:
                    np.savetxt("ql_scenario_" + str(scenario) + ".csv", self.logs, delimiter=',')
                    break
            if success:
                successes += 1
                if successes == done:
                    print("Success!")
                    break
            else:
                if failure:
                    self.path = []
                    self.robot = [0, 0]
                    self.path.append(self.robot)
                pass

        if success:
            print("Success!")
        else:
            print("Failed!")

    def task_execution(self, action):
        self.last_robot = self.robot
        if action == 0:  # forwards
            if self.robot[1] < 9:
                self.robot = np.add(self.robot, [0, 1])
        elif action == 1:  # right
            if self.robot[0] < 9:
                self.robot = np.add(self.robot, [1, 0])
        elif action == 2:  # left
            if self.robot[0] != 0:
                self.robot = np.add(self.robot, [-1, 0])
        else:  # backwards
            if self.robot[1] != 0:
                self.robot = np.add(self.robot, [0, -1])
        self.path.append(self.robot)
        print(self.robot)
        return q_learning.get_state(tuple(self.robot))

    def get_reward(self, action, state):
        reached = (self.robot[0] == self.target[0] and self.robot[1] == self.target[1])
        failed = list(self.robot) in list(self.obstacles)

        if reached:
            return 1
        else:
            if self.robot is self.last_robot:
                return -5
            if failed:
                return -10
            return -0.01


if __name__ == "__main__":

    # position of destinations
    _target = [5, 9]
    charging_station = [9, 5]

    # obstacles positions

    obstacles_total = [
        [[0, 1], [0, 3], [5, 0], [5, 2], [8, 2], [1, 3], [4, 3], [7, 4], [8, 4], [0, 5], [5, 5], [3, 9], [8, 6],
         [2, 7], [6, 7], [0, 8], [4, 8], [6, 9], [7, 9], [8, 9]],
        [[1, 1], [2, 1], [3, 1], [2, 2], [2, 3], [6, 0], [6, 1], [6, 2], [7, 2], [8, 2], [8, 5], [2, 4],
         [2, 6], [3, 6], [4, 6], [5, 6], [8, 6], [8, 7], [8, 8], [6, 8]],
        [[1, 1], [2, 1], [3, 1], [2, 2], [2, 3], [1, 2], [3, 3], [6, 1], [7, 1], [8, 1], [7, 2], [7, 3], [8, 3],
         [6, 1], [1, 5], [2, 5], [3, 5], [2, 6], [2, 7], [3, 7]],
        [[0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [1, 2], [2, 2], [3, 2], [2, 3], [2, 4], [4, 0], [5, 0],
         [6, 0], [6, 1], [6, 2], [7, 7], [6, 7], [7, 4], [7, 5], [8, 4]],
        [[1, 3], [2, 1], [3, 1], [2, 4], [2, 5], [3, 9], [3, 6], [3, 7], [4, 2], [5, 2], [5, 6], [5, 7],
         [5, 8], [6, 4], [6, 6], [7, 4], [7, 6], [8, 4], [8, 1], [8, 2]],
        [[1, 1], [1, 2], [2, 1], [3, 2], [3, 3], [4, 3], [5, 3], [1, 4], [0, 1], [2, 6], [4, 7], [3, 8],
         [2, 8], [1, 8], [7, 7], [7, 6], [4, 6], [7, 5], [8, 0], [8, 3]],
        [[1, 2], [1, 3], [2, 4], [3, 5], [3, 6], [5, 1], [7, 1], [7, 0], [8, 0], [8, 1], [7, 3], [6, 3],
         [5, 3], [6, 4], [6, 5], [8, 5], [5, 8], [7, 7], [3, 0], [6, 8]],
        [[0, 3], [3, 1], [2, 1], [7, 0], [7, 1], [8, 1], [3, 3], [3, 4], [4, 5], [4, 6], [2, 4], [2, 5],
         [3, 6], [4, 6], [1, 8], [2, 8], [2, 9], [7, 9], [7, 8], [8, 8]],
        [[2, 2], [3, 1], [2, 1], [7, 0], [7, 1], [8, 1], [2, 3], [3, 4], [4, 5], [4, 6], [2, 4], [2, 5],
         [3, 6], [4, 6], [1, 8], [2, 8], [2, 9], [7, 9], [7, 8], [8, 8]],
        [[1, 4], [2, 4], [3, 2], [4, 4], [4, 5], [6, 3], [5, 2], [6, 2], [6, 0], [5, 5], [1, 7], [5, 7],
         [5, 8], [6, 8], [4, 7], [3, 7], [6, 5], [7, 5], [2, 5], [7, 2]],
        [[1, 1], [1, 2], [1, 3], [3, 1], [4, 1], [5, 1], [7, 1], [7, 2], [7, 0], [5, 4], [6, 4], [7, 4],
         [1, 6], [2, 6], [3, 6], [2, 8], [6, 6], [6, 7], [6, 8], [8, 6]],
        [[1, 1], [1, 3], [0, 5], [0, 6], [1, 9], [2, 6], [3, 3], [3, 9], [4, 0], [4, 7], [5, 5], [6, 3],
         [7, 1], [7, 6], [6, 8], [6, 9], [8, 9], [9, 7], [9, 2], [8, 4]],
        [[1, 1], [1, 2], [1, 7], [1, 8], [2, 1], [3, 1], [4, 5], [4, 6], [5, 6], [5, 5], [5, 8], [6, 8],
         [7, 8], [7, 7], [9, 3], [9, 4], [2, 6], [2, 7], [1, 7], [0, 7]],
        [[0, 4], [1, 1], [1, 5], [1, 8], [2, 7], [2, 2], [3, 3], [4, 0], [5, 1], [6, 3], [8, 3], [8, 1],
         [7, 5], [7, 7], [7, 8], [3, 9], [6, 4], [7, 4], [6, 3], [7, 3]],
        [[0, 3], [0, 6], [1, 3], [1, 7], [2, 0], [2, 1], [4, 2], [4, 3], [3, 8], [3, 9], [3, 5], [3, 6],
         [4, 6], [4, 5], [6, 5], [7, 6], [8, 6], [7, 3], [8, 3], [9, 3]]

    ]

    _learning_rate = 0.01
    _discount_factor = 0.99
    _actions = 4
    _robot = [0, 0]
    _states = list(it.product(range(0, 10), range(0, 10)))

    for obs_index in range(len(obstacles_total)):
        q_learning = PathPlanningQLearning(4, _states, _learning_rate, _discount_factor, _target, obstacles_total[0])

        q_learning.start(1000000, 100000, 1, obs_index)
        #print(q_learning.q_table)
        # print(str(q_learning.path) + " reward " + str(q_learning.reward))
        np.savetxt('ql_path_scenario_' + str(obs_index) + '.csv', q_learning.path, delimiter=',')


