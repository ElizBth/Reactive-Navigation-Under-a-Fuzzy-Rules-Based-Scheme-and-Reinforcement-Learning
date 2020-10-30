from planning.RobotPathPlanning import RobotPathPlanning
from learning.Sarsa import Sarsa
import numpy as np
from ttictoc import tic, toc


class RobotBatteryBehaviorSarsa(Sarsa):

    def __init__(self, total_actions, total_states, learning_rate, discount_factor, _target, _charging_station,
                 _obstacles, max_success):
        super().__init__(total_actions, total_states, learning_rate, discount_factor)
        self.actions_selected = []
        self.destiny = 0
        self.reward = []
        self.log = []
        self.max_success = max_success
        self.battery = 100
        self.action_selected = 0
        repulsive_factor = 61.5
        att_factor = 2.3
        obstacle_radius = 0.5
        self.pathPlanning = RobotPathPlanning(_target, _charging_station, _obstacles, repulsive_factor, att_factor,
                                              obstacle_radius, 10, 10)
        self.distance_to_destination = 0
        self.distance_to_charging_station = 0
        self.last_distance_to_charging_station = 0
        self.last_distance_to_destination = 0
        self.init_dist_to_destination = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                 self.pathPlanning.goal)
        self.init_dist_to_station = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                             self.pathPlanning.charging_station)

    def start(self, epochs, steps):
        success = 0
        self.log = []
        self.log.append("nom,epoch,trial,battery,station,target,reward,state,action,destiny,done,fail,other")
        for epoch in range(epochs):

            step = 0
            self.battery = 100

            self.pathPlanning.reset()
            self.update_distance_to_charging_station()
            self.update_distance_to_destination()

            self.last_distance_to_charging_station = self.distance_to_charging_station
            self.last_distance_to_destination = self.distance_to_destination

            self.update_battery_level()
            state = self.get_state(self.battery)
            self.action_selected = self.select_action(0, state)
            while step < steps and self.battery > 0:
                state, self.action_selected, reward = self.step_execution(state, step, self.action_selected)
                self.reward.append(reward)
                self.log.append(
                    [0, epoch, step, self.battery, self.distance_to_charging_station, self.distance_to_destination,
                     self.reward.pop(), state, self.actions_selected.pop(), self.destiny,
                     self.pathPlanning.is_done(), self.pathPlanning.is_failed(),0])
                if self.pathPlanning.is_done():
                    success += 1
                    print("Success")
                    break
                if self.pathPlanning.is_failed():
                    print("Fail")
                    break
                if self.pathPlanning.end:
                    print("End")
                    break
                self.update_battery_level()
                step += 1

            print("Epoch " + str(epoch) + " steps " + str(step) + " battery level " + str(self.battery))
            if success == self.max_success:
                break

    def get_reward(self, action, state):
        if self.distance_to_destination < 35:
            return 10 # 2
        elif self.distance_to_charging_station < 35:
            return 5 # 1
        elif self.distance_to_destination > 50 and self.distance_to_charging_station > 50 and self.battery < 25: # agregué battery
            return -20 # -2
        else:
            return -1 # -1
        #if self.last_distance_to_destination > self.distance_to_destination and self.battery > 40:
        #    return 1
        #if self.last_distance_to_charging_station > self.distance_to_charging_station and self.battery > 40:
        #    return -0.2
        #if self.last_distance_to_charging_station > self.distance_to_charging_station and self.battery <= 40:
        #    return 0.5
        #if self.pathPlanning.is_done():
        #    return 2
        #else:
        #    return -0.5

    def task_execution(self, action):
        self.action_selected = action
        self.actions_selected.append(action)
        if action == 1:
            print("Go to station")
            self.pathPlanning.go_to_next_position(1)
            self.destiny = 1

        elif action == 2:
            print("Wait")
            self.destiny = 2
        else:
            print("Go to destination")
            self.pathPlanning.go_to_next_position(0)
            self.destiny = 0

        self.update_distance_to_charging_station()
        self.update_distance_to_destination()
        self.update_battery_level()
        self.last_distance_to_charging_station = self.distance_to_charging_station
        self.last_distance_to_destination = self.distance_to_destination

        return self.get_state(self.battery)

    def update_battery_level(self):
        battery = self.battery
        battery -= 0.75 if self.battery >= 40 else 1.25
        self.battery = round(battery) if battery > 0 else 0

    def update_distance_to_destination(self):
        self.distance_to_destination = 99 * self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                     self.pathPlanning.goal) / self.init_dist_to_destination
        self.distance_to_destination = max(0, min(99, self.distance_to_destination))

    def update_distance_to_charging_station(self):
        self.distance_to_charging_station = 99 * self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                          self.pathPlanning.charging_station) / self.init_dist_to_station
        self.distance_to_charging_station = max(0, min(99, self.distance_to_charging_station))




if __name__ == "__main__":
    _target = [5, 9]
    _charging_station = [9, 5]

    # obstacles = [[2, 4], [2, 3], [3, 3], [6, 1], [8, 9], [6, 9], [7, 2], [8, 4], [9, 3], [6, 5], [6, 4], [1, 2], [8, 6],
    #             [4, 8], [3, 8], [7, 7], [8, 1], [5, 9], [2, 0], [3, 0]]

    _total_actions = 4

    _states = list(range(101))

    _total_states = _total_actions * len(_states)

    _learning_rate = 0.01

    _discount_factor = 0.99
    print(len(_states))
    obstacles_total = [
        [[0, 1], [0, 3], [5, 0], [5, 2], [8, 2], [1, 3], [4, 3], [7, 4], [8, 4], [0, 5], [5, 5], [3, 6], [8, 6],
         [2, 7], [6, 7], [0, 8], [4, 8], [6, 9], [7, 9], [8, 9]],
        [[1, 1], [2, 1], [3, 1], [2, 2], [2, 3], [6, 0], [6, 1], [6, 2], [7, 2], [8, 2], [8, 5], [2, 4],
         [2, 6], [3, 6], [4, 6], [5, 6], [8, 6], [8, 7], [8, 8], [6, 8]],
        [[1, 1], [2, 1], [3, 1], [2, 2], [2, 3], [1, 2], [3, 3], [6, 1], [7, 1], [8, 1], [7, 2], [7, 3], [8, 3],
         [6, 3], [1, 5], [2, 5], [3, 5], [2, 6], [2, 7], [3, 7]],
        [[0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [1, 2], [2, 2], [3, 2], [2, 3], [2, 4], [4, 0], [5, 0],
         [6, 0], [6, 1], [6, 2], [6, 8], [6, 7], [7, 4], [7, 5], [8, 4]],
        [[1, 2], [2, 1], [3, 1], [2, 4], [2, 5], [2, 7], [3, 6], [3, 7], [4, 2], [5, 2], [5, 6], [5, 7],
         [5, 8], [6, 4], [6, 6], [7, 4], [7, 6], [8, 4], [8, 3], [8, 2]],
        [[1, 1], [1, 2], [2, 1], [3, 2], [3, 3], [4, 3], [5, 3], [1, 4], [0, 1], [2, 6], [4, 7], [3, 8],
         [2, 8], [1, 8], [6, 6], [7, 6], [4, 6], [7, 5], [8, 4], [8, 3]],
        [[1, 2], [1, 3], [2, 4], [3, 5], [3, 6], [5, 1], [6, 0], [7, 0], [8, 0], [8, 1], [7, 3], [6, 3],
         [5, 3], [6, 4], [6, 5], [8, 5], [5, 8], [7, 7], [7, 8], [6, 8]],
        [[3, 0], [3, 1], [2, 1], [7, 0], [7, 1], [8, 1], [3, 3], [3, 4], [4, 5], [4, 6], [2, 4], [2, 5],
         [3, 6], [4, 6], [1, 8], [2, 8], [2, 9], [7, 9], [7, 8], [8, 8]],
        [[3, 0], [3, 1], [2, 1], [7, 0], [7, 1], [8, 1], [3, 3], [3, 4], [4, 5], [4, 6], [2, 4], [2, 5],
         [3, 6], [4, 6], [1, 8], [2, 8], [2, 9], [7, 9], [7, 8], [8, 8]],
        [[1, 4], [2, 4], [3, 4], [4, 4], [4, 5], [5, 3], [5, 2], [6, 2], [6, 1], [5, 5], [1, 7], [5, 7],
         [5, 8], [6, 8], [4, 7], [3, 7], [6, 5], [7, 5], [2, 5], [7, 2]],
        [[1, 1], [1, 2], [1, 3], [3, 1], [4, 1], [5, 1], [7, 1], [7, 2], [7, 0], [5, 4], [6, 4], [7, 4],
         [1, 6], [2, 6], [3, 6], [2, 8], [6, 6], [6, 7], [6, 8], [8, 6]],
        [[1, 1], [1, 3], [0, 5], [0, 6], [1, 9], [2, 6], [3, 3], [3, 9], [4, 0], [4, 7], [5, 5], [6, 3],
         [7, 1], [7, 6], [6, 8], [6, 9], [8, 9], [9, 7], [9, 2], [8, 4]],
        [[1, 1], [1, 2], [1, 7], [1, 8], [2, 1], [3, 1], [4, 5], [4, 6], [5, 6], [5, 5], [5, 8], [6, 8],
         [7, 8], [7, 7], [8, 4], [9, 4], [2, 6], [2, 7], [1, 7], [0, 7]],
        [[0, 4], [1, 1], [1, 5], [1, 8], [2, 7], [2, 2], [3, 3], [4, 0], [5, 1], [6, 3], [7, 2], [8, 1],
         [7, 5], [7, 7], [7, 8], [3, 9], [6, 4], [7, 4], [6, 3], [7, 3]],
        [[0, 3], [0, 6], [1, 3], [1, 7], [2, 0], [2, 1], [4, 2], [4, 3], [3, 8], [3, 9], [3, 5], [3, 6],
         [4, 6], [4, 5], [6, 7], [7, 6], [8, 6], [7, 3], [8, 3], [9, 3]]

    ]

    SAVE_TIME = 0
    SAVE_LOG = 1
    _max_success = 10

    robotBatterySarsaSimulation = RobotBatteryBehaviorSarsa(_total_actions, _states, _learning_rate, _discount_factor,
                                                            _target, _charging_station, obstacles_total[0], _max_success)
    delay = np.zeros(15)
    for obs_index in range(15):
        obstacles = obstacles_total[obs_index]  # np.random.randint(10, size=(20, 2)).T
        rows = []
        for index in range(1):
            robotBatterySarsaSimulation.pathPlanning.obstacles = obstacles
            tic()
            robotBatterySarsaSimulation.start(1000, 10000)

            delay[obs_index] = toc()

        if SAVE_LOG:
            np.savetxt("log_rf2/log_sarsa_rf2_" + str(obs_index) + ".csv",
                   robotBatterySarsaSimulation.log,
                   delimiter=", ",
                   fmt='% s')
    if SAVE_TIME:
        np.savetxt("log_rf2/log_sarsa_delay_rf2.txt",
                   delay,
                   delimiter=", ",
                   fmt='% s')

    print("Game over")
