from learning.FuzzyQLearning import FQL
from fuzzy import FuzzySet as fuzzySet
from planning.RobotPathPlanning import RobotPathPlanning
import numpy as np
from ttictoc import tic, toc


class RobotBatteryBehaviorFQL(FQL):
    battery = 100
    distance_to_destination = 100
    distance_to_charging_station = 100

    # row = []

    def __init__(self, actions, output_functions, fuzzy_sets, fuzzy_sets_identifiers, _target, _charging_station,
                 _obstacles, max_success):
        super().__init__(0.01, 0.99, actions, output_functions, fuzzy_sets, fuzzy_sets_identifiers)
        repulsive_factor = 61.5
        att_factor = 2.3
        obstacle_radius = 0.5
        self.pathPlanning = RobotPathPlanning(_target, _charging_station, _obstacles, repulsive_factor, att_factor,
                                              obstacle_radius, 10, 10)
        self.actions_selected = []
        self.states_fallen = []
        self.reward = []
        self.input_values_per_step = []
        self.max_success = max_success
        self.init_dist_to_destination = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                 self.pathPlanning.goal)
        self.init_dist_to_station = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                             self.pathPlanning.charging_station)

        self.position = [0, 0]

        self.log = []
        # self.row = []

    def task_execution(self, action):
        self.actions_selected.append(action)
        if action == 1:
            print("Go to station battery " + str(self.battery))
            self.position = self.pathPlanning.go_to_next_position(1)
        elif action == 2:
            print("Wait battery " + str(self.battery))
        else:
            print("Go to destination battery " + str(self.battery))
            self.position = self.pathPlanning.go_to_next_position(0)
        return self.get_rule_index(self.get_fuzzy_values())

    def get_reward(self, rule, action):
        if self.distance_to_destination < 35:
            reward = 10 # 2
        elif self.distance_to_charging_station < 35:
            reward = 5 # 1
        elif self.distance_to_destination > 50 and self.distance_to_charging_station > 50 and self.battery < 25: # agregué battery
            reward = -20 # -2
        else:
            reward = -1 # -1

        self.reward.append(reward)
        return reward

    def start(self, epochs, max_trials):
        success = 0
        self.log = []
        self.log.append("nom,epoch,trial,battery,station,target,reward,state,action,x,y,done,fail")
        for epoch in range(0, epochs):
            trial = 0
            self.battery = 99
            self.pathPlanning.reset()
            while trial < max_trials and self.battery > 10:
                trial += 1
                fuzzy_values = self.get_fuzzy_values()
                self.input_values_per_step.append(fuzzy_values)
                state = self.get_rule_index(fuzzy_values)
                self.states_fallen.append(state)
                self.step_execution(state, trial)

                self.log.append(
                    [0, epoch, trial, self.battery, self.distance_to_charging_station, self.distance_to_destination,
                     self.reward.pop(), state, self.actions_selected.pop(), self.position[0],
                     self.position[1], self.pathPlanning.is_done(), self.pathPlanning.is_failed(),0])

                if self.pathPlanning.is_done():
                    success += 1
                    print("Success!")
                    break

                if self.pathPlanning.is_failed():
                    print("Game over!!")
                    break

            if success == self.max_success:
                print("Successes reached!")
                return trial
                # break

    def get_fuzzy_values(self):
        self.update_battery_level()
        self.update_distance_to_destination()
        self.update_distance_to_charging_station()
        return [round(self.battery), round(self.distance_to_destination), round(self.distance_to_charging_station)]

    def update_battery_level(self):
        battery = self.battery
        battery -= 0.75 if self.battery >= 40 else 1.25
        self.battery = battery if battery > 0 else 0

    def update_distance_to_destination(self):
        self.distance_to_destination = 99 * self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                      self.pathPlanning.goal) / self.init_dist_to_destination
        self.distance_to_destination = max(0, min(99, self.distance_to_destination))

    def update_distance_to_charging_station(self):
        self.distance_to_charging_station = 99 * self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                     self.pathPlanning.charging_station) / self.init_dist_to_station
        self.distance_to_charging_station = max(0, min(99, self.distance_to_charging_station))


if __name__ == "__main__":

    #version rf2
    target = [5, 9]
    charging_station = [9, 5]

    battery_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                         fuzzySet.FuzzyFunctionsIdentifiers.Triangular, fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
    battery_identifiers = ["very_low", "low", "medium", "high"]
    battery_parameters = [[0, 40, 0, 20, 1], [20, 60, 40, 1], [40, 80, 60, 1], [60, 100, 80, 100, 1]]

    target_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                        fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
    target_identifiers = ["close", "near", "far"]
    target_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

    station_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                         fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
    station_identifiers = ["close", "near", "far"]
    station_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

    battery_fuzzy_set = fuzzySet.FuzzySet(functions=battery_functions, identifiers=battery_identifiers,
                                          parameters=battery_parameters, step=1, end=100)

    target_fuzzy_set = fuzzySet.FuzzySet(functions=target_functions, identifiers=target_identifiers,
                                         parameters=target_parameters, step=1, end=100)

    station_fuzzy_set = fuzzySet.FuzzySet(functions=station_functions, identifiers=station_identifiers,
                                          parameters=station_parameters, step=1, end=100)
    fuzzy_set_list = [battery_fuzzy_set, target_fuzzy_set, station_fuzzy_set]

    fuzzy_identifiers_list = [battery_identifiers, target_identifiers, station_identifiers]

    output_function_list = [-10, 0, 10]

    total_actions = 3

    obstacles_total = [
        [[0, 1], [0, 3], [5, 0], [5, 2], [8, 2], [1, 3], [4, 3], [7, 4], [8, 4], [0, 5], [5, 5], [3, 6], [8, 6],
         [2, 7], [6, 7], [0, 8], [4, 8], [6, 9], [7, 9], [8, 9]],
        [[1, 1], [2, 1], [3, 1], [2, 2], [2, 3], [6, 0], [6, 1], [6, 2], [7, 2], [8, 2], [8, 5], [2, 4],
         [2, 6], [3, 6], [4, 6], [5, 6], [8, 6], [8, 7], [8, 8], [6, 8]],
        [[1, 1], [2, 1], [3, 1], [2, 2], [2, 3], [1, 2], [3, 3], [6, 1], [7, 1], [8, 1], [7, 2], [7, 3], [8, 3],
         [6, 3], [1, 5], [2, 5], [3, 5], [2, 6], [2, 7], [3, 7]],
        [[0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [1, 2], [2, 2], [3, 2], [2, 3], [2, 4], [4, 0], [5, 0],
         [6, 0], [6, 1], [6, 2], [6, 6], [6, 7], [7, 4], [7, 5], [8, 4]],
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

    _max_success = 10
    SAVE_TIME = 0
    SAVE_LOG = 1
    delay = np.zeros(15)
    robotBatteryFQLSimulation = RobotBatteryBehaviorFQL(total_actions, output_function_list, fuzzy_set_list,
                                                        fuzzy_identifiers_list, target, charging_station,
                                                        obstacles_total[0],
                                                        _max_success)
    for obs_index in range(10,11):
        obstacles = obstacles_total[obs_index]  # np.random.randint(10, size=(20, 2)).T
        rows = []
        for index in range(1):
            # obstacles = np.random.randint(10, size=(20, 2)).T

            # print(obstacles)
            tic()
            trial = robotBatteryFQLSimulation.start(10000, 1000)
            robotBatteryFQLSimulation.pathPlanning.obstacles = obstacles

            delay[obs_index] = toc()
            #data_string = str(trial) + "," + str(np.sum(robotBatteryFQLSimulation.reward))
            #rows.append(data_string)

        # print(robotBatteryFQLSimulation.log)
        if SAVE_LOG:
            np.savetxt("log_rf2/log_fql_rf2_" + str(obs_index) + ".csv",
                   robotBatteryFQLSimulation.log,
                   delimiter=", ",
                   fmt='% s')

    if SAVE_TIME:
        np.savetxt("log_rf2/log_fql_delay_rf2.txt",
                   delay,
                   delimiter=", ",
                   fmt='% s')



