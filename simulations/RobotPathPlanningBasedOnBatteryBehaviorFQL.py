from learning.FuzzyQLearning import FQL
from fuzzy import FuzzySet as fuzzySet
from planning.RobotPathPlanning import RobotPathPlanning


class RobotBatteryBehaviorFQL(FQL):
    battery = 100
    distance_to_destination = 100
    distance_to_charging_station = 100

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

    def task_execution(self, action):
        self.actions_selected.append(action)
        if action == 1:
            print("Go to station")
            self.pathPlanning.go_to_next_position(1)
        elif action == 2:
            print("Wait")
        else:
            print("Go to destination")
            self.pathPlanning.go_to_next_position(0)
        return self.fis.get_rule_index(self.get_fuzzy_values())

    def get_reward(self, rule, action):
        if 0 < rule < 7 or 10 <= rule <= 11 or 13 <= rule <= 14:
            reward = 1
        elif rule == 15 or rule == 12 or rule == 21 or 23 < rule < 28:
            reward = 5
        else:
            reward = -20
        self.reward.append(reward)
        return reward

    def start(self, epochs, max_trials):
        success = 0
        for epoch in range(0, epochs):
            trial = 0
            while trial < max_trials and self.battery > 10:
                trial += 0
                fuzzy_values = self.get_fuzzy_values()
                self.input_values_per_step.append(fuzzy_values)
                state = self.fis.get_rule_index(fuzzy_values)
                self.states_fallen.append(state)
                self.step_execution(state, trial)

                if self.pathPlanning.is_done():
                    success += 1
                    print("Success!")
                    break

                if self.pathPlanning.is_failed():
                    print("Game over!!")
                    break

            if success >= self.max_success:
                print("Successes reached!")
                break

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
        self.distance_to_destination = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                self.pathPlanning.goal)

    def update_distance_to_charging_station(self):
        self.distance_to_charging_station = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                     self.pathPlanning.charging_station)


if __name__ == "__main__":
    target = [9, 9]
    charging_station = [9, 6]

    obstacles = [[2, 4], [2, 3], [3, 3], [6, 1], [8, 9], [6, 9], [7, 2], [8, 4], [9, 3], [6, 5], [6, 4], [1, 2], [8, 6],
                 [4, 8], [3, 8], [7, 7], [8, 1], [5, 9], [2, 0], [3, 0]]

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

    _max_success = 1

    robotBatteryFQLSimulation = RobotBatteryBehaviorFQL(total_actions, output_function_list, fuzzy_set_list,
                                                        fuzzy_identifiers_list, target, charging_station, obstacles,
                                                        _max_success)
    robotBatteryFQLSimulation.start(1, 1)

