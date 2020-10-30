from planning.RobotPathPlanning import RobotPathPlanning
from learning.QLearning import QL


class RobotBatteryBehaviorQL(QL):

    def __init__(self, total_actions, total_states, learning_rate, discount_factor, _target, _charging_station,
                 _obstacles, max_success):
        super().__init__(total_actions, total_states, learning_rate, discount_factor)
        self.actions_selected = []
        self.reward = []
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

    def start(self, epochs, steps):
        success = 0
        for epoch in range(epochs):

            step = 0
            self.battery = 100

            self.pathPlanning.reset()
            self.update_distance_to_charging_station()
            self.update_distance_to_destination()

            self.last_distance_to_charging_station = self.distance_to_charging_station
            self.last_distance_to_destination = self.distance_to_destination

            while step < steps:
                self.update_battery_level()
                state = self.get_state(self.battery)
                self.step_execution(state, step)
                self.reward.append(self.get_reward(self.action_selected, state))
                if self.pathPlanning.is_done():
                    print("Success")
                    break
                if self.pathPlanning.is_failed():
                    print("Fail")
                    break
                if self.pathPlanning.end:
                    print("End")
                    break
                step += 1
            print("Epoch " + str(epoch) + " steps " + str(step) + " battery level " + str(self.battery))
            if success >= self.max_success:
                break

    def get_reward(self, action, state):
        if self.last_distance_to_destination > self.distance_to_destination and self.battery > 40:
            return 1
        if self.last_distance_to_charging_station > self.distance_to_charging_station and self.battery > 40:
            return -0.2
        if self.last_distance_to_charging_station > self.distance_to_charging_station and self.battery <= 40:
            return 0.5
        if self.pathPlanning.is_done():
            return 2
        else:
            return -0.5

    def task_execution(self, action):
        self.action_selected = action
        self.actions_selected.append(action)
        if action == 1:
            print("Go to station")
            self.pathPlanning.go_to_next_position(1)

        elif action == 2:
            print("Wait")
        else:
            print("Go to destination")
            self.pathPlanning.go_to_next_position(0)

        self.update_distance_to_charging_station()
        self.update_distance_to_destination()

        self.last_distance_to_charging_station = self.distance_to_charging_station
        self.last_distance_to_destination = self.distance_to_destination

        return self.get_state(self.battery)

    def update_battery_level(self):
        battery = self.battery
        battery -= 0.75 if self.battery >= 40 else 1.25
        self.battery = round(battery) if battery > 0 else 0

    def update_distance_to_destination(self):
        self.distance_to_destination = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                self.pathPlanning.goal)

    def update_distance_to_charging_station(self):
        self.distance_to_charging_station = self.pathPlanning.get_euclidean_distance(self.pathPlanning.robot,
                                                                                     self.pathPlanning.charging_station)


if __name__ == "__main__":
    _target = [9, 9]
    _charging_station = [9, 6]

    obstacles = [[2, 4], [2, 3], [3, 3], [6, 1], [8, 9], [6, 9], [7, 2], [8, 4], [9, 3], [6, 5], [6, 4], [1, 2], [8, 6],
                 [4, 8], [3, 8], [7, 7], [8, 1], [5, 9], [2, 0], [3, 0]]

    _total_actions = 4

    _max_success = 10

    _states = list(range(101))

    _total_states = _total_actions * len(_states)

    _learning_rate = 0.01

    _discount_factor = 0.99
    print(len(_states))
    robotBatteryFQLSimulation = RobotBatteryBehaviorQL(_total_actions, _states, _learning_rate, _discount_factor,
                                                       _target, _charging_station, obstacles, _max_success)
    robotBatteryFQLSimulation.start(200, 1000)

    print("Game over")
