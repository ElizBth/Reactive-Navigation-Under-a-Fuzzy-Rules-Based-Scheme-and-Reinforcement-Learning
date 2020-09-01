#
# This file contains the class RobotPathPlanning and Direction where
# Direction: contains a labels with possible robot movements
# RobotPathPlanning: is an extension of the class ArtificialPotentialFields, which serves for path generation
#
# Last rev. 1.2
# Date: 01/09/2020
# By: Elizabeth Lopez
#
import enum
import itertools
import numpy as np
from planning.ArtificialPotentialFields import ArtificialPotentialFields


class Direction(enum.Enum):
    FRONT = 0
    LEFT_90 = 1
    RIGHT_90 = 2
    BACK = 3


class RobotPathPlanning(ArtificialPotentialFields):
    _window = [[-1, 0], [0, 1], [0, -1], [1, 0]]

    def __init__(self, destination, charging_station, obstacles, repulsive_factor, attractive_factor, obstacle_radius,
                 max_grid_x, max_grid_y):
        super().__init__(repulsive_factor, attractive_factor, obstacle_radius)
        self.set_environment(destination, obstacles)
        self.robot = [0, 0]
        self.last_destination = 0
        self.goal = destination
        self.grid_dims = [[0, 0], [max_grid_x, max_grid_y]]
        self.direction = Direction.FRONT
        self.charging_station = charging_station
        self.path_to_destination = self.generate_path(destination)
        self.path_to_charging_station = self.generate_path(charging_station)
        self.path_to_follow = self.path_to_destination

    #
    # This method returns a list with the path generated to reach destination
    # destination -> [x, y] is destination position
    #
    def generate_path(self, destination):
        self.destination = destination
        path = []

        resultant_forces = np.zeros((10, 10), dtype=float)

        for position in itertools.product(range(self.grid_dims[1][0]), range(self.grid_dims[1][1])):
            resultant_forces[position[0]][position[1]] = self.get_resultant_force(position)

        count = 0
        tmp_robot_position = self.robot
        tmp_tmp_robot_position = self.robot
        while count < 50:
            resultant_force = -1000000
            for index in range(len(self._window)):

                tmp_position = np.add(tmp_robot_position, self._window[index])

                if 0 > tmp_position[0] or 10 <= tmp_position[0] or 0 > tmp_position[1] or 10 <= tmp_position[1]:
                    tmp_resultant_force = -1000000
                else:
                    tmp_resultant_force = resultant_forces[int(tmp_position[0])][int(tmp_position[1])]

                if tmp_resultant_force > resultant_force:
                    tmp_tmp_robot_position = tmp_position
                    resultant_force = tmp_resultant_force

            tmp_robot_position = tmp_tmp_robot_position
            path.append(tmp_tmp_robot_position)

            if resultant_forces[tmp_robot_position[0]][tmp_robot_position[1]] == 0:
                break
            count += 1

        return path

    #
    # This method updates the parameters of next position. If destination change to go to charging station
    # a new path is generated. Robot position is updated with the next position on the list with the path followed.
    # destination -> [x, y]
    #
    def go_to_next_position(self, destination):
        self.update_step_parameters(destination)
        self.update_position()

    #
    # This method verifies if destinations has changed and if it is true generate a new path to the new destination
    # destination -> [x, y]
    #
    def update_step_parameters(self, destination):
        if destination != self.last_destination:
            if destination:
                self.path_to_follow = self.generate_path(self.charging_station)
            else:
                self.path_to_follow = self.generate_path(self.destination)
            self.last_destination = destination

    #
    # This method gets next position and update robot's position
    #
    def update_position(self):
        self.robot = self.path_to_follow.pop(0)
        print(self.robot)



