#
# This file contains the class ArtificialPotentialFields where it can found the attractive and repulsive function
# for fields. Using a destination point and the coordinates of the obstacles.
# Last rev. 1.2
# By: Elizabeth Lopez
# Date: 01/09/2020
import numpy as np


class ArtificialPotentialFields:

    def __init__(self, repulsive_factor, attractive_factor, obstacle_radius):
        self.repulsive_factor = repulsive_factor
        self.attractive_factor = attractive_factor
        self.obstacle_radius = obstacle_radius
        self.robot = [0, 0]
        self.obstacles = []
        self.destination = np.zeros(2)

    #
    # This method is used to set the destinations and obstacles coordinates
    # destination -> [x, y]
    # obstacles -> [ [x1, y1], [ x2, y2], [ x3, y3], ..., [xn, yn]]
    #
    def set_environment(self, destination, obstacles):
        self.destination = destination
        self.obstacles = obstacles

    #
    # This method verifies if robot is on destination position
    # return True/False
    #
    def is_done(self):
        if self.destination[0] == self.robot[0] and self.destination[1] == self.robot[1]:
            return 1
        else:
            return 0

    #
    # This method verifies if robot is on a obstacle position
    # return True/False
    #
    def is_failed(self):
        boo = list(self.robot) in list(self.obstacles)
        return 1 if boo else 0

    #
    # This method appends a new obstacle position
    # position -> [x, y]
    #
    def add_obstacle(self, position):
        self.obstacles.append(position)

    #
    # This method returns the Euclidean distance between destination and robot
    # destination -> [x, y]
    #
    def get_distance_to_destination(self, destination):
        return self.get_euclidean_distance(self.robot, destination)

    #
    # This method sets destination position
    #
    def set_destination(self, destination):
        self.destination = destination

    #
    # This method returns the Euclidean distance between obj1 and obj2
    # obj1 -> [x, y]
    # obj2 -> [x, y]
    #
    def get_euclidean_distance(self,  obj1, obj2):
        return np.sqrt(((obj2[0] - obj1[0]) ** 2) + ((obj2[1] - obj1[1]) ** 2))

    #
    # This method returns the attractive force between robot and destination
    # position -> [x, y] is the robot position
    #
    def get_attractive_force(self,  position):
        distance_to_goal = self.get_euclidean_distance(position, self.destination)
        return -self.attractive_factor * distance_to_goal

    #
    # This method returns the repulsive force between robot and obstacles
    # position -> [x, y] is the robot position
    #
    def get_repulsive_force(self,  position):
        repulsive_force = 0
        if len(self.obstacles) > 0:
            for obsIndex in range(0, len(self.obstacles)):
                distance_to_obs = self.get_euclidean_distance(position, self.obstacles[obsIndex])

                if distance_to_obs == 0:
                    repulsive_force += self.repulsive_factor * (-1 / self.obstacle_radius)
                elif distance_to_obs <= self.obstacle_radius:
                    dist_factor = (self.obstacles[obsIndex][0] - position[0]) ** 2 + (
                                (self.obstacles[obsIndex][1] - position[1]) ** 2)
                    repulsive_force += self.repulsive_factor * (1 / distance_to_obs - 1 / self.obstacle_radius) * dist_factor / (
                                distance_to_obs ** 2)
                else:
                    repulsive_force += 0
        return repulsive_force

    #
    # This method returns the resultant force in the position
    # position -> [x, y] is robot position
    #
    def get_resultant_force(self,  position):
        return self.get_attractive_force(position) + self.get_repulsive_force(position)
