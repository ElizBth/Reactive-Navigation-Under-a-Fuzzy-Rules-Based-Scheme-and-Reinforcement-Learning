#
# By: Elizabeth Lopez
#

from planning import RobotPathPlanning
import matplotlib.pyplot as plt

# position of destinations
target = [9, 9]
charging_station = [9, 6]

# obstacles positions
obstacles = [[2, 4], [2, 3], [3, 3], [6, 1], [8, 9], [6, 9], [7, 2], [8, 4], [9, 3], [6, 5], [6, 4], [1, 2], [8, 6],
             [4, 8], [3, 8], [7, 7], [8, 1], [5, 9], [2, 0], [3, 0]]

# potential fields parameters
repulsive_factor = 61.5
attractive_factor = 2.3
obstacle_radius = 0.5

# path planning initialization
robotPlanning = RobotPathPlanning.RobotPathPlanning(target, charging_station, obstacles, repulsive_factor,
                                                    attractive_factor, obstacle_radius, 10, 10)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(*zip(*obstacles), s=500, marker='s', color='black')
ax.plot(*zip(*robotPlanning.path_to_destination), marker='o', color='gray', markersize=15, label='Planned path')

ax.scatter(robotPlanning.goal[0], robotPlanning.goal[1], marker='*', color='black', s=1200, label='Destination')
ax.scatter(robotPlanning.charging_station[0], robotPlanning.charging_station[1], marker='P', color='black', s=500, label='Charging station')

ax.grid()

ax.set_xticks(range(0, 11, 1))
ax.set_yticks(range(0, 11, 1))

ax.set_axisbelow(True)

plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)

plt.show()