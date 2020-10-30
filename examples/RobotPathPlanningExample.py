#
# By: Elizabeth Lopez
#

from planning import RobotPathPlanning
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from ttictoc import tic, toc
import numpy as np

# position of destinations
target = [9, 9]
charging_station = [9, 5]

# obstacles positions
# obstacles = [[2, 4], [2, 3], [3, 3], [6, 1], [8, 9], [6, 9], [7, 2], [8, 4], [9, 3], [6, 5], [6, 4], [1, 2], [8, 6],
#             [4, 8], [3, 8], [7, 7], [8, 1], [5, 9], [2, 0], [3, 0]]


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

#obstacles = [[0, 3], [0, 6], [1, 3], [1, 7], [2, 0], [2, 1], [4, 2], [4, 3], [3, 8], [3, 9], [3, 5], [3, 6],
#             [4, 6], [4, 5], [6, 7], [7, 6], [8, 6], [7, 3], [8, 3], [9, 3]]
# potential fields parameters
repulsive_factor = 61.5
attractive_factor = 2.3
obstacle_radius = 0.5

# path planning initialization
index_i = 0
index_j = 0


font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=12)

fig, ax = plt.subplots(4, 4, figsize=(9, 9))

scenarios_steps = []
for obs_index in range(len(obstacles_total)):
    obstacles = obstacles_total[obs_index]

    robotPlanning = RobotPathPlanning.RobotPathPlanning(target, charging_station, obstacles, repulsive_factor,
                                                        attractive_factor, obstacle_radius, 10, 10)

    ax[index_i, index_j].scatter(*zip(*obstacles), s=60, marker='s', color='black', label='Obstacles')
    ax[index_i, index_j].plot(*zip(*robotPlanning.path_to_destination), marker='o', color='gray', markersize=4,
                              label='Path to destination')
    ax[index_i, index_j].plot(*zip(*robotPlanning.path_to_charging_station), marker=">", color='gray', markersize=4,
                              label='Path to station')
    ax[index_i, index_j].scatter(robotPlanning.goal[0], robotPlanning.goal[1], marker='*', color='black', s=100,
                                 label='Destination',)
    ax[index_i, index_j].scatter(robotPlanning.charging_station[0], robotPlanning.charging_station[1], marker='P',
                                 color='black', s=100,
                                 label='Charging station')

    ax[index_i, index_j].grid(b=True, which='major',color='lightgrey')
    ax[index_i, index_j].set_axisbelow(True)

    ax[index_i, index_j].set_xticks(range(0, 11, 1))
    ax[index_i, index_j].set_yticks(range(0, 11, 1))

    ax[index_i, index_j].set_title("Scenario " + str(obs_index+1), fontproperties = font,  y=-0.18)

    plt.setp(ax[index_i, index_j].get_xticklabels(), visible=False)
    plt.setp(ax[index_i, index_j].get_yticklabels(), visible=False)

    tic()
    robotPlanning.generate_path(target)
    tm_des = toc()

    tic()
    robotPlanning.generate_path(charging_station)
    tm_stat = toc()

    print(len(robotPlanning.path_to_destination))
    scenarios_steps.append([len(robotPlanning.path_to_destination), tm_des, len(robotPlanning.path_to_charging_station), tm_stat])

    if index_j == 2 and index_i == 3:
        ax[index_i, index_j].legend(prop=font, loc='upper center', bbox_to_anchor=(1.0, 0.9, 1.3, .102), shadow=False, ncol=1, mode="expand")
    ">"
    if index_j == 3:
        index_j = 0
        index_i += 1
    else:
        index_j += 1


ax[3, 3].set_axis_off()
# ax[0, 0].set_axisbelow(True)

plt.axis('off')

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)

plt.show()

fig.savefig('scenarios_examples', format='png', dpi=600, bbox_inches = 'tight')

np.savetxt('scenarios_steps_5_9_2.csv', scenarios_steps, delimiter=',')
