import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
import pandas as pd

# epoch,trial,battery,station,target,reward,state,action,x,y,done,fail
#  0       1      2        3        4       5       6      7     8   9  10    11

no = 14
success = np.zeros(no)
failures = np.zeros(no)
reward = np.zeros(no)
epochs = np.zeros(no)
trials = np.zeros(no, dtype=int)
actions = np.zeros((3, no))
states = np.zeros((no, 100))
battery = np.zeros(no)
destiny = np.zeros(no)
station = np.zeros(no)
local = np.zeros(no)
rewards = np.zeros(no)


success_reduced = np.zeros(no)
failures_reduced = np.zeros(no)
reward_reduced = np.zeros(no)
epochs_reduced = np.zeros(no)
trials_reduced = np.zeros(no, dtype=int)
actions_reduced = np.zeros((3, no))
states_reduced = np.zeros((no, 100))
battery_reduced = np.zeros(no)
destiny_reduced = np.zeros(no)
station_reduced = np.zeros(no)
local_reduced = np.zeros(no)
rewards_reduced = np.zeros(no)


epochs_ql = np.zeros(no)
trials_ql = np.zeros(no, dtype=int)

epochs_sarsa = np.zeros(no)
trials_sarsa = np.zeros(no, dtype=int)

font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=12)

for index in range(no):
    log = pd.read_csv('../simulations/log_ext_1/log_ext_v2_14_' + str(index) + '.csv')
    success[index] = sum(log['done'])
    failures[index] = sum(log['fail'])
    reward[index] = sum(log['reward'])
    epochs[index] = log['epoch'].max()



    # if index != 0:
    #    reward[index] = sum(reward[0:index])
    #    reward_reduced[index] = sum(reward_reduced[0:index])

    dataFiltered = log.query('epoch == ' + str(epochs[index]))
    trials[index] = int(max(dataFiltered['trial']))
    actions[0, index] = len(dataFiltered.query('action == 0'))
    actions[1, index] = len(dataFiltered.query('action == 1'))
    actions[2, index] = len(dataFiltered.query('action == 2'))

    battery[index] = dataFiltered.query('trial == ' + str(trials[index]))['battery']

fig, ax = plt.subplots(figsize=(10, 2))
plt.rcdefaults()
bar_width = 0.35
opacity = 0.8

steps_plot = ax.plot(range(1, 15), trials, alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')

ax.set_xticks(range(1, 15, 1))

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax.set_ylabel('Steps', fontname="Times New Roman", fontsize=12)
ax.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)

for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)

plt.legend(prop=ticks_font)
plt.grid(True)
fig.savefig('fql_ext_cs_steps.png', format='png', dpi=600, bbox_inches = 'tight')

# plt.show()

fig2, ax1 = plt.subplots(figsize=(10, 2))
plt.rcdefaults()

action_0_plot = ax1.plot(range(1, 15), actions[0], alpha=0.35, color='dodgerblue', marker='o', label='Action 0')
action_1_plot = ax1.plot(range(1, 15), actions[1], alpha=0.45, color='darkmagenta', marker='d', label='Action 1')
plt.grid(True)
ax1.grid()
ax1.set_xticks(range(1, 15, 1))
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax1.set_ylabel('Cases', fontname="Times New Roman", fontsize=12)
ax1.set_xlabel('Scenarios \n (a)', fontname="Times New Roman", fontsize=12)
plt.grid(True)

for label in ax1.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax1.get_yticklabels():
    label.set_fontproperties(ticks_font)



plt.legend(prop=ticks_font)
plt.grid(True)

fig2.savefig('scenarios_actions_selected_ext_cs.png', format='png', dpi=600, bbox_inches = 'tight')

fig1, ax3 = plt.subplots(figsize=(10, 2))
plt.rcdefaults()

reward_reduced[5] /= 10000
reward_reduced[13] /= 100
reward[13] /= 10
reward_plot = ax3.plot(range(1, 15), reward, alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')

plt.legend(prop=ticks_font)
ax3.set_xticks(range(1, 15, 1))

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax3.set_ylabel('Reward', fontname="Times New Roman", fontsize=12)
ax3.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)
plt.grid()
for label in ax3.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax3.get_yticklabels():
    label.set_fontproperties(ticks_font)
plt.grid(True)

fig1.savefig('scenarios_reward_ext_cs.png', format='png', dpi=600, bbox_inches = 'tight')

#delay = pd.read_csv('../simulations/log_rf2/log_ql_delay_rf2.txt', header=None)
#delay_reduced = pd.read_csv('../simulations/log_rf2/log_fql_reduced_delay_rf2.txt', header=None)
fig3, ax3 = plt.subplots(figsize=(10, 2))
plt.rcdefaults()

battery_plot = ax3.plot(range(1, 15), battery, alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')

ax3.set_ylabel('Battery level', fontname="Times New Roman", fontsize=12)
ax3.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)

for label in ax3.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax3.get_yticklabels():
    label.set_fontproperties(ticks_font)

plt.grid(True)

plt.legend(prop=ticks_font)

fig3.savefig('scenarios_battery_ext_cs.png', format='png', dpi=600, bbox_inches = 'tight')

plt.show()
