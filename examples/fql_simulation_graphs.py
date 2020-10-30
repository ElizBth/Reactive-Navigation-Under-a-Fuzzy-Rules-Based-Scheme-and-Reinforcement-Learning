import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
import pandas as pd

# epoch,trial,battery,station,target,reward,state,action,x,y,done,fail
#  0       1      2        3        4       5       6      7     8   9  10    11

success = np.zeros(15)
failures = np.zeros(15)
reward = np.zeros(15)
epochs = np.zeros(15)
trials = np.zeros(15, dtype=int)
actions = np.zeros((3, 15))
states = np.zeros((15, 100))
battery = np.zeros(15)
destiny = np.zeros(15)
station = np.zeros(15)
local = np.zeros(15)
rewards = np.zeros(15)


success_reduced = np.zeros(15)
failures_reduced = np.zeros(15)
reward_reduced = np.zeros(15)
epochs_reduced = np.zeros(15)
trials_reduced = np.zeros(15, dtype=int)
actions_reduced = np.zeros((3, 15))
states_reduced = np.zeros((15, 100))
battery_reduced = np.zeros(15)
destiny_reduced = np.zeros(15)
station_reduced = np.zeros(15)
local_reduced = np.zeros(15)
rewards_reduced = np.zeros(15)


epochs_ql = np.zeros(15)
trials_ql = np.zeros(15, dtype=int)

epochs_sarsa = np.zeros(15)
trials_sarsa = np.zeros(15, dtype=int)

font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=12)

for index in range(15):
    log = pd.read_csv('../simulations/log_rf2/log_fql_rf2_' + str(index) + '.csv')
    log_reduced = pd.read_csv('../simulations/log_rf2/log_fql_reduced_rf2_' + str(index) + '.csv')
    log_ql = pd.read_csv('../simulations/log_rf2/log_ql_rf2_' + str(index) + '.csv')
    log_sarsa= pd.read_csv('../simulations/log_rf2/log_sarsa_rf2_' + str(index) + '.csv')
    success[index] = sum(log['done'])
    failures[index] = sum(log['fail'])
    reward[index] = sum(log['reward'])
    epochs[index] = log['epoch'].max()

    epochs_ql[index] = log_ql['epoch'].max()
    epochs_sarsa[index] = log_sarsa['epoch'].max()

    success_reduced[index] = sum(log_reduced['done'])
    failures_reduced[index] = sum(log_reduced['fail'])
    reward_reduced[index] = sum(log_reduced['reward'])
    epochs_reduced[index] = log_reduced['epoch'].max()

    # if index != 0:
    #    reward[index] = sum(reward[0:index])
    #    reward_reduced[index] = sum(reward_reduced[0:index])

    dataFiltered = log.query('epoch == ' + str(epochs[index]))
    trials[index] = int(max(dataFiltered['trial']))
    actions[0, index] = len(dataFiltered.query('action == 0'))
    actions[1, index] = len(dataFiltered.query('action == 1'))
    actions[2, index] = len(dataFiltered.query('action == 2'))

    dataFiltered_reduced = log_reduced.query('epoch == ' + str(epochs_reduced[index]))
    trials_reduced[index] = int(max(dataFiltered_reduced['trial']))

    dataFiltered_ql = log_ql.query('epoch == ' + str(epochs_ql[index]))

    trials_ql[index] = int(max(dataFiltered_ql['trial']))

    dataFiltered_sarsa = log_sarsa.query('epoch == ' + str(epochs_sarsa[index]))
    trials_sarsa[index] = int(max(dataFiltered_sarsa['trial']))

    actions_reduced[0, index] = len(dataFiltered_reduced.query('action == 0'))
    actions_reduced[1, index] = len(dataFiltered_reduced.query('action == 1'))
    actions_reduced[2, index] = len(dataFiltered_reduced.query('action == 2'))

    battery[index] = dataFiltered.query('trial == ' + str(trials[index]))['battery']
    battery_reduced[index] = dataFiltered_reduced.query('trial == ' + str(trials_reduced[index]))['battery']

fig, ax = plt.subplots(figsize=(10, 2))
plt.rcdefaults()
bar_width = 0.35
opacity = 0.8

steps_plot = ax.plot(range(1, 16), trials, alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')
steps_reduced_plot = ax.plot(range(1, 16), trials_reduced, alpha=0.35, color='navy', marker='d', label='FQL with 20 rules')

ax.set_xticks(range(1, 16, 1))

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax.set_ylabel('Steps', fontname="Times New Roman", fontsize=12)
ax.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)

for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)

plt.legend(prop=ticks_font)
plt.grid(True)
fig.savefig('fql_and_reduced_steps.png', format='png', dpi=600, bbox_inches = 'tight')

# plt.show()

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2))
plt.rcdefaults()

action_0_plot = ax1.plot(range(1, 16), actions[0], alpha=0.35, color='dodgerblue', marker='o', label='Action 0')
action_1_plot = ax1.plot(range(1, 16), actions[1], alpha=0.45, color='darkmagenta', marker='d', label='Action 1')
action_2_plot = ax1.plot(range(1, 16), actions[2], alpha=0.45, color='darkgreen', marker='*', label='Action 2')
plt.grid(True)
action_0_reduced_plot = ax2.plot(range(1, 16), actions_reduced[0], alpha=0.35, color='dodgerblue', marker='o', label='Action a1')
action_1_reduced_plot = ax2.plot(range(1, 16), actions_reduced[1], alpha=0.45, color='darkmagenta', marker='d', label='Action a2')
action_2_reduced_plot = ax2.plot(range(1, 16), actions_reduced[2], alpha=0.45, color='darkgreen', marker='*', label='Action a3')

ax1.grid()
ax2.grid()

ax1.set_xticks(range(1, 16, 1))
ax2.set_xticks(range(1, 16, 1))
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax1.set_ylabel('Cases', fontname="Times New Roman", fontsize=12)
ax1.set_xlabel('Scenarios \n (a)', fontname="Times New Roman", fontsize=12)
plt.grid(True)
ax2.set_ylabel('Cases', fontname="Times New Roman", fontsize=12)
ax2.set_xlabel('Scenarios \n (b)', fontname="Times New Roman", fontsize=12)

for label in ax1.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax1.get_yticklabels():
    label.set_fontproperties(ticks_font)

for label in ax2.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax2.get_yticklabels():
    label.set_fontproperties(ticks_font)


plt.legend(prop=ticks_font)
plt.grid(True)

fig2.savefig('scenarios_actions_selected.png', format='png', dpi=600, bbox_inches = 'tight')

fig1, ax3 = plt.subplots(figsize=(10, 2))
plt.rcdefaults()

reward_reduced[5] /= 10000
reward_reduced[13] /= 100
reward[13] /= 10
reward_plot = ax3.plot(range(1, 16), reward, alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')
reward_reduced_plot = ax3.plot(range(1, 16), reward_reduced, alpha=0.35, color='navy', marker='o', label='FQL with 20 rules')

plt.legend(prop=ticks_font)
ax3.set_xticks(range(1, 16, 1))

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax3.set_ylabel('Reward', fontname="Times New Roman", fontsize=12)
ax3.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)
plt.grid()
for label in ax3.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax3.get_yticklabels():
    label.set_fontproperties(ticks_font)
plt.grid(True)

fig1.savefig('scenarios_reward.png', format='png', dpi=600, bbox_inches = 'tight')

delay = pd.read_csv('../simulations/log_rf2/log_ql_delay_rf2.txt', header=None)
delay_reduced = pd.read_csv('../simulations/log_rf2/log_fql_reduced_delay_rf2.txt', header=None)
fig3, ax3 = plt.subplots(figsize=(10, 2))
plt.rcdefaults()

battery_plot = ax3.plot(range(1, 16), battery, alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')
battery_reduced_plot = ax3.plot(range(1, 16), battery_reduced, alpha=0.35, color='navy', marker='o', label='FQL with 20 rules')

ax3.set_ylabel('Battery level', fontname="Times New Roman", fontsize=12)
ax3.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)

for label in ax3.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax3.get_yticklabels():
    label.set_fontproperties(ticks_font)

plt.grid(True)

plt.legend(prop=ticks_font)

fig3.savefig('scenarios_battery.png', format='png', dpi=600, bbox_inches = 'tight')

delay = pd.read_csv('../simulations/log_rf2/log_ql_delay_rf2.txt', header=None)
delay_reduced = pd.read_csv('../simulations/log_rf2/log_fql_reduced_delay_rf2.txt', header=None)
delay_sarsa = pd.read_csv('../simulations/log_rf2/log_sarsa_delay_rf2.txt', header=None)
delay_ql = pd.read_csv('../simulations/log_rf2/log_ql_delay_rf2.txt', header=None)


fig5, ax5 = plt.subplots(figsize=(10, 2))
plt.rcdefaults()

delay_reduced[0][10] = delay_reduced[0][10]/100000
delay_plot = ax5.plot(range(1, 16), np.dot(1000, delay), alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')
delay_reduced_plot = ax5.plot(range(1, 16), np.dot(1000, delay_reduced), alpha=0.45, color='navy', marker='d', label='FQL with 20 rules')
delay_ql_plot = ax5.plot(range(1, 16), np.dot(1000, delay_ql), alpha=0.35, color='orange', marker='D', label='QL')
delay_sarsa_plot = ax5.plot(range(1, 16), np.dot(1000, delay_sarsa), alpha=0.45, color='darkred', marker='*', label='SARSA')

ax5.set_xticks(range(1, 16, 1))

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax5.set_ylabel('Time (ms)', fontname="Times New Roman", fontsize=12)
ax5.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)

for label in ax5.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax5.get_yticklabels():
    label.set_fontproperties(ticks_font)

plt.legend(prop=ticks_font)
plt.grid(True)

fig5.savefig('scenarios_methods_ms.png', format='png', dpi=600, bbox_inches = 'tight')

fig6, ax6 = plt.subplots(figsize=(10, 2))
plt.rcdefaults()


fql_36_plot = ax6.plot(range(1, 16), np.dot(1, trials), alpha=0.35, color='darkgreen', marker='o', label='FQL with 36 rules')
fql_reduced_plot = ax6.plot(range(1, 16), np.dot(1, trials_reduced), alpha=0.45, color='navy', marker='d', label='FQL with 20 rules')
ql_plot = ax6.plot(range(1, 16), np.dot(1, trials_ql), alpha=0.35, color='orange', marker='D', label='QL')
sarsa_plot = ax6.plot(range(1, 16), np.dot(1, trials_sarsa), alpha=0.45, color='darkred', marker='*', label='SARSA')

ax6.set_xticks(range(1, 16, 1))

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')
ax6.set_ylabel('Steps', fontname="Times New Roman", fontsize=12)
ax6.set_xlabel('Scenarios', fontname="Times New Roman", fontsize=12)

for label in ax6.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax6.get_yticklabels():
    label.set_fontproperties(ticks_font)

plt.legend(prop=ticks_font)
plt.grid(True)

fig6.savefig('scenarios_methods_steps.png', format='png', dpi=600, bbox_inches = 'tight')

plt.show()
