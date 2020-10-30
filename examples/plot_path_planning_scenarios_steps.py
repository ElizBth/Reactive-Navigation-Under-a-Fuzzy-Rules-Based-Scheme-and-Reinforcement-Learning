import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd

datos = pd.read_csv('scenarios_steps_5_9.csv', header=None)

size = len(datos)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2))

plt.rcdefaults()
#plt.rcParams['text.usetex'] = True
# fig.suptitle('Horizontally stacked subplots')

bar_width = 0.35
opacity = 0.8

barra1 = ax1.bar(range(1, size+1), datos[0], bar_width, alpha=0.35, color='m', label='Path to the destination')

barra2 = ax1.bar(np.add(bar_width,range(1, size+1)), datos[2], bar_width, alpha=0.45, color='b',label='Path to the charging station')

barra3 = ax2.plot(range(1, size+1), datos[1]*1000, alpha=0.35, color='m', marker='o')

barra4 = ax2.plot(range(1, size+1), datos[3]*1000, alpha=0.45, color='b', marker='o')

#plt.ylabel('Número Iteraciones para alcanzar el objetivo', fontsize=12)
#plt.title('Número de obstáculos', fontsize=18)

#plt.legend(prop={'size': 14})

#autolabel(barra1, "center")
#autolabel(barra2, "center")

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', weight='normal', stretch='normal')


ax1.set_axisbelow(True)
ax1.set_yticks(range(3, 20, 3))
ax1.set_xticks(range(1, 16, 1))
ax2.set_xticks(range(1, 16, 1))
#ax1.xaxis.grid()
ax1.yaxis.grid()
ax2.yaxis.grid()

ax2.xaxis.grid()


for label in ax1.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax1.get_yticklabels():
    label.set_fontproperties(ticks_font)

for label in ax2.get_xticklabels():
    label.set_fontproperties(ticks_font)

for label in ax2.get_yticklabels():
    label.set_fontproperties(ticks_font)

ax1.set_ylabel('Number of steps',fontname="Times New Roman")
ax2.set_ylabel('Time (ms)', fontname="Times New Roman")
ax1.set_xlabel('Scenarios \n (a)', fontname="Times New Roman")
ax2.set_xlabel('Scenarios \n (b)', fontname="Times New Roman")

ax1.legend(prop=ticks_font, loc='upper center', bbox_to_anchor=(0.47, 1.13, 1.3, .102), shadow=False, ncol=4, mode="expand")
plt.show()

fig.savefig('path_planning_steps_time_1', format='png', dpi=600, bbox_inches = 'tight')
