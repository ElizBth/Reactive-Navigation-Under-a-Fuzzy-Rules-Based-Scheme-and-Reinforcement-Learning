import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

battery_level = 100
index = 0
battery_levels = []
while battery_level > 0:
    battery_levels.append(battery_level)
    battery = battery_level
    battery -= 0.75 if battery_level >= 40 else 1.25
    battery_level = battery if battery > 0 else 0

battery_levels.append(battery_level)

plt.plot(battery_levels)

plt.show()