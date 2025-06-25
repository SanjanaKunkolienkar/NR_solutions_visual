import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

cwd = os.getcwd()
path = os.path.join(cwd, 'Inputs\\New_PW_solutions_2bus_Polar.csv')

# --------------------------
# Load CSV
# --------------------------
df = pd.read_csv(path, skiprows=1)
voltages = df.iloc[:, 0].values
angles = df.columns[1:].astype(float)
Z = df.iloc[:, 1:].values.astype(int)

A, V = np.meshgrid(angles, voltages)
max_iter = np.max(np.abs(Z))

# --------------------------
# Create custom diverging colormap
# --------------------------

# Define colors: from negative to positive
# Order: black -> dark yellow -> light yellow -> white -> light red -> dark red -> black
# cmap_colors = [
#     #(0.0, 'black'),      # start (negative extreme)
#     (0.0, '#CCCC00'),   # dark yellow
#     (0.45, '#FFFF99'),   # light yellow
#     (0.5, 'white'),      # midpoint (zero)
#     (0.55, '#FF9999'),   # light red
#     (1.0, '#990000'),   # dark red
#     #(1.0, 'black')       # end (positive extreme)
# ]

cmap_colors = [
    (0.0, '#4B0082'),  # deep purple (negative extreme)
    (0.2, '#6A4C93'),
    (0.35, '#9F86C0'),
    (0.5, '#E0D4A3'),  # beige/light at zero
    (0.65, '#F6AE2D'),
    (0.8, '#F26419'),
    (1.0, '#B23A00')   # dark brown/orange (positive extreme)
]

custom_cmap = LinearSegmentedColormap.from_list('yellow_red_black', cmap_colors)

# --------------------------
# Plot
# --------------------------
fig, ax = plt.subplots(figsize=(14, 7))
# set font size
plt.rcParams['font.size'] = 20  # Set the desired font size (e.g., 12)

c = ax.pcolormesh(A, V, Z,
                  cmap=custom_cmap,
                  vmin=-max_iter,
                  vmax=max_iter,
                  shading='auto')

cb = fig.colorbar(c, ax=ax, label='Iterations')

ax.set_xlabel('V angle [deg]', fontsize=20)
ax.set_ylabel('V mag [pu]', fontsize=20)
# ax.set_title('NR ROC', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

plt.tight_layout()
plt.savefig('NR_ROC_2bus.png', dpi=600, bbox_inches='tight')
plt.show()


###########################
# ANGLE SWEEP #
###########################

import numpy as np
import matplotlib.pyplot as plt

row_index = np.argmin(np.abs(voltages - 1.0))
selected_voltage = voltages[row_index]
iterations = Z[row_index, :]
volt_angles = angles

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot: X = voltage angles, Y = iterations
ax.plot(volt_angles, iterations, label=f"{selected_voltage:.3f}")

# Labels, title, legend
ax.set_xlabel('Volt Angle Deg.', fontsize=20)
ax.set_ylabel('Iteration', fontsize=20)
ax.set_title(f"{selected_voltage:.3f}", fontsize=20)
ax.legend(fontsize=15)

ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.show()
