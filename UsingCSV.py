import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# --------------------------
# Load CSV
# --------------------------
df = pd.read_csv('New_PW_solutions.csv', skiprows=1)
voltages = df.iloc[:, 0].values
angles = df.columns[1:].astype(float)
Z = df.iloc[:, 1:].values.astype(int)

A, V = np.meshgrid(angles, voltages)

max_iter = np.max(np.abs(Z))

# --------------------------
# 1. Create combined normalized magnitude + label mask
# --------------------------

norm_mag = np.zeros_like(Z, dtype=float)   # normalized magnitude [0,1]
label = np.zeros_like(Z, dtype=int)        # 1=LV, 2=HV, 0=NC

lv_mask = (Z < 0) & (np.abs(Z) < max_iter)
hv_mask = (Z > 0) & (Z < max_iter)

norm_mag[lv_mask] = np.abs(Z[lv_mask]) / max_iter
norm_mag[hv_mask] = Z[hv_mask] / max_iter

label[lv_mask] = 1  # LV
label[hv_mask] = 2  # HV
# Everything else stays 0 => NC

# --------------------------
# 2. Build a custom colormap:
# - LV shades (YlOrBr) for label 1
# - HV shades (Reds) for label 2
# - Black for NC (label 0)
# --------------------------

n_colors = 100

LV_cmap = plt.cm.YlOrBr(np.linspace(0.3, 1, n_colors))
HV_cmap = plt.cm.Reds(np.linspace(0.3, 1, n_colors))
black = np.array([[0, 0, 0, 1]])

# Concatenate: black + LV shades + HV shades
combined_colors = np.vstack((black, LV_cmap, HV_cmap))
combined_cmap = ListedColormap(combined_colors)

# Map each point to:
#   index = 0 -> black
#   1..n_colors -> LV shades
#   n_colors+1..2n_colors -> HV shades

index_grid = np.zeros_like(Z, dtype=int)

# LV: index = 1 + normalized_level * (n_colors - 1)
index_grid[lv_mask] = 1 + (norm_mag[lv_mask] * (n_colors - 1)).astype(int)

# HV: index = n_colors + 1 + normalized_level * (n_colors - 1)
index_grid[hv_mask] = n_colors + 1 + (norm_mag[hv_mask] * (n_colors - 1)).astype(int)

# --------------------------
# 3. Plot single contourf
# --------------------------

fig, ax = plt.subplots(figsize=(14, 7))

cf = ax.contourf(A, V, index_grid, levels=np.arange(0, 2 * n_colors + 2),
                 cmap=combined_cmap)

# --------------------------
# 4. Add two separate colorbars
# --------------------------

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# LV bar: index 1 .. n_colors
lv_sm = ScalarMappable(norm=Normalize(vmin=0, vmax=max_iter - 1), cmap=plt.cm.YlOrBr)
cb1 = fig.colorbar(lv_sm, ax=ax, location='right', shrink=0.8,
                   label='LV iterations')
cb1.set_ticks(np.linspace(1, max_iter - 1, 5))
cb1.set_ticklabels([str(int(t)) for t in np.linspace(1, max_iter - 1, 5)])

# HV bar: index 1 .. n_colors
hv_sm = ScalarMappable(norm=Normalize(vmin=0, vmax=max_iter - 1), cmap=plt.cm.Reds)
cb2 = fig.colorbar(hv_sm, ax=ax, location='right', shrink=0.8,
                   label='HV iteraitons')
cb2.set_ticks(np.linspace(1, max_iter - 1, 5))
cb2.set_ticklabels([str(int(t)) for t in np.linspace(1, max_iter - 1, 5)])
# --------------------------
# 5. Labels
# --------------------------

ax.set_xlabel('V angle deg')
ax.set_ylabel('V mag pu')
ax.set_title('NR ROC')

plt.tight_layout()
plt.show()
