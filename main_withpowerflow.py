import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# settings
baseMVA = 100
max_iter = 30
# Slack bus is bus 1
V_slack = 1.0

# Bus 2 load
P_load = -200 / baseMVA
Q_load = -100 / baseMVA

# Line impedance between bus 1 and 2
r = 0.0000001
x = 0.1000000015
Z = r + 1j * x
Y = 1 / Z
G = Y.real
B = Y.imag

def power_mismatch(Vm, Va):
    P = Vm * V_slack * (G * np.cos(Va) + B * np.sin(Va)) - Vm**2 * G
    Q = Vm * V_slack * (G * np.sin(Va) - B * np.cos(Va)) + Vm**2 * B
    dP = -P_load - P
    dQ = -Q_load - Q
    return np.array([dP, dQ])

def jacobian(Vm, Va):
    dP_dVm = V_slack * (G * np.cos(Va) + B * np.sin(Va)) - 2 * Vm * G
    dP_dVa = -Vm * V_slack * (G * np.sin(Va) - B * np.cos(Va))
    dQ_dVm = V_slack * (G * np.sin(Va) - B * np.cos(Va)) + 2 * Vm * B
    dQ_dVa = Vm * V_slack * (G * np.cos(Va) + B * np.sin(Va))
    return np.array([[dP_dVm, dP_dVa],
                     [dQ_dVm, dQ_dVa]])

def NR(Vm0, Va0, max_iter, tol=1e-6):
    Vm = Vm0
    Va = Va0
    for it in range(max_iter):
        mis = power_mismatch(Vm, Va)
        if np.linalg.norm(mis, ord=2) < tol:
            return Vm, Va, it + 1, True
        J = jacobian(Vm, Va)
        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return Vm, Va, it + 1, False
        Vm += dx[0]
        Va += dx[1]
    return Vm, Va, max_iter, False

# find the high and low solution
Vm_high, Va_high, _, _ = NR(1.0, 0.0, max_iter)
Vm_low,  Va_low,  _, _ = NR(0.2, 0.0, max_iter)
print(f"High voltage solution: |V|={Vm_high:.4f}")
print(f"Low voltage solution: |V|={Vm_low:.4f}")
#TODO: later print all stable solutions automatically

# make a grid for the NR region of convergence
Vm_range = np.linspace(0.0, 1.2, 500)
Va_range = np.deg2rad(np.linspace(-180, 180, 500))
Vm_grid, Va_grid = np.meshgrid(Vm_range, Va_range)

#run pf
Iterations = np.full_like(Vm_grid, np.nan)
Solution = np.full_like(Vm_grid, np.nan)
sols = pd.DataFrame(columns=['Vm', 'Va', 'Convergence'])
print("Running NR for all grid points...")
for i in range(Vm_grid.shape[0]):
    for j in range(Vm_grid.shape[1]):
        Vm0 = Vm_grid[i, j]
        Va0 = Va_grid[i, j]
        Vm_final, Va_final, iters, success = NR(Vm0, Va0, max_iter)
        # In your loop:
        if success:
            dist_high = abs(Vm_final - Vm_high)
            dist_low = abs(Vm_final - Vm_low)
            Solution[i, j] = 1 if dist_high < dist_low else 0
            Iterations[i, j] = iters
            sols = sols._append({'Vm': Vm0, 'Va': Va0, 'Convergence': True}, ignore_index=True)
        else:
            Iterations[i, j] = np.nan
            Solution[i,j] = np.nan
            sols = sols._append({'Vm': Vm0, 'Va': Va0, 'Convergence': False}, ignore_index=True)

sols.to_csv('2bus_solutions.csv', index=False)
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

Normalized = np.where(np.isnan(Iterations), np.nan, Iterations / max_iter)

# Negative for LV
Z_plot = np.zeros_like(Normalized)
Z_plot[:] = np.nan

# Mask for low and high, with sign:
Z_plot[(Solution == 0) & ~np.isnan(Normalized)] = -Normalized[(Solution == 0) & ~np.isnan(Normalized)]
Z_plot[(Solution == 1) & ~np.isnan(Normalized)] = Normalized[(Solution == 1) & ~np.isnan(Normalized)]

Z_plot = ma.masked_invalid(Z_plot)

levels = np.linspace(-1, 1, 100)

cmap_colors = [
    (0.0, 'black'),      # start (negative extreme)
    (0.25, '#CCCC00'),   # dark yellow
    (0.45, '#FFFF99'),   # light yellow
    (0.5, 'white'),      # midpoint (zero)
    (0.55, '#FF9999'),   # light red
    (0.75, '#990000'),   # dark red
    (1.0, 'black')       # end (positive extreme)
]
custom_cmap = LinearSegmentedColormap.from_list('yellow_red_black', cmap_colors)

fig, ax = plt.subplots(figsize=(15, 7))
ax.set_facecolor('black')

cf = ax.contourf(
    np.rad2deg(Va_grid), Vm_grid, Z_plot,
    levels=levels, cmap=custom_cmap, linestyles="None"
)

cb = fig.colorbar(cf, ax=ax, shrink=0.8, label='Iterations (negative=LV, positive=HV)')
cb.ax.tick_params(labelsize=15)

# Show real iteration ticks: -max_iter ... 0 ... +max_iter
raw_ticks = np.linspace(-1, 1, 5)
cb.set_ticks(raw_ticks)
cb.set_ticklabels([f"{int(t * max_iter)}" for t in raw_ticks])

ax.set_ylabel('Voltage Magnitude (p.u.)')
ax.set_xlabel('Voltage Angle (degrees)')
ax.set_title('NR ROV')
plt.savefig('NR_ROC_2bus_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

# # interactive plot - comment later
# Normalized = np.full_like(Iterations, np.nan)
# mask = ~np.isnan(Iterations)
# Normalized[mask] = Iterations[mask] / max_iter
#
# Combined = np.full_like(Normalized, np.nan)
# Combined[Solution == 1] = Normalized[Solution == 1]       # High: positive
# Combined[Solution == 0] = -Normalized[Solution == 0]      # Low: negative
#
# print("Check Combined min/max:", np.nanmin(Combined), np.nanmax(Combined))
#
#
# Z = Combined.T
# Va_deg = np.rad2deg(Va_grid)
# import plotly.graph_objects as go
#
# # normalized iertions
# Z_high = np.full_like(Combined, np.nan)
# Z_low  = np.full_like(Combined, np.nan)
# Z_high[Solution == 1] = Normalized[Solution == 1]
# Z_low[Solution == 0] = Normalized[Solution == 0]
# Z_high = Z_high.T
# Z_low = Z_low.T
#
# Iter_high = np.full_like(Combined, np.nan)
# Iter_low  = np.full_like(Combined, np.nan)
# Iter_high[Solution == 1] = Iterations[Solution == 1]
# Iter_low[Solution == 0]  = Iterations[Solution == 0]
# Iter_high = Iter_high.T
# Iter_low = Iter_low.T
#
#
# fig = go.Figure()
# # Choose tick marks in raw iteration units:
# ticks_raw = np.arange(0, max_iter + 1, 5)
# ticks_normalized = ticks_raw / max_iter
# fig = go.Figure()
#
# fig.add_trace(go.Heatmap(
#     z=Z_high,
#     customdata=Iter_high,
#     y=Vm_range,
#     x=np.rad2deg(Va_range),
#     colorscale="Reds",
#     zmin=0, zmax=1,
#     colorbar=dict(
#         title="Iterations (High)",
#         len=0.5,
#         y=0.75,
#         tickmode='array',
#         tickvals=ticks_normalized,
#         ticktext=[f"{t}" for t in ticks_raw]
#     ),
#     hovertemplate="Vm: %{y:.2f}<br>Va: %{x:.2f}°<br>",
# ))
#
# fig.add_trace(go.Heatmap(
#     z=Z_low,
#     customdata=Iter_low,   # raw count for low basin too
#     y=Vm_range,
#     x=np.rad2deg(Va_range),
#     colorscale="YlOrBr",
#     zmin=0, zmax=1,
#     colorbar=dict(
#         title="Iterations (Low)",
#         len=0.5,
#         y=0.25,
#         tickmode='array',
#         tickvals=ticks_normalized,
#         ticktext=[f"{t}" for t in ticks_raw]
#     ),
#     hovertemplate="Vm: %{y:.2f}<br>Va: %{x:.2f}°<br>",
# ))
# fig.update_layout(
#     title="NR Region of Convergence",
#     yaxis=dict(title="Voltage Magnitude (p.u.)", showgrid=False),
#     xaxis=dict(title="Voltage Angle (degrees)", showgrid=False),
#     plot_bgcolor='black'
# )
# fig.show()
# fig.write_html("NR_ROC_2bus_interactive.html")
