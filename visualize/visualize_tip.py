import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors

def create_spinning_shell(radius=1, max_height=100, num_points=80):
    theta = np.linspace(0, 2*np.pi, num_points)
    r     = np.linspace(0, radius,    num_points)
    theta, r = np.meshgrid(theta, r)
    z_norm = np.exp(-r)
    z_min  = np.exp(-radius)
    z      = (z_norm - z_min) / (1 - z_min) * max_height
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z, z_min

def invert_r(z_vals, H, z_min):
    return -np.log((z_vals/H) * (1 - z_min) + z_min)

radius, max_h, N = 1, 100, 80
lower_h = 55
upper_h = max_h - lower_h

x_lo, y_lo, z_lo, zmin = create_spinning_shell(radius, lower_h, N)
z_li = lower_h - z_lo

x_hi, y_hi, z_hi_raw, _ = create_spinning_shell(radius, upper_h, N)
z_ho = z_hi_raw + lower_h

rank_path = os.path.join(
    os.path.dirname(__file__),'..',
    'tblogs/protect_vs_invade_adjusting_43',
    'agent_rankings.csv'
)
lines = open(rank_path).read().splitlines()
tracker_lines, target_lines = [], []
mode = None
for L in lines:
    if L.startswith('=== Tracker Rankings'):
        mode = 't'; continue
    if L.startswith('=== Target Rankings'):
        mode = 'g'; continue
    if mode == 't' and L:
        tracker_lines.append(L)
    if mode == 'g' and L:
        target_lines.append(L)

df_t = pd.read_csv(io.StringIO('\n'.join(tracker_lines)))
df_g = pd.read_csv(io.StringIO('\n'.join(target_lines)))

# ensure 'model' column exists
if 'model' not in df_t.columns:
    df_t['model'] = df_t.iloc[:, 0]
if 'model' not in df_g.columns:
    df_g['model'] = df_g.iloc[:, 0]

df_t['wr'] = df_t.overall_win_rate * 100
df_g['wr'] = df_g.overall_win_rate * 100

Mt, Mg = len(df_t), len(df_g)
θt = np.random.rand(Mt) * 2*np.pi
θg = np.random.rand(Mg) * 2*np.pi

zt = df_t.wr.values
zg = df_g.wr.values

def r_max_at_z(z_vals):
    # 只对需要的部分调用 invert_r
    temp = np.empty_like(z_vals, dtype=float)
    bottom = z_vals <= lower_h
    top    = ~bottom
    # 屏蔽 log 的 invalid warning
    with np.errstate(invalid='ignore'):
        temp[bottom] = invert_r(lower_h - z_vals[bottom], lower_h, zmin)
        temp[top]    = invert_r(z_vals[top] - lower_h, upper_h, zmin)
    return temp


rmax_t = r_max_at_z(zt)
rmax_g = r_max_at_z(zg)

rt = np.sqrt(np.random.rand(Mt)) * rmax_t * 0.9
rg = np.sqrt(np.random.rand(Mg)) * rmax_g * 0.9

xt = rt * np.cos(θt);  yt = rt * np.sin(θt)
xg = rg * np.cos(θg);  yg = rg * np.sin(θg)

head_df = pd.read_csv(os.path.join(
    os.path.dirname(__file__),'..',
    'tblogs/protect_vs_invade_adjusting_43',
    'head_to_head_results.csv'
))

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

for (X, Y, Z) in [(x_lo, y_lo, z_li),(x_hi, y_hi, z_ho)]:
    ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        alpha=0.08,
        color=(0.2,0.4,1,0.08),
        edgecolors=(0,0,0,0.005),
        linewidth=0.05
    )

st = ax.scatter(xt, yt, zt, c='blue',   s=8, label='tracker agents')
sg = ax.scatter(xg, yg, zg, c='red', s=8, label='target agents')
ax.legend(loc='upper right')

ax.set_zlim(0, max_h)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Win-rate (%)')
ax.set_title('Agent Rankings in Spinning Tip')
ax.view_init(azim=60, elev=30)

lines_list = []
def clear_lines():
    global lines_list
    for ln in lines_list:
        ln.remove()
    lines_list.clear()

cursor = mplcursors.cursor([st, sg], hover=True)
@cursor.connect("add")
def on_hover(sel):
    clear_lines()
    artist, idx = sel.artist, sel.index
    if artist is st:
        model = df_t['model'].iloc[idx]
        x0,y0,z0 = xt[idx], yt[idx], zt[idx]
        df_h2h = head_df[head_df.tracker_model == model]
        for _, row in df_h2h.iterrows():
            opp = row.target_model
            j = df_g.index[df_g['model'] == opp][0]
            x1,y1,z1 = xg[j], yg[j], zg[j]
            color = 'blue' if row.tracker_wins > row.target_wins else 'red'
            ln, = ax.plot([x0,x1],[y0,y1],[z0,z1], c=color, linewidth=0.5)
            lines_list.append(ln)
        sel.annotation.set_text(model)
    else:
        model = df_g['model'].iloc[idx]
        x0,y0,z0 = xg[idx], yg[idx], zg[idx]
        df_h2h = head_df[head_df.target_model == model]
        for _, row in df_h2h.iterrows():
            opp = row.tracker_model
            i = df_t.index[df_t['model'] == opp][0]
            x1,y1,z1 = xt[i], yt[i], zt[i]
            color = 'red' if row.target_wins > row.tracker_wins else 'blue'
            ln, = ax.plot([x1,x0],[y1,y0],[z1,z0], c=color, linewidth=0.5)
            lines_list.append(ln)
        sel.annotation.set_text(model)

plt.show()