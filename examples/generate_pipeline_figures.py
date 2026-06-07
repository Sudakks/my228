"""
generate_pipeline_figures_v3.py
────────────────────────────────
Generates 5 pipeline panels using the EXACT same functions and parameters
as utils.py and demo.py. No custom reimplementations.

Panels:
  (a) Binary occupancy map
  (b) Value function V(x, g)
  (c) Signed distance function d_S(x)
  (d) Total risk map R(x) with dynamic obstacles
  (e) Final replanning frame with A* path

Usage:  python generate_pipeline_figures_v3.py
(place next to models/, results/, dataset/, utils.py)
"""
import os, sys, math

_script_dir = os.path.dirname(os.path.abspath(__file__))

_astar_candidates = [
    os.path.join(_script_dir, "../2D_Neural_Heuristics"),
]
for _cand in _astar_candidates:
    _cand = os.path.abspath(_cand)
    if os.path.isdir(os.path.join(_cand, "astar")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        print(f"[path] astar found at {_cand}")
        break
else:
    print("[warning] astar directory not found, A* may fail")
sys.path.insert(0, _script_dir)
from utils import (
    get_risk_colormap,
    smooth_chi,
    load_pno_models,
    compute_total_risk,
    step_obstacles,
    run_pno_inference,
    plan_path_astar,
)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── import utils.py directly (same directory) ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    get_risk_colormap,
    smooth_chi,
    load_pno_models,
    compute_total_risk,
    step_obstacles,
    run_pno_inference,
    plan_path_astar,
)

# ── A* planner path (same as utils.py) ───────────────────────────────────────
heuristics_dir = os.path.abspath("../src")
if os.path.isdir(heuristics_dir) and heuristics_dir not in sys.path:
    sys.path.append(heuristics_dir)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — must match demo.py exactly
# ════════════════════════════════════════════════════════════════════════════
MAP_SIZE        = 64
DATA_DIR        = f"dataset/synthetic/{MAP_SIZE}x{MAP_SIZE}/"
if not os.path.exists(DATA_DIR):
    DATA_DIR    = f"examples/dataset/synthetic/{MAP_SIZE}x{MAP_SIZE}/"

MODEL_DIR       = "examples/models/"
if not os.path.exists(MODEL_DIR):
    MODEL_DIR   = "./models/"

MAP_IDX         = 7               # same as demo.py
OUTPUT_DIR      = "pipeline_panels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# obstacle / risk params — identical to demo.py
OPEN_CELL_DIST  = {64: 3.0, 256: 15.0}
SIGMA_STATIC    = 1.0
SIGMA_DYNAMIC   = 1.0
ALPHA_DYNAMIC   = 0.3
DYNAMIC_SPEED   = 1.0
PERC_OPEN_CELLS = 0.05
TOTAL_STEPS     = 50
PLAN_INTERVAL   = 1              # replan every N steps for panel (e)
FIGURE_STEP     = 30             # save panel (e) from this step instead of the final step
RISK_WEIGHT     = 10.0
SEED            = 42

# visual style — exactly as demo.py animation
RISK_CMAP  = get_risk_colormap()  # green → yellow → red
OBS_STYLE  = dict(marker="o", color="blue", markersize=8,
                  markeredgecolor="black", markeredgewidth=0.5,
                  linestyle="none")
PATH_STYLE = dict(color="blue", linewidth=2, linestyle="-")
AGENT_STYLE= dict(marker="s", color="blue", markersize=6,
                  markeredgecolor="black", markeredgewidth=0.5,
                  linestyle="none")
GOAL_STYLE = dict(marker="*", color="blue", markersize=10, linestyle="none")

# ════════════════════════════════════════════════════════════════════════════
# load data
# ════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")

actual_masks = np.load(os.path.join(DATA_DIR, "mask.npy"))
dist_maps    = np.load(os.path.join(DATA_DIR, "dist_in.npy"))
goal_maps    = np.load(os.path.join(DATA_DIR, "goal.npy"))
output_maps  = np.load(os.path.join(DATA_DIR, "output.npy"))

binary_map  = actual_masks[MAP_IDX]       # 1=free, 0=obstacle
static_dist = dist_maps[MAP_IDX]
goal_data   = goal_maps[MAP_IDX][::-1]    # [row, col]
H, W        = binary_map.shape
cost_map_astar = (1.0 - binary_map)       # A* convention: 1=obstacle
print(f"[data] map={MAP_IDX}, size={H}×{W}, goal={goal_data}")

# ════════════════════════════════════════════════════════════════════════════
# load models via utils.load_pno_models  (same as demo.py)
# ════════════════════════════════════════════════════════════════════════════
modelSDF, modelPNO = load_pno_models(device, MODEL_DIR)

mask_tensor = torch.tensor(binary_map, dtype=torch.float).reshape(1,H,W,1).to(device)
goal_coord  = torch.tensor([goal_data], dtype=torch.int).to(device)

with torch.no_grad():
    sdf_pred   = modelSDF(mask_tensor)
    chi_tensor = smooth_chi(mask_tensor, sdf_pred, 5.0)

sdf_np = sdf_pred.squeeze().cpu().numpy()
print("[models] loaded + SDF predicted")

# ════════════════════════════════════════════════════════════════════════════
# static risk  (demo.py params: SIGMA_STATIC=1.5)
# ════════════════════════════════════════════════════════════════════════════
static_risk = np.exp(-(static_dist ** 2) / (2 * SIGMA_STATIC ** 2))
static_risk = (static_risk - static_risk.min()) / (static_risk.max() - static_risk.min() + 1e-8)

# ════════════════════════════════════════════════════════════════════════════
# obstacle init — exactly as demo.py
# ════════════════════════════════════════════════════════════════════════════
DIRECTIONS = np.array([
    [0, DYNAMIC_SPEED], [DYNAMIC_SPEED, 0],
    [0, -DYNAMIC_SPEED], [-DYNAMIC_SPEED, 0],
    [DYNAMIC_SPEED, DYNAMIC_SPEED], [DYNAMIC_SPEED, -DYNAMIC_SPEED],
    [-DYNAMIC_SPEED, DYNAMIC_SPEED], [-DYNAMIC_SPEED, -DYNAMIC_SPEED]
], dtype=float)

np.random.seed(SEED)

open_cells    = np.argwhere(static_dist > OPEN_CELL_DIST[MAP_SIZE])
NUM_OBSTACLES = math.floor(len(open_cells) * PERC_OPEN_CELLS)
print(f"[obstacles] {len(open_cells)} open cells → {NUM_OBSTACLES} obstacles")

random_idx      = np.random.choice(len(open_cells), NUM_OBSTACLES + 1, replace=False)
agent_pos_init  = open_cells[random_idx[0]].copy()
obs_positions   = [open_cells[i].astype(float) for i in random_idx[1:]]
obs_velocities  = [DIRECTIONS[np.random.choice(len(DIRECTIONS))].copy()
                   for _ in range(NUM_OBSTACLES)]

# ════════════════════════════════════════════════════════════════════════════
# simulation loop — calls utils functions directly
# ════════════════════════════════════════════════════════════════════════════
print(f"[sim] {TOTAL_STEPS} steps …")

last_replan = None
figure_replan = None
agent_pos   = agent_pos_init.copy()

# static path for reference (panel e global plan)
static_value = run_pno_inference(modelPNO, static_risk, chi_tensor,
                                 goal_coord, mask_tensor, device)
global_path, _ = plan_path_astar(agent_pos_init, goal_data,
                                  cost_map_astar, static_value,
                                  risk_weight=RISK_WEIGHT)

for t in range(TOTAL_STEPS):
    # ── move obstacles (utils.step_obstacles does wall-collision bouncing) ──
    step_obstacles(obs_positions, obs_velocities, binary_map, H, W)

    # ── compute total risk (utils.compute_total_risk: np.maximum, local_sigma) ──
    total_risk = compute_total_risk(
        obs_positions, obs_velocities, static_risk, H, W,
        sigma_dyn=SIGMA_DYNAMIC, alpha_dyn=ALPHA_DYNAMIC)

    # ── replan every PLAN_INTERVAL steps ──
    if t % PLAN_INTERVAL == 0:
        value_fn = run_pno_inference(modelPNO, total_risk, chi_tensor,
                                     goal_coord, mask_tensor, device)
        path, nodes = plan_path_astar(agent_pos, goal_data,
                                       cost_map_astar, value_fn,
                                       risk_weight=RISK_WEIGHT)
        # agent takes one step along path
        if len(path) > 1:
            agent_pos = path[1].copy()

        last_replan = {
            "step":      t + 1,
            "risk_map":  total_risk.copy(),
            "value_fn":  value_fn.copy(),
            "path":      path.copy() if len(path) else np.array([]),
            "agent_pos": agent_pos.copy(),
            "nodes":     nodes,
            "obs_pos":   np.array([p.copy() for p in obs_positions]),
            "obs_vel":   np.array([v.copy() for v in obs_velocities]),
        }

        if figure_replan is None and last_replan["step"] >= FIGURE_STEP:
            figure_replan = last_replan.copy()

    if (t + 1) % 10 == 0:
        print(f"  step {t+1}/{TOTAL_STEPS}  "
              f"path_len={len(last_replan['path']) if last_replan else 0}  "
              f"nodes={last_replan['nodes'] if last_replan else 0}")

fr = figure_replan if figure_replan is not None else last_replan
print(f"[sim] done  figure replan at step {fr['step']}")

# ── value function without dynamic risk (pure PNO output for panel b) ────
zero_risk  = np.zeros((H, W), dtype=np.float32)
value_static = run_pno_inference(modelPNO, zero_risk, chi_tensor,
                                  goal_coord, mask_tensor, device)

# ════════════════════════════════════════════════════════════════════════════
# helper: uniform axis style
# ════════════════════════════════════════════════════════════════════════════
def style_ax(ax, title):
    ax.set_title(title, fontsize=10, pad=7, fontweight="bold",
                 fontfamily="sans-serif")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#BBBBBB"); sp.set_linewidth(0.7)

def add_cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.92)
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

def letter(ax, c):
    ax.text(0.03, 0.97, f"({c})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", alpha=0.45, lw=0),
            zorder=9)

def plot_obs(ax, obs_pos_arr, obs_vel_arr):
    """Plot obstacles exactly as demo.py: blue dots + no colour-coding."""
    ax.plot(obs_pos_arr[:, 1], obs_pos_arr[:, 0], **OBS_STYLE, zorder=6)

def plot_goal(ax):
    ax.plot(goal_data[1], goal_data[0], **GOAL_STYLE, zorder=7)

def format_timestamp(step):
    return f"t = {step}"

# ════════════════════════════════════════════════════════════════════════════
# 5-panel combined figure
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(27, 5.5), facecolor="white",
                         gridspec_kw={"wspace": 0.05})

# (a) binary map
ax = axes[0]
ax.imshow(binary_map, origin="lower", cmap="gray", vmin=0, vmax=1)
plot_goal(ax)
style_ax(ax, "Binary Occupancy Map\n$1/c(x)$")
letter(ax, "a")

# (b) value function
ax = axes[1]
im = ax.imshow(value_static, origin="lower", cmap="inferno_r")
plot_goal(ax)
add_cb(fig, ax, im, "Cost-to-go")
style_ax(ax, "Value Function $V(x,\\,g)$\n(PNO w/ PINN)")
letter(ax, "b")

# (c) SDF
ax = axes[2]
im = ax.imshow(sdf_np, origin="lower", cmap="viridis")
add_cb(fig, ax, im, "Distance")
style_ax(ax, "Signed Distance Function\n$d_S(x)$  (FNOSDF)")
letter(ax, "c")

# (d) risk map with obstacles  — risk_cmap = green→yellow→red
ax = axes[3]
im = ax.imshow(fr["risk_map"], origin="lower", cmap=RISK_CMAP, vmin=0, vmax=1)
plot_obs(ax, fr["obs_pos"], fr["obs_vel"])
plot_goal(ax)
add_cb(fig, ax, im, "Risk $R(x)$")
style_ax(ax, f"Total Risk Map $R(x)$\n"
             f"(σ={SIGMA_DYNAMIC}, α={ALPHA_DYNAMIC},  step {fr['step']})")
letter(ax, "d")

# (e) replanning frame
ax = axes[4]
im = ax.imshow(fr["risk_map"], origin="lower", cmap=RISK_CMAP, vmin=0, vmax=1)
path = fr["path"]
if len(path) > 0:
    ax.plot(path[:, 1], path[:, 0], **PATH_STYLE, zorder=5, label="A* path")
ax.plot(fr["agent_pos"][1], fr["agent_pos"][0], **AGENT_STYLE,
        zorder=7, label="Agent")
plot_goal(ax)
plot_obs(ax, fr["obs_pos"], fr["obs_vel"])
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.65,
          markerscale=0.9, handlelength=1.5)
add_cb(fig, ax, im, "Risk $R(x)$")
style_ax(ax, f"Dynamic Replanning  (step {fr['step']})\n"
             f"A* + PNO heuristic  (nodes={fr['nodes']})")
letter(ax, "e")

# caption
fig.text(0.5, -0.03,
    "(a) Binary occupancy map $1/c(x)$.  "
    "(b) Predicted value function $V(x,g)$ from PNO w/ PINN.  "
    "(c) Signed distance function $d_S(x)$ from FNOSDF.  "
    "(d) Continuous risk map $R(x)$ with "
    f"{NUM_OBSTACLES} dynamic obstacles "
    f"(σ={SIGMA_DYNAMIC}, α={ALPHA_DYNAMIC}).  "
    "(e) A* replanning using PNO value function as heuristic "
    f"(risk weight={RISK_WEIGHT:.0f}).",
    ha="center", fontsize=8.5, color="#555555",
    fontfamily="sans-serif", style="italic")

out = os.path.join(OUTPUT_DIR, "pipeline_panels.png")
fig.savefig(out, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"\n[saved] {out}")

# ════════════════════════════════════════════════════════════════════════════
# individual panels
# ════════════════════════════════════════════════════════════════════════════
def save_panel(fname, draw_fn):
    f2, a2 = plt.subplots(figsize=(4.5, 4.5), facecolor="white")
    draw_fn(f2, a2)
    a2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in a2.spines.values():
        sp.set_edgecolor("#BBBBBB"); sp.set_linewidth(0.7)
    p = os.path.join(OUTPUT_DIR, fname)
    f2.savefig(p, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(f2)
    print(f"  [saved] {p}")

def draw_a(f2, a2):
    a2.imshow(binary_map, origin="lower", cmap="gray", vmin=0, vmax=1)
    plot_goal(a2)
    a2.set_title("Binary Occupancy Map  $1/c(x)$",
                 fontsize=10, pad=6, fontweight="bold")

def draw_b(f2, a2):
    im2 = a2.imshow(value_static, origin="lower", cmap="inferno_r")
    plot_goal(a2)
    cb = f2.colorbar(im2, ax=a2, fraction=0.046, pad=0.04, shrink=0.88)
    cb.set_label("Cost-to-go", fontsize=9)
    a2.set_title("Value Function $V(x, g)$",
                 fontsize=10, pad=6, fontweight="bold")

def draw_c(f2, a2):
    im2 = a2.imshow(sdf_np, origin="lower", cmap="viridis")
    cb = f2.colorbar(im2, ax=a2, fraction=0.046, pad=0.04, shrink=0.88)
    cb.set_label("Distance", fontsize=9)
    a2.set_title("Signed Distance Function $d_S(x)$",
                 fontsize=10, pad=6, fontweight="bold")

def draw_d(f2, a2):
    im2 = a2.imshow(fr["risk_map"], origin="lower",
                    cmap=RISK_CMAP, vmin=0, vmax=1)
    plot_obs(a2, fr["obs_pos"], fr["obs_vel"])
    plot_goal(a2)
    cb = f2.colorbar(im2, ax=a2, fraction=0.046, pad=0.04, shrink=0.88)
    cb.set_label("Risk $R(x)$", fontsize=9)
    a2.set_title("Total Risk Map  $R(x)$",
                 fontsize=10, pad=6, fontweight="bold")

def draw_e(f2, a2):
    im2 = a2.imshow(fr["risk_map"], origin="lower",
                    cmap=RISK_CMAP, vmin=0, vmax=1)
    path = fr["path"]
    if len(path) > 0:
        a2.plot(path[:, 1], path[:, 0], **PATH_STYLE, zorder=5)
    a2.plot(fr["agent_pos"][1], fr["agent_pos"][0],
            **AGENT_STYLE, zorder=7)
    plot_goal(a2)
    plot_obs(a2, fr["obs_pos"], fr["obs_vel"])
    cb = f2.colorbar(im2, ax=a2, fraction=0.046, pad=0.04, shrink=0.88)
    cb.set_label("Risk $R(x)$", fontsize=9)
    a2.set_title(f"Dynamic Replanning  ({format_timestamp(fr['step'])})",
                 fontsize=10, pad=6, fontweight="bold")

save_panel("1_binary_map.png",     draw_a)
save_panel("2_value_function.png", draw_b)
save_panel("3_sdf.png",            draw_c)
save_panel("4_risk_map.png",       draw_d)
save_panel("5_replanning.png",     draw_e)

print("\nAll done.")