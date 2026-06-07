import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import *

# Output directory for images and animations
OUTPUT_DIR = "demonstration_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

risk_cmap = get_risk_colormap()

METHOD = "whole_map"
MODEL = "original"

# --- 1. Load Static Map ---
MAP_SIZE = 64
open_cell_dist = {64: 3.0, 256: 15.0}
data_dir = f"dataset/synthetic/{MAP_SIZE}x{MAP_SIZE}/"
if not os.path.exists(data_dir):
    data_dir = f"examples/dataset/synthetic/{MAP_SIZE}x{MAP_SIZE}/"
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset not found at {data_dir}")

MAP_IDX = 2
actual_masks = np.load(os.path.join(data_dir, "mask.npy"))
dist_maps = np.load(os.path.join(data_dir, "dist_in.npy"))
binary_map = actual_masks[MAP_IDX]
static_dist = dist_maps[MAP_IDX]
H, W = binary_map.shape
print(f"Map size: {H}x{W}")

# Add risk to static map
SIGMA_STATIC = 1.0
static_risk = np.exp(-(static_dist ** 2) / (2 * SIGMA_STATIC ** 2))
static_risk = (static_risk - static_risk.min()) / (static_risk.max() - static_risk.min() + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Binary Map", "Distance Field (SDF)", "Static Risk"]
maps = [binary_map, static_dist, static_risk]
cmaps = ["gray", "viridis", risk_cmap]

for ax, m, title, cmap in zip(axes, maps, titles, cmaps):
    im = ax.imshow(m, origin='lower', cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_static_map_analysis.png"), bbox_inches='tight')
print("Static map analysis plot saved.")
plt.close()

# --- 2. Settings & Initializations ---
PERC_OPEN_CELLS = 0.02
TOTAL_STEPS = 50
SIGMA_DYNAMIC = 1.0
ALPHA_DYNAMIC = 0.3
DYNAMIC_SPEED = 1.0

# Obstacle movement directions
DIRECTIONS = np.array([
    [0, DYNAMIC_SPEED], [DYNAMIC_SPEED, 0], [0, -DYNAMIC_SPEED], [-DYNAMIC_SPEED, 0],
    [DYNAMIC_SPEED, DYNAMIC_SPEED], [DYNAMIC_SPEED, -DYNAMIC_SPEED],
    [-DYNAMIC_SPEED, DYNAMIC_SPEED], [-DYNAMIC_SPEED, -DYNAMIC_SPEED]]).astype(float)

SEED = 0
np.random.seed(SEED)

open_cells = np.argwhere(static_dist > open_cell_dist[MAP_SIZE])

num_open_cells = open_cells.shape[0]
print(num_open_cells, "open cells")
NUM_OBSTACLES = math.floor(num_open_cells * PERC_OPEN_CELLS)
print(NUM_OBSTACLES, "obstacles")

# Load Start and Goal positions
start_positions = np.load(os.path.join(data_dir, "start_positions.npy"))
agent_pos_init = start_positions[MAP_IDX]

goal_maps = np.load(os.path.join(data_dir, "goal.npy"))
goal_data = goal_maps[MAP_IDX][::-1]

# Spawn obstacles: Filter out start and goal from candidates
start_cell = tuple(agent_pos_init.astype(int))
goal_cell = tuple(goal_data.astype(int))

candidate_indices = []
for i, cell in enumerate(open_cells):
    c_tuple = tuple(cell)
    if c_tuple != start_cell and c_tuple != goal_cell:
        candidate_indices.append(i)

obs_indices = np.random.choice(candidate_indices, NUM_OBSTACLES, replace=False)
obs_positions = [open_cells[idx].astype(float) for idx in obs_indices]
obs_velocities = [DIRECTIONS[np.random.choice(len(DIRECTIONS))] for _ in range(NUM_OBSTACLES)]

initial_risk = compute_total_risk(obs_positions, obs_velocities, static_risk, H, W, sigma_dyn=SIGMA_DYNAMIC,
                                  alpha_dyn=ALPHA_DYNAMIC)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
titles = ["Binary Map with Obstacles", "Risk Map with Obstacles"]
maps = [binary_map, initial_risk]
cmaps = ["gray", risk_cmap]

for ax, m, title, cmap in zip(axes, maps, titles, cmaps):
    ax.imshow(m, origin='lower', cmap=cmap)
    ax.plot(np.array(obs_positions)[:, 1], np.array(obs_positions)[:, 0], 'ko', label='Obstacles')
    ax.set_title(title)
    ax.legend()
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_initial_obstacle_placement.png"), bbox_inches='tight')
print("Initial obstacle placement plot saved.")
plt.close()

# --- 3. Run Risk Simulation & Animation ---
sim_data = run_risk_simulation(obs_positions, obs_velocities, static_risk, binary_map, H, W,
                               TOTAL_STEPS, SIGMA_DYNAMIC, ALPHA_DYNAMIC)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(binary_map, origin='lower', cmap='gray')
obs_dots_binary, = ax1.plot([], [], 'ko', markersize=8)
ax1.set_title("Binary Environment")
ax1.axis('off')

im_risk = ax2.imshow(sim_data['risk'][0], origin='lower', cmap=risk_cmap, vmin=0, vmax=1.0)
obs_dots_risk, = ax2.plot([], [], 'ko', markersize=8, markeredgecolor='black')
ax2.set_title("Dynamic Continuous Risk")
ax2.axis('off')

artists = {
    'im_risk': im_risk,
    'obs_dots_bin': obs_dots_binary,
    'obs_dots_risk': obs_dots_risk
}

update_fn = partial(animation_step, data_dict=sim_data, artists=artists)

ani = FuncAnimation(fig, update_fn, frames=TOTAL_STEPS, interval=100, blit=False)
gif_path = os.path.join(OUTPUT_DIR, "03_risk_simulation.gif")
print(f"Saving risk animation to {gif_path}...")
ani.save(gif_path, writer=PillowWriter(fps=7))
plt.close()
print("Risk simulation complete.")

# --- 4. Global PNO Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Models
modelSDF, modelPNO = load_pno_models(device)

# Pre-compute SDF and mask tensor
mask_tensor = torch.tensor(binary_map, dtype=torch.float).reshape(1, H, W, 1).to(device)
with torch.no_grad():
    sdf_pred = modelSDF(mask_tensor)
    chi_tensor = smooth_chi(mask_tensor, sdf_pred, 5.0)
print("Global setup complete.")

# --- 5. Static Planning with Pre-trained PNO ---

# 2. A* Planning
print("Planning global static path...")
goal_coord_static = torch.tensor(np.array([goal_data]), dtype=torch.int).to(device)
cost_map_static = run_pno_inference(modelPNO, static_risk, chi_tensor, goal_coord_static, mask_tensor, device)
global_path, _ = plan_path_astar(agent_pos_init, goal_data, static_risk, cost_map_static, risk_weight=10.0)

if len(global_path) == 0:
    print("Warning: No global static path found! The goal may be unreachable.")

# 3. Visualization
plt.figure(figsize=(12, 5))

maps = [binary_map, static_risk]
cmaps = ["gray", risk_cmap]
titles = ["Path Planning in Static Binary Map", "Path Planning in Static Risk Map"]

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(maps[i], cmap=cmaps[i], origin='lower')
    if len(global_path) > 0:
        plt.plot(global_path[:, 1], global_path[:, 0], 'b-', linewidth=2, label='A* Path')
    plt.plot(agent_pos_init[1], agent_pos_init[0], 'bs', markersize=6, label='Start')
    plt.plot(goal_data[1], goal_data[0], 'b*', markersize=10, label='Goal')
    plt.title(titles[i])
    plt.legend()
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_static_path_planning.png"), bbox_inches='tight')
plt.close()
print("Static planning plot saved.")

# --- 6. Dynamic Planning Simulation ---
plan_data = run_planning_simulation(obs_positions, obs_velocities, agent_pos_init, goal_data,
                                    modelPNO, chi_tensor, mask_tensor, device,
                                    static_risk, binary_map, H, W,
                                    SIGMA_DYNAMIC, ALPHA_DYNAMIC, global_path=global_path)

# Animation Setup
fig_dyn, (ax_bin, ax_risk) = plt.subplots(1, 2, figsize=(14, 6))

ax_bin.imshow(binary_map, origin='lower', cmap='gray')
gp_line_bin, = ax_bin.plot([], [], 'b--', alpha=0.5, label='Global Plan')
obs_dots_bin, = ax_bin.plot([], [], 'ko', markersize=6)
agent_dot_bin, = ax_bin.plot([], [], 'bs', markersize=6)
collision_dots_bin, = ax_bin.plot([], [], 'bx', markersize=5)
ax_bin.plot(goal_data[1], goal_data[0], 'b*', markersize=10)
ax_bin.set_title("Binary Environment")
ax_bin.axis('off')

im_risk_dyn = ax_risk.imshow(plan_data['risk'][0], origin='lower', cmap=risk_cmap, vmin=0, vmax=1.0)
path_line, = ax_risk.plot([], [], 'b-', linewidth=2)
obs_dots_risk, = ax_risk.plot([], [], 'ko', markersize=6)
agent_dot_risk, = ax_risk.plot([], [], 'bs', markersize=6)
collision_dots_risk, = ax_risk.plot([], [], 'bx', markersize=5)
ax_risk.plot(goal_data[1], goal_data[0], 'b*', markersize=10)
ax_risk.set_title("Dynamic Neural Planning")
ax_risk.axis('off')

artists_dyn = {
    'im_risk': im_risk_dyn,
    'obs_dots_bin': obs_dots_bin,
    'obs_dots_risk': obs_dots_risk,
    'agent_dot_bin': agent_dot_bin,
    'agent_dot_risk': agent_dot_risk,
    'collision_dots_bin': collision_dots_bin,
    'collision_dots_risk': collision_dots_risk,
    'path_line': path_line,
    'global_path_bin': gp_line_bin
}

update_dyn_fn = partial(animation_step, data_dict=plan_data, artists=artists_dyn)

ani_dyn = FuncAnimation(fig_dyn, update_dyn_fn, frames=len(plan_data['risk']), interval=150, blit=False)
gif_path = os.path.join(OUTPUT_DIR, "05_dynamic_planning.gif")
print(f"Saving planning animation to {gif_path}...")
ani_dyn.save(gif_path, writer=PillowWriter(fps=7))
plt.close()
print("Dynamic planning simulation complete.")
