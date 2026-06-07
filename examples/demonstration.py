import math
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import *


def run_demo(
        map_size=64,
        map_idx=2,
        method="whole_map",
        model="original",
        perc_obs=0.02,
        dynamic_speed=1.0,
        sigma_static=1.0,
        sigma_dynamic=1.0,
        alpha_dynamic=0.3,
        seed=0,
        output_dir="data"
):
    np.random.seed(seed)

    # Output Setup
    os.makedirs(output_dir, exist_ok=True)
    risk_cmap = get_risk_colormap()

    # --- 1. Load Static Map ---
    open_cell_dist = {64: 3.0, 256: 15.0}
    data_dir = f"dataset/synthetic/{map_size}x{map_size}/"
    if not os.path.exists(data_dir):
        data_dir = f"examples/dataset/synthetic/{map_size}x{map_size}/"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}")

    actual_masks = np.load(os.path.join(data_dir, "mask.npy"))
    dist_maps = np.load(os.path.join(data_dir, "dist_in.npy"))
    binary_map = actual_masks[map_idx]
    static_dist = dist_maps[map_idx]
    H, W = binary_map.shape
    print(f"Map size: {H}x{W}, Method: {method}, Model: {model}")

    static_risk = np.exp(-(static_dist ** 2) / (2 * sigma_static ** 2))
    static_risk = (static_risk - static_risk.min()) / (static_risk.max() - static_risk.min() + 1e-8)

    # --- 2. Initialize Scenario ---
    DIRECTIONS = np.array([
        [0, dynamic_speed], [dynamic_speed, 0], [0, -dynamic_speed], [-dynamic_speed, 0],
        [dynamic_speed, dynamic_speed], [dynamic_speed, -dynamic_speed],
        [-dynamic_speed, dynamic_speed], [-dynamic_speed, -dynamic_speed]]).astype(float)

    open_cells = np.argwhere(static_dist > open_cell_dist[map_size])
    num_open_cells = open_cells.shape[0]
    num_obstacles = math.floor(num_open_cells * perc_obs)
    print(f"{num_open_cells} open cells, {num_obstacles} obstacles")

    start_positions = np.load(os.path.join(data_dir, "start_positions.npy"))
    agent_pos_init = start_positions[map_idx]

    goal_maps = np.load(os.path.join(data_dir, "goal.npy"))
    goal_data = goal_maps[map_idx][::-1]

    start_cell = tuple(agent_pos_init.astype(int))
    goal_cell = tuple(goal_data.astype(int))

    candidate_indices = []
    for i, cell in enumerate(open_cells):
        c_tuple = tuple(cell)
        if c_tuple != start_cell and c_tuple != goal_cell:
            candidate_indices.append(i)

    obs_indices = np.random.choice(candidate_indices, num_obstacles, replace=False)
    obs_positions = [open_cells[idx].astype(float) for idx in obs_indices]
    obs_velocities = [DIRECTIONS[np.random.choice(len(DIRECTIONS))] for _ in range(num_obstacles)]

    # --- 3. Global PNO Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if model == "new":
        modelSDF, modelPNO = load_new_model(device)
        multi_channel = True
    else:
        modelSDF, modelPNO = load_pno_models(device)
        multi_channel = False

    mask_tensor = torch.tensor(binary_map, dtype=torch.float).reshape(1, H, W, 1).to(device)
    with torch.no_grad():
        sdf_pred = modelSDF(mask_tensor)
        chi_tensor = smooth_chi(mask_tensor, sdf_pred, 5.0)
    print("Model setup complete.")

    # --- 4. Static Path Planning (Global Guide) ---
    print("Planning global static path...")
    goal_coord_static = torch.tensor(np.array([goal_data]), dtype=torch.int).to(device)
    #cost_map_static = run_pno_inference(modelPNO, static_risk, chi_tensor, goal_coord_static, mask_tensor, device)
    cost_map_static = run_pno_inference(modelPNO, static_risk, chi_tensor, goal_coord_static, mask_tensor, device, multi_channel=multi_channel)
    global_path, _ = plan_path_astar(agent_pos_init, goal_data, static_risk, cost_map_static, risk_weight=10.0)

    if len(global_path) == 0:
        print("Warning: No global static path found! The goal may be unreachable.")

    # --- 5. Run Simulation ---
    if method == "sliding_window":
        print("Running Dynamic Window Simulation (Path-Guided)...")
        plan_data = run_dynamic_window_simulation(obs_positions, obs_velocities, agent_pos_init, goal_data,
                                                  modelPNO, chi_tensor, mask_tensor, device,
                                                  static_risk, binary_map, H, W,
                                                  sigma_dynamic, alpha_dynamic,
                                                  global_path=global_path, multi_channel=multi_channel)
    else:
        print("Running Whole Map Simulation...")
        plan_data = run_planning_simulation(obs_positions, obs_velocities, agent_pos_init, goal_data,
                                            modelPNO, chi_tensor, mask_tensor, device,
                                            static_risk, binary_map, H, W,
                                            sigma_dynamic, alpha_dynamic,
                                            global_path=global_path, multi_channel=multi_channel)

    # --- 6. Save Results ---
    final_output_dir = os.path.join(output_dir, str(map_size), f"{method}_{model}")
    save_metrics(plan_data['metrics'], final_output_dir, f"{map_idx}.json")

    # Animation Setup
    fig, (ax_bin, ax_risk) = plt.subplots(1, 2, figsize=(14, 6))

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
    ax_risk.set_title(f"Dynamic Planning ({method})")
    ax_risk.axis('off')

    artists = {
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

    update_fn = partial(animation_step, data_dict=plan_data, artists=artists)

    interval, fps = 100, 10

    ani = FuncAnimation(fig, update_fn, frames=len(plan_data['risk']), interval=interval, blit=False)
    gif_path = os.path.join(final_output_dir, f"{map_idx}.gif")
    print(f"Saving animation to {gif_path}...")
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Demo for Map {map_idx} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single path planning demo.")

    # Core Settings
    parser.add_argument("--map_size", type=int, default=64, help="Map size (64 or 256).")
    parser.add_argument("--map_idx", type=int, default=2, help="Index of the map to test.")
    parser.add_argument("--method", type=str, default="whole_map", choices=["whole_map", "sliding_window"],
                        help="Planning method to use.")
    parser.add_argument("--model", type=str, default="original", choices=["original", "new"],
                        help="Model weights to load ('original' or 'new').")

    # Scenario Parameters
    parser.add_argument("--perc_obs", type=float, default=0.02, help="Percentage of open cells occupied by obstacles.")
    parser.add_argument("--dynamic_speed", type=float, default=1.0, help="Movement speed of obstacles.")
    parser.add_argument("--sigma_static", type=float, default=1.0, help="Risk sigma for static obstacles.")
    parser.add_argument("--sigma_dynamic", type=float, default=1.0, help="Risk sigma for dynamic obstacles.")
    parser.add_argument("--alpha_dynamic", type=float, default=0.3, help="Alpha parameter for dynamic risk.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    # Output Settings
    parser.add_argument("--output_dir", type=str, default="data", help="Root directory for output data.")

    args = parser.parse_args()

    # Pass all arguments to the run_demo function
    run_demo(**vars(args))
