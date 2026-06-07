import torch
import numpy as np
import os
import sys
import json
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import LinearSegmentedColormap

# Add heuristics dir to path for A*
heuristics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if heuristics_dir not in sys.path:
    sys.path.append(heuristics_dir)

from astar.astar import AStar
from astar.environment_simple import Environment2D


def get_risk_colormap():
    colors = ["green", "yellow", "red"]
    return LinearSegmentedColormap.from_list("risk_map", colors)


def smooth_chi(mask, dist, smooth_coef=5.0):
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5


def check_collision(agent_pos, obs_positions, binary_map):
    """
    Checks for collisions.
    Static: 0 in binary_map represents an obstacle.
    Dynamic: Moving obstacles are checked at the agent's 4 neighbors (H/V).
    """
    r, c = int(round(agent_pos[0])), int(round(agent_pos[1]))
    H, W = binary_map.shape
    
    # Convert dynamic obstacle positions to grid coordinates
    obs_grid = set([(int(round(o[0])), int(round(o[1]))) for o in obs_positions])
    
    # Check current and 4 neighbors (up, down, left, right)
    for dr, dc in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            # Check static map (0=obstacle)
            if binary_map[nr, nc] == 0:
                return True
            # Check dynamic obstacles
            if (nr, nc) in obs_grid:
                return True
    return False


def save_metrics(metrics, output_dir, filename):
    """
    Saves the metrics dictionary to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{filename}"
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")


def load_new_model(device, model_dir="examples/models/"):
    # Source of truth: dynamic_planning_with_moving_obstacles.ipynb
    # Logic: modelPNO from results/alexm/best_model.pt
    if not os.path.exists(model_dir):
        model_dir = "./models/"
        
    from models.fno import FNO2d
    from models.deepnormMultiGoal import DEEPNORM2dMultiGoal

    modelSDF = FNO2d(4, 1, 8, 8, 16).to(device)
    modelPNO = DEEPNORM2dMultiGoal(4, 8, 8, 16, in_channels=2).to(device)

    try:
        sdf_path = os.path.join(model_dir, "FNOSDF/best_model.pt")
        # Alex's model path as per notebook
        alex_path = os.path.join(os.path.dirname(model_dir), "results/alexm/best_model.pt")
        if not os.path.exists(alex_path):
            alex_path = "./results/alexm/best_model.pt"
        
        modelSDF.load_state_dict(torch.load(sdf_path, map_location=device, weights_only=True))
        modelPNO.load_state_dict(torch.load(alex_path, map_location=device, weights_only=True))
        print(f"Alex's models loaded successfully from {alex_path}.")
    except Exception as e:
        print(f"Warning: Alex's model loading failed ({e}). Using random weights.")

    modelSDF.eval()
    modelPNO.eval()
    return modelSDF, modelPNO


def load_pno_models(device, model_dir="examples/models/"):
    if not os.path.exists(model_dir):
        model_dir = "./models/"
    from models.fno import FNO2d
    from models.deepnormMultiGoal import DEEPNORM2dMultiGoal
    modelSDF = FNO2d(4, 1, 8, 8, 16).to(device)
    modelPNO = DEEPNORM2dMultiGoal(4, 8, 8, 16, in_channels=2).to(device)
    try:
        sdf_path = os.path.join(model_dir, "FNOSDF/best_model.pt")
        pno_path = os.path.join(model_dir, "PNOwPINN/best_model.pt")
        modelSDF.load_state_dict(torch.load(sdf_path, map_location=device, weights_only=True))
        modelPNO.load_state_dict(torch.load(pno_path, map_location=device, weights_only=True), strict=False)
        print(f"Pre-trained models loaded from {model_dir} (partial match).")
    except Exception as e:
        print(f"Warning: Pre-trained weights loading failed ({e}). Using random weights.")
    modelSDF.eval(); modelPNO.eval()
    return modelSDF, modelPNO


def compute_total_risk(obs_positions, obs_velocities, static_risk, H, W, sigma_dyn=1.0, alpha_dyn=0.6):
    total_risk = static_risk.copy()
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    for pos, vel in zip(obs_positions, obs_velocities):
        dx, dy = xs - pos[1], ys - pos[0]
        d_sq = dx ** 2 + dy ** 2
        d_norm = np.sqrt(d_sq) + 1e-6
        speed = np.linalg.norm(vel)
        cos_theta = (dx * vel[1] + dy * vel[0]) / (speed * d_norm + 1e-6)
        local_sigma = sigma_dyn * (speed + 1.0)
        w = 1 + alpha_dyn * cos_theta
        denom = 2 * (local_sigma * w) ** 2 + 1e-6
        risk = np.exp(-(d_sq) / denom)
        total_risk = np.maximum(total_risk, risk)
    return np.clip(total_risk, 0, 1)


def step_obstacles(obs_positions, obs_velocities, binary_map, H, W):
    for i in range(len(obs_positions)):
        pos = obs_positions[i].astype(float); vel = obs_velocities[i].astype(float)
        max_travel = max(abs(vel[0]), abs(vel[1])); num_steps = int(np.ceil(max_travel))
        if num_steps == 0: continue
        step_vel = vel / num_steps
        for _ in range(num_steps):
            bounce_y, bounce_x = False, False
            next_r = pos[0] + step_vel[0]
            grid_r_check, grid_c_current = int(np.clip(next_r, 0, H-1)), int(np.clip(pos[1], 0, W-1))
            if next_r < 0 or next_r >= H or binary_map[grid_r_check, grid_c_current] == 0:
                step_vel[0] *= -1; vel[0] *= -1; bounce_y = True; next_r = pos[0] + step_vel[0]
            next_c = pos[1] + step_vel[1]
            grid_r_current, grid_c_check = int(np.clip(pos[0], 0, H-1)), int(np.clip(next_c, 0, W-1))
            if next_c < 0 or next_c >= W or binary_map[grid_r_current, grid_c_check] == 0:
                step_vel[1] *= -1; vel[1] *= -1; bounce_x = True; next_c = pos[1] + step_vel[1]
            if not bounce_y and not bounce_x:
                grid_r_diag, grid_c_diag = int(np.clip(next_r, 0, H-1)), int(np.clip(next_c, 0, W-1))
                if binary_map[grid_r_diag, grid_c_diag] == 0:
                    step_vel *= -1; vel *= -1; next_r, next_c = pos[0] + step_vel[0], pos[1] + step_vel[1]
            pos[0], pos[1] = next_r, next_c
        obs_positions[i], obs_velocities[i] = pos, vel


def run_pno_inference(modelPNO, risk_map, chi_tensor, goal_coord, mask_tensor, device):
    # Order for Alex's model (from notebook): [Chi, Risk]
    risk_t = torch.tensor(risk_map, dtype=torch.float).reshape(1, risk_map.shape[0], risk_map.shape[1], 1).to(device)
    chi_t = chi_tensor.to(device)
    with torch.no_grad():
        input_tensor = torch.cat([chi_t, risk_t], dim=-1)
        cost_to_go = modelPNO(input_tensor, goal_coord)
    return (cost_to_go * mask_tensor).squeeze().cpu().numpy()


def plan_path_astar(start_node, goal_node, cost_map, value_function, risk_weight=10.0):
    env = Environment2D(goal_node, cost_map, valuefunction=value_function, risk_weight=risk_weight)
    path_cost, path, actions, expands, _ = AStar.plan(start_node, env)
    return np.asarray(path), expands


def run_risk_simulation(obs_pos_init, obs_vel_init, static_risk, binary_map, H, W, steps, sigma_dyn, alpha_dyn):
    obs_pos = [p.copy() for p in obs_pos_init]; obs_vel = [v.copy() for v in obs_vel_init]
    risk_frames, obs_frames = [], []
    for _ in range(steps):
        risk = compute_total_risk(obs_pos, obs_vel, static_risk, H, W, sigma_dyn, alpha_dyn)
        risk_frames.append(risk.copy()); obs_frames.append(np.array(obs_pos).copy())
        step_obstacles(obs_pos, obs_vel, binary_map, H, W)
    return {'risk': risk_frames, 'obs': obs_frames}


def run_planning_simulation(obs_pos_init, obs_vel_init, agent_pos_init, goal_data,
                            modelPNO, chi_tensor, mask_tensor, device,
                            static_risk, binary_map, H, W, sigma_dyn, alpha_dyn, global_path=None):
    obs_pos = [p.copy() for p in obs_pos_init]; obs_vel = [v.copy() for v in obs_vel_init]
    agent_pos = agent_pos_init.copy(); goal_coord = torch.tensor([goal_data], dtype=torch.int).to(device)
    risk_frames, obs_frames, agent_frames, path_frames = [], [], [], []
    
    # Metric initialization
    total_nodes_expanded = 0
    cumulative_risk = 0.0
    collision_count = 0
    
    max_steps = 500
    while not np.array_equal(agent_pos, goal_data) and len(agent_frames) < max_steps:
        risk = compute_total_risk(obs_pos, obs_vel, static_risk, H, W, sigma_dyn, alpha_dyn)
        risk_frames.append(risk.copy()); obs_frames.append(np.array(obs_pos).copy())
        
        cost_map = run_pno_inference(modelPNO, risk, chi_tensor, goal_coord, mask_tensor, device)
        path, expands = plan_path_astar(agent_pos, goal_data, risk, cost_map)
        
        path_frames.append(path.copy())
        if len(path) > 1: agent_pos = path[1]
        
        # Accumulate metrics after moving
        total_nodes_expanded += expands
        cumulative_risk += risk[int(agent_pos[0]), int(agent_pos[1])]
        if check_collision(agent_pos, obs_pos, binary_map):
            collision_count += 1
            
        agent_frames.append(agent_pos.copy()); step_obstacles(obs_pos, obs_vel, binary_map, H, W)
        
    metrics = {
        'path_length': len(agent_frames),
        'total_nodes_expanded': total_nodes_expanded,
        'cumulative_risk': float(cumulative_risk),
        'mean_risk_per_step': float(cumulative_risk / max(1, len(agent_frames))),
        'hard_collisions': collision_count,
        'success': np.array_equal(agent_pos, goal_data)
    }
    
    return {'risk': risk_frames, 'obs': obs_frames, 'agent': agent_frames, 'path': path_frames, 
            'global_path': global_path, 'metrics': metrics}


def run_dynamic_window_simulation(obs_pos_init, obs_vel_init, agent_pos_init, goal_data,
                                 modelPNO, chi_tensor, mask_tensor, device,
                                 static_risk, binary_map, H, W, sigma_dyn, alpha_dyn, global_path=None):
    import torch.nn.functional as F
    obs_pos = [p.copy() for p in obs_pos_init]; obs_vel = [v.copy() for v in obs_vel_init]
    agent_pos = agent_pos_init.copy()
    window_center = agent_pos.copy() # Stable center
    risk_frames, obs_frames, agent_frames, path_frames = [], [], [], []
    
    # Metric initialization
    total_nodes_expanded = 0
    cumulative_risk = 0.0
    collision_count = 0
    
    max_steps = 500; base_window_size = 20

    while not np.array_equal(agent_pos, goal_data) and len(agent_frames) < max_steps:
        risk_global = compute_total_risk(obs_pos, obs_vel, static_risk, H, W, sigma_dyn, alpha_dyn)
        risk_frames.append(risk_global.copy()); obs_frames.append(np.array(obs_pos).copy())

        max_obs_speed = max([np.linalg.norm(v) for v in obs_vel]) if obs_vel else 0
        window_size = int(base_window_size + 4 * max_obs_speed)
        half_w = window_size // 2

        if np.linalg.norm(agent_pos - window_center) > (half_w // 2):
            window_center = agent_pos.copy()
        
        r_min = int(np.clip(window_center[0] - half_w, 0, H - 2))
        r_max = int(np.clip(window_center[0] + half_w, r_min + 2, H))
        c_min = int(np.clip(window_center[1] - half_w, 0, W - 2))
        c_max = int(np.clip(window_center[1] + half_w, c_min + 2, W))

        local_risk = risk_global[r_min:r_max, c_min:c_max]
        local_risk_t = torch.tensor(local_risk, dtype=torch.float).reshape(1, 1, r_max-r_min, c_max-c_min).to(device)
        # Order: Chi, Risk. Upscale to 256x256.
        input_risk = F.interpolate(local_risk_t, size=(256, 256), mode='bilinear')
        
        chi_crop = chi_tensor[:, r_min:r_max, c_min:c_max, :]
        input_chi = F.interpolate(chi_crop.permute(0, 3, 1, 2), size=(256, 256), mode='bilinear')
        
        # Local Goal Selection
        if global_path is not None:
            local_goal = None
            for point in reversed(global_path):
                if r_min <= point[0] < r_max and c_min <= point[1] < c_max:
                    local_goal = np.array([point[0] - r_min, point[1] - c_min]); break
            if local_goal is None: local_goal = np.array([goal_data[0] - r_min, goal_data[1] - c_min])
        else:
            if r_min <= goal_data[0] < r_max and c_min <= goal_data[1] < c_max:
                local_goal = np.array([goal_data[0] - r_min, goal_data[1] - c_min])
            else:
                dr, dc = goal_data[0] - agent_pos[0], goal_data[1] - agent_pos[1]
                norm = max(abs(dr), abs(dc), 1e-6)
                local_goal = np.array([(agent_pos[0]-r_min) + (dr/norm)*(half_w-2), (agent_pos[1]-c_min) + (dc/norm)*(half_w-2)])
                local_goal[0], local_goal[1] = np.clip(local_goal[0], 0, (r_max-r_min)-1), np.clip(local_goal[1], 0, (c_max-c_min)-1)

        model_goal = torch.tensor(np.array([[local_goal[0] * (256.0/(r_max-r_min)), 
                                             local_goal[1] * (256.0/(c_max-c_min))]]), dtype=torch.int).to(device)
        
        input_tensor = torch.cat([input_chi, input_risk], dim=1).permute(0, 2, 3, 1) # [1, 256, 256, 2]
        with torch.no_grad():
            cost_to_go_upscaled = modelPNO(input_tensor, model_goal)
            cost_to_go_local = F.interpolate(cost_to_go_upscaled.permute(0, 3, 1, 2), 
                                             size=(r_max-r_min, c_max-c_min), mode='bilinear').squeeze().cpu().numpy()

        agent_local_pos = np.array([agent_pos[0] - r_min, agent_pos[1] - c_min])
        path_local, expands = plan_path_astar(agent_local_pos.astype(int), local_goal.astype(int), local_risk, cost_to_go_local)
        
        if len(path_local) > 0:
            path_global = path_local + np.array([r_min, c_min])
            path_frames.append(path_global.copy())
            if len(path_local) > 1: agent_pos = path_global[1]
        else:
            path_frames.append(np.array([]))
            
        # Accumulate metrics after moving
        total_nodes_expanded += expands
        cumulative_risk += risk_global[int(agent_pos[0]), int(agent_pos[1])]
        if check_collision(agent_pos, obs_pos, binary_map):
            collision_count += 1
            
        agent_frames.append(agent_pos.copy()); step_obstacles(obs_pos, obs_vel, binary_map, H, W)
        
    metrics = {
        'path_length': len(agent_frames),
        'total_nodes_expanded': total_nodes_expanded,
        'cumulative_risk': float(cumulative_risk),
        'mean_risk_per_step': float(cumulative_risk / max(1, len(agent_frames))),
        'hard_collisions': collision_count,
        'success': np.array_equal(agent_pos, goal_data)
    }
        
    return {'risk': risk_frames, 'obs': obs_frames, 'agent': agent_frames, 'path': path_frames, 
            'global_path': global_path, 'metrics': metrics}


def animation_step(frame, data_dict, artists):
    updated = []
    if 'im_risk' in artists:
        artists['im_risk'].set_array(data_dict['risk'][frame]); updated.append(artists['im_risk'])
    if 'obs_dots_bin' in artists:
        artists['obs_dots_bin'].set_data(data_dict['obs'][frame][:, 1], data_dict['obs'][frame][:, 0]); updated.append(artists['obs_dots_bin'])
    if 'obs_dots_risk' in artists:
        artists['obs_dots_risk'].set_data(data_dict['obs'][frame][:, 1], data_dict['obs'][frame][:, 0]); updated.append(artists['obs_dots_risk'])
    
    if 'global_path' in data_dict and data_dict['global_path'] is not None:
        gp = data_dict['global_path']
        if 'global_path_bin' in artists:
            artists['global_path_bin'].set_data(gp[:, 1], gp[:, 0]); updated.append(artists['global_path_bin'])
        if 'global_path_risk' in artists:
            artists['global_path_risk'].set_data(gp[:, 1], gp[:, 0]); updated.append(artists['global_path_risk'])

    if 'agent' in data_dict:
        if 'agent_dot_bin' in artists:
            artists['agent_dot_bin'].set_data([data_dict['agent'][frame][1]], [data_dict['agent'][frame][0]]); updated.append(artists['agent_dot_bin'])
        if 'agent_dot_risk' in artists:
            artists['agent_dot_risk'].set_data([data_dict['agent'][frame][1]], [data_dict['agent'][frame][0]]); updated.append(artists['agent_dot_risk'])
        if 'path_line' in artists:
            p = data_dict['path'][frame]
            if len(p) > 0: artists['path_line'].set_data(p[:, 1], p[:, 0])
            else: artists['path_line'].set_data([], [])
            updated.append(artists['path_line'])
    return updated
