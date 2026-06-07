import argparse
import os
import numpy as np
from demonstration import run_demo


def main():
    parser = argparse.ArgumentParser(description="Run batch path planning experiments.")

    # Core Experiment Settings
    parser.add_argument("--map_size", type=int, default=64, help="Map size (64 or 256).")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting map index.")
    parser.add_argument("--end_idx", type=int, default=10, help="Ending map index (exclusive).")
    parser.add_argument("--method", type=str, default="whole_map", choices=["whole_map", "sliding_window"],
                        help="Planning method to use.")
    parser.add_argument("--model", type=str, default="original", choices=["original", "new"],
                        help="Model weights to load ('original' or 'new').")

    # Scenario Parameters (Passed to demo.run_demo)
    parser.add_argument("--perc_obs", type=float, default=0.02, help="Percentage of open cells occupied by obstacles.")
    parser.add_argument("--dynamic_speed", type=float, default=1.0, help="Movement speed of obstacles.")
    parser.add_argument("--sigma_static", type=float, default=1.0, help="Risk sigma for static obstacles.")
    parser.add_argument("--sigma_dynamic", type=float, default=1.0, help="Risk sigma for dynamic obstacles.")
    parser.add_argument("--alpha_dynamic", type=float, default=0.3, help="Alpha parameter for dynamic risk.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    # Output Settings
    parser.add_argument("--output_dir", type=str, default="data", help="Root directory for output data.")

    args = parser.parse_args()

    # Get total number of maps available
    data_dir = f"dataset/synthetic/{args.map_size}x{args.map_size}/"
    if not os.path.exists(data_dir):
        data_dir = f"examples/dataset/synthetic/{args.map_size}x{args.map_size}/"

    actual_masks = np.load(os.path.join(data_dir, "mask.npy"))
    total_maps = actual_masks.shape[0]

    end_idx = min(args.end_idx, total_maps)
    print(f"Starting batch experiment: Maps {args.start_idx} to {end_idx - 1} ({args.map_size}x{args.map_size})")

    # Iterate through maps
    for idx in range(args.start_idx, end_idx):
        print(f"\n--- Running Map {idx}/{total_maps - 1} ---")
        try:
            # Convert args to dictionary and override map_idx
            experiment_kwargs = vars(args).copy()
            del experiment_kwargs['start_idx']
            del experiment_kwargs['end_idx']
            experiment_kwargs['map_idx'] = idx

            run_demo(**experiment_kwargs)

        except Exception as e:
            print(f"Error on Map {idx}: {e}")
            continue

    print("\nBatch experiment complete.")


if __name__ == "__main__":
    main()
