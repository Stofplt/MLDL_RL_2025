import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

def plot(file_name, custom_env, eval_freq=5000):
    """Plot training results with proper reward scaling (-50 to 0)"""
    graph_title = file_name.replace("_", " ")

    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    outfile_stem = f"./plots/{file_name}"

    # Load results and create timesteps array
    results_path = f"./results/{file_name}.npy"
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}")
        return
        
    results = np.load(results_path)
    x = np.arange(0, len(results) * eval_freq, eval_freq)

    # Check if this is a curriculum learning run by checking for stage files
    stage_files = sorted(glob(f"./results/{file_name}_stage*.npy"))
    has_curriculum = len(stage_files) > 0

    if has_curriculum:
        # Plot global progress (measured against full task)
        plt.figure(figsize=(10, 5))
        plt.fill_between(x, results[:, 0] - results[:, 1], results[:, 0] + results[:, 1], alpha=0.3)
        plt.plot(x, results[:, 0], linewidth=2, label='Global Success Rate')
        plt.ylim(-50, 0)  # Fix y-axis range for rewards
        plt.xlabel("Total Timesteps")
        plt.ylabel("Returns")
        plt.title(f"{graph_title}\nGlobal Progress (Full Task)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{outfile_stem}_global.png", bbox_inches='tight')
        print(f"Output to: {outfile_stem}_global.png")
        plt.close()
        
        # Find the latest stage file (current stage)
        latest_stage = max([int(f.split('_stage')[-1].split('.')[0]) for f in stage_files])
        latest_stage_path = f"./results/{file_name}_stage{latest_stage}.npy"
        
        # Plot local progress (current stage only)
        if os.path.exists(latest_stage_path):
            stage_results = np.load(latest_stage_path)
            x_stage = np.arange(len(stage_results)) * eval_freq
            
            plt.figure(figsize=(10, 5))
            plt.plot(x_stage, np.clip(stage_results[:, 0], -50, 0), 'g-', 
                    label=f'Stage {latest_stage} Returns', linewidth=2)
            plt.fill_between(x_stage, 
                           np.clip(stage_results[:, 0] - stage_results[:, 1], -50, 0),
                           np.clip(stage_results[:, 0] + stage_results[:, 1], -50, 0),
                           alpha=0.3)
            plt.axhline(y=-10, color='r', linestyle='--', label='Success Threshold')
            plt.ylim(-50, 0)  # Match global plot scale
            plt.xlabel("Stage Timesteps")
            plt.ylabel("Returns")
            plt.title(f"{graph_title}\nCurrent Stage Progress (Stage {latest_stage})")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{outfile_stem}_current_stage.png", bbox_inches='tight')
            print(f"Output to: {outfile_stem}_current_stage.png")
            plt.close()
    else:
        # For non-curriculum runs, just plot the overall progress
        plt.figure(figsize=(10, 5))
        plt.fill_between(x, results[:, 0] - results[:, 1], results[:, 0] + results[:, 1], alpha=0.3)
        plt.plot(x, results[:, 0], linewidth=2, label='Mean Return')
        plt.ylim(-50, 0)  # Fix y-axis range for all plots
        plt.xlabel("Timesteps")
        plt.ylabel("Returns")
        plt.title(f"{graph_title}\nTraining Progress")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{outfile_stem}.png", bbox_inches='tight')
        print(f"Output to: {outfile_stem}.png")
        plt.close()
