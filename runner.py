"""
Experiment runner script for automated hyperparameter search and model training.

This script reads a runner configuration file that defines multiple hyperparameter
combinations and automatically runs training experiments for each combination.
It supports grid search over multiple parameters and handles experiment failures gracefully.
"""

# Standard library imports
import pdb
import traceback
from itertools import product

# Third-party imports
import yaml

# Local imports
from train import main

if __name__ == "__main__":
    # Load runner configuration that defines hyperparameter combinations
    runner_opt = yaml.safe_load(open(f"configs/runner.yaml", "r"))
    # runner_opt = {k: v for k, v in runner_opt.items() if k!='model'}

    # Prepare keys and values for Cartesian product (grid search)
    keys = [[k] * len(v) for k, v in runner_opt.items()]
    values = list(runner_opt.values())

    # Iterate through all combinations of hyperparameters
    for kt, vt in zip(product(*keys), product(*values)):
        # Create dictionary for current hyperparameter combination
        item_dict = {k: v for k, v in zip(kt, vt)}

        # Load base configuration for the specified model
        opt = yaml.safe_load(open(f"configs/train_cfg/{item_dict['model']}.yaml", "r"))

        # Update configuration with current hyperparameter values
        for k, v in item_dict.items():
            if k != "model":
                flag = False
                # Search for the key in training, model, or dataset sections
                for mk in ["training", "model", "dataset"]:
                    if k in opt[mk]:
                        opt[mk][k] = v
                        flag = True
                if not flag:
                    print("key not found")
                    exit(1)

        # Save updated configuration and run training
        try:
            # Write updated configuration back to file
            with open(f"configs/train_cfg/{item_dict['model']}.yaml", "w") as f:
                yaml.dump(opt, f)

            print("running item_dict", item_dict, "\n\n")

            # Start training with the current configuration
            main(f"configs/train_cfg/{item_dict['model']}.yaml")

        except Exception as e:
            # Handle training failures gracefully and continue with next experiment
            print(traceback.format_exc())
            continue
