#!/usr/bin/env python3
"""
ensemble_train.py

Train multiple instances of the same model architecture using different random seeds
(bagging/ensemble approach) by leveraging the modified run_training_trial function.
Logs the paths of the successfully trained models under a parent MLflow run.
"""

import argparse
import torch
import os
import mlflow
import json # To save the list of model paths

# Import necessary components from train.py
from train import run_training_trial, parse_arguments, DEFAULT_OUTPUT_DIR

# --- Configuration ---
RANDOM_SEEDS = [42, 123, 456, 789, 1011] # Define the seeds for the ensemble members
ENSEMBLE_RUN_NAME = "hvac_ensemble_training"
MODEL_PATHS_FILENAME = "ensemble_model_paths.json"

def main():
    # --- Parse Base Arguments ---
    # Use the same parser as train.py to get shared hyperparameters
    args = parse_arguments()
    
    print(f"Starting ensemble training with {len(RANDOM_SEEDS)} seeds: {RANDOM_SEEDS}")
    print(f"Base arguments: {args}")

    # --- Determine Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Parent MLflow Run ---
    mlflow.set_experiment("HVAC_Prediction") # Optional: Set experiment name if desired
    with mlflow.start_run(run_name=ENSEMBLE_RUN_NAME) as parent_run:
        print(f"Parent MLflow Run ID: {parent_run.info.run_id}")
        mlflow.log_params(vars(args)) # Log base hyperparameters
        mlflow.log_param("random_seeds_used", RANDOM_SEEDS)
        mlflow.log_param("num_ensemble_members", len(RANDOM_SEEDS))

        successful_model_paths = []
        failed_seeds = []

        # --- Loop Through Seeds ---
        for seed in RANDOM_SEEDS:
            print(f"\n--- Training Ensemble Member: Seed {seed} ---")
            try:
                # run_training_trial now returns (metric, model_path)
                # We don't need the metric here, just the path
                _, model_path = run_training_trial(args, device, trial_num=None, random_seed=seed) 
                
                if model_path and os.path.exists(model_path):
                    print(f"Successfully trained model for seed {seed}. Path: {model_path}")
                    # Store relative path for portability if possible
                    relative_path = os.path.relpath(model_path, start=os.getcwd()) # Or use absolute path if needed
                    successful_model_paths.append(relative_path) 
                else:
                     print(f"Training failed or model path not returned for seed {seed}.")
                     failed_seeds.append(seed)

            except Exception as e:
                print(f"Error training model for seed {seed}: {e}")
                failed_seeds.append(seed)
                # Decide if you want to stop the whole ensemble run on one failure
                # continue 

        # --- Log Results ---
        print("\n--- Ensemble Training Summary ---")
        print(f"Successfully trained models: {len(successful_model_paths)}")
        print(f"Failed seeds: {failed_seeds}")
        print("Successful model paths:")
        for path in successful_model_paths:
            print(f"  - {path}")

        # Log the list of successful model paths to a file within the parent run artifacts
        output_dir = getattr(args, 'output_dir', DEFAULT_OUTPUT_DIR) # Use output dir for this file too
        os.makedirs(output_dir, exist_ok=True) 
        paths_filepath = os.path.join(output_dir, MODEL_PATHS_FILENAME)
        
        try:
            with open(paths_filepath, 'w') as f:
                json.dump(successful_model_paths, f, indent=4)
            print(f"Saved list of model paths to: {paths_filepath}")
            mlflow.log_artifact(paths_filepath, artifact_path="ensemble_info")
            print(f"Logged '{MODEL_PATHS_FILENAME}' to MLflow artifacts path 'ensemble_info'")
        except Exception as e:
            print(f"Error saving or logging model paths file: {e}")
            
        # Log overall status
        mlflow.log_metric("num_successful_members", len(successful_model_paths))
        mlflow.log_metric("num_failed_members", len(failed_seeds))
        if failed_seeds:
             mlflow.set_tag("status", "completed_with_failures")
        elif not successful_model_paths:
             mlflow.set_tag("status", "failed")
        else:
             mlflow.set_tag("status", "completed")

if __name__ == '__main__':
    main() 