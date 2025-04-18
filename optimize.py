import optuna
import torch
import argparse
import os
from train import run_training_trial, parse_arguments, DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_DIR # Import the refactored function and defaults

# --- Configuration ---
N_TRIALS = 30 # Number of trials to run
STUDY_NAME = "hvac_optimization" # Optional: For organizing studies, especially with storage
# Use a database for storage if you want to resume studies or run distributed
# STORAGE_URL = "sqlite:///hvac_optuna.db" 
STORAGE_URL = None 

# --- Objective Function ---
def objective(trial):
    """Optuna objective function to train and evaluate the model."""
    
    # --- Device ---
    # Determine device for each trial (can be done once outside if always the same)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Suggest Hyperparameters ---
    # Create a namespace or dict to hold parameters for run_training_trial
    # Start with defaults from parse_arguments and override with suggestions
    args = parse_arguments() # Get default args first
    
    # Override with suggested values
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.num_layers = trial.suggest_int("num_layers", 1, 4) # Example range
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    args.hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16) # Example range/step
    
    # TCN Params - Example: Suggest a uniform channel size for the default 2 layers
    tcn_channel_size = trial.suggest_categorical("tcn_channel_size", [32, 64, 96])
    args.tcn_num_channels = [tcn_channel_size] * 2 # Assuming 2 layers as per default
    args.tcn_kernel_size = trial.suggest_categorical("tcn_kernel_size", [2, 3, 5])
    
    # You can add more parameters here, e.g.:
    # args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    # args.seq_len = trial.suggest_int("seq_len", 8, 32) 
    # args.loss_function = trial.suggest_categorical("loss_function", ['MSE', 'Huber'])
    
    # --- Fixed Parameters (already set by parse_arguments defaults) ---
    # Ensure essential parameters like file paths, epochs etc. are correctly set.
    # We rely on parse_arguments() defaults here, but you could override them:
    # args.num_epochs = 50 # Or make it tunable: trial.suggest_int("num_epochs", ...)
    # args.train_csv = 'path/to/train.csv'
    # args.test_csv = 'path/to/test.csv'
    # args.output_dir = DEFAULT_OUTPUT_DIR # Use default or specify
    # args.results_dir = DEFAULT_RESULTS_DIR # Use default or specify
    # ... and so on for any other args needed by run_training_trial

    # --- Run the Training Trial ---
    # Pass the trial number for unique naming/logging
    test_mae = run_training_trial(args, device, trial_num=trial.number)
    
    # --- Return Metric to Optimize ---
    # Optuna will minimize this value
    return test_mae

# --- Main Optimization Script ---
if __name__ == "__main__":
    # Create or load the study
    # Sampler: TPE is the default and usually works well.
    # Pruner: MedianPruner stops unpromising trials early.
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1) # Configure pruner
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL, 
        direction="minimize", # We want to minimize MAE
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True # Set to True if using storage and want to resume
    )

    print(f"Starting optimization with {N_TRIALS} trials.")
    print(f"Sampler: {study.sampler.__class__.__name__}, Pruner: {study.pruner.__class__.__name__}")
    print(f"Using storage: {STORAGE_URL or 'In-memory'}")
    
    # Start the optimization
    # Use try-except to catch KeyboardInterrupt and still print results
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None) # Adjust timeout (seconds) if needed
    except KeyboardInterrupt:
        print("Optimization stopped manually.")

    # --- Results ---
    print("\nOptimization Finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"  Value (MAE): {best_trial.value:.6f}")

    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Optional: Save results or visualize
    # study.trials_dataframe().to_csv("optuna_results.csv")
    # if optuna.visualization.is_available():
    #     fig1 = optuna.visualization.plot_optimization_history(study)
    #     fig1.show()
    #     fig2 = optuna.visualization.plot_param_importances(study)
    #     fig2.show()
    
    print("\nMLflow UI might show nested runs for each trial under the main script run (if run via `mlflow run`).")
    print(f"Access best trial's artifacts via MLflow run corresponding to trial {best_trial.number}.") 