#!/usr/bin/env python3
"""
predict_hvac.py

Load a trained Hybrid model (LSTM+TCN) and scalers saved by train.py's MLflow run.
Predict the single 'puissance_cvac_future' value for each ID
in test_features.csv and save results in Kaggle submission format.
Logs the submission file to a new MLflow run.
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import joblib
import mlflow # Added MLflow
import tempfile # To download artifacts
import ast # To evaluate string representations of lists/bools from params

# Import necessary components
from core_components import LSTMModel
from tcn_model import TCNModel
from hybrid_model import HybridModel
from utils import add_time_features, safe_literal_eval # Import from utils

# Copy the add_time_features function from train.py
def add_time_features(df):
    """Adds time-based features to the DataFrame."""
    # Ensure 'date' column is datetime type
    if 'date' not in df.columns:
        raise ValueError("'date' column is required for time features but not found in test data.")
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    return df

# Helper to safely evaluate MLflow params which are strings
def safe_literal_eval(val):
    try:
        # Handle boolean strings explicitly
        if isinstance(val, str):
            if val.lower() == 'true':
                return True
            if val.lower() == 'false':
                return False
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # If it's not a literal, return the original string
        return val

def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict future HVAC consumption using Hybrid Model (LSTM+TCN) from an MLflow run')
    # --- MLflow Input ---
    parser.add_argument('--training_run_id', required=True, help='MLflow Run ID of the training run to load artifacts and parameters from.')

    # --- Data Input/Output ---
    parser.add_argument('--test_csv', default=os.path.join('dataset', 'test_features.csv'), help='Path to test features CSV file')
    parser.add_argument('--results_dir', default='results', help='Directory to save prediction results locally')
    parser.add_argument('--output_filename', default='submission.csv', help='Filename for the Kaggle submission CSV (will also be logged to MLflow)')

    # --- Prediction Specific Args (Optional overrides, but typically inferred) ---
    # Sequence length might be needed if test data prep differs significantly, but usually inferred.
    # parser.add_argument('--seq_len', type=int, help='Sequence length (optional, will try to load from training params)')

    # Model architecture parameters are now loaded from MLflow run, removed from args
    return parser.parse_args()

def load_training_info(client, training_run_id):
    print(f"Loading parameters and artifact URI from training run: {training_run_id}")
    try:
        training_run = client.get_run(training_run_id)
        train_params = training_run.data.params
        train_artifact_uri = training_run.info.artifact_uri
        print(f"Successfully loaded info for run: {training_run_id}")
        return train_params, train_artifact_uri
    except Exception as e:
        raise ValueError(f"Could not fetch training run '{training_run_id}' from MLflow. Error: {e}")

def extract_training_params(train_params):
    print("Extracting and parsing parameters logged during training...")
    try:
        # Data related
        seq_len = int(train_params['seq_len'])
        use_time_features = safe_literal_eval(train_params.get('use_time_features', 'True'))

        # LSTM related
        hidden_size = int(train_params['hidden_size'])
        num_layers = int(train_params['num_layers'])
        dropout = float(train_params['dropout'])

        # TCN related
        tcn_num_channels_str = train_params['tcn_num_channels']
        tcn_num_channels = safe_literal_eval(tcn_num_channels_str)
        if not isinstance(tcn_num_channels, list):
             raise ValueError(f"Could not parse tcn_num_channels: '{tcn_num_channels_str}'")
        tcn_kernel_size = int(train_params['tcn_kernel_size'])
        tcn_dropout = float(train_params['tcn_dropout'])
        
        # Extract feature columns used during training
        all_feature_cols_str = train_params.get('features') # It was logged as a string representation of a list
        if not all_feature_cols_str:
            raise ValueError("'features' parameter not found in the training run.")
        all_feature_cols = safe_literal_eval(all_feature_cols_str)
        if not isinstance(all_feature_cols, list):
             raise ValueError(f"Could not parse 'features' from training params: '{all_feature_cols_str}'")
        input_size = len(all_feature_cols)

        print(f"  Using seq_len: {seq_len}, use_time_features: {use_time_features}")
        print(f"  LSTM: hidden={hidden_size}, layers={num_layers}, dropout={dropout}")
        print(f"  TCN: channels={tcn_num_channels}, kernel={tcn_kernel_size}, dropout={tcn_dropout}")
        print(f"  Features ({input_size}): {all_feature_cols}")

        parsed_params = {
            'seq_len': seq_len,
            'use_time_features': use_time_features,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'tcn_num_channels': tcn_num_channels,
            'tcn_kernel_size': tcn_kernel_size,
            'tcn_dropout': tcn_dropout,
            'all_feature_cols': all_feature_cols,
            'input_size': input_size
        }
        return parsed_params

    except KeyError as e:
        raise ValueError(f"Missing expected parameter '{e}' in training run parameters.")
    except (ValueError, TypeError, SyntaxError) as e:
         raise ValueError(f"Error parsing parameter from training run. Error: {e}")

def download_artifacts(client, training_run_id, tmpdir):
    print(f"Downloading artifacts from run {training_run_id} to {tmpdir}")
    try:
        # Correct artifact paths based on how train.py logged them
        model_artifact_path = "best_model/hybrid_hvac_best.pth" # Use the actual filename logged
        feature_scaler_artifact_path = "scalers/feature_scaler.save"
        target_scaler_artifact_path = "scalers/target_scaler.save"

        model_local_path = client.download_artifacts(training_run_id, model_artifact_path, tmpdir)
        feature_scaler_local_path = client.download_artifacts(training_run_id, feature_scaler_artifact_path, tmpdir)
        target_scaler_local_path = client.download_artifacts(training_run_id, target_scaler_artifact_path, tmpdir)

        print(f"  Downloaded model state dict to: {model_local_path}")
        print(f"  Downloaded feature scaler to: {feature_scaler_local_path}")
        print(f"  Downloaded target scaler to: {target_scaler_local_path}")
        return model_local_path, feature_scaler_local_path, target_scaler_local_path

    except Exception as e:
        raise RuntimeError(f"Failed to download artifacts from run '{training_run_id}'. Check paths and permissions. Error: {e}")

def prepare_test_data(args, train_params, feature_scaler_local_path):
    print("Loading and preparing test data...")
    test_csv_path = args.test_csv
    use_time_features = train_params['use_time_features']
    all_feature_cols = train_params['all_feature_cols']
    input_size = train_params['input_size']
    seq_len = train_params['seq_len']

    # --- Load Test Data ---
    parse_dates = ['date'] if use_time_features else None
    df_test = pd.read_csv(test_csv_path, parse_dates=parse_dates)

    if 'ID' not in df_test.columns:
        raise ValueError("Test CSV must contain an 'ID' column.")
    ids = df_test['ID'].values

    # --- Add Time Features (if applicable) ---
    if use_time_features:
        print("Adding time features...")
        df_test = add_time_features(df_test)
    else:
        print("Not using time features (as per training run).")

    # --- Validate and Select Features ---
    if not set(all_feature_cols).issubset(df_test.columns):
        missing_cols = set(all_feature_cols) - set(df_test.columns)
        raise ValueError(f'Test CSV missing required feature columns used during training: {missing_cols}')
    features = df_test[all_feature_cols].values
    print(f"Using {input_size} features (determined from training run): {all_feature_cols}")

    # --- Load Scaler and Scale Features ---
    print(f"Loading feature scaler from: {feature_scaler_local_path}...")
    feat_scaler = joblib.load(feature_scaler_local_path)

    num_features_expected_by_scaler = feat_scaler.n_features_in_
    if input_size != num_features_expected_by_scaler:
         raise ValueError(
             f"Feature mismatch: Prepared {input_size} features based on training params, "
             f"but the loaded scaler expects {num_features_expected_by_scaler} features. "
             f"Inconsistency between training run parameters and saved scaler artifact."
         )

    print(f"Scaling {input_size} features...")
    features_scaled = feat_scaler.transform(features)

    # --- Input Sequence Preparation ---
    print(f"Preparing input sequences using seq_len={seq_len}...")
    num_samples = features_scaled.shape[0]
    num_features = features_scaled.shape[1]
    X_test_seq = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    for i in range(num_samples):
        X_test_seq[i, :, :] = np.tile(features_scaled[i, :], (seq_len, 1))
    print(f"Prepared input data shape: {X_test_seq.shape}")
    
    return X_test_seq, ids

def build_model_from_params(train_params):
    print("Initializing Hybrid Model structure based on training parameters...")
    input_size = train_params['input_size']
    lstm_model_instance = LSTMModel(
        input_size=input_size,
        hidden_size=train_params['hidden_size'],
        num_layers=train_params['num_layers'],
        dropout=train_params['dropout']
    )
    tcn_model_instance = TCNModel(
        input_size=input_size,
        output_size=1,
        num_channels=train_params['tcn_num_channels'],
        kernel_size=train_params['tcn_kernel_size'],
        dropout=train_params['tcn_dropout']
    )
    model = HybridModel(lstm_model=lstm_model_instance, tcn_model=tcn_model_instance)
    print("Hybrid Model structure initialized.")
    return model

def run_prediction(model, X_test_seq, target_scaler_local_path, device):
    print("Running predictions...")
    model.eval() # Ensure model is in eval mode
    preds_scaled_list = []
    
    # Create a simple DataLoader for prediction batches
    X_tensor = torch.from_numpy(X_test_seq).float()
    pred_dataset = torch.utils.data.TensorDataset(X_tensor)
    # Use a reasonable batch size, can be adjusted based on memory
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
         for Xb, in pred_loader: # Unpack the single tensor from dataset
            Xb = Xb.to(device)
            batch_preds_scaled = model(Xb).cpu().numpy() # Shape: (batch_size, 1)
            preds_scaled_list.append(batch_preds_scaled)

    preds_scaled = np.concatenate(preds_scaled_list) # Shape: (num_samples, 1)

    # --- Load Target Scaler and Inverse Transform ---
    print(f"Loading target scaler from: {target_scaler_local_path}...")
    target_scaler = joblib.load(target_scaler_local_path)
    preds = target_scaler.inverse_transform(preds_scaled).flatten() # Shape: (num_samples,)
    print(f"Generated {len(preds)} predictions.")
    return preds

def save_and_log_predictions(args, ids, preds, prediction_run_info):
    print("Formatting predictions for submission...")
    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, args.output_filename)
    df_out = pd.DataFrame({
        'ID': ids,
        'puissance_cvac_future': preds
    })
    df_out.to_csv(output_path, index=False)
    print(f'Submission file saved locally to {output_path}')

    # --- Log Submission File to MLflow ---
    print(f"Logging submission file '{args.output_filename}' to MLflow run {prediction_run_info.run_uuid}...")
    mlflow.log_artifact(output_path, artifact_path="predictions")
    print("Submission artifact logged.")

def main():
    args = parse_arguments()

    # --- Start Prediction MLflow Run ---
    # This run will log the prediction output and link to the training run
    with mlflow.start_run(run_name="prediction") as prediction_run:
        prediction_run_info = prediction_run.info # Store info for logging
        print(f"Started Prediction MLflow Run: {prediction_run_info.run_uuid}")
        mlflow.log_param("training_run_id", args.training_run_id)
        mlflow.set_tag("mlflow.note.content", f"Predictions based on training run: {args.training_run_id}")


        # --- Get Training Run Info ---
        client = mlflow.tracking.MlflowClient()
        train_params_raw, train_artifact_uri = load_training_info(client, args.training_run_id)

        # --- Extract Parameters from Training Run ---
        train_params = extract_training_params(train_params_raw)


        # --- Download Artifacts ---
        with tempfile.TemporaryDirectory() as tmpdir:
            model_local_path, feature_scaler_local_path, target_scaler_local_path = download_artifacts(client, args.training_run_id, tmpdir)

            # --- Load Test Data ---
            X_test_seq, ids = prepare_test_data(args, train_params, feature_scaler_local_path)

            # --- Load Model State ---
            print(f"Loading model state dict from downloaded artifact: {model_local_path}...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

            # Initialize model structure using parameters from training run
            model = build_model_from_params(train_params)

            # Load the state dict
            model.load_state_dict(torch.load(model_local_path, map_location=device))
            model.to(device)
            print("Hybrid Model state loaded successfully.")

            # --- Run Predictions ---
            preds = run_prediction(model, X_test_seq, target_scaler_local_path, device)

            # --- Format and Save Submission Locally ---
            # print("Formatting predictions for submission...")
            # os.makedirs(args.results_dir, exist_ok=True)
            # output_path = os.path.join(args.results_dir, args.output_filename)
            # df_out = pd.DataFrame({
            #     'ID': ids,
            #     'puissance_cvac_future': preds
            # })
            # df_out.to_csv(output_path, index=False)
            # print(f'Submission file saved locally to {output_path}')

            # # --- Log Submission File to MLflow ---
            # print(f"Logging submission file '{args.output_filename}' to MLflow run {prediction_run.info.run_uuid}...")
            # mlflow.log_artifact(output_path, artifact_path="predictions")
            save_and_log_predictions(args, ids, preds, prediction_run_info)
            
            print("Prediction script finished.")

if __name__ == '__main__':
    main() 