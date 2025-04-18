#!/usr/bin/env python3
"""
train.py

Train an enhanced LSTM model with options for feature engineering, regularization,
LR scheduling, and early stopping to predict 'puissance_cvac_future'.
Evaluates on validation and test sets, saves the best model, scalers,
and evaluation results.
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split # For train/val split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from tqdm import tqdm

# Import core components from hvac_lstm_prototype
from hvac_lstm_prototype import HVACDataset, LSTMModel, create_sequences
from tcn_model import TCNModel
from hybrid_model import HybridModel

def add_time_features(df):
    # Ensure 'date' column is datetime type
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    # Optionally add sine/cosine transformations for cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    return df

def load_series(csv_path, feature_cols, target_col, use_time_features=True):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Add time features before selecting columns
    if use_time_features:
        df = add_time_features(df)
        # Update feature_cols to include new time features
        time_feature_cols = ['hour_sin', 'hour_cos', 'dayofweek', 'month_sin', 'month_cos']
        all_feature_cols = feature_cols + time_feature_cols
    else:
        all_feature_cols = feature_cols
        
    # Ensure all feature columns exist
    missing_cols = [col for col in all_feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required feature columns: {missing_cols}")
        
    features = df[all_feature_cols].values
    target = df[target_col].values
    return features, target, df, all_feature_cols # Return updated feature list

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced LSTM for single-step future prediction')
    # Data/Paths
    parser.add_argument('--train_csv', default='dataset/train_split.csv', help='Path to training data CSV')
    parser.add_argument('--test_csv', default='dataset/test_split.csv', help='Path to test data CSV')
    parser.add_argument('--output_dir', default='models', help='Directory to save model and scalers')
    parser.add_argument('--results_dir', default='results', help='Directory to save evaluation results')
    parser.add_argument('--model_filename', default='lstm_hvac_best.pth', help='Filename for saved BEST model state dict')
    parser.add_argument('--feature_scaler_filename', default='feature_scaler.save', help='Filename for saved feature scaler')
    parser.add_argument('--target_scaler_filename', default='target_scaler.save', help='Filename for saved target scaler')
    
    # Model Hyperparameters
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length for LSTM input')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for LSTM and FC layers')
    
    # TCN Hyperparameters - Added
    parser.add_argument('--tcn_num_channels', type=int, nargs="+", default=[64, 64], help="Number of channels in each TCN layer (list)")
    parser.add_argument('--tcn_kernel_size', type=int, default=3, help="Kernel size for TCN convolutions")
    parser.add_argument('--tcn_dropout', type=float, default=0.2, help="Dropout rate for TCN layers")

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--loss_function', type=str, default='Huber', choices=['MSE', 'Huber'], help='Loss function to use (MSE or Huber)')
    parser.add_argument('--val_split_ratio', type=float, default=0.15, help='Fraction of training data to use for validation')

    # Feature Engineering
    parser.add_argument('--use_time_features', action='store_true', default=True, help='Add time-based features (hour, day, month)')
    parser.add_argument('--no_time_features', action='store_false', dest='use_time_features', help='Do not add time-based features')

    # Training Enhancements
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True, help='Use ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--no_lr_scheduler', action='store_false', dest='use_lr_scheduler', help='Do not use LR scheduler')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau scheduler')
    
    parser.add_argument('--use_early_stopping', action='store_true', default=True, help='Use early stopping based on validation loss')
    parser.add_argument('--no_early_stopping', action='store_false', dest='use_early_stopping', help='Do not use early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')

    parser.add_argument('--use_gradient_clipping', action='store_true', default=True, help='Use gradient clipping during training')
    parser.add_argument('--no_gradient_clipping', action='store_false', dest='use_gradient_clipping', help='Do not use gradient clipping')
    parser.add_argument('--gradient_clip_max_norm', type=float, default=1.0, help='Max norm for gradient clipping')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Define feature and target columns ---
    base_feature_cols = [
        'temperature_exterieure', 'humidite', 'ensoleillement',
        'vitesse_vent', 'direction_vent', 'temperature_interieure'
    ]
    target_col = 'puissance_cvac'

    # --- Create output directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # --- Data Loading and Preparation ---
    print("Loading data...")
    # Load train data and get updated feature list
    feat_train_val, targ_train_val, df_train_val, all_feature_cols = load_series(
        args.train_csv, base_feature_cols, target_col, args.use_time_features
    )
    # Load test data using the same feature list
    feat_test, targ_test, df_test, _ = load_series(
        args.test_csv, base_feature_cols, target_col, args.use_time_features
    )
    print(f"Loaded {len(df_train_val)} train/val samples and {len(df_test)} test samples.")
    print(f"Using features: {all_feature_cols}")

    # --- Scale Features ---
    print("Scaling features...")
    feat_scaler = MinMaxScaler()
    feat_train_val_s = feat_scaler.fit_transform(feat_train_val)
    feat_test_s = feat_scaler.transform(feat_test)

    # --- Scale Targets ---
    print("Scaling targets...")
    targ_scaler = MinMaxScaler()
    targ_train_val_s = targ_scaler.fit_transform(targ_train_val.reshape(-1,1)).flatten()
    targ_test_s = targ_scaler.transform(targ_test.reshape(-1,1)).flatten()

    # --- Create Sequences ---
    print("Creating sequences...")
    X_train_val, y_train_val = create_sequences((feat_train_val_s, targ_train_val_s), seq_len=args.seq_len)
    X_test, y_test = create_sequences((feat_test_s, targ_test_s), seq_len=args.seq_len)

    # --- Split Train/Validation ---
    print(f"Splitting training data into train/validation ({1-args.val_split_ratio:.0%}/{args.val_split_ratio:.0%})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_split_ratio, shuffle=False # Time series data, no shuffle
    )
    print(f"Created {len(X_train)} training sequences, {len(X_val)} validation sequences, {len(X_test)} test sequences.")

    # --- DataLoaders ---
    train_ds = HVACDataset(X_train, y_train)
    val_ds = HVACDataset(X_val, y_val)
    test_ds = HVACDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model, Loss, Optimizer, Scheduler ---
    print("Initializing models...")
    lstm_model_instance = LSTMModel(
        input_size=len(all_feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    tcn_model_instance = TCNModel(
        input_size=len(all_feature_cols),
        output_size=1, # Ensure TCN also outputs a single value
        num_channels=args.tcn_num_channels,
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout
    )
    
    model = HybridModel(lstm_model=lstm_model_instance, tcn_model=tcn_model_instance)
    model.to(device)
    print("Initialized Hybrid Model (LSTM + TCN).")

    if args.loss_function == 'MSE':
        criterion = nn.MSELoss()
        print("Using MSELoss")
    elif args.loss_function == 'Huber':
        criterion = nn.HuberLoss()
        print("Using HuberLoss")
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=args.lr_scheduler_factor, 
            patience=args.lr_scheduler_patience, 
            verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler (patience={args.lr_scheduler_patience}, factor={args.lr_scheduler_factor})")

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(args.output_dir, args.model_filename)

    for epoch in range(1, args.num_epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False)
        for Xb, yb in train_pbar:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            if args.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_max_norm)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_train_loss = train_loss / len(train_ds)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds_scaled_list = []
        val_trues_scaled_list = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                out_scaled = model(Xb)
                loss = criterion(out_scaled, yb)
                val_loss += loss.item() * Xb.size(0)
                val_preds_scaled_list.append(out_scaled.cpu().numpy())
                val_trues_scaled_list.append(yb.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_ds)
        print(f'Epoch {epoch}/{args.num_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}')

        # LR Scheduling
        if scheduler:
            scheduler.step(avg_val_loss)

        # Early Stopping & Saving Best Model
        if avg_val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.6f} --> {avg_val_loss:.6f}). Saving model...')
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve for {epochs_no_improve} epoch(s).')
            if args.use_early_stopping and epochs_no_improve >= args.early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs.')
                break

    print("Training finished.")

    # --- Final Model Evaluation on Test Set ---
    print("Loading best model for final evaluation on test set...")
    if not os.path.exists(best_model_path):
         print("Warning: Best model file not found. Evaluating with the last state.")
         # If no best model was saved (e.g., training stopped early or never improved)
         # evaluate with the current model state. Consider saving last state as fallback?
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    model.eval()
    test_preds_scaled_list = []
    test_trues_scaled_list = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            out_scaled = model(Xb).cpu().numpy()
            test_preds_scaled_list.append(out_scaled)
            test_trues_scaled_list.append(yb.numpy())

    preds_scaled = np.concatenate(test_preds_scaled_list)
    trues_scaled = np.concatenate(test_trues_scaled_list)

    # Inverse transform
    preds = targ_scaler.inverse_transform(preds_scaled).flatten()
    trues = targ_scaler.inverse_transform(trues_scaled).flatten()

    # Calculate final metrics on test set
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    print(f'Final Test MSE: {mse:.4f}, MAE: {mae:.4f}')

    # --- Saving Artifacts ---
    # Best model already saved during training
    print(f'Best model state dict saved to {best_model_path}')

    # Save scalers (trained on full train+val data)
    feature_scaler_path = os.path.join(args.output_dir, args.feature_scaler_filename)
    target_scaler_path = os.path.join(args.output_dir, args.target_scaler_filename)
    joblib.dump(feat_scaler, feature_scaler_path)
    joblib.dump(targ_scaler, target_scaler_path)
    print(f'Saved feature scaler to {feature_scaler_path}')
    print(f'Saved target scaler to {target_scaler_path}')

    # Save test set evaluation results
    results_df = pd.DataFrame({'y_true': trues, 'y_pred': preds})
    eval_path = os.path.join(args.results_dir, 'test_set_eval.csv')
    results_df.to_csv(eval_path, index=False)
    print(f'Saved test set evaluation results to {eval_path}')

    # Load best model state
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    # Print the learned weight
    final_alpha = model.alpha.item()
    final_weight = torch.sigmoid(model.alpha).item()
    print(f"Final learned weight parameter (alpha): {final_alpha:.4f}")
    print(f"Final LSTM weight (sigmoid(alpha)): {final_weight:.4f}")
    print(f"Final TCN weight (1 - sigmoid(alpha)): {1 - final_weight:.4f}")

    # --- Final Evaluation on Test Set ---
    print("Evaluating on test set using the best model...")

if __name__ == '__main__':
    main() 