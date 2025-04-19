#!/usr/bin/env python3
"""
split_dataset.py

Split the original HVAC dataset into 80% train and 20% test (time-based) and save to CSV files.
"""
import os
import pandas as pd

def main():
    # Load and sort by date
    input_csv = os.path.join('dataset', 'train_dataset.csv')
    df = pd.read_csv(input_csv, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # Compute split index (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    # Output files
    train_out = os.path.join('dataset', 'train_split.csv')
    test_out = os.path.join('dataset', 'test_split.csv')
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f'Saved training split ({len(train_df)} rows) to {train_out}')
    print(f'Saved test split ({len(test_df)} rows) to {test_out}')

if __name__ == '__main__':
    main()