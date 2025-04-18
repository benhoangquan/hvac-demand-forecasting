#!/usr/bin/env python3
"""
utils.py

Utility functions shared across the project.
"""
import pandas as pd
import numpy as np
import ast # Added for safe_literal_eval

def add_time_features(df):
    """Adds time-based features to the DataFrame."""
    # Ensure 'date' column is datetime type
    if 'date' not in df.columns:
        raise ValueError("'date' column is required for time features but not found.")
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

# Add other utility functions here as needed, e.g., safe_literal_eval from predict_hvac.py

def safe_literal_eval(val):
    """Safely evaluate a string containing a Python literal (list, tuple, number, bool, string, etc.)."
    """
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