import pandas as pd
import numpy as np

# Try to load transition and raw data
try:
    df = pd.read_parquet('../Data/data_2024-04-10_2026-04-10.parquet')
    print(f"Loaded parquet with shape {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Is it transition or raw?
    if 'reward' in df.columns and 'time_to_deadline' in df.columns:
        print("Data is a TRANSITION dataset.")
        
        # Check deadline fear
        # How many episodes actually experienced a deadline miss penalty?
        # A miss usually has time_to_deadline == 0 and queue_size > 0
        missed = df[(df['time_to_deadline'] == 0) & (df['queue_size'] > 0)]
        print(f"\nEpisodes ending in deadline miss: {len(missed)}")
        if len(missed) > 0 and 'reward' in missed.columns:
            print(f"Average reward on miss: {missed['reward'].mean():.2f}")
            
        # Check action behavior near deadline
        near_deadline = df[df['time_to_deadline'] <= 1]
        far_deadline = df[df['time_to_deadline'] > 12]
        
        print("\n=== ACTION ANALYSIS ===")
        print(f"Avg Action when FAR from deadline (> 12h): {far_deadline['action'].mean():.4f}")
        print(f"Avg Action when NEAR deadline (<= 1h): {near_deadline['action'].mean():.4f}")
        
        print("\n=== REWARD SKEWNESS ===")
        print(f"Min reward: {df['reward'].min():.2f}")
        print(f"Max reward: {df['reward'].max():.2f}")
        print(f"Median reward: {df['reward'].median():.2f}")
        
    else:
        print("Data is RAW dataset.")
        # If it's raw, show the action proxy logic
        if 'gas_used' in df.columns and 'gas_limit' in df.columns:
            action_proxy = df['gas_used'] / df['gas_limit']
            print("\nIf action = gas_used / gas_limit:")
            print(f"Min: {action_proxy.min():.4f}, Max: {action_proxy.max():.4f}, Mean: {action_proxy.mean():.4f}")
        
except Exception as e:
    print("Error:", e)
