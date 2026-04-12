#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_parquet('./Final_Project/Data/transitions_proxy.parquet')
print('Shape:', df.shape)
print('\nColumns:', df.columns.tolist())
print('\n--- queue_size stats ---')
print(df['queue_size'].describe())
print('\n--- executed_volume_proxy stats ---')
print(df['executed_volume_proxy'].describe())
print('\n--- action value counts ---')
print(df['action'].value_counts())
print('\n--- queue_size when action=1 ---')
print(df[df['action']==1]['queue_size'].describe())
print('\n--- executed_volume_proxy when action=1 ---')
print(df[df['action']==1]['executed_volume_proxy'].describe())
print('\n--- First 10 rows ---')
print(df[['action', 'queue_size', 'executed_volume_proxy', 'gas_t']].head(10))
print('\n--- Sample rows where action=1 ---')
print(df[df['action']==1][['action', 'queue_size', 'executed_volume_proxy', 'gas_t']].head(20))
