import pandas as pd
df = pd.read_parquet('../Data/transitions_hardened_v2.parquet')
print("Total rows:", len(df))
print("Mean Action:", df['action'].mean())
corr_action_queue = df['action'].corr(df['queue_size'])
corr_action_time = df['action'].corr(df['time_to_deadline'])
print("Corr Action-Queue:", corr_action_queue)
print("Corr Action-Time:", corr_action_time)
missed_target = df[(df['time_to_deadline'] <= 0.1) & (df['queue_size'] > 0)]
print("Missed Target Count:", len(missed_target))
print("Rewards Min:", df['reward'].min())
