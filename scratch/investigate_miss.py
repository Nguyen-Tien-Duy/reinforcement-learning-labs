
import pandas as pd
import sys

def investigate_episode(file_path, ep_id, n_steps=20):
    df = pd.read_parquet(file_path)
    ep_df = df[df['episode_id'] == ep_id].sort_values('step_index')
    
    if ep_df.empty:
        print(f"Không tìm thấy Episode {ep_id}")
        return
        
    print(f"\n=== ĐIỀU TRA EPISODE {ep_id} (20 Bước cuối cùng) ===")
    cols = ['step_index', 'gas_t', 'queue_size', 'action', 'reward', 'done']
    # Show last n_steps
    last_n = ep_df.tail(n_steps)[cols]
    
    # Add a column to show physical execution result
    action_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    exec_cap = 500.0
    last_n['ratio'] = last_n['action'].map(lambda a: action_bins[int(a)])
    last_n['executed'] = last_n.apply(lambda r: min(int(r['ratio'] * r['queue_size']), exec_cap), axis=1)
    last_n['remaining'] = last_n['queue_size'] - last_n['executed']
    
    print(last_n.to_string(index=False))

if __name__ == "__main__":
    path = sys.argv[1]
    ep_id = int(sys.argv[2])
    investigate_episode(path, ep_id)
