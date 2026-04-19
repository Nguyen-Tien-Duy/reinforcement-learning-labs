import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import d3rlpy
from d3rlpy.algos import DiscreteDecisionTransformerConfig
from d3rlpy.preprocessing import MinMaxObservationScaler, StandardRewardScaler

STATE_COLS = [
    "s_gas_t0", "s_gas_t1", "s_gas_t2",       
    "s_congestion", "s_momentum", "s_accel",   
    "s_surprise", "s_backlog",               
    "s_queue", "s_time_left", "s_gas_ref"
]

def _to_bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype(np.int64).isin([1]).to_numpy(dtype=bool)

def build_d3rlpy_dataset(df: pd.DataFrame, *, mode="strict"):
    from d3rlpy import ActionSpace
    from d3rlpy.dataset import MDPDataset

    observations = df[STATE_COLS].to_numpy(dtype=np.float32)
    rewards = pd.to_numeric(df["reward"], errors="coerce").to_numpy(dtype=np.float32)
    
    terminals = _to_bool_array(df["done"])
    if "truncated" in df.columns:
        timeouts = _to_bool_array(df["truncated"])
    else:
        timeouts = np.zeros(len(df), dtype=bool)
    
    terminals = terminals & ~timeouts
    actions = df["action"].to_numpy(dtype=np.int64)

    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        action_space=ActionSpace.DISCRETE
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent

def load_normalization_params(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    params_path = data_dir / "state_norm_params.json"
    with open(params_path, "r") as f:
        params = json.load(f)
    return np.array(params["mins"], dtype=np.float32), np.array(params["maxs"], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description=" Train Discrete Decision Transformer on Cloud GPU")
    parser.add_argument("--input", type=Path, required=True, help="Đường dẫn tới file parquet V27 (vd: transitions_discrete_v27.parquet)")
    parser.add_argument("--outdir", type=Path, default=Path("dt_output"), help="Thư mục xuất log và model")
    # L4 config: 100,000 steps * 256 batch_size = 25.6 triệu chuỗi được xem. 
    # Tổng data là 4.5 triệu -> AI sẽ coi qua toàn bộ dữ liệu khoảng 5.6 LẦN (Đúng chuẩn 5 passes)
    parser.add_argument("--n-steps", type=int, default=100000, help="Tổng số bước huấn luyện")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (TỐI ƯU CHO L4 24GB)")
    parser.add_argument("--context-size", type=int, default=20, help="Độ dài chuỗi ngữ cảnh cho Transformer (K)")
    parser.add_argument("--oracle-only", action="store_true", help="Chỉ dạy cho Agent bằng dữ liệu chuẩn của Oracle")
    parser.add_argument("--debug", action="store_true", help="Chế độ Debug: Chạy Toy model cực nhẹ để test lỗi")
    args = parser.parse_args()

    logging.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    
    # === INVERSE NORMALIZATION (Khai nhãn Transformer) ===
    # Phục hồi dữ liệu vật lý (RAW) để Attention Mechanism hoạt động được.
    try:
        current_mins, current_maxs = load_normalization_params(args.input.parent)
        logging.info(f"[*] Đang khôi phục vật lý cho {len(STATE_COLS)} đặc trưng trạng thái (Cloud Mode)...")
        for i, col in enumerate(STATE_COLS):
            if col in df.columns:
                # Safe Guard: Chỉ khôi phục nếu dữ liệu thực sự đang bị nén [0, 1]
                if df[col].max() <= 1.01 and df[col].min() >= -1.01:
                    df[col] = df[col] * (current_maxs[i] - current_mins[i]) + current_mins[i]
                else:
                    logging.info(f"  > Bỏ qua {col}: Đã là giá trị vật lý (Max={df[col].max():.2f})")
        logging.info("✅ Khải thuật vật lý hoàn tất. H100 đã sẵn sàng với RAW DATA.")
    except Exception as e:
        logging.warning(f"⚠️ Cảnh báo Inverse Normalization thất bại: {e}")
        logging.info("Filtering for oracle-only transitions...")
        df = df[df["policy_type"] == 1].copy()

    # Log-Transformation cho Reward (Rất quan trọng với DT để không bị nổ attention)
    logging.info("Applying Log-Transformation to Rewards...")
    raw_rewards = df["reward"].to_numpy()
    log_rewards = np.sign(raw_rewards) * np.log1p(np.abs(raw_rewards))
    df["reward"] = log_rewards

    # Tiền xử lý Scaling Parameters
    median = float(np.median(log_rewards))
    q75, q25 = np.percentile(log_rewards, [75, 25])
    iqr = float(q75 - q25) if float(q75 - q25) > 0 else 1.0

    logging.info("Force action as int32...")
    df['action'] = df['action'].astype(np.int32)
    import gc
    
    # 4) Chia tập Train/Eval THEO THỜI GIAN (Quy tắc thép: Test trên Quá Khứ, Thi Tương Lai)
    unique_episodes = sorted(df["episode_id"].unique())
    # Tuyệt đối KHÔNG SHUFFLE để đảm bảo đồng nhất Cloud và Local
    split_idx = int(len(unique_episodes) * 0.9)
    train_ids = unique_episodes[:split_idx]
    eval_ids = unique_episodes[split_idx:]
    
    train_df = df[df["episode_id"].isin(train_ids)].copy()
    eval_df = df[df["episode_id"].isin(eval_ids)].copy()
    
    # [ANTI-OOM] Dọn rác dataframe GỐC ngay lập tức để cứu RAM
    del df
    gc.collect()
    logging.info("♻️ Đã dọn rác DataFrame gốc để giải phóng RAM.")
    
    logging.info(f"Multi-level split: {len(train_ids)} train vs {len(eval_ids)} eval episodes.")

    train_dataset = build_d3rlpy_dataset(train_df, mode="strict")
    # [ANTI-OOM] Xóa tiếp train_df sau khi convert
    del train_df
    gc.collect()

    eval_dataset = build_d3rlpy_dataset(eval_df, mode="strict")
    # [ANTI-OOM] Xóa nốt eval_df
    del eval_df
    gc.collect()
    logging.info("♻️ Đã dọn rác hoàn toàn. Bộ nhớ đang ở trạng thái tối ưu nhất!")
    
    # Load Strict State Normalization
    mins, maxs = load_normalization_params(args.input.parent)
    obs_scaler = MinMaxObservationScaler(minimum=mins, maximum=maxs)
    rew_scaler = StandardRewardScaler(mean=median, std=iqr)

    # -------------------------------------------------------------
    # 🧠 CẤU HÌNH DECISION TRANSFORMER - CHẾ ĐỘ LINH HOẠT
    # -------------------------------------------------------------
    if args.debug:
        logging.info("🛠️ [DEBUG MODE] Đang dùng cấu hình TOY SIÊU NHẸ để test lỗi...")
        config = DiscreteDecisionTransformerConfig(
            batch_size=32,
            learning_rate=1e-4,
            context_size=5,
            max_timestep=10000,  # [BẮT BUỘC] Giống SOTA để không bị out of index
            num_heads=2,
            num_layers=2,
            observation_scaler=obs_scaler,
            reward_scaler=rew_scaler,
            compile_graph=False, # Debug không cần JIT
        )
        args.n_steps = 200
        n_steps_per_epoch = 100
    else:
        logging.info("🚀 [SOTA MODE] Đang dùng cấu hình KHỔNG LỒ cho H100...")
        config = DiscreteDecisionTransformerConfig(
            batch_size=args.batch_size,
            learning_rate=1e-4,
            context_size=args.context_size,
            max_timestep=10000,
            num_heads=8,
            num_layers=10,
            warmup_tokens=102400,
            final_tokens=6500000000,
            observation_scaler=obs_scaler,
            reward_scaler=rew_scaler,
            compile_graph=True,
        )
        n_steps_per_epoch = 5000

    # Nếu có GPU, ép chạy trên GPU (Dùng PyTorch gốc cho tương thích d3rlpy v2)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training on Device: {device}")
    
    algo = config.create(device=device)
    algo.build_with_dataset(train_dataset)

    args.outdir.mkdir(parents=True, exist_ok=True)
    run_name = f"DT_V28_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    from d3rlpy.metrics import DiscreteActionMatchEvaluator
    
    # [ĐỘ LẠI - V28] Callback chẩn đoán 'Sáng bừng trí tuệ' cho Decision Transformer
    class DTDiagnosticCallback:
        def __init__(self, train_dataset, eval_dataset, interval=5000, context_size=20):
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.interval = interval
            self.context_size = context_size

        def __call__(self, algo, epoch, total_step):
            if total_step % self.interval == 0 and total_step > 0:
                import random
                import numpy as np
                import torch
                
                print(f"\n[Step {total_step}] --- 🧠 DT BRAIN SCAN REPORT ---")
                
                # Đo khả năng dự đoán trên tập EVAL (Dùng thực tế 20 bước lịch sử)
                match_count = 0
                test_samples = 50
                
                # Pick ngẫu nhiên 50 tình huống để kiểm tra "IQ"
                for _ in range(test_samples):
                    ep = random.choice(self.eval_dataset.episodes)
                    if len(ep.observations) <= self.context_size + 1: continue
                    
                    # Cắt một lát cắt lịch sử (Context)
                    idx = random.randint(self.context_size, len(ep.observations) - 2)
                    obs_slice = ep.observations[idx-self.context_size+1 : idx+1]
                    act_slice = ep.actions[idx-self.context_size+1 : idx+1]
                    rew_slice = ep.rewards[idx-self.context_size+1 : idx+1]
                    
                    # Bắt AI đoán action tiếp theo dựa trên lịch sử này
                    # (Dùng API nội bộ của d3rlpy để dự đoán chuỗi)
                    try:
                        # Convert to tensors
                        in_obs = torch.tensor(obs_slice, dtype=torch.float32).unsqueeze(0).to(algo.device)
                        in_act = torch.tensor(act_slice, dtype=torch.long).unsqueeze(0).to(algo.device)
                        in_rew = torch.tensor(rew_slice, dtype=torch.float32).unsqueeze(0).to(algo.device)
                        in_ret = torch.ones((1, self.context_size, 1)).to(algo.device) * 1.0 # Pseudo return
                        in_tim = torch.arange(idx-self.context_size+1, idx+1).unsqueeze(0).to(algo.device)
                        
                        pred_actions = algo.impl._predict_best_action(in_obs, in_act, in_rew, in_ret, in_tim)
                        dt_action = int(pred_actions[0][-1].item())
                        oracle_action = int(ep.actions[idx])
                        
                        if dt_action == oracle_action:
                            match_count += 1
                    except:
                        continue
                
                acc = (match_count / test_samples) * 100
                print(f"  > Contextual Match Rate (IQ Test): {acc:.2f}%")
                
                # In thử 1 mẫu cho bác soi
                print(f"  > Sample Detail: AI Predicted={dt_action} | Oracle={oracle_action} " + ("✅" if dt_action == oracle_action else "❌"))
                print("--------------------------------------------------\n", flush=True)

    dt_callback = DTDiagnosticCallback(train_dataset, eval_dataset, interval=5000, context_size=args.context_size)

    logging.info(f"🚀 Bắt đầu HUẤN LUYỆN L4 CLOUD ({args.n_steps} steps - ~5.6 Passes qua dữ liệu)...")
    algo.fit(
        train_dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        show_progress=True,
        experiment_name=run_name,
        callback=dt_callback
    )
    
    logging.info(" Huấn luyện Decision Transformer hoàn tất!")

if __name__ == "__main__":
    main()
