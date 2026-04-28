from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import d3rlpy
import torch
import uvicorn
from typing import List
import os
import numpy as np

app = FastAPI(title="Gas Optimization AI Sidecar")

# 1. Nạp Model (Load một lần duy nhất khi khởi động)
MODEL_PATH = "d3rlpy_logs/DiscreteCQL_V6_20260420_2345_20260420234508/model_120000.d3"

if not os.path.exists(MODEL_PATH):
    print(f"KHÔNG TÌM THẤY MODEL TẠI: {MODEL_PATH}")
    # Có thể nạp model khác dự phòng hoặc báo lỗi
else:
    print(f"[*] Đang nạp AI Model từ: {MODEL_PATH}")
    algo = d3rlpy.load_learnable(MODEL_PATH)

class PredictionRequest(BaseModel):
    state: List[float]

@app.get("/health")
def health():
    return {"status": "ready", "model": MODEL_PATH}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Chuyển đổi list sang numpy array để d3rlpy xử lý
        state_np = np.array([request.state], dtype=np.float32)
        action = algo.predict(state_np)[0]
        return {"action": int(action)}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Chạy trên port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
