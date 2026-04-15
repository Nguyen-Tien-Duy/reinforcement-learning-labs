# Tại Sao Dữ Liệu Huấn Luyện Khiến RL Không Học Được — và Cách Khắc Phục

## Bối cảnh

Agent IQL (Implicit Q-Learning) được huấn luyện Offline trên bộ dữ liệu `transitions_hardened_v2.parquet` (~5.2 triệu transitions) để tối ưu gas cho giao dịch Ethereum. Kết quả: **Policy Collapse** — Agent luôn chọn `action ≈ 0` (không xả giao dịch) bất kể trạng thái, dẫn đến hàng đợi ùn ứ và vi phạm deadline 100%.

---

## 3 Lỗi Gốc Rễ Trong Pipeline Dữ Liệu

### Lỗi #1: Cột Action Không Có Quan Hệ Nhân Quả Với State

> [!CAUTION]
> Đây là lỗi nghiêm trọng nhất — nó khiến toàn bộ quá trình học trở nên vô nghĩa.

**Hiện tượng:** Cột `action` trong dataset V2 được tính bằng công thức proxy:

```python
action = gas_used / gas_limit   # ← Tỷ lệ gas tiêu thụ của block
```

Đây là một đại lượng **quan sát vật lý** của mạng Ethereum, KHÔNG phải quyết định tối ưu của bất kỳ agent nào. Nó không phản ứng với:
- **Queue (hàng đợi):** Corr(Action, Queue) = **+0.0067** ≈ 0 (ngẫu nhiên hoàn toàn)
- **Time to Deadline:** Action trung bình = 0.38 bất kể còn 24h hay còn 1h

**Minh chứng toán học:**
| Cặp biến | Pearson Correlation | Ý nghĩa |
|----------|:-------------------:|---------|
| Action ↔ Queue Size | +0.0067 | Không tồn tại quan hệ tuyến tính |
| Action ↔ Time Left | −0.0002 | Agent "mù thời gian" |
| Action ↔ Gas Price | +0.8900 | Chỉ phản ánh network load, KHÔNG phải chiến thuật |

**Hậu quả cho IQL:** Thuật toán cố gắng tìm hàm $Q(s,a)$ nhưng không có pattern nào trong data thể hiện rằng "khi queue cao → nên xả nhiều" hay "khi sắp hết giờ → nên xả gấp". IQL nhận tín hiệu ngẫu nhiên thuần túy, và sụp đổ về action hằng số.

---

### Lỗi #2: Deadline Penalty Quá Yếu — Bị Nuốt Chửng Bởi Chi Phí Gas

> [!WARNING]
> Kinh tế học phần thưởng bị lệch khiến Agent thà "ngồi im chịu phạt" còn hơn trả phí Gas.

**Cấu hình cũ:**
```python
deadline_penalty = 2,000,000.0   # 2 triệu
reward_scale     = 1,000,000.0   # Chia cho 1 triệu
# → Penalty thực tế = -2.0 điểm
```

**So sánh với chi phí xả hàng:**
```
Chi phí xả 500 tx ở gas 100 Gwei:
  = 500 × 15000 × 100 / 1,000,000
  = 750.0 điểm (chi phí)

Penalty vỡ deadline: -2.0 điểm
```

> **Tỷ lệ: Chi phí Gas / Penalty = 375x.** Agent chọn phương án kinh tế hơn: không làm gì cả, ăn phạt 2 điểm thay vì trả 750 điểm gas.

---

### Lỗi #3: Bug Nhân Quả Trong Pipeline — State↔Action Mất Đồng Bộ

> [!IMPORTANT]
> Ngay cả khi bật Oracle, pipeline cũ vẫn tạo ra dữ liệu rác.

**Luồng cũ (bị lỗi):**
```
1. build_state_action()  →  Tính queue_size từ action GỐC (proxy)
2. apply_oracle()        →  Ghi đè action bằng DP tối ưu
3. build_reward()        →  Tính reward + next_state
```

Vấn đề: Sau bước 2, `action` đã thay đổi nhưng `queue_size` trong state vector (index 8) vẫn phản ánh action cũ. Kết quả:
- State nói "queue = 100" (tính từ proxy action cũ)
- Action = 0.8 (từ Oracle mới)
- Nhưng thực tế nếu action = 0.8 thì queue phải giảm mạnh ở bước tiếp theo

→ **Agent nhận tín hiệu mâu thuẫn:** "Queue cao nhưng action cao vẫn không giảm queue?" → Học được bài học sai.

---

## Giải Pháp Đã Triển Khai (V4 Pipeline)

### Fix #1: Thay Proxy Bằng Hindsight Oracle (DP)

Thay vì dùng `gas_used/gas_limit`, ta chạy **Quy Hoạch Động ngược (Backward Induction)** với tầm nhìn toàn tri (God-View) để tìm action tối ưu cho từng episode:

```python
# oracle_builder.py — Numba JIT-compiled DP
for t in range(T-2, -1, -1):     # Backward sweep
    for Q in range(Q_max + 1):    # Mỗi mức queue
        # Tìm n* = argmin(exec_cost + urgency + V[t+1])
        ...
```

Oracle trộn vào dataset theo tỷ lệ:
- **50%** Oracle actions (tối ưu)
- **20%** Suboptimal actions (1 - optimal, để IQL học tránh)
- **30%** Behavior gốc (đa dạng hóa)

### Fix #2: Tăng Deadline Penalty Lên 5 Tỷ

```python
# config.py
deadline_penalty = 5,000,000,000.0   # 5 tỷ (was 2 triệu)
# → Penalty thực tế = -5000.0 điểm
```

**Tỷ lệ mới: Chi phí Gas / Penalty = 0.15x.** Penalty áp đảo hoàn toàn — Agent BẮT BUỘC phải xả hàng trước deadline hoặc chịu "cái chết kinh tế".

### Fix #3: Đồng Bộ Nhân Quả State↔Action

Chèn bước `recalculate_queue_and_state()` SAU Oracle, TRƯỚC Reward:

```python
# transition_builder.py — Luồng mới
1. build_state_action()        →  state + action gốc
2. apply_oracle()              →  ghi đè action
3. recalculate_queue_and_state()  →  ★ TÁI TÍNH queue từ action MỚI
4. build_reward()              →  reward + next_state nhất quán
```

### Fix #4: Băm Phẳng Cấu Trúc State (Hiệu Năng)

**Vấn đề:** Cột `state` chứa Python List (5.2 triệu objects) → PyArrow serialize đơn luồng → RAM 1.7GB rác → Swap → Đóng băng máy.

**Giải pháp:** Tách thành 11 cột Float32 có tên ngữ nghĩa:

| Cột | Ý nghĩa | Vai trò |
|-----|---------|---------|
| `s_gas_t0` | Gas price hiện tại | Market signal |
| `s_gas_t1` | Gas price t-1 | History |
| `s_gas_t2` | Gas price t-2 | History |
| `s_congestion` | Block congestion $(g_{used} - g_{target}) / g_{target}$ | Network load |
| `s_momentum` | Log-return gas $\Delta \ln(g_t)$ | Trend |
| `s_accel` | Gia tốc gas $\Delta^2 \ln(g_t)$ | Trend change |
| `s_surprise` | Z-score tx count | Mempool anomaly |
| `s_backlog` | EWMA backlog pressure | Cumulative stress |
| `s_queue` | Pending transactions | **Agent's debt** |
| `s_time_left` | Hours to deadline | **Urgency signal** |
| `s_gas_ref` | Rolling mean gas (128 blocks) | Reference price |

**Kết quả:** RAM giảm 1.7GB → 200MB. Thời gian lưu Parquet: phút → giây.

---

## Trạng Thái Hiện Tại

| Hạng mục | Trạng thái |
|----------|-----------|
| Pipeline V4 (named columns) | 🔄 Đang build (`build_v4_named.log`) |
| Oracle DP solver | ✅ Đã fix boundary condition |
| Deadline Penalty 5B | ✅ Đã áp dụng |
| Causal recalculation | ✅ `recalculate_queue_and_state()` đã chèn |
| State schema flattening | ✅ 11 cột ngữ nghĩa `s_*` / `ns_*` |
| Validation suite | ✅ 6 file test đã cập nhật schema mới |

## Bước Tiếp Theo

1. **Chờ build V4 hoàn tất** → Chạy `validate_v3_data.py` + `logic_integrity_suite.py`
2. **Kiểm tra Causal Signal:** Corr(Action, s_queue) phải > 0.001
3. **Train IQL** trên dataset V4 với tham số từ `SIMULATED_FEE_USAGE.md`
4. **Đánh giá:** WIS, Doubly Robust, CVaR theo evaluation suite
