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
| Cặp biến            | Pearson Correlation | Ý nghĩa                                           |
| ------------------- | :-----------------: | ------------------------------------------------- |
| Action ↔ Queue Size |       +0.0067       | Không tồn tại quan hệ tuyến tính                  |
| Action ↔ Time Left  |       −0.0002       | Agent "mù thời gian"                              |
| Action ↔ Gas Price  |       +0.8900       | Chỉ phản ánh network load, KHÔNG phải chiến thuật |

**Hậu quả cho IQL:** Thuật toán cố gắng tìm hàm $Q(s,a)$ nhưng không có pattern nào trong data thể hiện rằng "khi queue cao → nên xả nhiều" hay "khi sắp hết giờ → nên xả gấp". IQL nhận tín hiệu ngẫu nhiên thuần túy, và sụp đổ về action hằng số.

---

### Lỗi #2: Deadline Penalty Quá Yếu — B bị Nuốt Chửng Bởi Chi Phí Gas

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

| Cột            | Ý nghĩa                                                 | Vai trò            |
| -------------- | ------------------------------------------------------- | ------------------ |
| `s_gas_t0`     | Gas price hiện tại                                      | Market signal      |
| `s_gas_t1`     | Gas price t-1                                           | History            |
| `s_gas_t2`     | Gas price t-2                                           | History            |
| `s_congestion` | Block congestion $(g_{used} - g_{target}) / g_{target}$ | Network load       |
| `s_momentum`   | Log-return gas $\Delta \ln(g_t)$                        | Trend              |
| `s_accel`      | Gia tốc gas $\Delta^2 \ln(g_t)$                         | Trend change       |
| `s_surprise`   | Z-score tx count                                        | Mempool anomaly    |
| `s_backlog`    | EWMA backlog pressure                                   | Cumulative stress  |
| `s_queue`      | Pending transactions                                    | **Agent's debt**   |
| `s_time_left`  | Hours to deadline                                       | **Urgency signal** |
| `s_gas_ref`    | Rolling mean gas (128 blocks)                           | Reference price    |

**Kết quả:** RAM giảm 1.7GB → 200MB. Thời gian lưu Parquet: phút → giây.

---

## 🚀 Cập nhật Phiên bản V25/V26 (Bản Master)
*Ngày ghi nhận: 17/04/2026*

### Bài học #5: Sức mạnh của sự kiên nhẫn (Scale 0.1 / Beta 0.001)
Trong quá trình test độ nhạy, chúng ta tìm thấy cấu hình "Smart++" tạo ra sự đa dạng hành động lớn nhất:
- Dù áp lực hàng đợi thấp (`Scale=0.1`), nhưng mức phạt chờ siêu nhỏ (`Beta=0.001`) giúp Oracle dám ngồi đợi giá gas đáy thay vì xả ngay lập tức.
- Kết quả: Đạt **28% Chờ (Action 0)**, giúp Agent học được cả kỹ năng "quan sát vùng đáy".

### Bài học #6: Lỗi "Điểm mù Deadline" (The Triple-Sync Principle)
Chúng ta phát hiện Oracle để sót hàng (Miss Rate 1.23%) do sự lệch pha logic:
- **Nguyên nhân**: Oracle dùng Linear Penalty (`500 * queue`), nhưng Training & Env dùng Flat Penalty (`500`). 
- **Giải pháp (V26)**: Đồng bộ hóa 100% logic phạt Deadline và chi phí chờ cho Oracle khớp với môi trường huấn luyện. 

---
*Ghi chú: Luôn Audit lại Action Distribution và Solvability Rate (Miss Rate) sau mỗi bản build để đảm bảo Agent không học từ một "ông thầy" sai lầm.*

### Bài học #7 (V27 Lịch sử): Trò lừa kinh tế của Flat Penalty & Sự lên ngôi của Linear Penalty
- **Hiện tượng**: Ngay cả khi sức xả (capacity) dư sức đáp ứng, Oracle ở V26 vẫn quyết định "ôm" 18,000 txs và chịu trễ hạn.
- **Nguyên lý Toán Kinh Tế**: Vì V26 dùng Flat Penalty (-500 điểm cố định), Oracle tính toán thấy việc *trả phí Gas đắt đỏ* để xả 18,000 txs tốn kém gấp hàng ngàn lần so với việc *đóng 500 điểm bồi thường hợp đồng*. Quyết định "bỏ cuộc" của nó không phải vì yếu, mà vì quá khôn (chọn con đường phạt rẻ nhất)!
- **Giải pháp**: Ở bản V27, chuyển sang **Linear Penalty** ($500 \times Queue$). Phạt tỉ lệ thuận với lượng dư, dồn Agent vào bước đường cùng: "Nợ bao nhiêu đền bấy nhiêu". Kết quả: Oracle ngoan ngoãn xả sạch 100% không dám bùng kèo.

### Bài học #8: Vá lỗi nảy số (Quantization Error) tại Biên Số 0
- Khi giải bài toán DP trên không gian liên tục được chia lưới (Discretized Bins), hàng đợi nhỏ hơn `0.5 * Q_step` sẽ bị làm tròn thành bến 0 ($q\_idx = 0$). Oracle tưởng nhầm là bồn rỗng nên chọn Action 0 (không làm gì), dẫn tới bị lọt kẽ vài đơn hàng.
- **Cách vá**: Gài logic `if current_Q > 0.0 and q_idx == 0: q_idx = 1`. Ép những lượng dư siêu nhỏ vào Bin 1 thay vì Bin 0 để Oracle nhận diện được rác thải và bơm xả hoàn toàn. Từ đó chốt hạ Miss Rate 0.00%.

### Bài học #9: Phân phối lệnh tối ưu là "Bang-Bang Control"
- Nỗi sợ ban đầu: Tại sao Oracle toàn chọn lệnh 0 (15%) và lệnh 4 (35%) mà không phân bổ đều? 
- Bệ phóng lý thuyết: Vì chi phí phạt tuyến tính đụng độ với sức xả vật lý, lý thuyết Điều khiển Tối ưu chỉ định **Bang-Bang Control** là nghiệm đỉnh cao:
  - Giá Gas cực đắt (Đu Đỉnh) $\rightarrow$ Ép công suất Action 0 (Đóng băng).
  - Giá Gas chạm đáy (Local Min) $\rightarrow$ Nhấp nhả lút ga Action 4 (All-in xả kho).
  Đây không phải là tự hủy, mà là nghệ thuật chớp thời cơ và cắt lỗ của một Siêu Trí tuệ tài chính.

---

## 🚀 Kỷ Nguyên Hồi Sinh: Khai Nhãn & Điều Khiển Transformer (V28+)
*Ngày ghi nhận: 18/04/2026*

### Bài học #10: Lỗi "Mù" Dữ Liệu do Double Normalization
> [!CAUTION]
> Một lỗi hệ thống tinh vi khiến AI nhìn thấy thế giới chỉ toàn những con số 0 vô nghĩa.

- **Hiện tượng**: Dù dữ liệu đã chuẩn hóa [0, 1], nhưng bộ `ObservationScaler` trong training lại thực hiện chuẩn hóa thêm một lần nữa. Kết quả là mọi biến động tỷ Gwei của thị trường đều bị nén về mức 0.00000001.
- **Giải pháp - Inverse Normalization (Khai nhãn)**: 
  - Khôi phục giá trị vật lý thô (Hàng tỷ Gwei) ngay khi nạp Parquet.
  - Cho phép AI "nhìn" thấy sự biến động thực tế để ra quyết định chính xác.
- **Kết quả**: Đưa Policy từ sụp đổ hoàn toàn (Miss Rate 100%) về mức tiệm cận chuyên gia (Miss Rate 5.5%).

### Bài học #11: `--target` - "Nút vặn" tham vọng của Decision Transformer
> [!IMPORTANT]
> Khác với BCQ, Decision Transformer học theo mục tiêu (Conditioning on Return).

- **Phát hiện**: Nếu đặt Target quá thấp (số âm lớn), AI sẽ lười biếng và chấp nhận trễ hạn. Nếu đặt Target cao (số dương lớn), AI sẽ trở nên "tham vọng" và bắt chước quỹ đạo của Oracle.
- **Cơ chế**: AI không học "Cái gì là tốt nhất", mà học "Làm thế nào để đạt được mức điểm X". Tham số `--target` chính là cách chúng ta điều khiển "IQ" của AI mà không cần huấn luyện lại.

### Bài học #12: Sự thức tỉnh về Độ sâu và Trí nhớ (SOTA Architecture)
- **Thực tế**: Một mô hình nông (3 layers) và trí nhớ ngắn (20 context) chỉ giúp AI đạt được trình độ "Chuyên gia tập sự" (Vẫn còn 40% trễ hạn).
- **Chìa khóa SOTA**: Để đạt mức Miss Rate 0%, bắt buộc phải sử dụng kiến trúc SOTA:
  - **10-12 Layers**: Tăng khả năng xử lý các quy luật giá phức tạp.
  - **Context 64**: Trí nhớ đủ dài để nhận diện xu hướng Gas của cả 1 giờ lịch sử.

---
*Ghi chú: Thất bại trong thực nghiệm là nền tảng cho sự đột phá trong lý thuyết. Hiểu được tại sao AI sai mới là đỉnh cao của nghiên cứu RL.*
