# 📐 Chứng minh Toán học — Toàn bộ Giả định & Công thức trong Đồ án

## Mục lục
1. [MDP & Bellman Equation](#1)
2. [Q-Learning & Optimal Policy](#2)
3. [CQL Loss — Tại sao đè OOD xuống?](#3)
4. [Oracle DP — Backward Induction](#4)
5. [Reward Function — 3 thành phần](#5)
6. [Backlog Pressure — AR Model](#6)
7. [n_critics=1 — Double Pessimism](#7)
8. [Reward Shaping — Tại sao giảm penalty?](#8)
9. [Safety Layer 20% — Trade-off](#9)

---

<a id="1"></a>
## 1. MDP & Phương trình Bellman

### 1.1 Định nghĩa MDP

MDP = bộ 5 $(S, A, P, R, \gamma)$

**Giả định Markov:** Trạng thái tương lai chỉ phụ thuộc vào trạng thái hiện tại, không phụ thuộc quá khứ:

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

**Tại sao bài toán Gas thỏa mãn Markov?**
- State vector $s_t \in \mathbb{R}^{11}$ đã encode đủ thông tin: giá gas hiện tại + 2 bước trước ($g_t, g_{t-1}, g_{t-2}$), momentum ($m_t$), acceleration ($acc_t$), backlog pressure ($b_t$).
- Đặc biệt $b_t$ là biến AR tích lũy lịch sử tắc nghẽn → bù đắp cho việc không có LSTM.
- Queue $Q_t$ và deadline $\tau_t$ là đủ để xác định ràng buộc tương lai.

### 1.2 Mục tiêu tối ưu

Tìm policy $\pi^*$ tối đa hóa tổng reward chiết khấu:

$$J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

**Tại sao $\gamma = 0.99$?** Vì mỗi episode có ~7200 steps (blocks), effective horizon = $\frac{1}{1-\gamma} = 100$ steps. Nghĩa là agent "nhìn xa" khoảng 100 blocks (~20 phút) — đủ để capture xu hướng giá gas ngắn hạn.

### 1.3 Bellman Equation — Chứng minh

**Q-function:**
$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R(s_{t+k}, a_{t+k}) \mid s_t=s, a_t=a\right]$$

**Chứng minh tính đệ quy:**

$$Q^\pi(s, a) = \mathbb{E}\left[R(s,a) + \gamma \sum_{k=1}^{\infty} \gamma^{k-1} R(s_{t+k}, a_{t+k})\right]$$

Tách R(s,a) ra ngoài kỳ vọng (nó đã xác định khi biết s, a):

$$= R(s,a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)}\left[\mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R(s_{t+k+1}, a_{t+k+1}) \mid s_{t+1}=s'\right]\right]$$

Nhận ra phần trong ngoặc chính là $Q^\pi(s', \pi(s'))$:

$$\boxed{Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \cdot Q^\pi(s', \pi(s'))}$$

**Bellman Optimality:**

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \cdot \max_{a'} Q^*(s', a')$$

Policy tối ưu: $\pi^*(s) = \arg\max_a Q^*(s, a)$

---

<a id="2"></a>
## 2. Q-Learning — Tại sao hội tụ?

### 2.1 Update rule

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\right]$$

Phần trong ngoặc vuông gọi là **TD Error** $\delta_t$:

$$\delta_t = r + \gamma \max_{a'} Q(s', a') - Q(s,a)$$

### 2.2 Chứng minh hội tụ (tóm tắt)

**Định lý (Watkins & Dayan, 1992):** Q-Learning hội tụ đến $Q^*$ với xác suất 1 nếu:
1. Mỗi cặp $(s, a)$ được thăm vô hạn lần
2. Learning rate $\alpha_t$ thỏa: $\sum_t \alpha_t = \infty$ và $\sum_t \alpha_t^2 < \infty$

**Ý nghĩa cho bài toán:** Trong Offline RL, điều kiện 1 bị vi phạm — không phải mọi $(s, a)$ đều có trong dataset. Đây chính là lý do cần CQL.

---

<a id="3"></a>
## 3. CQL Loss — Chứng minh tại sao đè OOD xuống

### 3.1 Công thức CQL Loss

$$\mathcal{L}_{\text{CQL}}(\theta) = \underbrace{\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(Q_\theta(s,a) - y)^2\right]}_{\mathcal{L}_{\text{TD}}} + \alpha \underbrace{\mathbb{E}_{s \sim \mathcal{D}}\left[\log\sum_{a'} e^{Q_\theta(s,a')} - \mathbb{E}_{a \sim \pi_\beta}[Q_\theta(s,a)]\right]}_{\text{Conservative Penalty}}$$

Với $y = r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a')$ (target network).

### 3.2 Phân tích Conservative Penalty

Gọi penalty = $\mathcal{R}(\theta)$:

$$\mathcal{R}(\theta) = \mathbb{E}_s\left[\log\sum_{a'} e^{Q_\theta(s,a')} - \mathbb{E}_{a \sim \pi_\beta}[Q_\theta(s,a)]\right]$$

**Số hạng 1:** $\log\sum_{a'} e^{Q(s,a')}$ — đây là **LogSumExp**, xấp xỉ $\max_a Q(s,a)$ nhưng smooth hơn. Nó tác dụng như lực kéo **tất cả** Q-values xuống.

**Số hạng 2:** $\mathbb{E}_{a \sim \pi_\beta}[Q(s,a)]$ — trung bình Q-value của các actions **có trong dataset**. Bị trừ đi → tác dụng như lực đẩy Q-values của in-distribution actions **lên**.

**Hiệu quả ròng:**
- Actions **trong dataset** (in-distribution): bị kéo xuống bởi số hạng 1, nhưng được đẩy lên bởi số hạng 2 → **giữ nguyên** hoặc tăng nhẹ.
- Actions **ngoài dataset** (OOD): chỉ bị kéo xuống bởi số hạng 1, không có lực đẩy lên → **bị đè xuống mạnh**.

### 3.3 Chứng minh Q-value bị lower-bound

**Định lý (Kumar et al., 2020):** Với $\alpha$ đủ lớn, Q-function học được $\hat{Q}^\pi$ thỏa mãn:

$$\hat{Q}^\pi(s, a) \leq Q^\pi(s, a) \quad \forall (s, a)$$

Nghĩa là CQL luôn **đánh giá thấp hơn** giá trị thực → policy chọn action an toàn hơn, tránh chọn OOD actions bị overestimate.

### 3.4 Tại sao $\alpha = 1.0$?

- $\alpha$ quá nhỏ (0.1): Conservative penalty yếu → vẫn bị OOD overestimation.
- $\alpha$ quá lớn (10): Đè Q-value xuống quá mạnh → agent sợ hãi, không dám làm gì (Pessimistic Collapse).
- $\alpha = 1.0$: Cân bằng giữa Bellman accuracy và conservative safety. Giá trị mặc định được khuyến nghị trong paper gốc cho discrete action space.

---

<a id="4"></a>
## 4. Oracle DP — Backward Induction

### 4.1 Tại sao DP cho kết quả tối ưu toàn cục?

**Nguyên lý Tối ưu Bellman (Bellman's Principle of Optimality):**

> *"Một policy tối ưu có tính chất: bất kể trạng thái ban đầu và quyết định đầu tiên là gì, các quyết định còn lại phải tạo thành một policy tối ưu đối với trạng thái sinh ra từ quyết định đầu tiên."*

### 4.2 Backward Pass — Chứng minh tính đúng đắn

**Khởi tạo** (bước cuối $T$):
$$V[T, Q] = \begin{cases} 0 & \text{nếu } Q = 0 \text{ (hoàn tất)} \\ \text{penalty} & \text{nếu } Q > 0 \text{ (miss deadline)} \end{cases}$$

**Bước quy nạp** ($t = T-1, \ldots, 0$):
$$V[t, Q] = \min_{b \in \{0,1,2,3,4\}} \left\{c(t, Q, b) + V[t+1, f(Q, b, t)]\right\}$$

Với chi phí tức thời:
$$c(t, Q, b) = \frac{g_t}{\sigma}\left(C_{\text{base}} \cdot \mathbf{1}[n_b > 0] + C_{\text{mar}} \cdot n_b\right) + \frac{\beta}{\sigma} \cdot (Q - n_b) \cdot e^{\alpha(1-\tau_t)}$$

Và hàm chuyển trạng thái queue:
$$f(Q, b, t) = \max(0, Q - n_b) + w_{t+1}$$

**Chứng minh bằng quy nạp ngược:**

*Bước cơ sở:* Tại $t = T$, $V[T, Q]$ đúng vì không còn quyết định nào cần thực hiện.

*Giả sử quy nạp:* $V[t+1, Q]$ là chi phí tối thiểu từ bước $t+1$ đến $T$, với mọi $Q$.

*Bước quy nạp:* Tại bước $t$, agent chọn bin $b$ chịu chi phí $c(t, Q, b)$ và chuyển sang state $(t+1, f(Q,b,t))$. Tổng chi phí = $c(t,Q,b) + V[t+1, f(Q,b,t)]$. Chọn $b$ minimize tổng này → $V[t, Q]$ là chi phí tối thiểu từ bước $t$. QED.

### 4.3 Độ phức tạp

$$O(Q_{\max} \times T \times |\mathcal{A}|) = O(Q_{\max} \times 7200 \times 5)$$

Với $Q_{\max} \approx 100{,}000$: ~3.6 tỷ phép tính/episode → cần Numba JIT ($\approx 100\times$ speedup).

---

<a id="5"></a>
## 5. Reward Function — Phân tích từng thành phần

### 5.1 Công thức đầy đủ

$$r_t = \underbrace{-\frac{g_t}{\sigma}\left(C_{\text{base}} \cdot \mathbf{1}[n_t > 0] + C_{\text{mar}} \cdot n_t\right)}_{\text{(1) Gas Cost}} - \underbrace{\frac{\beta}{\sigma} \cdot Q_{\text{remain}} \cdot e^{\alpha(1-\tau_t)}}_{\text{(2) Urgency}} - \underbrace{\frac{\lambda}{\sigma} \cdot \mathbf{1}[Q_T > 0]}_{\text{(3) Deadline}}$$

### 5.2 Thành phần (1): Gas Cost

$$\text{Gas Cost} = -\frac{g_t}{\sigma}\left(21000 \cdot \mathbf{1}[n_t > 0] + 15000 \cdot n_t\right)$$

- $C_{\text{base}} = 21{,}000$ gas: Chi phí cố định cho 1 transaction trên Ethereum (theo EIP-2718).
- $C_{\text{mar}} = 15{,}000$ gas: Chi phí biên cho mỗi giao dịch thêm (trung bình ERC-20 transfer).
- $\sigma = 10^9$: Scale factor (1 Gwei = $10^9$ wei) để reward không quá lớn.
- Dấu âm: vì đây là **chi phí** — reward càng âm = càng tốn kém.

### 5.3 Thành phần (2): Urgency Penalty

$$\text{Urgency} = -\frac{\beta}{\sigma} \cdot Q_{\text{remain}} \cdot e^{\alpha(1-\tau_t)}$$

**Tại sao hàm mũ?**

| $\tau_t$ (thời gian còn lại) | $e^{\alpha(1-\tau_t)}$ với $\alpha=3.5$ |
|---|---|
| 1.0 (mới bắt đầu) | $e^0 = 1.0$ |
| 0.5 (nửa thời gian) | $e^{1.75} = 5.75$ |
| 0.2 (gần deadline) | $e^{2.8} = 16.4$ |
| 0.0 (hết thời gian) | $e^{3.5} = 33.1$ |

→ Penalty tăng **phi tuyến**, tạo gradient mạnh khi gần deadline, "ép" agent phải xả hàng.

**Nhân với $Q_{\text{remain}}$:** Penalty tỷ lệ thuận với số giao dịch còn lại → agent bị phạt nặng hơn nếu "ôm" nhiều hàng khi gần deadline.

### 5.4 Thành phần (3): Deadline Penalty

$$\text{Deadline} = -\frac{\lambda}{\sigma} \cdot \mathbf{1}[Q_T > 0]$$

Với $\lambda = 5 \times 10^9$ (ban đầu) hoặc $\lambda_{\text{shaped}} = 20{,}000$ (sau Reward Shaping).

$\mathbf{1}[Q_T > 0]$ = 1 nếu còn giao dịch tồn đọng khi hết thời gian = **miss deadline**.

---

<a id="6"></a>
## 6. Backlog Pressure — Mô hình AR

### 6.1 Công thức

$$b_t = \max\left(0, \; \rho \cdot b_{t-1} + \alpha_b \cdot p_t + \beta_b \cdot u_t\right)$$

### 6.2 Tại sao đây là mô hình Autoregressive?

Vì $b_t$ phụ thuộc vào **chính nó ở bước trước** ($b_{t-1}$). Khai triển đệ quy:

$$b_t = \rho^t b_0 + \sum_{k=0}^{t-1} \rho^k \left(\alpha_b \cdot p_{t-k} + \beta_b \cdot u_{t-k}\right)$$

→ $b_t$ là tổng có trọng số của **toàn bộ lịch sử** tắc nghẽn, với trọng số giảm theo hàm mũ $\rho^k$.

### 6.3 Half-life = 14 blocks

$$\rho^{t_{1/2}} = 0.5 \implies t_{1/2} = \frac{\ln 0.5}{\ln 0.95} = \frac{-0.693}{-0.0513} \approx 13.5 \approx 14 \text{ blocks}$$

→ Sau 14 blocks (~2.8 phút), ảnh hưởng của một cú sốc tắc nghẽn giảm còn 50%. Phù hợp với thời gian trung bình của MEV spike hoặc NFT drop trên Ethereum.

### 6.4 Tại sao $\max(0, \cdot)$?

Đảm bảo $b_t \geq 0$ (áp lực không thể âm). Khi mạng bình thường ($p_t < 0$, $u_t < 0$), backlog pressure tự nhiên giảm về 0.

---

<a id="7"></a>
## 7. n_critics = 1 — Chứng minh Double Pessimism

### 7.1 Double Q-Learning (n_critics = 2)

Dùng 2 mạng $Q_1, Q_2$, target:

$$y = r + \gamma \min(Q_1(s', a^*), Q_2(s', a^*))$$

**Mục đích:** Chống Overestimation Bias trong Online RL.

### 7.2 Nhưng CQL đã có Conservative Penalty!

CQL đã chủ động **underestimate** Q-value qua penalty term. Nếu kết hợp thêm Double Q:

$$\hat{Q}_{\text{Double-CQL}} \leq \min(\hat{Q}_1^{\text{CQL}}, \hat{Q}_2^{\text{CQL}}) \leq Q^* - \underbrace{\epsilon_{\text{CQL}}}_{\text{CQL bias}} - \underbrace{\epsilon_{\text{Double}}}_{\text{Double Q bias}}$$

→ **Bi quan kép (Double Pessimism)**: Q-values bị đè xuống 2 lần.

### 7.3 Hậu quả: Pessimistic Collapse

Khi $Q(s, a) \to -\infty$ cho mọi action $a \neq 0$:
- Agent chỉ còn chọn Action 0 (giữ hàng) vì nó có Q-value "ít âm nhất"
- Giữ hàng mãi → miss 100% deadline
- Policy collapse hoàn toàn

### 7.4 Kết luận

$$\texttt{n\_critics = 1} + \text{CQL penalty} = \text{Đủ một lớp pessimism}$$

Không phải thiếu sót, mà là **thiết kế có chủ ý** để tránh Double Pessimism.

---

<a id="8"></a>
## 8. Reward Shaping — Chứng minh tại sao giảm penalty

### 8.1 Vấn đề: Risk-Reward Asymmetry

| | Giá trị | Đơn vị |
|---|---|---|
| Max timing profit | $\sim 10^3$ | reward units |
| Deadline penalty (ban đầu) | $\sim 10^8$ | reward units |
| **Tỷ lệ** | **1 : 223,000** | |

### 8.2 Phân tích bằng Lý thuyết Quyết định

Agent rational sẽ tối đa Expected Value:

$$\mathbb{E}[V] = p_{\text{win}} \cdot \text{Profit} - p_{\text{miss}} \cdot \text{Penalty}$$

Để "chờ đợi" có lợi, cần:

$$p_{\text{win}} \cdot 10^3 > p_{\text{miss}} \cdot 10^8$$
$$\frac{p_{\text{win}}}{p_{\text{miss}}} > 10^5$$

→ Agent phải **chắc chắn 99.999%** rằng sẽ không miss deadline mới dám chờ. Đây là yêu cầu bất khả thi khi agent mới bắt đầu học.

### 8.3 Sau Reward Shaping (V32)

| Tham số | Trước | Sau | Tác động |
|---|---|---|---|
| deadline_penalty | 1,000,000 | 20,000 | $50\times$ giảm |
| urgency_alpha | 1.0 | 3.5 | $3.5\times$ tăng |
| urgency_beta | 0.0001 | 0.0005 | $5\times$ tăng |

Tỷ lệ Risk/Reward mới: $\frac{20{,}000}{10^3} = 20:1$ — agent chỉ cần chắc 95% → dám exploration.

### 8.4 Định lý Reward Shaping (Ng et al., 1999)

> *Potential-based reward shaping $F(s, s') = \gamma \Phi(s') - \Phi(s)$ không thay đổi optimal policy.*

Trong bài toán của chúng ta, việc scale deadline penalty **không phải** potential-based shaping (nó thay đổi magnitude tuyệt đối), nhưng vì Urgency penalty đã tăng $3.5\times$ để bù lại → agent vẫn bị ép xả hàng khi gần deadline, chỉ là với gradient "mềm" hơn thay vì bị "đập" một cú chí mạng.

---

<a id="9"></a>
## 9. Safety Layer 20% — Phân tích Trade-off

### 9.1 Cơ chế

$$a_t^{\text{final}} = \begin{cases} \pi_{\text{CQL}}(s_t) & \text{nếu } \tau_t > 0.2 \quad \text{(80\% đầu)} \\ \text{Action 4 (xả 100\%)} & \text{nếu } \tau_t \leq 0.2 \quad \text{(20\% cuối)} \end{cases}$$

### 9.2 Tại sao 20%?

Với episode 7200 blocks, 20% = 1440 blocks. Với $C_{\text{cap}} = 500$ tx/block:

$$\text{Max throughput 20\%} = 1440 \times 500 = 720{,}000 \text{ transactions}$$

Trong khi $Q_{\max} \approx 104{,}000$ → Safety Layer có khả năng xả **gấp 7 lần** queue lớn nhất → đảm bảo 0% Miss.

### 9.3 Trade-off định lượng

| Metric | CQL thuần | CQL + Safety 20% |
|---|---|---|
| Gas Savings | 22.4% | **11.1%** |
| Miss Rate | 0.4% | **0.0%** |
| Win Rate | 93.8% | **91.7%** |

Mất 11.3% savings nhưng đổi lại **tuyệt đối 0% miss** — trong tài chính, 1 lần miss có thể gây thiệt hại lớn hơn tổng savings của cả năm.

---

## 📝 Tóm tắt: 9 Điểm Toán học cần nắm vững

| # | Chủ đề | Câu hỏi hội đồng có thể hỏi |
|---|---|---|
| 1 | Bellman Equation | "Chứng minh tính đệ quy?" |
| 2 | Q-Learning convergence | "Tại sao Q-Learning hội tụ?" |
| 3 | CQL Loss | "Conservative penalty hoạt động thế nào?" |
| 4 | Oracle DP | "Chứng minh DP tối ưu toàn cục?" |
| 5 | Reward Function | "Tại sao dùng hàm mũ cho urgency?" |
| 6 | Backlog Pressure | "Half-life 14 blocks từ đâu?" |
| 7 | n_critics = 1 | "Tại sao không dùng Double Q?" |
| 8 | Reward Shaping | "Giảm penalty có thay đổi optimal policy?" |
| 9 | Safety Layer | "20% có cơ sở toán học gì?" |
