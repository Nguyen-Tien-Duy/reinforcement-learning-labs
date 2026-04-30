# Phân Tích Toán Học: Tại Sao Offline RL Tích Trữ Thay Vì Tối Ưu Hóa Gas

> **Tác giả:** Phân tích tự động từ dữ liệu thực nghiệm V33  
> **Ngày:** 2026-04-30  
> **Mục đích:** Chứng minh bằng toán học rằng cấu trúc Reward bất đối xứng là nguyên nhân gốc rễ  
> khiến mọi thuật toán Offline RL (CQL, BCQ, Decision Transformer) đều thất bại trước chiến lược Greedy đơn giản.

---

## 1. Định Nghĩa Bài Toán

Hệ thống Relayer quản lý một hàng đợi giao dịch (Transaction Queue) trên mạng Ethereum.  
Tại mỗi bước thời gian $t$, Agent chọn một trong 5 hành động:

$$a_t \in \{0, 0.25, 0.5, 0.75, 1.0\}$$

trong đó $a_t$ là tỷ lệ phần trăm hàng đợi được xử lý (0 = giữ hết, 1.0 = xả hết).

Số lượng giao dịch được thực thi tại bước $t$:

$$n_t = a_t \times \min(q_t, C_{exec})$$

với $q_t$ là kích thước hàng đợi và $C_{exec} = 500$ là công suất tối đa.

---

## 2. Cấu Trúc Hàm Reward

Reward tại mỗi bước $t$ gồm 3 thành phần:

$$R_t = \frac{R_{eff}(t) - R_{urg}(t) - R_{cat}(t)}{s_{reward}}$$

### 2.1. Hiệu suất (Efficiency)

$$R_{eff}(t) = \frac{n_t \cdot (g_{ref} - g_t) - \frac{C_{base}}{10^9} \cdot g_t \cdot \mathbf{1}[n_t > 0]}{s_g}$$

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| $g_t$ | ~2.69 Gwei (trung bình) | Giá Gas hiện tại |
| $g_{ref}$ | ~2.70 Gwei (trung bình) | Giá Gas tham chiếu (MA-128) |
| $s_g$ | 10.0 | Hệ số chia tỷ lệ |
| $C_{base}$ | 21,000 gas | Chi phí cố định mỗi batch |

**Nhận xét quan trọng:**  
Chênh lệch trung bình $g_{ref} - g_t \approx 0.003$ Gwei — **cực kỳ nhỏ**.  
Với $n_t = 500$ (xả tối đa): $R_{eff} \approx \frac{500 \times 0.003}{10} = 0.15$ điểm/bước.

### 2.2. Áp lực Khẩn cấp (Urgency)

$$R_{urg}(t) = \beta \cdot q_t \cdot e^{\alpha(1 - \tau_t / D)}$$

| Tham số | Giá trị |
|---|---|
| $\beta$ | 0.0001 |
| $\alpha$ | 1.0 |

**Nhận xét:** Với $\beta = 0.0001$, urgency gần như **không có tác dụng** ép Agent xả hàng.  
Ví dụ: $q_t = 10{,}000$, $\tau_t/D = 0.5$: $R_{urg} = 0.0001 \times 10{,}000 \times e^{0.5} \approx 1.65$ — quá nhỏ.

### 2.3. Thảm họa (Catastrophe) — **ĐÂY LÀ CHỖ CHẾT**

$$R_{cat} = \lambda_d \times q_T \times \mathbf{1}[t = T]$$

| Tham số | Giá trị |
|---|---|
| $\lambda_d$ | 1,000,000 |

Phạt chỉ áp dụng **MỘT LẦN DUY NHẤT** tại bước cuối cùng $T$, tỷ lệ thuận với số hàng còn lại $q_T$.

---

## 3. Chứng Minh Toán Học: Tại Sao Agent Tích Trữ

### Định lý: Bất đẳng thức Risk-Reward

Agent phải so sánh 2 chiến lược:

**Chiến lược A (Greedy — Xả Ngay):**

Tổng reward từ hiệu suất:

$$\sum_{t=0}^{T} R_{eff}(t) = \sum_{t=0}^{T} \frac{n_t \cdot (g_{ref} - g_t)}{s_g}$$

> **Dữ liệu thực nghiệm (Episode 0, 7172 bước):**  
> $\sum R_{eff} = +1{,}538$ điểm (tổng cộng, sau khi trừ chi phí overhead).  
> Trong đó **43% các bước có $R_{eff} < 0$** (bán lỗ khi $g_t > g_{ref}$).

**Chiến lược B (Tích trữ — Giữ Hết):**

$$R_{total} = 0 - R_{cat} = -\lambda_d \times q_T$$

> **Dữ liệu thực nghiệm:**  
> $q_T = 343$ giao dịch còn lại → $R_{cat} = 1{,}000{,}000 \times 343 = 343{,}000{,}000$

### So sánh trực tiếp

| Chiến lược | Thua lỗ | Tỷ lệ |
|---|---|---|
| Xả Ngay (Greedy) | 13,036 điểm (bán lỗ lúc gas cao) | 1× |
| Giữ Hết (Hoarder) | 343,000,000 điểm (bị phạt deadline) | **26,315×** |

> **Kết luận:** Penalty lớn gấp **26,315 lần** so với chi phí bán lỗ.  
> → Bất kỳ Agent nào có IQ > 0 đều phải chọn Greedy.

### Vậy tại sao Agent vẫn tích trữ?

Vì Agent **KHÔNG** so sánh 2 chiến lược cực đoan như trên.  
Agent học từ **DỮ LIỆU**, và dữ liệu Oracle nói rằng:

---

## 4. Bẫy Class Imbalance Trong Dữ Liệu Oracle

### 4.1. Phân phối hành động của Oracle

| Action | Tỷ lệ | Log Reward TB | Ý nghĩa |
|---|---|---|---|
| **Action 0 (Giữ)** | **69.6%** | -0.89 | Oracle giữ hàng 70% thời gian |
| Action 1 (Xả 25%) | 9.7% | +2.77 | Lợi nhuận cao nhất |
| Action 2 (Xả 50%) | 2.5% | +1.14 | |
| Action 3 (Xả 75%) | 2.0% | +0.73 | |
| Action 4 (Xả 100%) | 16.2% | +0.15 | Lợi nhuận thấp nhất |

### 4.2. Tại sao Oracle giữ hàng 70%?

Oracle có khả năng **nhìn thấu tương lai** (Hindsight Dynamic Programming).  
Nó biết chính xác giá Gas sẽ giảm vào lúc nào → kiên nhẫn chờ → xả đúng đáy.

Nhưng Agent RL **KHÔNG** nhìn thấy tương lai. Nó chỉ thấy:
- "70% các bước, câu trả lời đúng là GIỮ"
- "Khi nào xả thì reward cao, nhưng điều kiện xả rất phức tạp"

### 4.3. Hậu quả cho từng thuật toán

#### CQL (Conservative Q-Learning)
- CQL tính Q-value cho mỗi action, sau đó **trừ phạt** các action OOD (ngoài phân phối dữ liệu).
- Vì Action 0 chiếm 69.6% → Q(s, a=0) được ước lượng chính xác nhất.
- Q(s, a=4) bị CQL **ép xuống thêm** (conservative penalty) vì nó xuất hiện ít hơn.
- Kết quả: CQL **LUÔN** chọn Action 0 → Tích trữ → Deadline Miss.

$$Q_{CQL}(s, a) = Q(s,a) - \alpha_{CQL} \cdot \left[\log \sum_a e^{Q(s,a)} - \mathbb{E}_{a \sim \hat{\pi}} [Q(s,a)]\right]$$

Số hạng phạt $\alpha_{CQL}$ đẩy Q-value của các action hiếm (1-4) xuống thấp hơn nữa.

#### BCQ (Batch-Constrained Q-Learning)
- BCQ giới hạn Agent chỉ được chọn action mà behavior policy đã từng làm.
- Vì Action 0 chiếm đa số → BCQ bị "nhốt" trong vùng Action 0.
- `action_flexibility` quá thấp → Agent không dám chọn Action 3-4.

#### Decision Transformer (DT)
- DT sử dụng **Cross-Entropy Loss** để phân loại action.
- Cross-Entropy trên dữ liệu mất cân bằng (70%/30%) → **Majority Class Bias**.
- Model luôn dự đoán Action 0 vì đó là cách **giảm Loss nhanh nhất** (đúng 70% thời gian).

$$\mathcal{L}_{CE} = -\sum_{t} \log P(a_t^{oracle} | s_{1:t}, a_{1:t-1}, R_{target})$$

Khi $a_t^{oracle} = 0$ chiếm 70%, minimize $\mathcal{L}_{CE}$ đồng nghĩa với **luôn đoán 0**.

---

## 5. Bất Đẳng Thức Cốt Lõi

### Điều kiện để RL có thể thắng Greedy:

Gọi:
- $\Delta g = g_{ref} - g_t$ : Chênh lệch Gas (trung bình ≈ 0.003 Gwei)
- $N_{total}$ : Tổng số TX phải xử lý trong episode (~160,000)  
- $\lambda_d$ : Mức phạt deadline (1,000,000)
- $p_{miss}$ : Xác suất Agent bị trễ deadline khi "chờ đợi"

**Điều kiện cần:**

$$\underbrace{N_{total} \cdot \Delta g / s_g}_{\text{Lợi nhuận tối đa từ timing}} > \underbrace{p_{miss} \cdot \lambda_d \cdot \mathbb{E}[q_T]}_{\text{Kỳ vọng thiệt hại từ penalty}}$$

Thay số:
- Vế trái: $160{,}000 \times 0.003 / 10 = 48$ điểm
- Vế phải: $p_{miss} \times 1{,}000{,}000 \times q_T$

→ Chỉ cần $p_{miss} \times q_T > 0.000048$ là RL đã thua Greedy!

Với $p_{miss} = 1\%$ và $q_T = 100$: Vế phải = $0.01 \times 1{,}000{,}000 \times 100 = 1{,}000{,}000$ ≫ 48.

> **Kết luận:** Bất đẳng thức này **KHÔNG BAO GIỜ** thỏa mãn với cấu trúc tham số hiện tại.  
> Lợi nhuận từ việc "timing thị trường" (48 điểm) nhỏ hơn rủi ro penalty  
> hàng TRIỆU lần. **Greedy là nghiệm gần tối ưu (Near-Optimal).**

---

## 6. Bảng Tổng Hợp Siêu Tham Số & Tác Động

| Siêu tham số | Giá trị hiện tại | Tác động | Giải pháp tiềm năng |
|---|---|---|---|
| $\lambda_d$ (deadline_penalty) | 1,000,000 | Quá lớn → mọi chiến lược chờ đều có kỳ vọng âm | Giảm xuống tỷ lệ với $R_{eff}$ |
| $\beta$ (urgency_beta) | 0.0001 | Quá nhỏ → không ép Agent xả sớm | Tăng lên 1.0 - 10.0 |
| $s_g$ (gas_scaling_factor) | 10.0 | Chia nhỏ $R_{eff}$ thêm 10 lần | Giảm xuống 1.0 |
| $\Delta g$ (gas spread) | ~0.003 Gwei | Biên lợi nhuận quá mỏng | Cấu trúc thị trường, không thay đổi được |
| Oracle Action 0 ratio | 69.6% | Class Imbalance → Majority Bias | Oversampling Action 1-4 hoặc Focal Loss |

---

## 7. Kết Luận Khoa Học

### 7.1. Phát hiện chính

1. **Bất đối xứng Reward (Asymmetric Risk-Reward):** Lợi nhuận từ tối ưu hóa Gas ($\sim 48$ điểm/episode) nhỏ hơn rủi ro deadline penalty ($\sim 343{,}000{,}000$ điểm) **hơn 7 triệu lần**. Trong điều kiện này, chiến lược Greedy (xả ngay) là nghiệm gần tối ưu theo lý thuyết quyết định (Decision Theory).

2. **Class Imbalance trong Expert Data:** Oracle giữ hàng 69.6% thời gian, tạo ra dữ liệu mất cân bằng nghiêm trọng. Mọi thuật toán Offline RL đều bị bias về phía hành động đa số (Action 0 = Giữ).

3. **Conservative Collapse (CQL):** Cơ chế phạt bảo thủ của CQL ($\alpha_{CQL}$) đẩy Q-value của các action hiếm (xả hàng) xuống thấp hơn nữa, tạo vòng xoáy tự tăng cường: càng ít xả → Q-value xả càng thấp → càng không dám xả.

### 7.2. Hướng phát triển tương lai

1. **Reward Shaping:** Thiết kế lại hàm thưởng sao cho $R_{eff}$ và $R_{cat}$ cùng bậc độ lớn (same order of magnitude).
2. **Safety Layer:** Kết hợp RL với một tầng an toàn cứng (Hard Constraint) bắt buộc xả hết khi $\tau_t / D < 0.1$.
3. **Focal Loss cho DT:** Thay Cross-Entropy bằng Focal Loss để giảm ảnh hưởng của class đa số.
4. **Curriculum Learning:** Bắt đầu train với $\lambda_d$ nhỏ, tăng dần để Agent học cách xả trước khi bị "sốc" bởi penalty.

---

## Phụ Lục: Dữ Liệu Thực Nghiệm

```
Episode 0 (Oracle, 7172 bước):
  s_gas_t0:  mean=2.6922 Gwei, range=[2.44, 4.69]
  s_gas_ref: mean=2.6950 Gwei, range=[2.59, 3.40]
  s_queue:   mean=2,334 TX,    range=[24, 16,227]
  
  R_eff tổng (Greedy):  +1,538 điểm
  R_eff lỗ (43% bước):  -13,036 điểm  
  R_cat (Giữ hết):      -343,000,000 điểm

Toàn bộ Oracle (4.6M transitions):
  Reward RAW:  mean=-2.985, median=-0.138
  Reward LOG:  mean=-0.281, median=-0.130
  Action 0:    69.6% (Giữ hàng)
  Action 1-4:  30.4% (Xả hàng)
```
