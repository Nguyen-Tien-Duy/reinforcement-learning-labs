# Kiến trúc hệ thống Continuous Oracle & Offline RL

Tài liệu này định nghĩa cấu trúc để nâng cấp hệ thống tối ưu phí Gas sang không gian hành động liên tục (Continuous Action Space) kết hợp với thuật toán dán nhãn Thần thánh (Hindsight Oracle Labeling). Định hướng này giúp hệ thống mô phỏng chính xác "Lợi thế quy mô" (Economies of Scale) và tối ưu điểm hòa vốn.

---

## 1. Cơ sở kinh tế học

### 1.1. Lợi thế quy mô (Economies of Scale)
Cấu trúc phí cho một Smart Contract Multisender trên Ethereum bao gồm 2 phần:
*   **Fixed Overhead ($C_{base}$):** Mức phí gas cố định để mở kết nối hợp đồng (ví dụ: 21,000 gas).
*   **Marginal Cost ($C_{mar}$):** Phí gas tăng thêm cho mỗi ví nhận (ví dụ: 15,000 gas).
$\rightarrow$ Gửi 100 người trong 1 lệnh luôn rẻ hơn việc gửi lẻ 100 lệnh do tiết kiệm được 99 lần $C_{base}$. Điều này sinh ra một bài toán tồn kho: AI có động lực (incentive) "Ôm hàng" đợi Batch lớn để gỡ lại Overhead.

### 1.2. Thuyết Quyền chọn Thực (Real Options Theory)
Giữ lại hàng đợi ở action $a=0$ là việc nắm giữ một quyền chọn (Call Option) để gửi giao dịch ở mức gas rẻ hơn trong tương lai. Khi tiến về thời hạn chót (Deadline), giá trị của quyền chọn này ngót dần về 0.

---

## 2. Mô hình hóa Toán học (Continuous MDP)

**Trạng thái (State):** 
$$s_t = [g_t, p_t, m_t, u_t, b_t, Q_t, \tau_t]$$
Bao gồm giá Gas thô và các thuộc tính động học Pseudo-mempool, cùng với độ lớn hàng đợi hiện tại và thời gian dồn ép.

**Hành động (Action):**
$$a_t \in [0.0, 1.0]$$
Đại diện cho "Tỷ lệ % hàng đợi sẽ được đem đi xả". Việc này mở ra không gian cho kỹ thuật "Cost Averaging" (rải khối lượng xả ra để trung bình hóa biến động phí Gas).

**Động lực học quá trình (Transition):**
Khối lượng giải ngân: $n_t = \lfloor a_t \cdot Q_t \rfloor$
Mùi vị thời gian: $\tau_{t+1} = \tau_t - 1$
Hàng đợi: $Q_{t+1} = Q_t - n_t + W_t$ (Với $W_t$ là yêu cầu mới xen ngang).

---

## 3. Thuật toán Dán nhãn Oracle (Hindsight Optimization)

Để tránh nhược điểm của dữ liệu hành vi Offline RL (AI bắt chước theo một behavior policy cẩu thả của người thường), ta dùng **Quy hoạch động (Dynamic Programming)** tìm ra chuỗi hành động tối ưu tuyệt đối của quá khứ. Nó thiết lập một Policy "Thần thánh", đóng vai trò kim chỉ nam lý tưởng nhất.

### 3.1. Phương trình Bellman Backward Induction
Lưu giá trị tối thiểu để xử lý $q$ giao dịch còn lại từ thời điểm $t$ đến mức hạn chót $T$:

$$V_t^*(q) = \min_{n \in [0, q]} \left\{ C(n, g_t) + V_{t+1}^*(q - n + W_t) \right\}$$
Trong đó Immediate Cost tại hiện tại bị chi phối bởi độ giãn đoạn (Lợi thế quy mô):
$$C(n, g_t) = g_t \cdot \left[ C_{base} \cdot \mathbb{1}_{(n > 0)} + n \cdot C_{mar} \right]$$

### 3.2. Mã giả thuật toán (Dynamic Matrix):
- Build DP Table `V` kích thước `[T+1, Q_max+1]`. Khởi tạo Infinity.
- Điểm chốt `V[T, q] = +Penalty` đối với các deadline trễ.
- Lần ngược từ $T-1$ về $0$. Duyệt vòng for kép trên lượng `q` hiện tại và lượng test `n` mang xả. Lưu kết quả làm `Best_N`.
- Lần xuôi Forward pass từ $0 \rightarrow T$ dựa vào lượng `incoming` để thu thập chuỗi $a_t^*$ liên tục.

---

## 4. Ứng dụng Training Offline RL (IQL Continuous)

Sau khi tạo dữ liệu gán nhãn chuẩn chỉ (Ví dụ: Mix tỷ lệ 50% Oracle DP, 30% Non-optimal Logging, 20% Thảm họa - Fatal Failure), bài toán sẽ được huấn luyện thông qua Implicit Q-Learning.

1. **Expectile Regression ($\tau$):** Ta dùng Expectile cao (ví dụ: $\tau=0.8$) để Agent chỉ lấy ý tưởng từ những đường lối $a_t$ sáng nhất.
2. **Actor-Critic Continuous:** 
   Thay vì Discrete, Policy Network nhả dải tỷ lệ Action từ một lớp ráng Sigmoid/Tanh, hoặc ra phân phối Gaussian $(\mu, \sigma)$ để thực thi việc giải ngân Tỷ lệ liên tục.
