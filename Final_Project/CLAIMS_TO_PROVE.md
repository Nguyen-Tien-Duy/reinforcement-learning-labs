# 📋 Danh Sách Giả Định & Khẳng Định Cần Chứng Minh Trước Khi Publish Paper

> Tài liệu này liệt kê **tất cả** các giả định (assumptions), khẳng định (claims),
> và kết luận (conclusions) được đưa ra trong đồ án. Mỗi mục cần có bằng chứng
> thực nghiệm hoặc chứng minh toán học trước khi submit bài báo khoa học.

---

## Nhóm A: Mô Hình Hóa Bài Toán (Problem Formulation)

### A1. ❓ MDP Formulation là đúng đắn
- **Khẳng định:** Bài toán điều phối giao dịch có thể được mô hình hóa dưới dạng MDP với 11 chiều state, 5 action rời rạc.
- **Cần chứng minh:**
  - [ ] Tính Markov (Markov Property): Trạng thái $s_t$ có chứa đủ thông tin để dự đoán $s_{t+1}$ không? Hay cần thêm lịch sử dài hơn (POMDP)?
  - [ ] Autocorrelation test: Kiểm tra xem reward $r_t$ có phụ thuộc vào $r_{t-k}$ (k > 1) sau khi đã condition trên $s_t$ không.
  - [ ] Ablation: Thử bỏ từng feature trong 11 chiều, đo tác động lên hiệu năng CQL → xác nhận tất cả 11 chiều đều cần thiết.

### A2. ❓ Không gian Hành động Rời rạc (5 bins) là đủ tốt
- **Khẳng định:** 5 bins {0%, 25%, 50%, 75%, 100%} là đủ để xấp xỉ hành vi tối ưu.
- **Cần chứng minh:**
  - [ ] So sánh Discrete Oracle (5 bins) vs Continuous Oracle (0-500 tx) trên cùng tập dữ liệu → Đo mức chênh lệch Gas Cost giữa 2 Oracle.
  - [ ] Nếu chênh lệch < 3%, kết luận 5 bins đã gần-tối-ưu. Nếu > 5%, cần thử 10 bins hoặc 20 bins.
  - [ ] Phân tích phân phối hành động của Continuous Oracle → Xem nó có tập trung quanh 5 đỉnh {0, 0.25, 0.5, 0.75, 1.0} không.

### A3. ❓ Hàm Reward thiết kế đúng mục tiêu
- **Khẳng định:** Hàm reward $r_t = R_{eff} + R_{urg} + R_{cat}$ align với mục tiêu thực tế (giảm gas, không trễ deadline).
- **Cần chứng minh:**
  - [ ] Correlation test: Tổng reward của episode có tương quan âm với tổng Gas Cost thực tế không? (Kỳ vọng: $\rho < -0.8$)
  - [ ] Reward Hacking check: Agent có exploit được reward mà không thực sự giảm Gas không? (VD: giữ hàng để tránh urgency penalty nhưng cuối cùng xả hết ở block cuối với gas cao)

---

## Nhóm B: Oracle & Data Pipeline

### B1. ❓ Discrete Oracle thực sự là tối ưu toàn cục trong không gian 5 bins
- **Khẳng định:** DP Backward Induction tìm ra chính sách tối ưu **toàn cục** (global optimum).
- **Cần chứng minh:**
  - [ ] Chứng minh toán học: DP trên không gian hữu hạn $(T \times Q_{max} \times 5)$ đảm bảo tối ưu toàn cục (theo định lý Bellman).
  - [ ] Sanity check: So sánh Oracle cost vs Greedy cost trên mọi episode → Oracle phải luôn $\leq$ Greedy (không bao giờ tệ hơn).
  - [ ] Edge case: Kiểm tra episode có $Q_0 = 0$ (không có giao dịch) → Oracle cost phải = 0.

### B2. ❓ Queue Physics đúng (Causal Consistency)
- **Khẳng định:** $Q_{t+1} = \max(0, Q_t - n_t) + w_{t+1}$ mô phỏng chính xác hàng đợi thực.
- **Cần chứng minh:**
  - [ ] Unit test: Với $Q_t = 100$, action = 4 (100%), $n_t = \min(100, 500) = 100$ → $Q_{t+1} = 0 + w_{t+1}$.
  - [ ] Unit test: Với $Q_t = 1000$, action = 2 (50%), $n_t = \min(500, 500) = 500$ → $Q_{t+1} = 500 + w_{t+1}$.
  - [ ] Invariant check: $Q_t \geq 0$ tại mọi thời điểm trong mọi episode.
  - [ ] Conservation law: Tổng $\sum n_t + Q_T = Q_0 + \sum w_t$ (bảo toàn giao dịch).

### B3. ❓ Dataset không bị Data Leakage
- **Khẳng định:** 20% episodes cuối (ID 964-1205) không bị rò rỉ vào tập train.
- **Cần chứng minh:**
  - [ ] Xác nhận split dựa trên episode_id (theo thời gian), không phải random split.
  - [ ] Kiểm tra: Không có episode nào xuất hiện trong cả train và test.
  - [ ] Observation Scaler (MinMax) được fit trên tập TRAIN, không phải toàn bộ dataset.

### B4. ❓ Dataset Mix (70% Oracle + 10% Suboptimal + 20% Behavior) là hợp lý
- **Khẳng định:** Tỷ lệ mix 70/10/20 cho kết quả tốt nhất.
- **Cần chứng minh:**
  - [ ] Ablation study: So sánh 100% Oracle vs 70/10/20 vs 50/25/25 → Đo hiệu năng CQL.
  - [ ] Action coverage check: Kiểm tra mọi action {0,1,2,3,4} đều xuất hiện với tần suất > 5% trong dataset mix.

---

## Nhóm C: Kết Quả Thực Nghiệm (Empirical Claims)

### C1. ✅ CQL + Safety 20% tiết kiệm 11.1% so với Greedy (đã chứng minh)
- **Bằng chứng hiện có:**
  - [x] `reproduce_sota.py` chạy ra đúng kết quả: Gas = 65,525, Miss = 0.0%, vs Greedy = +11.1%.
  - [x] Tái lập 100% trên máy local (CPU).
- **Cần bổ sung:**
  - [ ] Statistical significance test (t-test hoặc Wilcoxon): p-value < 0.05 cho sự khác biệt giữa CQL và Greedy trên 242 episodes.
  - [ ] Confidence interval: Tính 95% CI cho mức tiết kiệm (VD: 11.1% ± 2.3%).
  - [ ] Per-episode analysis: Bao nhiêu episodes CQL thắng Greedy? Bao nhiêu thua? (Win Rate)

### C2. ❓ CQL thuần (không Safety) tiết kiệm 22.4% nhưng Miss = 0.4%
- **Cần chứng minh:**
  - [ ] Chạy CQL model_160000 KHÔNG CÓ Safety Layer trên 242 episodes.
  - [ ] Ghi lại chính xác: Gas Cost, Miss Rate, số episode bị miss.
  - [ ] So sánh trực tiếp: CQL thuần vs CQL + Safety → Đo chính xác "giá" của Safety Layer.

### C3. ❓ Oracle tiết kiệm 38.8% so với Greedy
- **Cần chứng minh:**
  - [ ] Chạy Oracle trên 242 episodes test → Tính Gas Cost trung bình.
  - [ ] Xác nhận Oracle Miss Rate = 0% (DP phải giải xong queue trước deadline).
  - [ ] Lưu ý: Oracle dùng Discrete (5 bins), CẦN ghi rõ trong paper là "Discrete Oracle upper bound", không phải "Theoretical optimum".

### C4. ❓ Smart Rule + Safety 20% tệ hơn Greedy 4.3%
- **Cần chứng minh:**
  - [ ] Chạy Smart Rule trên 242 episodes → Xác nhận Gas Cost = 76,892.
  - [ ] Giải thích rõ logic Smart Rule là gì (VD: chờ Gas < Moving Average rồi mới bán).
  - [ ] Phân tích TẠI SAO Smart Rule thua Greedy: Queue Buildup → Overhead 21,000 gas/batch → Mất nhiều hơn tiết kiệm.

### C5. ❓ Decision Transformer thất bại hoàn toàn (Miss 90-100%)
- **Cần chứng minh:**
  - [ ] Chạy DT trên ít nhất 20 episodes test → Ghi lại Miss Rate chính xác.
  - [ ] Action distribution analysis: Đếm tần suất action {0,1,2,3,4} mà DT chọn → Chứng minh nó bị Majority Bias (>90% Action 0).
  - [ ] So sánh với action distribution trong training data (Oracle) → Chứng minh DT chỉ copy tần suất, không học value.

---

## Nhóm D: Phân Tích & Giải Thích (Analytical Claims)

### D1. ❓ Risk/Reward Asymmetry = 1:223,000 gây Pessimistic Collapse
- **Khẳng định:** Tỷ lệ Risk/Reward quá lớn khiến Agent không dám explore.
- **Cần chứng minh:**
  - [ ] Tính toán chính xác: $\frac{\text{Deadline Penalty max}}{\text{Gas Savings max per step}}$ với các tham số V33.
  - [ ] Ablation: Train CQL với penalty = {1K, 10K, 20K, 100K, 1M} → Vẽ đồ thị Miss Rate và Gas Cost theo penalty.
  - [ ] Chứng minh: Tồn tại ngưỡng penalty $\lambda^*$ mà dưới đó Agent bắt đầu explore thành công.

### D2. ❓ Reward Shaping là yếu tố quyết định (không phải Data Coverage đơn thuần)
- **Khẳng định:** Thay đổi deadline_penalty từ 1M xuống 20K là nguyên nhân chính khiến model V32 thắng.
- **Cần chứng minh (Ablation thực sự):**
  - [ ] **Thí nghiệm kiểm soát 1:** Train CQL trên V32 DATA + V33 REWARD (penalty=1M) → Nếu thua = Reward Shaping quan trọng hơn Data.
  - [ ] **Thí nghiệm kiểm soát 2:** Train CQL trên V33 DATA + V32 REWARD (penalty=20K) → Nếu thắng = Reward Shaping quan trọng hơn Data.
  - [ ] **Thí nghiệm kiểm soát 3:** Giữ nguyên V32 REWARD nhưng đổi CQL alpha từ 1.0 sang 0.1 → Đo tác động riêng của CQL alpha.

### D3. ❓ Class Imbalance (69.6% Action 0) gây ra Majority Bias ở DT
- **Cần chứng minh:**
  - [ ] Đếm chính xác tần suất action trong tập Oracle: Action 0 = ?%, Action 1 = ?%, ..., Action 4 = ?%.
  - [ ] So sánh với tần suất action mà DT dự đoán khi test → Xem có giống phân phối train không.
  - [ ] Thử Class Rebalancing (oversampling Action 1-4, undersampling Action 0) → Xem DT có cải thiện không.

### D4. ❓ CQL có khả năng Trajectory Stitching (ghép quỹ đạo) mà DT không có
- **Khẳng định:** CQL ghép các phần tốt từ nhiều trajectory khác nhau nhờ Bellman equation.
- **Cần chứng minh:**
  - [ ] Phân tích hành vi CQL trên một episode cụ thể: Vẽ biểu đồ action theo thời gian → So sánh với Oracle → Xem CQL có chọn action khác Oracle nhưng VẪN tiết kiệm gas không (= sáng tạo ra chiến lược mới).
  - [ ] Hoặc trích dẫn Levine et al. (2020) / Kumar et al. (2020) như bằng chứng lý thuyết.

### D5. ❓ Safety Layer là "bảo hiểm" cần thiết, không phải "hack" kết quả
- **Cần chứng minh:**
  - [ ] Chạy Safety-Only (không có CQL, chỉ có: if time < 20% → action=4, else action=random) → Chứng minh Safety Layer đơn thuần KHÔNG đủ để thắng Greedy.
  - [ ] Phân tích: Trong 242 episodes, Safety Layer được kích hoạt bao nhiêu lần? Bao nhiêu % thời gian CQL thực sự điều khiển?
  - [ ] Đo: Nếu đổi threshold từ 20% sang {10%, 15%, 25%, 30%} → Vẽ Pareto curve (Gas Cost vs Miss Rate).

---

## Nhóm E: Tính Tổng Quát & Giới Hạn (Generalizability)

### E1. ❓ Kết quả có tổng quát hóa sang thời kỳ thị trường khác không?
- **Cần chứng minh:**
  - [ ] Chia 242 episodes theo đặc tính thị trường (high volatility vs low volatility) → CQL thắng đều ở cả 2 nhóm hay chỉ thắng ở 1 nhóm?
  - [ ] Walk-forward test: Train trên tháng 1-6, test trên tháng 7-8. Rồi train trên tháng 1-7, test trên tháng 8. → Kết quả có nhất quán không?

### E2. ❓ Model có bị overfit vào checkpoint 160,000 không?
- **Cần chứng minh:**
  - [ ] Đánh giá các checkpoint lân cận: model_150000, model_170000, model_180000 trên cùng 242 episodes.
  - [ ] Nếu hiệu năng dao động mạnh giữa các checkpoint → Dấu hiệu overfit.
  - [ ] Nếu hiệu năng ổn định trong khoảng 140K-200K → Mô hình robust.

### E3. ❓ Kết quả có phụ thuộc vào random seed không?
- **Cần chứng minh:**
  - [ ] Kiểm tra: CQL inference có deterministic không? (Không có dropout, không có sampling → YES, deterministic).
  - [ ] Kiểm tra: Oracle DP có deterministic không? (Backward induction trên bảng giá cố định → YES, deterministic).
  - [ ] Nếu cả 2 đều deterministic → Kết quả tái lập 100% mà KHÔNG cần chạy nhiều seed (đã chứng minh ở reproduce_sota.py).

---

## Nhóm F: So Sánh với Related Work

### F1. ❓ Baseline Greedy là "Dynamic" chứ không phải "Static"
- **Khẳng định:** Greedy baseline bám sát giá thị trường, khác với "static fee" trong các bài báo khác.
- **Cần chứng minh:**
  - [ ] Mô tả chính xác Greedy: Tại mỗi block, xả 100% queue với giá gas hiện tại (Base Fee + Priority Fee).
  - [ ] So sánh: Greedy KHÔNG bao giờ trả giá cao hơn thị trường (không overpay) → Đây đã là baseline cực kỳ mạnh.

### F2. ❓ Bài toán "Timing" khác bản chất với "Gas Used Optimization"
- **Cần chứng minh:**
  - [ ] Phân loại rõ ràng: Gas Used optimization = sửa code Solidity để giảm opcode. Gas Price optimization = canh thời điểm gửi giao dịch.
  - [ ] Trích dẫn: Các bài báo 20-30% savings đều thuộc loại Gas Used hoặc Batching, KHÔNG phải Timing.
  - [ ] Khẳng định: Bài toán của chúng ta là Timing trên hệ thống ĐÃ batch sẵn → Baseline đã rất mạnh.

---

## 📊 Tổng Kết Tiến Độ

| Nhóm | Tổng | Đã chứng minh | Còn lại |
|------|------|----------------|---------|
| A. Mô hình hóa | 3 claims | 0 | 3 |
| B. Oracle & Data | 4 claims | 0 | 4 |
| C. Kết quả thực nghiệm | 5 claims | 1 (C1) | 4 |
| D. Phân tích & Giải thích | 5 claims | 0 | 5 |
| E. Tổng quát & Giới hạn | 3 claims | 0 | 3 |
| F. So sánh Related Work | 2 claims | 0 | 2 |
| **TỔNG** | **22 claims** | **1** | **21** |

---

## 🎯 Độ Ưu Tiên Khi Publish Paper

### Bắt buộc phải có (Must-have):
1. **C1** ✅ Statistical significance test (p-value, CI)
2. **D2** Ablation: Reward Shaping vs Data Coverage (thí nghiệm kiểm soát)
3. **C3** Oracle upper bound chính xác (+ ghi rõ là Discrete Oracle)
4. **B1** Chứng minh Oracle tối ưu toàn cục trong 5 bins
5. **D5** Safety Layer ablation (Pareto curve)
6. **E2** Checkpoint stability test

### Nên có (Should-have):
7. **A2** So sánh 5 bins vs Continuous Oracle
8. **D1** Penalty ablation (vẽ đồ thị)
9. **C2** CQL thuần vs CQL + Safety (chính xác hóa)
10. **D3** Class Imbalance analysis cho DT

### Tốt nếu có (Nice-to-have):
11. **E1** Walk-forward test
12. **A1** Markov Property test
13. **B4** Dataset Mix ablation
14. **D4** Trajectory Stitching visualization
