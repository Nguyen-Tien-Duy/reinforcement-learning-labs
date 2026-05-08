# 📋 Script Thuyết Trình — Final Project
## Ứng dụng Offline Reinforcement Learning Tối ưu Chi phí Gas trên Ethereum

> **Tổng thời gian:** ~14–15 phút  
> **Số slide chính:** 14 (không tính backup)  
> **Nguyên tắc:** Mỗi slide nói khoảng 1 phút. Slide quan trọng (Leaderboard, Safety Layer) nói 1.5–2 phút.

---

## Slide 1 — Trang bìa ⏱️ 30 giây

> *[Đứng dậy, chào hội đồng]*

**Nói:**

"Kính chào Cô và các bạn. Hôm nay nhóm em xin trình bày đề tài **Ứng dụng Offline Reinforcement Learning để Tối ưu Chi phí Gas trên mạng Ethereum**. Đây là bài toán thực tế trong hệ thống blockchain, nơi mỗi giao dịch đều phải trả một khoản phí gọi là Gas Fee."

---

## Slide 2 — Nội dung trình bày ⏱️ 20 giây

**Nói:**

"Bài thuyết trình gồm 5 phần chính: Đầu tiên em sẽ giới thiệu **bài toán**, sau đó trình bày **phương pháp** tiếp cận, tiếp theo là **kết quả** thực nghiệm, rồi **phân tích** các phát hiện quan trọng, và cuối cùng là **kết luận**."

> *[Chuyển slide nhanh, không dừng lâu]*

---

## Slide 3 — Bối cảnh: Chi phí Gas trên Ethereum ⏱️ 1 phút 30 giây

**Nói:**

"Trước tiên, em xin giải thích bài toán. Trên mạng Ethereum, mỗi giao dịch đều phải trả một khoản phí gọi là **Gas Fee**. Phí này **không cố định** mà dao động liên tục theo mức độ tắc nghẽn của mạng.

Vào giờ cao điểm — ví dụ khi có sự kiện NFT drop hoặc nhiều người giao dịch cùng lúc — phí có thể tăng lên **gấp 5 đến 10 lần** so với bình thường.

*(Chỉ vào hình bên phải)* Như các bạn thấy ở biểu đồ này, đường Gas Price dao động rất mạnh trong suốt 24 giờ. Vậy câu hỏi đặt ra là: **khi nào nên gửi giao dịch để tiết kiệm nhất?**

Tuy nhiên, có một ràng buộc rất quan trọng: tất cả giao dịch phải hoàn tất trước **deadline 24 giờ**, tức là chúng ta phải đạt **0% Miss Rate** — không được bỏ lỡ bất kỳ giao dịch nào."

---

## Slide 4 — Mô hình hóa: Markov Decision Process ⏱️ 1 phút 15 giây

**Nói:**

"Để giải bài toán này, nhóm em mô hình hóa nó thành một **Markov Decision Process**.

Tại mỗi block trên blockchain, hệ thống cần quyết định gửi bao nhiêu giao dịch để tối thiểu tổng chi phí Gas mà vẫn đảm bảo deadline.

Về **State**, mỗi trạng thái có **11 chiều**, bao gồm: giá gas hiện tại và 2 bước trước, mức tắc nghẽn, momentum giá, số giao dịch còn trong hàng đợi, thời gian còn lại trước deadline, và một đặc trưng mới mà nhóm tự thiết kế gọi là **Backlog Pressure**.

Về **Action**, agent chọn 1 trong 5 bin rời rạc: giữ hàng, hoặc xả 25%, 50%, 75%, hay 100% hàng đợi.

**Reward** được thiết kế gồm 3 thành phần: chi phí gas âm, urgency penalty tăng theo hàm mũ khi gần deadline, và deadline penalty nếu bỏ lỡ."

---

## Slide 5 — Hindsight Oracle ⏱️ 1 phút 15 giây

**Nói:**

"Một vấn đề lớn trong Offline RL là chất lượng dataset. Nếu chúng ta chỉ thu thập dữ liệu giao dịch thực tế, phần lớn hành vi sẽ là **suboptimal** — tức là con người không phải lúc nào cũng chọn thời điểm tốt nhất.

Để giải quyết, nhóm xây dựng một **Hindsight Oracle** — một bộ giải tối ưu toàn cục bằng **Quy hoạch Động**. Oracle này biết trước toàn bộ giá gas trong 24 giờ, nên nó có thể tìm ra chiến lược **tối ưu tuyệt đối**.

*(Chỉ vào công thức)* Bằng thuật toán Backward Pass, Oracle tính giá trị tối ưu tại mỗi trạng thái và tạo ra **quỹ đạo chuyên gia** — đây chính là nhãn huấn luyện cho agent.

*(Chỉ vào pipeline bên phải)* Toàn bộ pipeline đi từ dữ liệu thô qua Feature Engineering, Oracle DP, tính lại Queue, tính Reward, và cuối cùng tạo ra MDPDataset chuẩn."

---

## Slide 6 — Dữ liệu: 8.6 Triệu Transitions ⏱️ 1 phút

**Nói:**

"Dataset cuối cùng gồm **hơn 8.6 triệu transitions** từ **1,206 episodes**, mỗi episode tương ứng 1 ngày giao dịch trên Ethereum Mainnet.

Nhóm chia **80% train, 20% test** — tương đương 242 episodes để đánh giá.

Điểm đặc biệt là nhóm mix dataset theo tỷ lệ: **70% Oracle** tối ưu, **10% Suboptimal** làm phản ví dụ, và **20% Behavior** thực tế. Sự pha trộn này giúp agent không chỉ biết hành vi tốt mà còn phân biệt được hành vi xấu."

---

## Slide 7 — Tại sao Offline RL? Tại sao CQL? ⏱️ 1 phút 15 giây

**Nói:**

"Tại sao nhóm chọn **Offline RL** thay vì Online RL? Vì trên blockchain, mỗi lần thử-sai là **mất tiền thật**. Chúng ta không có quyền tương tác tự do với môi trường như trong game. Agent phải học hoàn toàn từ dataset tĩnh đã thu thập.

Trong các thuật toán Offline RL, nhóm chọn **Conservative Q-Learning** vì nó giải quyết được vấn đề cốt lõi: **OOD Overestimation** — khi agent đánh giá quá cao những hành động mà nó chưa từng thấy trong dataset.

*(Chỉ vào công thức CQL Loss)* Công thức Loss của CQL gồm 2 phần: **Bellman TD Loss** như Q-Learning thông thường, cộng thêm một **Conservative Penalty** với hệ số alpha. Penalty này chủ động **đè Q-value của các hành động ngoài phân bố xuống**, giúp agent an toàn hơn khi ra quyết định."

---

## Slide 8 — Kiến trúc và Huấn luyện ⏱️ 45 giây

**Nói:**

"Về kiến trúc, nhóm sử dụng một Q-Network đơn giản với 2 lớp ẩn, mỗi lớp 512 neuron, kết nối ReLU, đầu vào 11 chiều và đầu ra 5 action values.

Một điểm đáng chú ý là nhóm chỉ dùng **1 Q-network** thay vì Double Q thông thường — lý do sẽ giải thích ở phần sau.

*(Chỉ vào Training Curves bên phải)* Biểu đồ training cho thấy model hội tụ ổn định sau khoảng 500,000 steps."

---

## Slide 9 — Bảng Xếp Hạng Hiệu Năng ⏱️ 1 phút 30 giây

> **[SLIDE QUAN TRỌNG NHẤT — nói chậm và rõ]**

**Nói:**

"Đây là bảng xếp hạng hiệu năng trên toàn bộ **242 episodes** test set. Nhóm so sánh 5 chiến lược:

- **Hạng 1: Oracle** — trần tối ưu lý thuyết, tiết kiệm 38.8% so với Greedy, nhưng đây là kết quả không khả thi trong thực tế vì phải biết trước tương lai.

- *(Chỉ vào dòng highlight)* **Hạng 2: CQL kết hợp Safety Layer** — đây là model của nhóm — tiết kiệm **11.1% Gas** so với Greedy, và quan trọng nhất là **Miss Rate bằng 0%**.

- **Hạng 3: Greedy** — baseline đơn giản, gửi hết ngay lập tức.

- **Hạng 4: Smart Rule** — luật thông minh do con người viết, nhưng thực tế lại **thua cả Greedy** 4.3%.

- **Hạng 5: Decision Transformer** — thất bại hoàn toàn với Miss Rate 90–100%.

Tóm lại: model CQL của nhóm thắng Greedy trong **222 trên 242 episodes**, đạt Win Rate **91.7%**, và khai thác được **28.6%** tiềm năng tối ưu của Oracle."

---

## Slide 10 — Phân tích Chi tiết Từng Episode ⏱️ 45 giây

**Nói:**

"Biểu đồ này phân tích chi tiết từng episode. Mỗi điểm là một episode, trục Y thể hiện phần trăm tiết kiệm so với Greedy.

Điểm đáng chú ý: CQL khai thác rất mạnh ở những ngày có Gas biến động lớn — savings cao nhất đạt **79.7%**.

Và quan trọng hơn: khi CQL thua, mức thua rất nhỏ — tối đa chỉ khoảng **âm 0.2%**. Nghĩa là rủi ro downside rất thấp — đây là đặc tính rất tốt cho một hệ thống tài chính."

---

## Slide 11 — Tại sao Decision Transformer thất bại? ⏱️ 1 phút 15 giây

**Nói:**

"Phần phân tích này giải thích tại sao Decision Transformer thất bại hoàn toàn.

**Lý do thứ nhất: Majority Bias.** Decision Transformer học bằng Cross-Entropy Loss — bản chất là phân loại. Trong dataset Oracle, Action 0 (giữ hàng) chiếm tới **69.6%** — vì phần lớn thời gian, giá gas chưa đủ thấp để gửi. DT bị bias nặng, luôn đoán Action 0, kết quả là **giữ hàng mãi mãi** và bỏ lỡ 100% deadline.

CQL vượt qua được vì Q-Learning đánh giá **giá trị thực** của từng hành động, không bị ảnh hưởng bởi tần suất xuất hiện.

**Lý do thứ hai: Risk-Reward Asymmetry.** Lợi nhuận timing chỉ ở mức 10 mũ 3, trong khi hình phạt deadline lên tới 10 mũ 8 — tỷ lệ **1 chọi 223 nghìn**. Agent ban đầu sợ hãi đến mức không dám làm gì cả. Nhóm giải quyết bằng **Reward Shaping**: giảm penalty 50 lần, tăng urgency 3.5 lần, giúp agent **dám khám phá** thay vì chỉ tránh né rủi ro."

---

## Slide 12 — Kiến trúc Hệ thống Lai: RL + Safety Layer ⏱️ 1 phút 30 giây

> **[SLIDE QUAN TRỌNG — điểm nhấn thiết kế hệ thống]**

**Nói:**

"Dù CQL đã mạnh, nhưng nó vẫn có xác suất miss deadline khoảng 0.4%. Trong hệ thống tài chính, con số này là **không chấp nhận được**.

Nhóm thiết kế một **kiến trúc Lai** gồm 2 module:

*(Chỉ vào sơ đồ)* Tại mỗi bước, hệ thống kiểm tra thời gian còn lại. Nếu còn **hơn 20% thời gian** — tức 80% thời gian đầu — CQL agent toàn quyền quyết định, nó 'lướt sóng' giá gas để tìm điểm mua rẻ nhất.

Khi thời gian còn lại **dưới 20%**, Safety Layer tự động kích hoạt, ép bán hết 100% hàng đợi — bất kể giá gas cao hay thấp. Điều này đảm bảo **tuyệt đối không bỏ lỡ deadline**.

Đánh đổi là savings giảm từ 22.4% xuống 11.1%, nhưng Miss Rate giảm từ 0.4% xuống **đúng 0%**. Trong bài toán tài chính, sự đánh đổi này hoàn toàn xứng đáng."

---

## Slide 13 — Kết luận ⏱️ 1 phút

**Nói:**

"Tổng kết lại, nhóm rút ra 4 bài học quan trọng:

**Thứ nhất**, Reward Shaping là chìa khóa. Việc giảm deadline penalty 50 lần giúp agent thoát khỏi trạng thái Pessimistic Collapse.

**Thứ hai**, Data Coverage quyết định hiệu năng. Chỉ một khác biệt nhỏ 5% trong phạm vi phủ của queue đã thay đổi kết quả tới 40%.

**Thứ ba**, hệ thống Lai AI + Safety Layer là chuẩn mực cho ứng dụng thực tế, cân bằng giữa tối ưu và độ tin cậy.

**Và thứ tư**, model đạt hiệu năng **11.1% savings, 0% Miss Rate, Win Rate 91.7%**.

Về hướng phát triển, nhóm đề xuất tích hợp cấu trúc EIP-1559, xây dựng Continuous Oracle, và thử nghiệm LSTM hoặc Transformer cho bài toán dự báo chuỗi thời gian."

---

## Slide 14 — Thank you! ⏱️ 15 giây

**Nói:**

"Trên đây là toàn bộ bài thuyết trình của nhóm. Em xin cảm ơn Cô và các bạn đã lắng nghe. Kính mời Cô và các bạn đặt câu hỏi ạ."

> *[Cúi đầu nhẹ, ngồi xuống hoặc đứng chờ câu hỏi]*

---

## 🛡️ Chuẩn bị cho Câu hỏi Hội đồng

### Câu hỏi 1: "Tại sao chỉ dùng 1 Q-network (n_critics=1)?"
> **Dùng Backup Slide 15**

**Trả lời:** "Thưa Cô, trong Online RL, Double Q-Learning dùng min(Q1, Q2) để chống Overestimation Bias. Tuy nhiên, CQL đã có sẵn Conservative Penalty — nó chủ động đè Q-value xuống rồi. Nếu kết hợp thêm Double Q, sẽ xảy ra hiện tượng **bi quan kép** (Double Pessimism) — Q-value bị đè xuống 2 lần dẫn đến Pessimistic Collapse, agent sợ hãi và chỉ chọn giữ hàng mãi. Do đó, n_critics = 1 là lựa chọn **có chủ ý** để tránh hiện tượng này."

### Câu hỏi 2: "Backlog Pressure là gì? Tại sao cần feature này?"
> **Dùng Backup Slide 16**

**Trả lời:** "Backlog Pressure là một đặc trưng Autoregressive do nhóm tự thiết kế. Nó tích lũy áp lực tắc nghẽn qua các bước thời gian thông qua hệ số decay rho = 0.95, tương đương half-life 14 blocks. Feature này giúp agent 'cảm nhận' được các đợt sóng tắc nghẽn — như khi có NFT drop — mà **không cần mạng LSTM hay Transformer** phức tạp."

### Câu hỏi 3: "Làm sao chứng minh Reward Shaping có tác dụng?"
> **Dùng Backup Slide 17**

**Trả lời:** "Nhóm có một Natural Experiment: so sánh V33 (deadline penalty 1 triệu) và V32 (deadline penalty 20 nghìn). V33 thua Greedy hơn 30%, trong khi V32 thắng 11.1%. Sự khác biệt chính chỉ nằm ở Reward Shaping — giảm penalty 50 lần giúp agent dám khám phá thay vì chỉ tránh né rủi ro."

### Câu hỏi 4: "Tại sao Decision Transformer không hoạt động?"

**Trả lời:** "Có 2 lý do chính. Thứ nhất, DT dùng Cross-Entropy Loss nên bị Majority Bias — Action 0 chiếm 69.6% dataset khiến DT luôn đoán giữ hàng. Thứ hai, bài toán có Risk-Reward Asymmetry cực đoan — tỷ lệ 1:223,000 giữa lợi nhuận và hình phạt. CQL vượt qua nhờ Q-Learning đánh giá giá trị thực của hành động, không bị ảnh hưởng bởi tần suất."

### Câu hỏi 5: "Safety Layer 20% có cơ sở gì?"

**Trả lời:** "Nhóm đã thử nghiệm nhiều ngưỡng khác nhau. 20% là điểm cân bằng tối ưu — nếu thấp hơn (10%) thì vẫn có risk miss deadline, nếu cao hơn (30%) thì agent không đủ thời gian 'lướt sóng' để tiết kiệm. Tại 20%, savings đạt 11.1% và Miss Rate chính xác bằng 0%."

### Câu hỏi 6: "So với các nghiên cứu khác thì kết quả này thế nào?"

**Trả lời:** "Đa phần các nghiên cứu về gas optimization tập trung vào dự báo giá gas (price prediction), không giải bài toán scheduling dưới ràng buộc deadline. Đề tài của nhóm là một trong những nghiên cứu đầu tiên áp dụng **Offline RL** cho bài toán gas timing optimization với ràng buộc Zero-Miss SLA, đạt kết quả cạnh tranh: khai thác được 28.6% tiềm năng của Oracle toàn tri."

---

## ⏱️ Phân bổ Thời gian Tổng hợp

| Phần | Slides | Thời gian |
|------|--------|-----------|
| Mở đầu (Bìa + Mục lục) | 1–2 | 50 giây |
| Bài toán | 3–4 | 2 phút 45 giây |
| Phương pháp | 5–8 | 4 phút 15 giây |
| Kết quả | 9–10 | 2 phút 15 giây |
| Phân tích | 11–12 | 2 phút 45 giây |
| Kết luận + Thank you | 13–14 | 1 phút 15 giây |
| **Tổng** | **14** | **~14 phút** |
