# Kế Hoạch Đăng Paper ICAIF 2026

## 1. Phân Loại Hệ Thống Journal/Conference

Bảng phân loại cho lĩnh vực **AI/ML + Finance/Blockchain**, xếp từ cao đến thấp.

### Tier S — "Nobel Prize" của CS (Accept rate 15-25%)

| Venue | Loại | Đặc điểm | Dự án của mình? |
|---|---|---|---|
| **NeurIPS** | Conference | ML/RL lý thuyết nặng, cần proof toán | ❌ Thiếu đóng góp thuật toán mới |
| **ICML** | Conference | Tương tự NeurIPS | ❌ |
| **ICLR** | Conference | Thực nghiệm mạnh, vẫn cần novelty lý thuyết | ❌ |
| **Nature / Science** | Journal | Cần câu chuyện "thay đổi thế giới" | ❌ |

### Tier A — Rất uy tín, thực tế hơn (Accept rate 20-30%)

| Venue | Loại | Đặc điểm | Dự án của mình? |
|---|---|---|---|
| **AAAI** | Conference | AI tổng quát, chấp nhận ứng dụng nếu có insight | ⚠️ Khó, cần thêm ablation |
| **IJCAI** | Conference | Tương tự AAAI, hơi dễ hơn | ⚠️ |
| **KDD** | Conference | Data Mining + ứng dụng thực tế | ⚠️ |
| **Nature Machine Intelligence** | Journal | ML ứng dụng high-impact | ❌ |

### Tier B — Ứng dụng chuyên ngành (Accept rate 25-40%) ✅ TARGET

| Venue | Loại | Đặc điểm | Dự án của mình? |
|---|---|---|---|
| **ICAIF** (ACM) | Conference | **AI in Finance** — đúng 100% topic | ✅ **TARGET #1** |
| **IEEE Blockchain** | Conference | Blockchain + AI | ✅ Phù hợp |
| **AAMAS** | Conference | Multi-agent, có track RL ứng dụng | ⚠️ |
| **Expert Systems with Applications** | Journal (Q1) | IF ~8, chấp nhận applied work | ✅ Backup |
| **Knowledge-Based Systems** | Journal (Q1) | IF ~8 | ✅ Backup |
| **IEEE Access** | Journal (Q1/Q2) | Open access, review nhanh | ✅ Dễ nhất |

### Tier C — Workshop tại conference lớn (Accept rate 40-60%)

| Venue | Loại | Đặc điểm | Dự án của mình? |
|---|---|---|---|
| **NeurIPS Workshop** (RL4RealLife, SafeRL) | Workshop | Muốn thấy RL ứng dụng thực tế, kết quả khiêm tốn nhưng trung thực | ✅ |
| **ICML Workshop** (AI4Finance) | Workshop | Tương tự | ✅ |
| **AAAI Workshop** (AI in FinTech) | Workshop | Dễ nhất, paper 4-6 trang | ✅ Backup tốt |

### Tier D — Tạp chí "dễ thở" (Accept rate 40-60%)

| Venue | Loại | Đặc điểm |
|---|---|---|
| **Scientific Reports** (Nature) | Journal | Có chữ "Nature" nhưng accept ~50% |
| **PLOS ONE** | Journal | Chỉ cần methodology đúng |
| **Applied Sciences (MDPI)** | Journal | Open access, review nhanh |
| **Frontiers in AI** | Journal | Tương tự |

> [!WARNING]
> Scientific Reports thuộc nhà Nature nhưng chất lượng rất dao động. Nhiều bài "không hay" lên Nature là do publish ở sub-journal này, không phải Nature chính.

---

## 2. ICAIF 2026 — Thông Tin Chi Tiết

### Logistics

| Hạng mục | Chi tiết |
|---|---|
| **Tên đầy đủ** | 7th ACM International Conference on AI in Finance |
| **Địa điểm** | Milan, Italy (Bocconi University) |
| **Thời gian** | 14-17 tháng 11, 2026 |
| **Deadline submit** | **2 tháng 8, 2026** (11:59 PM AoE) |
| **Hệ thống submit** | [Microsoft CMT](https://cmt3.research.microsoft.com/ICAIF2026/) |
| **Website** | [icaif2026.org](https://icaif2026.org/) |

### Topic phù hợp (trích từ Call for Papers)

ICAIF '26 liệt kê rõ ràng các track sau — dự án của mình khớp với **ít nhất 2 track**:
- **"AI Agents & Reinforcement Learning"** — RL, sequential decision-making, meta-learning
- **"Blockchain & DeFi"** — Blockchain technology, cryptocurrency, decentralized finance

### Chi phí

| Hạng mục | Giá | Ghi chú |
|---|---|---|
| **Submit paper** | **$0** | Hoàn toàn miễn phí |
| **Publication fee (APC)** nếu accepted | ~$700-1000 | Chỉ trả khi paper được chấp nhận |
| **APC sau giảm giá** (VN = Lower-Middle-Income) | **~$350-500** | Tự động giảm 50% |
| **APC nếu apply Financial Hardship Waiver** | **$0** | Cần apply sau khi accept |
| **Registration (Student, In-Person)** | ~$300-350 | Tham khảo ICAIF 2025 |
| **Registration (Student, Virtual/Online)** | **~$100** | Trình bày online |

> [!TIP]
> **Chiến lược chi phí tối thiểu:**
> 1. Submit miễn phí ($0)
> 2. Nếu accepted → Apply ACM Financial Hardship Waiver cho APC ($0)
> 3. Đăng ký trình bày Virtual ($100)
> 4. **Tổng chi phí: ~$100 (~2.5 triệu VNĐ)**

> [!NOTE]
> Việt Nam được World Bank xếp hạng **Lower-Middle-Income Country**, nên tự động được giảm 50% phí APC. Ngoài ra, ACM có chính sách discretionary waiver cho trường hợp khó khăn tài chính — cần apply sau khi paper được accept.

---

## 3. Literature Review — 13 Papers Cần Đọc

### Mảng 1: Offline RL — Nền tảng lý thuyết

| # | Paper | Tác giả | Năm | Nguồn | Vai trò trong paper của mình |
|---|---|---|---|---|---|
| 1 | **Offline RL: Tutorial, Review, and Perspectives on Open Problems** | Levine, Kumar, Tucker, Fu | 2020 | [arXiv:2005.01643](https://arxiv.org/abs/2005.01643) | **BẮT BUỘC.** Survey gốc định nghĩa Offline RL, distribution shift, extrapolation error. Dùng trong phần Introduction + Related Work |
| 2 | **Conservative Q-Learning for Offline RL (CQL)** | Kumar, Zhou, Tucker, Levine | 2020 | [arXiv:2006.04779](https://arxiv.org/abs/2006.04779) | **BẮT BUỘC.** Paper gốc thuật toán CQL mà mình dùng. Giải thích conservative penalty, lower-bound Q-value |
| 3 | **Off-Policy Deep RL without Exploration (BCQ)** | Fujimoto, Meger, Precup | 2019 | [arXiv:1812.02900](https://arxiv.org/abs/1812.02900) | So sánh CQL vs BCQ. Giải thích tại sao chọn CQL (discrete action space phù hợp hơn) |
| 4 | **Decision Transformer: RL via Sequence Modeling** | Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas, Mordatch | 2021 | [arXiv:2106.01345](https://arxiv.org/abs/2106.01345) | **BẮT BUỘC.** Giải thích tại sao DT thất bại trong bài toán này (Majority Bias từ class imbalance — Action 0 chiếm 69.6%) |

### Mảng 2: Ethereum & Gas Fee Mechanism

| # | Paper | Tác giả | Năm | Nguồn | Vai trò |
|---|---|---|---|---|---|
| 5 | **Transaction Fee Mechanism Design for Ethereum: An Economic Analysis of EIP-1559** | Roughgarden | 2021 | [arXiv:2012.00854](https://arxiv.org/abs/2012.00854) | **BẮT BUỘC.** Lý thuyết kinh tế về gas fee mechanism. Giải thích tại sao gas price dao động và không thể predict chính xác |
| 6 | **EIP-1559: Fee market change for ETH 1.0 chain** | Buterin, Conner, Dudley, Slipper, Norden | 2019 | [EIP-1559](https://eips.ethereum.org/EIPS/eip-1559) | Tài liệu kỹ thuật gốc. Cite khi mô tả base fee, priority fee |
| 7 | **Analysis of Information Propagation in Ethereum Network Using GAT and RL** | Behfar, Crowcroft | 2023 | [arXiv:2311.01406](https://arxiv.org/abs/2311.01406) | Paper duy nhất trên arxiv kết hợp Ethereum + RL. So sánh: họ optimize network-level (gas limit), mình optimize application-level (transaction timing) |

### Mảng 3: Optimal Execution — Bài toán tương đồng nhất

| # | Paper | Tác giả | Năm | Nguồn | Vai trò |
|---|---|---|---|---|---|
| 8 | **Optimal Execution of Portfolio Transactions** | Almgren, Chriss | 2001 | Journal of Risk, 3(2) | **KINH ĐIỂN.** Định nghĩa bài toán "optimal execution" (timing + cost + deadline constraint). Bài toán Gas scheduling của mình là phiên bản blockchain của bài toán này |
| 9 | **A Reinforcement Learning Approach to Optimal Execution** | Ning, Lin, Jaimungal | 2021 | [arXiv:2006.02936](https://arxiv.org/abs/2006.02936) | RL cho optimal trade execution. So sánh: họ dùng Online RL + Simulated market, mình dùng Offline RL + Real Ethereum data |
| 10 | **Universal Trading for Order Execution with Oracle Policy Distillation** | Fang, Li, Wei, He, Wang | 2021 | AAAI 2021 | **RẤT GIỐNG MÌNH.** Họ cũng dùng Oracle (Hindsight) để tạo expert data rồi distill vào RL agent. So sánh methodology trực tiếp |

### Mảng 4: Safe RL & Reward Shaping

| # | Paper | Tác giả | Năm | Nguồn | Vai trò |
|---|---|---|---|---|---|
| 11 | **Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping** | Ng, Harada, Russell | 1999 | ICML 1999 | **Paper gốc Reward Shaping.** Justify toán học cho việc tune reward (giảm deadline penalty 50×, tăng urgency) |
| 12 | **A Comprehensive Survey on Safe Reinforcement Learning** | García, Fernández | 2015 | JMLR 16(1) | Survey Safe RL. Justify tại sao dùng Safety Layer (rule-based hard constraint) thay vì constrained optimization |
| 13 | **Recovery RL: Safe RL with Learned Recovery Zones** | Thananjeyan, Balakrishna, Nair, Luo, Srinivasan, Hwang, Gonzalez, Goldberg | 2021 | [arXiv:2010.15920](https://arxiv.org/abs/2010.15920) | Approach khác cho Safe RL. So sánh: họ dùng learned safety (neural net), mình dùng rule-based safety (đơn giản, có mathematical guarantee) |

---

## 4. So Sánh Kết Quả Với Các Paper Liên Quan

| Paper / Approach | Bài toán | Kết quả vs Baseline | Safety | Data |
|---|---|---|---|---|
| **DRL Optimal Execution (2025)** | Timing mua/bán cổ phiếu | **+0.17% vs TWAP** | Không | Simulated |
| **FPPO (IEEE, 2025)** | Portfolio execution | **+3% ~ 15%** | Không | Simulated |
| **Behfar & Crowcroft (2023)** | Ethereum gas limit optimization | "Superior results" (ko có % cụ thể) | Không | Ethereum data |
| **Dự án của mình** | **Ethereum gas transaction timing** | **+9.5% vs Greedy** | **0% Miss Rate** | **Ethereum mainnet 8 tháng, 1206 episodes** |

> [!IMPORTANT]
> **Selling points chính của paper:**
> 1. **Bài toán mới:** Transaction scheduling as Offline MDP — chưa ai làm
> 2. **Dữ liệu thật:** 8.6 triệu transitions từ Ethereum mainnet (không phải simulated)
> 3. **Phân tích thất bại:** DT (Majority Bias), Risk-Reward Asymmetry (1:223,000)
> 4. **Hybrid Architecture:** CQL + Rule-based Safety Layer → 0% Miss Rate
> 5. **Honest Evaluation:** Hold-out Validation loại bỏ Selection Bias, báo cáo Generalization Gap (24.2% → 9.5%)

---

## 5. Timeline Hành Động

| Thời gian | Công việc | Output |
|---|---|---|
| **10/5 → 31/5** | Đọc 13 papers, ghi chú key findings | Literature review notes |
| **1/6 → 15/6** | Viết Literature Review + Introduction (3-4 trang) | Sections 1-2 của paper |
| **15/6 → 30/6** | Viết Methodology + Experiments (4-5 trang) | Sections 3-4 |
| **1/7 → 15/7** | Chạy thêm ablation study nếu cần (ví dụ: adaptive safety threshold, model ensemble) | Bảng kết quả bổ sung |
| **15/7 → 25/7** | Viết Results + Conclusion, polish toàn bộ paper | Draft hoàn chỉnh |
| **25/7 → 1/8** | Nhờ GVHD review, proofread tiếng Anh | Final version |
| **2/8** | **SUBMIT** trên [Microsoft CMT](https://cmt3.research.microsoft.com/ICAIF2026/) | 🎉 |

> [!CAUTION]
> Deadline là **2 tháng 8, 2026** — chỉ còn **~3 tháng**. Cần bắt tay vào đọc paper ngay lập tức.

---

## 6. Cấu Trúc Paper Đề Xuất (ACM Format, 8-10 trang)

```
1. Introduction (1 trang)
   - Bài toán Gas scheduling trên Ethereum
   - Tại sao Online RL không khả thi (mất tiền thật)
   - Đóng góp: Offline RL + Safety Layer + Honest Evaluation

2. Related Work (1.5 trang)
   - Offline RL (CQL, BCQ, DT)
   - Ethereum gas mechanism (EIP-1559)
   - Optimal execution (Almgren-Chriss, RL approaches)
   - Safe RL

3. Problem Formulation (1.5 trang)
   - MDP formulation (State 11D, Action Discrete(5))
   - Hindsight Oracle (DP Solver)
   - Reward function design + Reward Shaping justification

4. Method (2 trang)
   - Data pipeline (Feature Eng → Oracle → Queue Recompute → Reward)
   - CQL training configuration
   - Safety Layer architecture

5. Experiments (2 trang)
   - Dataset description (1206 eps, 8.6M transitions)
   - Hold-out Validation protocol (Train/Val/Test split)
   - Results table (Leaderboard)
   - Per-episode analysis
   - Failure analysis (DT, Risk Asymmetry)

6. Conclusion (0.5 trang)
```
