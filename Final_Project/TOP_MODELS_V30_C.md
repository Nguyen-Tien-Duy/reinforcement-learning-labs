# Danh sách các Model xuất sắc nhất (0.0% Miss Rate) - Sàng lọc lần 1

Dựa trên kết quả đánh giá 10 episodes (Screening test), dưới đây là Top các mô hình đạt độ ổn định tuyệt đối (0.0% Miss Rate) được sắp xếp theo Chi phí (True Cost) từ thấp đến cao để chuẩn bị cho đợt test trên tập Holdout 20%.

> **Baseline Oracle (Expert) Cost:** 129,231

## Top 10 Models Tốt Nhất

1. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_80000.d3` (Cost: 133,246)
2. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_140000.d3` (Cost: 133,357)
3. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_50000.d3` (Cost: 134,374)
4. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_150000.d3` (Cost: 135,020)
5. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_60000.d3` (Cost: 135,041)
6. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_180000.d3` (Cost: 135,054)
7. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_200000.d3` (Cost: 135,183)
8. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_260000.d3` (Cost: 135,330)
9. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_70000.d3` (Cost: 136,190)
10. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_110000.d3` (Cost: 136,401)

## Danh sách các Model 0% Miss Rate khác (Dự phòng)
11. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_90000.d3` (Cost: 136,750)
12. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_170000.d3` (Cost: 136,836)
13. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_100000.d3` (Cost: 138,181)
14. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_130000.d3` (Cost: 138,462)
15. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_220000.d3` (Cost: 138,529)
16. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_240000.d3` (Cost: 138,917)
17. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_120000.d3` (Cost: 139,608)
18. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_270000.d3` (Cost: 139,648)
19. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_190000.d3` (Cost: 140,508)
20. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_250000.d3` (Cost: 140,830)
21. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_160000.d3` (Cost: 141,169)
22. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_230000.d3` (Cost: 142,904)
23. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_210000.d3` (Cost: 142,946)
24. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_350000.d3` (Cost: 144,382)
25. `d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_430000.d3` (Cost: 144,504)

---

## Lệnh Evaluate trên tập Holdout (150 Episodes ~ 20% Test)

Để chạy đánh giá chuyên sâu cho **toàn bộ 25 models xuất sắc nhất** (0% Miss Rate) với hiệu suất tối đa, bạn có thể copy nguyên khối lệnh dưới đây:

```bash
nohup env LD_PRELOAD="/opt/intel/oneapi/mkl/2026.0/lib/libmkl_rt.so.3:/usr/lib/libmimalloc.so" \
taskset -c 0,1,2,3 ./venv/bin/python Final_Project/visualize/leaderboard_v28.py \
    --models \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_80000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_140000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_50000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_150000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_60000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_180000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_200000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_260000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_70000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_110000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_90000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_170000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_100000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_130000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_220000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_240000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_120000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_270000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_190000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_250000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_160000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_230000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_210000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_350000.d3 \
        d3rlpy_logs/DiscreteCQL_V6_20260427_1845_20260427184554/model_430000.d3 \
    --data Final_Project/Data/transitions_v30_C.parquet \
    --episodes 150 > eval_v30_C_all25_holdout.log 2>&1 &
```

> **Giải thích:**
> Lệnh này sẽ sử dụng file `leaderboard_v28.py` (đã được sửa lỗi Nén Kép và cực kỳ tối ưu) để đánh giá **toàn bộ 25 models** trên **150 episodes** (khoảng 20% Data Holdout). Mọi thứ đều được chạy đa luồng trên 4 nhân thực của máy.
