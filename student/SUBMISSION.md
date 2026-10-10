# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Thanh Hòa
- MSSV: 2A202602559
- Email: 26ai.hoant@vinuni.edu.vn
- Link repo (fork): https://github.com/Jikay-070203/K4-L2L3-DAY23-NguyenThanhHoa-2A202602559-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): nộp trên LMS (hash của commit cuối cùng chứa file này)

## Tóm tắt kết quả

- `fusion_mode`, `frames`, `segment`, `seed`: compare, [0, 198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.970, 0.700, 519/16/222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.150 m, 502, 11.344, 0, 239, 2.523
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.136 m, 502, 9.267, 0, 239, 2.523
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: RMSE LiDAR = 0.150 m, fused = 0.136 m (fused − lidar = -0.014 m; cổng ≤ 0.05 m: đạt). Camera update làm RMSE giảm 0.014 m so với chỉ LiDAR. RMSE chỉ tính trên các cặp confirmed track–GT được ghép, nên phải đọc cùng số ghép: LiDAR có 502 cặp, fused có 502 cặp; ghost lần lượt 0 và 0; miss lần lượt 239 và 239. Với `precision_track = matches/(matches+ghost)` và `coverage = matches/det_tp` (det_tp = 519): LiDAR 1.000 / 0.967, fused 1.000 / 0.967 (ngưỡng đầy điểm 0.75 / 0.70). Số miss (239) chủ yếu do detector bỏ sót: detection.fn = 222, phần còn lại 17 là xe đã được detector phát hiện nhưng chưa có confirmed track ghép (trễ xác nhận vài frame đầu của mỗi track), nên không phải lỗi gán của tracker (coverage đã chỉ tính trên det_tp). Việc tạo, xác nhận và xoá track chỉ dựa vào LiDAR và detection dùng chung cho hai mode, nên số ghost/miss giữa hai mode gần như chỉ khác khi camera làm trạng thái EKF thay đổi (ví dụ vị trí track lệch ra ngoài hoặc vào cổng 2 m khi ghép GT). Giới hạn: đo camera là tâm hộp 2D ground-truth cộng nhiễu theo seed, không phải camera detector, nên kết quả fused không chứng minh chất lượng một perception độc lập.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H)

1. **Đo lidar 3D và camera 2D trong EKF.** Lidar: `z = (x, y, z)` tính bằng mét (tâm hộp 3D), `R = diag(0.1², 0.1², 0.1²)`, `h(x)` tuyến tính (đổi vị trí xe sang hệ cảm biến) nên `H` là ma trận 3×6 cố định (phần quay, cột vận tốc bằng 0). Camera: `z = (u, v)` tính bằng pixel, `R = diag(5², 5²)` px², `h(x)` là phép chiếu pinhole phi tuyến `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` nên `H` là Jacobian 2×6 (đạo hàm chiếu nhân ma trận quay) và phải tính lại quanh trạng thái hiện tại. Camera chỉ có 2 chiều nên không quan sát được độ sâu; lidar cho vị trí 3D trực tiếp (`kalman.py`, `camera_fusion.py`).
2. **Vì sao cần gating Mahalanobis trước khi gán.** Khoảng cách `d² = γᵀS⁻¹γ` với `S = HPHᵀ + R` chuẩn hoá sai số đo theo độ bất định của track và của cảm biến, nên khi `P` lớn thì chấp nhận đo xa hơn, khi `P` nhỏ thì khắt khe hơn, còn Euclid coi mọi hướng và mọi track như nhau. Cổng χ² (`chi2.ppf(0.995, dim_meas)`, dim 3 → 12.84, dim 2 → 10.60) loại các cặp phi lý trước khi greedy chọn cặp nhỏ nhất, tránh EKF bị kéo bởi đo của xe khác hoặc nhiễu (`association.py`: `chi2_gate`, `association_cost_matrix`).
3. **Track-then-fuse hay fuse-then-track.** Là **track-then-fuse**: chỉ một danh sách track; mỗi frame predict một lần, rồi gán + update LiDAR (AssocL), rồi gán + update camera (AssocC) trên cùng track; không trộn dữ liệu thô trước khi detect. Trên log `grade_run.log`, mỗi `(mode, frame)` có một record; các trường detection (`det_tp/fp/fn`, `valid_gt`) giống nhau giữa `lidar` và `fused` vì detector dùng chung, chỉ phần tracking (`confirmed`, `matches`, `ghosts`, `misses`, `sum_sq_err`) khác nhau — camera chỉ tinh chỉnh trạng thái track.
4. **Camera lệch calibration.** `h(x)` chiếu sai nên innovation `γ = z − h(x)` có sai số hệ thống (trung bình khác 0, lớn dần theo độ lệch và khoảng cách), `S` không đổi nên `d²` tăng. Nếu `d²` vượt cổng χ² thì cặp bị loại và camera không còn tác dụng; nếu chưa vượt, EKF bị kéo lệch nên vị trí track và RMSE xấu đi. Đây là phân tích lý thuyết, chưa đo trong bài này (không làm bonus).
5. **Vì sao cần `sensor` tường minh ở frame rỗng; vai trò lidar vs camera.** Khi `meas_list` rỗng không còn `meas.sensor` để biết đây là lượt LiDAR hay camera, trong khi `manage_tracks` phải biết: lượt LiDAR rỗng vẫn phải trừ score các track nằm trong FOV (miss) và xoá track hết score, nếu không track ma sống mãi. Chỉ LiDAR quyết định score/khởi tạo/xoá; camera chỉ gọi `ekf_update` (không `update_track_score`, không tạo, không xoá), vì camera ở lab là hộp 2D ground-truth cộng nhiễu, chỉ 2 chiều, không phải detector độc lập, và để nó quyết định tồn tại sẽ làm track tồn tại nhờ nhãn (`manager.handle_updated_track`, `manage_tracks`).
6. **Xác nhận, giữ confirmed, xoá track.** Khởi tạo: `score = 1/window = 1/6`, `state = initialized`. Mỗi lần LiDAR hit cộng `1/6` (tối đa 1), miss trong FOV trừ `1/6`. `score > 0.8` thì `confirmed` (score khởi tạo 1/6 nên cần thêm 4 hit LiDAR liên tiếp, tức 5 lần LiDAR thấy gồm cả lần sinh track: 5/6 ≈ 0.833); chưa confirmed mà hit thì `tentative`; đã `confirmed` thì một miss không hạ trạng thái. Xoá khi `P[0,0]` hoặc `P[1,1] > max_P = 9`, hoặc confirmed có `score < 0.6` (từ 1.0 cần 3 miss liên tiếp: 0.5), hoặc chưa confirmed có `score <= 0` (`track_management.py: should_delete_track`).

## Bonus (không bắt buộc)

- Không

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Claude Code, Anthropic)
- Dùng cho phần nào (hàm, câu hỏi, debug): viết code Part E–H (`kalman.py`, `association.py`, `camera_fusion.py`, `track_management.py`), tạo notebook chạy Kaggle, và soạn nháp báo cáo này (phần số liệu được sinh tự động từ `metrics.json`).
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): chạy `pytest student/tests` (toàn bộ pass), chạy `fusion-run-lab --fusion compare --seed 0` trên toàn segment và dùng `metrics.json`/`grade_run.log` sinh ra, đối chiếu công thức EKF/Mahalanobis/pinhole với tài liệu và test.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
