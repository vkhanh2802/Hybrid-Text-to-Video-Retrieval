# Protocol chạy cross-encoder CPU — chốt trước inference

## Phạm vi và cấu hình

Người dùng đã cho phép tải checkpoint và chạy CPU smoke/development. Không dùng
GPU/VLM, không gọi lại retrieval, API planner, Drive, không rebuild/index/caption.

- Model `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Download chỉ config/tokenizer/safetensors; inference cache-only, CPU FP32.
- Giữ nguyên module đã triển khai: pool100, batch16, max_pairs1200, max_chunks4,
  chunk_chars1200, max_length512, query_chars1600; CPU4 threads, interop1.
- Giữ scored-slot permutation, rank ladder, max-chunk/min-event aggregation;
  không chọn lại threshold, pool hoặc prompt sau kết quả.
- Production vẫn mode off. Không có automatic promotion dù điểm tăng.

## Trình tự

1. Smoke bằng3 synthetic pairs + batch16; kiểm tra finite raw logits, lặp lại,
   model ở CPU, thiếu caption và lỗi injected giữ fallback. Synthetic scores
   không phải bằng chứng chất lượng retrieval hoặc nhãn mới.
2. Prepare/freeze development trước khi chấm candidate thật: **15 query units
   (11 ảnh,4 chuỗi) ×3 retrieval arms =45 cases**, đủ mọi denominator.
3. Replay chính xác candidate/plan từ `full_pipeline_ablation_20261102` với
   baseline đã lưu và text rerank. Không rerun solve_kis/retrieval/DP/encoder.
4. Chỉ sau đọc kết quả dev, nếu có tín hiệu hữu ích có thể chạy human-v3 bằng
   cấu hình/code không đổi như **observed regression**, không dùng để retune.
5. Không chạy VLM mới trong protocol này; không reuse điểm VLM cho shortlist khác.

## Provenance và cách đo

- CPU cases kiểm tra lại fingerprint theo private bindings của packet đã công bố;
  qrels hash phải khớp protocol gốc, baseline metrics phải tính lại khớp.
- Source code mới có thể khác code đã tạo CPU cache do đã thêm text reranker.
  Không sửa fingerprint của run cũ để làm như đã đo lại retrieval hiện tại.
- Giữ inline captions gốc. Temporal/missing text lấy từ caption artifacts có
  checksum lịch sử khớp; đóng băng lookup trước inference. Thiếu caption vẫn unknown.
- Freeze protocol, source snapshot, model file hashes, raw input hashes, prepared
  inputs trước run; kiểm tra lại sau. Không ghi đè output và không sửa qrels.
- Metrics: known-frame/path Hit@1/5/20, image MRR@20, target-video top1 phụ,
  paired wins/losses và case IDs, coverage/errors, thời gian chấm từng case.
- Không lọc query score lỗi/thiếu; giữ ranking baseline cho candidate không chấm.
- Không memoize score giữa arms, báo work thực tế. Cold first case gồm model load;
  timing không gồm retrieval/planner/caption preparation, không phải end-to-end.

## Giới hạn và quyết định

Dev là bộ nhỏ AI quan sát ảnh đã dùng nhiều lần; không phải human-independent
holdout. Ba arms dùng chung query units, không coi là45 thử nghiệm độc lập.
Human-v3 đã quan sát, chỉ regression. Qrels thưa: kết quả ngoài nhãn không tự thành
negative. Caption/logit không xác nhận pixels, temporal support, chuyển động hay
quan hệ nhân quả. Text reranker không sửa candidate pool/gate recall.

Giữ production off; chọn triển khai tiếp/kiểm tra regression dựa trên mô tả dev,
không tuyên bố hiệu quả khái quát hoặc chỉnh tham số theo human-v3.
