# Qwen batch độc lập và benchmark 4/8/12/20

## Cách chạy

`VLMReranker.score_batch` xử lý **mỗi ảnh là một conversation riêng**, với đúng
query/caption/OCR của ảnh đó. Processor nhận B text và B ảnh, model `generate`
trả B output. Không ghép collage hoặc yêu cầu một câu trả lời chung cho 20 ảnh.
Decoder-only tokenizer dùng **left padding** khi các prompt dài ngắn khác nhau.

- Giữ nguyên pinned model/revision, NF4 double quant, compute float16, SDPA,
  pixel limits và prompt production. `max_new_tokens=8`, greedy/no sampling.
- Pipeline chỉ xét **20 candidate đầu**, không lấy ảnh rank 21 trở đi để bù ảnh
  thiếu. Batch thực tế có thể nhỏ hơn 20 vì missing image hoặc nguồn ảnh lỗi.
- Caption, OCR và điểm đầu ra giữ cùng index; malformed output vẫn null/invalid,
  không bịa 0 hoặc điểm trung bình để làm batch pass.
- Chỉ dùng batching cho scorer khai báo `supports_batch is True`; scorer cũ/test
  vẫn dùng interface scalar. Qwen production và LazyVLM đã hỗ trợ batch.
- Giữ stream ảnh mở qua batch, đóng toàn bộ sau chấm; không cache ảnh runtime
  lên disk. Dữ liệu ảnh phục vụ benchmark được freeze riêng, không sửa nguồn.

Batch size mặc định 20, có thể chọn qua `KIS_VLM_BATCH_SIZE` (1–20). Launcher
`start_school.ps1` đọc thêm `vlm_batch_size` trong config private nếu được chọn
sau benchmark. Đây là kích thước microbatch, không phải thay budget candidate.

## Khi thiếu VRAM

Production có thể chia đôi batch bị CUDA OOM và thử lại các subbatch, ghi rõ
`attempts`, `oom_splits`, `effective_batches`. Một ảnh vẫn OOM ở size 1 được giữ
unknown với lỗi, không bỏ âm thầm hoặc chấm bù candidate khác. Chỉ lỗi OOM được
chia nhỏ; batch return sai số phần tử là error, không retry scalar che lỗi.

Trong benchmark so size, **`oom_fallback=False`**: batch OOM là không chạy được,
không được lấy thời gian subbatch để gọi là thành tích batch 20.

## Protocol timing

`tools/benchmark_qwen_batch_sizes.py`:

1. Chọn 20 ảnh distinct có sẵn từ output query công khai cũ, trước inference;
   không dùng đề DRES hoặc score labels để chọn. Giữ bytes/ID/caption và SHA-256.
2. Bài chọn timing packet không thay chính sách production top20/no-backfill.
3. Web tạm dừng theo xác nhận người dùng để chỉ một bản Qwen giữ GPU 6GB;
   worker và SSH tunnel vẫn giữ. Phiên DRES RAM mất, cần đăng nhập lại sau đó.
4. Nạp một model duy nhất, tách thời gian load. Mỗi size 1/4/8/12/20 có warm-up
   20 ảnh riêng. Ba vòng timed đổi thứ tự size để giảm thiên lệch nhiệt/cache.
5. Mỗi call vẫn chấm **cùng 20 ảnh**: size 4 = 5 microbatch; size 8 = 8+8+4;
   size 12 = 12+8; size 20 = một batch.
6. Đo CUDA-synchronized wall, peak allocated/reserved VRAM, score hợp lệ, OOM
   và agreement so với scalar. Không chạy suite/job phụ trong lúc đo GPU.
7. Chỉ ba lượt đủ 20 score hợp lệ mới đủ điều kiện về mặt validity; điểm số
   trùng scalar không đồng nghĩa relevance đúng. Không cam kết optimum toàn corpus.
8. Thời gian đo gồm mở/giải mã packet ảnh local và processor/generate/decode,
   không gồm retrieval, Groq, VPN tải ảnh hoặc browser rendering.

Không giảm resolution/token budget, đổi prompt/model/precision để làm một size
thắng; nếu cần thay những thứ đó phải thành experiment riêng.

## Kiểm thử và kết quả

Full suite cuối: **956 tests — 940 pass, 16 skip**, 155,195 giây.
Bao gồm strict OOM vs fallback, chunk sizes thật, mỗi conversation/score riêng,
left padding, OCR prompt chung và giữ budget/no-backfill trong pipeline.

Benchmark thật đã hoàn tất tại
`benchmark_results/qwen_batch_sizes_20261115_run03/`. Run01 chỉ có packet chuẩn bị
trước khi sửa fixture prompt hook; chưa chạy GPU. Run02 bị Windows native crash
`torch_cpu.dll`, exception `c0000005`, trong lúc load weight (~19%), trước bất kỳ
batch nào; không dùng làm số đo OOM hoặc tốc độ batch. Run03 dùng một CPU thread
chỉ khi load, phục hồi bốn thread trước warm-up/timing, cùng model/pixel/prompt
và cùng bytes của 20 ảnh. Có faulthandler, process log và exit receipt riêng;
không khẳng định đã xác định root cause của native crash chỉ từ một lượt thành công.

Median chấm 20 ảnh: scalar **27,51s**, batch4 **17,65s**, batch8 **17,15s**,
batch12 **16,49s**, batch20 **16,52s**. Cả 300 timed scores hợp lệ, không OOM.
Size12 và20 gần như hòa, nhưng12 peak reserved4,11GiB thay vì5,29GiB, nên cấu
hình school runtime hiện chọn **12**. Có một ảnh ở size12 đổi điểm7→3 so scalar;
size20 trùng cả20 điểm trong packet. Không coi validity là relevance equivalence.

Web đã chạy lại, Qwen đã warm qua query public. Một warm end-to-end sample
**22,46s** (18 ảnh có sẵn, generate12+6), không phải median hoặc A/B giữ planner
output cố định. Sau restart người dùng cần đăng nhập DRES lại.

[Báo cáo đầy đủ](../benchmark_results/qwen_batch_sizes_20261115_run03/SUMMARY.md).
