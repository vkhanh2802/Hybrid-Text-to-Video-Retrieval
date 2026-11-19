# Benchmark pipeline mở rộng, gồm VLM — thiết kế và cách chạy

> **Cập nhật03/11/2026:** đã nhận ZIP GPU thật và hoàn tất replay. Kết quả:
> [VLM_KAGGLE_RESULTS_20261103.md](VLM_KAGGLE_RESULTS_20261103.md). Các mục nói
> “chưa có VLM” bên dưới ghi lại trạng thái tại lúc thiết kế/chạy CPU, không
> phải trạng thái hiện tại. Không cần chạy lại Kaggle cho packet này.

## Mục tiêu và trạng thái

Kiểm tra cải thiện ở cấp retrieval/ranking thật, không chỉ event-support pairs.
Không đổi production config, không promote phương pháp từ kết quả nhỏ.

- Phần CPU chạy `solve_kis`: SigLIP/Milvus, BM25, MiniLM, fusion profile hiện
  hành, object evidence nếu có entity, temporal pooling/scoring/DP khi là chuỗi.
- Phần GPU dùng Qwen2.5-VL-3B đã pin revision, tách thành job **Kaggle**.
- Chưa chạy hoặc tải VLM trên máy hiện tại. Máy là RTX3050 6GB; VLM chưa có cache.
- LLM planner Groq chưa cấu hình API key. Lượt mặc định dùng kế hoạch event đã
  duyệt và object entities khớp literal từ câu; **không coi là đã đo LLM planner**.
- Các kết quả GPU chỉ được kết luận sau khi nhận file scores thật từ Kaggle.

## 1. Bộ câu hỏi và giới hạn suy luận

- Development:11 event queries của support-v2 và4 chuỗi dev-v1, nhãn AI quan sát
  ảnh trước đây, chưa human verified.
- Regression:26 event queries và10 chuỗi của human-v3. Nhãn do người dùng gán,
  tiếng Anh đã duyệt, nhưng **holdout này đã được quan sát** ở lượt trước.
- Tổng51 đơn vị ×3 retrieval arms =153 cases trước VLM.
- Không dùng các holdout-v1/v2 khác để chọn rule/profile. Không rename human
  holdout thành development; dữ liệu giữ split gốc và có `role=observed_regression`.
- Có thể dùng development để so sánh/chọn hướng thử, nhưng chỉ được tuyên bố
  khả năng khái quát sau một holdout thực sự mới. Human-v3 chỉ là regression.

## 2. Ba retrieval arms, có đối chứng ngân sách

| Arm | Ảnh đơn | Seed mỗi event của chuỗi |
|---|---|---|
| `full` | Câu đầy đủ,100 hits/stream | Câu đầy đủ,50 hits/stream |
| `full_deep` | Câu đầy đủ,200 hits/stream | Câu đầy đủ,100 hits/stream |
| `full_compact` | Full + compact,100 hits/variant/stream | Full + compact,50 hits/variant/stream |

Compact lấy tối đa20 token không phải stopword, giữ thứ tự và chữ gốc. **Không
thay câu xác minh**, không tự thêm danh từ/thuộc tính. Câu full vẫn nằm trong
retrieval variants; local scorer/VLM luôn nhận yêu cầu đầy đủ. Đây là thử nghiệm
keyword view đơn giản, không gọi nó là phân rã ngữ nghĩa hoàn chỉnh.

`full_deep` là đối chứng quan trọng: nếu tăng recall chỉ do lấy nhiều hits hơn,
không quy toàn bộ gain cho compact query. Với câu quá ngắn, compact có thể trùng
full nên số variants thực tế <2; stream queries thực tế nằm trong diagnostics.

Fusion/query-policy và nprobe lấy từ profile runtime, không đổi theo arm.
Temporal vẫn12 video, tối đa1500 frames/video, gate0,25/0,45, gap60s,
penalty0,001. Compact/deep chỉ thay seed retrieval, không sửa DP hoặc support.
Ảnh đơn đi qua object evidence với **cùng entities** cho cả3 arms; fixed-plan
entities chỉ là các tên detector đã xuất hiện nguyên văn trong câu.

**Khác với benchmark trước:** ảnh đơn ở đây chạy nhánh ảnh đơn `solve_kis`,
không dùng bounded event-frame diagnostic. Temporal có thể fallback về nhánh
ảnh đơn nếu không có path, giống production; fallback không được coi là path hit.

## 3. VLM thực sự được đo ở đâu?

### 3.1. Ảnh đơn: đúng cơ chế rerank production

- Chấm20 candidates đầu mỗi case bằng `VLMReranker.score` hiện có.
- Giữ đầy đủ candidate list CPU để replay; các candidate ngoài top20 không
  được tự gán VLM score.
- Áp dụng đúng `finalize_vlm_ranking`, weight0,5; không verified-first mới.
- Query VLM là `raw_query` của đơn vị, như nhánh ảnh đơn production. Với human
  queries đó là tiếng Việt gốc; với dev-v2 là câu tiếng Anh được tác giả nhãn ghi.

### 3.2. Chuỗi: joint ordered-evidence rerank thử nghiệm

Production hiện bypass VLM khi đã có temporal result. Vì vậy đây là **stage
thử nghiệm bổ sung**, không nói production vốn đã có sequence VLM verification.

- Chấm tối đa5 paths đầu, chỉ khi path đủ các event, mỗi event có ảnh/timestamp.
- Cung cấp **tất cả ảnh của path** theo thứ tự, event action đầy đủ và timestamp
  cho Qwen, không dùng ảnh đại diện để chấm toàn query nhiều bước.
- VLM trả mức phù hợp0–10 cho bộ evidence; kết hợp bằng cùng công thức ranking.
- Không dựng alternate path, không thay support gate, không thêm video đáp án
  vào pool. Vì vậy không thể cứu mọi lỗi retrieval hoặc known-positive bị gate loại.
- `verified=false` luôn giữ nguyên: sparse keyframes không chứng minh chuyển
  động liên tục, nhân quả hoặc mọi thời điểm trong clip.

### 3.3. Pair diagnostics: VLM có nhìn ra điều kiện đúng hay không?

Chấm39 development pairs +130 human-regression pairs bằng câu event đầy đủ.
Các ảnh được chọn từ qrels **chỉ phục vụ classification diagnostic**, không
được thêm vào candidate retrieval. Điều này giúp phân biệt “VLM nhận ra đúng
nhưng retrieval không tìm thấy” và “VLM cũng không phân biệt được”.

Ngưỡng thử cố định8/10 theo vùng “strong match” của rubric, **không calibrate
theo regression**. Báo coverage, số cặp có score hợp lệ cho cả hai caption modes
và macro TPR/FPR trên tập paired-valid đó. Missing/invalid scores không đổi thành
negative, unknown judgments không vào confusion matrix.

### 3.4. Có caption và chỉ dùng ảnh

Mọi loại task có hai mode:

- `with_caption`: như reranker production, caption được đưa vào prompt dưới dạng
  dữ liệu không tin cậy, không phải ground truth.
- `image_only`: không đưa caption vào prompt, để đo ảnh hưởng từ văn bản sinh.

Cùng model/revision,4-bit NF4,SDPA,giới hạn ảnh256²–512²,greedy generation.
Các request giống hệt query/images/caption/mode được dedup, nên không trả phí
GPU lặp lại chỉ vì candidate chung giữa các retrieval arms.

## 4. Gói Kaggle và bảo mật

CPU artifacts:

`benchmark_results/full_pipeline_ablation_20261102/`

Gói export dự kiến:

`benchmark_results/full_pipeline_vlm_packet_20261102/`

- `vlm_packet/manifest.json` + `images/`: **chỉ query, ảnh evidence, caption,
  model/code hashes**, không nhãn đáp án hoặc target video IDs.
- `bindings_private.json`: ánh xạ task với case/nhãn, **chỉ giữ trên máy**,
  không vào ZIP Kaggle.
- `export_summary.json`: số task, ảnh thực tế đọc được, missing/ambiguous image,
  fingerprints. Không in Drive IDs/token hoặc exception body chứa credentials.
- Các ảnh dùng original bytes và SHA-256. Pair image đổi hash so với nhãn cũ
  bị đánh dấu không dùng; không âm thầm chấm một ảnh khác rồi sửa ground truth.

ZIP chỉ từ allowlist code và các ảnh content-addressed trong packet. Không
đưa `.env`, `config/images.local.json`, `config/llm.local.json`, Google
credentials, qrels, full dataset/index hoặc `units_private.json` lên Kaggle.

**Dataset Kaggle phải để Private** vì ZIP chứa keyframe/caption của corpus.
Không cần đưa model weights vào ZIP; notebook tải đúng pinned public Qwen model
trên Kaggle khi bạn bật Internet.

## 5. Phần cần bạn làm trên Kaggle

Khi ZIP/notebook đã được tạo, dùng:

1. `dist/kis-vlm-benchmark.zip` — upload làm **private Kaggle Dataset**. Giữ
   nguyên tên file ZIP khi thêm vào Dataset.
2. `notebooks/vlm_pipeline_benchmark.ipynb` — import thành Kaggle Notebook.
3. Notebook Settings: chọn **GPU T4** (một T4 là đủ cho cấu hình NF4; nếu chọn
   T4×2, code vẫn chỉ dùng GPU0), **Internet ON**.
4. Add Input: Dataset chứa ZIP ở bước1. Không cần Milvus hoặc khóa Drive trên Kaggle.
5. Chạy các cell lần lượt. Notebook nhận được cả ZIP còn nguyên lẫn Dataset
   đã được Kaggle tự giải nén. Không dùng `requirements.txt` của Windows; cell cài
   riêng dependencies VLM và giữ CUDA torch/NumPy đã có trên Kaggle.
6. Có thể đặt `MAX_TASKS = 100` để smoke-test, sau đó đổi thành `None` và chạy lại
   **cell scoring trong cùng session/output**. Các task đã ghi sẽ tự skip;
   không thay model/prompt/quantization giữa các lần resume.
7. Kiểm tra `summary.json`: `complete=true`. `unscored_tasks>0` được ghi rõ và
   xử lý fallback, không xóa chúng để làm kết quả đẹp hơn.
8. Cell cuối tạo **`/kaggle/working/vlm-results.zip`**. Tải file đó và gửi lại
   trong cuộc trò chuyện; không cần gửi model weights hoặc ảnh lần nữa.

Thời gian phụ thuộc số task thực tế/GPU; không suy từ unit tests ra runtime GPU.
Model download/startup được ghi riêng trong `hardware.json`; task timing không
gộp thời gian download model. Tổng CPU/GPU trên hai máy không phải production
end-to-end latency benchmark.

### Nếu hết phiên Kaggle

Tải `vlm-results.zip` trước khi mất output, dù chưa complete. Lần sau dùng đúng
code/packet, giải nén checkpoint để khôi phục đúng:

```text
/kaggle/working/vlm-results/run_context.json
/kaggle/working/vlm-results/scores.jsonl
```

Rồi chạy lại cell scoring. Không trộn checkpoint của packet, source code hoặc
model khác. Nếu file JSONL bị cắt giữa một dòng do ngắt phiên, gửi file cho tôi
để kiểm tra; không tự xóa/chọn score thấp hoặc lấy score tốt nhất của nhiều lượt.

### Nếu gặp lỗi khi chạy

- **Không có GPU:** chọn GPU trong Settings rồi khởi động lại session. Không
  chuyển sang CPU để cố chạy model3B trên máy cá nhân.
- **Tải model lỗi/không có Internet:** bật Internet và chạy lại; scorer dừng
  trước khi ghi score nếu khởi tạo model thất bại. Không điền score0 thay lỗi.
- **CUDA OOM ở task:** task đó được lưu unscored, giữ baseline khi replay.
  Không tự hạ độ phân giải, đổi model hoặc chạy lại chỉ các score thấp rồi trộn
  chung. Gửi log và ZIP cho tôi; nếu cần cấu hình khác sẽ tạo run version riêng.
- **Cài package xung đột/import lỗi:** gửi traceback; không thay hàng loạt CUDA
  torch/NumPy hoặc chạy `requirements.txt` Windows để chữa thử.
- **Notebook không tìm input:** kiểm tra Dataset input là đúng gói benchmark
  và chỉ có một bản. Notebook nhận cả `kis-vlm-benchmark.zip` lẫn nội dung tự
  giải nén có `vlm_packet/manifest.json` và `tools/kaggle_vlm_benchmark.py`, rồi
  copy/giải nén vào `/kaggle/working/benchmark_code`.

## Kiểm thử đã làm trước khi chạy VLM thật

- Full suite cuối sau khi hoàn tất gói: **329 tests,315 đạt,14 skipped**;
  lượt targeted các công cụ ablation/VLM/packet: **15/15 đạt**.
- Kiểm tra bằng score/ảnh giả lập: task IDs/hashes, ZIP không chứa qrels/private
  bindings, không tải VLM trên máy cá nhân, invalid score không biến thành0,
  resume skip task đã ghi, paired coverage, từ chối kết quả giả/incomplete,
  multi-image input và parser chỉ nhận số0–10.
- Một ảnh đại diện hoặc incomplete temporal path không tạo sequence VLM task.
- Model inference GPU thực tế **chưa được chạy**, nên các unit tests không phải
  bằng chứng Qwen sẽ cải thiện ranking hoặc bảo đảm phù hợp VRAM/runtime.
- Hash production sources, config và nhãn human gốc/frozen được đối chiếu sau
  CPU run, giữ nguyên. Không rebuild index, commit hoặc push.

## 6. Nếu muốn đo cả LLM planner

Phần này hiện **chưa chạy** do Groq chưa có API key. Bạn có thể cấu hình provider/
model/key trong form API của ứng dụng, hoặc đặt key trong môi trường của phiên
chạy trên máy. **Không gửi API key trong chat và không đưa key lên Dataset.**

Sau khi cấu hình xong, báo tôi để chạy một run version mới với `--planner api`.
Runner gọi `create_structured_query` đúng code production và lưu output một
lần/query để3 retrieval arms dùng cùng plan. Lỗi API/planner vẫn nằm trong mẫu
đánh giá, không dùng plan tự viết thay thế âm thầm.

Khi planner đổi số event hoặc không tạo sequence, known-path metric giữ yêu cầu
nhãn gốc; không map lại event bằng đáp án để che lỗi planner. API planner là
ablation riêng với fixed-plan control, không nhập nhằng hai nguồn query.

## 7. Phần tôi làm sau khi nhận GPU results

- Kiểm tra model/revision,prompt,code hashes,packet ID và task IDs đầy đủ.
- Từ chối simulated/test scorer, kết quả thiếu task hoặc thay code/score-range.
- Replay đúng candidate lists đã chốt, không gọi search lại sau khi nhìn scores.
- Giữ mọi query trong retrieval denominator; score lỗi giữ retrieval ranking,
  báo riêng coverage. Pair diagnostics dùng paired-valid coverage như mô tả.
- So sánh3 arms ×2 caption modes với CPU baseline tương ứng; báo Hit@1/5/20,
  MRR,known-path hit,candidate coverage và chi phí. Development/regression tách riêng.
- Không auto-promote, không biến regression đã quan sát thành new holdout.

## 8. Công cụ tái lập

```powershell
python -B -u -m tools.pipeline_ablation --dev-labels benchmark_results/event_support_v2_seed2026110202/frozen_support_labels_v1.json --dev-sequence-labels benchmark_results/drive_groundtruth_20261102_seed20261102/frozen_labels_v1.json --human-packet benchmark_results/human_holdout_v3_seed2026110203_v2 --human-labels benchmark_results/human_holdout_v3_seed2026110203_v2/frozen_human_labels_v1.json --output-dir benchmark_results/full_pipeline_NEW --planner fixed
python -B -u -m tools.vlm_benchmark_packet export --source-run benchmark_results/full_pipeline_NEW --output-dir benchmark_results/vlm_packet_NEW
python -B -m tools.vlm_benchmark_packet package --packet-root benchmark_results/vlm_packet_NEW --output dist/kis-vlm-benchmark.zip --notebook notebooks/vlm_pipeline_benchmark.ipynb
python -B -m tools.vlm_benchmark_packet replay --source-run benchmark_results/full_pipeline_NEW --packet-root benchmark_results/vlm_packet_NEW --result-dir PATH_TO_EXTRACTED_GPU_RESULTS --output-dir benchmark_results/vlm_replay_NEW
```

Không ghi đè artifact đã công bố. Các bundle full-ingestion và keyframe/caption-only
trước đây vẫn tách riêng và không bị thay thế bởi bundle benchmark VLM này.

## 9. Kết quả CPU đã hoàn tất (chưa có VLM)

| Split/task | Metric | Full | Full deep | Full + compact |
|---|---|---:|---:|---:|
| Dev,11 ảnh | Known-frame Hit@1 | 2/11 | 3/11 | 4/11 |
| Dev,11 ảnh | Known-frame Hit@20 | 5/11 | 5/11 | 6/11 |
| Dev,11 ảnh | MRR@20 | 0,24242 | 0,28687 | 0,37942 |
| Human regression,26 ảnh | Known-frame Hit@1 | 3/26 | 3/26 | 3/26 |
| Human regression,26 ảnh | Known-frame Hit@20 | 6/26 | 7/26 | 6/26 |
| Human regression,26 ảnh | MRR@20 | 0,13138 | 0,13629 | 0,12850 |
| Dev,4 chuỗi | Known-path Hit@20 | 0/4 | 0/4 | 0/4 |
| Dev,4 chuỗi | Đúng target video top1 | 4/4 | 4/4 | 4/4 |
| Human regression,10 chuỗi | Known-path Hit@20 | 2/10 | 2/10 | 2/10 |
| Human regression,10 chuỗi | Đúng target video top1 | 6/10 | 6/10 | 6/10 |

Không có lỗi local scoring hoặc temporal fallback trong các case chuỗi. Cùng
điểm metric không có nghĩa cùng pool/path: ví dụ compact giữ target video trong
pool7/10 human sequences, so với8/10 ở full/deep. **Không promote compact** chỉ
vì dev Hit@1 tốt hơn trên11 câu.

Coverage trong candidate list ảnh đơn đã lưu tối đa300 kết quả (không phải toàn
corpus relevance):

| Tập | Full | Full deep | Full + compact |
|---|---:|---:|---:|
| Dev: có ít nhất1 known-positive ở candidate list | 7/11 | 7/11 | 7/11 |
| Human regression: có known-positive ở candidate list | 12/26 | 14/26 | 12/26 |

VLM chỉ chấm20 ảnh đầu, không thể tự tìm lại frame hoàn toàn ngoài candidate
list. Với paths, nó chỉ rerank path DP đã chọn, không dựng path known-positive
khác từ raw frame tables. Giới hạn này phải đi kèm mọi báo cáo VLM sau đó.

CPU times lưu trong `cpu_summary.json` là tổng lượt tuần tự có cache reuse, không
là benchmark latency công bằng: arm chạy sau hưởng cache text encoder/mapping.
Không dùng chênh lệch nhỏ về thời gian để tuyên bố phương pháp nhanh hơn.

Hiện chỉ có kết quả CPU. **Chưa có score Qwen thật, chưa đo gain do VLM và chưa
đo LLM planner chưa cấu hình**. Sau khi người dùng chạy Kaggle và gửi ZIP, mới
đối chiếu các hiệu ứng này bằng tool replay.

## 10. Gói GPU đã sẵn sàng

Hai file thực tế đã tạo, không phải đường dẫn dự kiến:

```text
C:\Users\Khanh\Desktop\KISBranche\dist\kis-vlm-benchmark.zip
C:\Users\Khanh\Desktop\KISBranche\notebooks\vlm_pipeline_benchmark.ipynb
```

- ZIP **162,87 MiB**, chứa1.011 ảnh unique cùng code allowlist và manifest.
- **2.376 tasks** sau dedup:2.296 image tasks và80 multi-image path tasks,
  cho153 retrieval cases +169 pair diagnostics.
- Trong1.097 unique Drive image requests:1.011 thành công,86 failed. Không
  đổi candidate lists để bỏ ảnh thiếu; file `availability_audit.json` ghi rõ
  coverage. Không tự thay ảnh hoặc retry để chọn dữ liệu đẹp hơn.
- Có **2.192 tasks đủ ảnh** và**184 tasks thiếu ít nhất một ảnh**. Scorer sẽ
  ghi184 task này là `missing_image`, không chạy model cho chúng và không gán0.
  Vì thế `unscored_tasks` ít nhất184 là dự kiến ở gói này, không đồng nghĩa job
  chưa complete. `complete=true` nghĩa đủ2.376 task IDs được ghi, có cả lỗi/thiếu.
- **169/169 pair diagnostics có ảnh đầy đủ**, gồm39 dev và130 human-regression.
  Model vẫn có thể trả invalid/OOM; coverage GPU thật phải báo sau khi chạy.
- Kiểm tra ZIP CRC, danh sách file đúng allowlist, code/manifest hashes,
  notebook schema và cú pháp cell đều đạt. Đã chạy test input cell bằng gói giả
  cho cả trường hợp ZIP và Dataset tự giải nén; không gọi pip/GPU trong test.
- Ground truth,private bindings,target-video IDs,Drive credentials/API key
  không nằm trong ZIP. Query/caption có nội dung corpus nên vẫn để Dataset **Private**.

ZIP SHA-256:

```text
647cf1e99d33722156973b507d3e601929a160ebe161706a650ea9dda4090603
```

Manifest SHA-256:

```text
6fed1d767c369126af02d0ae4653779dacaa3640c9294d803b3b89a4dd5780eb
```

**Việc cần người dùng làm tiếp:** chạy notebook này trên Kaggle như mục5 và
gửi lại `vlm-results.zip`. Chưa cần gán nhãn thêm hoặc chuyển model sang máy
cá nhân. Nếu muốn đo thêm API planner, cấu hình riêng như mục6; gói này chưa
bao gồm planner chưa được cấu hình.
