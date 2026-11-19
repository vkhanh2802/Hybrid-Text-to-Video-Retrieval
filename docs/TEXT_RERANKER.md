# Nâng cấp module P1: reranker nhẹ trước VLM

> Cập nhật 04/11/2026: checkpoint đã cache, smoke thật đạt và hoàn tất153 cached
> CPU cases. **Kết quả mixed, giữ mode off**; xem
> [báo cáo đo thật](TEXT_RERANKER_CPU_RESULTS_20261104.md) và
> [protocol CPU](TEXT_RERANKER_CPU_PROTOCOL_20261104.md). Không chạy VLM mới.

## Trạng thái

Đã triển khai **module chạy trong pipeline**, không chỉ là script thử nghiệm:

- `kis/text_reranker.py`: config, lazy CPU inference, chunking, event-aware text
  scoring, fallback và ranking có ngân sách.
- `kis/solve_kis.py`: chạy sau fusion/object và trước diversity/VLM ở ảnh đơn;
  sau DP và trước trả kết quả ở chuỗi. Không gọi VLM ảnh đơn để xác nhận chuỗi.
- `kis/config.py`, `config/pipeline.json`: cấu hình dùng chung.
- `kis/web_app.py`: SearchService, status, diagnostics và metadata từng kết quả.
- Notebook đang dùng SearchService nên nhận cùng nâng cấp, không có pipeline riêng.
- `tools/check_text_reranker.py`: kiểm tra cache, **không nạp/tải model**.

**Mặc định `off`.** Feature đã có code/integration và đã đo bằng model thật trên
cached development/human regression. Gain cục bộ kèm loss, chưa đủ để promote.
Checkpoint hiện đã có trong cache; không tuyên bố gain từ unit tests hay coi
human-v3 là independent holdout.

Theo outline: [SEARCH_QUALITY_UPGRADE_OUTLINE.md](SEARCH_QUALITY_UPGRADE_OUTLINE.md),
mục6; [HCMC_AI_CHALLENGE_RESEARCH_AND_UPGRADE_OUTLINE.md](HCMC_AI_CHALLENGE_RESEARCH_AND_UPGRADE_OUTLINE.md),
mục7/P2. Temporal, gate và pool mặc định không được thay theo kết quả regression.

## Luồng mới khi bật

```text
Ảnh đơn:
  per-stream search -> fusion -> hydrate caption -> object evidence
  -> text reranker (pool100) -> optional diversity -> VLM (top20) -> kết quả

Chuỗi:
  per-event retrieval -> local evidence/gate -> ordered DP
  -> text rerank những complete paths có đủ caption từng event -> kết quả
```

Module này **xếp lại candidate đã có**, không tìm được video ngoài pool và không
cứu frame bị gate loại. Caption vẫn do model sinh, không phải ground truth.
OCR/ASR độc lập chưa được triển khai trong lượt này; không dùng caption giả làm OCR.

## Model và giới hạn tài nguyên

Checkpoint English-oriented: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`.
Revision được đối chiếu metadata model khi triển khai và đã tải/chạy CPU thật
ngày04/11/2026 trên môi trường hiện tại; xem hashes/runtime trong benchmark artifacts.

- CPU FP32, `eval()` và `torch.inference_mode()`; không giữ thêm model trên GPU6GB.
- Dùng safetensors; `local_files_only=True`, `trust_remote_code=False`.
- Không import transformers/torch khi import config, khởi tạo service hay xem status.
- Lazy-load khi có ít nhất một cặp query/text đủ điều kiện để chấm.
- Lỗi load được nhớ lại; sau khi chuẩn bị model cần restart service/kernel.
- Không thêm dependency: dùng transformers/torch đã có trong requirements.

| Cấu hình | Mặc định | Trần validation |
|---|---:|---:|
| pool_size | 100 candidates | 300 |
| batch_size | 16 pairs | 64 |
| max_pairs | 1200/query | 4096 |
| chunk_chars | 1200 | 4000 |
| max_chunks | 4/text | 8 |
| max_length | 512 tokens/pair | 512 |
| query_chars | 1600 | 8000 |

Text dài được chunk có overlap và lấy mẫu đều từ đầu tới cuối nếu vượt budget,
không chỉ lấy phần đầu. Khi max_chunks=1 chỉ có chunk đầu. Diagnostics báo số
chunk bỏ qua và truncation theo ký tự; tokenizer còn có `longest_first` truncation
ở max_length. Không coi text đã được kiểm tra đầy đủ khi có truncation.

## Query và aggregation

Ảnh đơn ưu tiên **câu đầy đủ tiếng Anh có sẵn** tại `stream_queries.visual[0]`,
sau đó visual_description, query_variants[0], raw_query. Không tự tạo rewrite,
dịch hay thêm điều kiện mới. Nếu fallback là tiếng Việt, checkpoint English
không được bảo đảm phù hợp; diagnostics giữ query_source để phân tích.

- Ảnh: lấy logit cao nhất qua các caption chunks.
- Chuỗi: chấm riêng `(event action, caption của chính event frame)`; lấy max
  qua chunks của event, rồi **min qua các event** để event mạnh không che event yếu.
- Path partial hoặc thiếu caption ở bất kỳ event nào: không chấm text, giữ vị trí cũ.
- Đây là chính sách kỹ thuật ban đầu chưa được calibration; không suy ra thứ tự
  từ caption hoặc dùng text để điền frame/event thiếu.

## Ranking: không trộn trực tiếp logits với cosine/RRF

Logits có thể âm, không phải xác suất. Module không sigmoid rồi gọi đó là
confidence, không gắn `verified=true`, không áp hard threshold để loại candidate.

Chính sách v1 là **scored-slot permutation + rank ladder**:

1. Xác định những slot trong top-pool có đủ text và toàn bộ score hợp lệ.
2. Chỉ hoán vị candidate giữa các slot đó theo text logit, tie giữ thứ tự cũ.
3. Với temporal, chỉ hoán vị trong cùng số event được hỗ trợ; không để text
   đảo ưu tiên support-count của DP. Event/frame/timestamp/coverage không đổi.
4. Candidate không chấm được giữ đúng slot. Không đổi missing/error thành0.
5. Giữ nguyên `candidate.score` và fusion evidence. Lưu điểm retrieval của slot
   mới trong **`text_ranking_score`** để finalizer/VLM không vô tình sắp lại theo
   score cũ và làm mất kết quả text rerank.

Ví dụ retrieval A=.9, B=.8 (thiếu caption), C=.7; text ưu tiên C hơn A:

```text
Vị trí mới:       C     B     A
score gốc:      .7    .8    .9
điểm ranking:   .9    .8    .7
```

B giữ nguyên slot2 và điểm. Điểm ranking là mức điểm của slot, **không phải
cosine của C, không phải xác suất relevance của C**. VLM sau đó dùng điểm ranking
làm base theo cùng công thức cũ. Tắt text thì finalizer vẫn dùng `score` như cũ.
VLM có thể tiếp tục đổi ranking; cam kết giữ slot nói về text stage, không phải
khóa slot sau VLM. Ảnh không có caption vẫn đi qua visual/VLM route bình thường.

## Bật/tắt và kiểm tra

Kiểm tra cache không chạy model:

```powershell
python -B -m tools.check_text_reranker
```

`cache_ready=false` và exit code khác0 là checkpoint chưa sẵn sàng, không phải
lý do tự tải model hoặc rebuild corpus. Presence check không bảo đảm compatibility.

Chế độ qua environment, trước khi khởi động app hoặc tạo SearchService:

```powershell
$env:KIS_TEXT_RERANKER_MODE = "off"          # giữ baseline, không chạy model
# Sau khi đã chuẩn bị checkpoint và đồng ý chạy inference:
$env:KIS_TEXT_RERANKER_MODE = "diagnostics"  # chấm/log, không đổi ranking
$env:KIS_TEXT_RERANKER_MODE = "rerank"       # chấm và xếp lại
.\start_kis.ps1
```

Chỉ chọn **một** mode; các dòng trên minh họa lựa chọn, không yêu cầu chạy cả ba.
Mode sai như `true` bị từ chối thay vì ngầm bật. Có thể sửa object text_reranker
trong config/pipeline.json; environment mode được ưu tiên. Không nhận model path
hoặc repo tùy ý qua request API.

Gọi trực tiếp từ Python cần truyền config và giữ scorer để tái sử dụng model:

```python
from kis.config import load_runtime_config
from kis.text_reranker import LazyCrossEncoder

runtime = load_runtime_config()
scorer = LazyCrossEncoder(runtime.text_reranker)  # chưa nạp model
# solve_kis(..., text_reranker_config=runtime.text_reranker,
#           text_reranker=scorer)
```

Các benchmark cũ gọi solve_kis mà không truyền option vẫn mặc định off. Không
ngầm bật feature trong old/frozen benchmark chỉ vì environment của web đổi.

### Chuẩn bị weights trên máy khác khi được phép

Máy hiện tại đã có checkpoint, không cần tải lại. Model public, không cần gửi
API key. Lệnh dưới **có tải file**, không tự chạy inference; dùng cùng Python
environment của app, chỉ khi người dùng đồng ý:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cross-encoder/ms-marco-MiniLM-L6-v2', revision='233902d25c440f23af6f7d6e94d2946bac0bee0a', allow_patterns=['config.json','tokenizer_config.json','tokenizer.json','vocab.txt','special_tokens_map.json','model.safetensors'])"
python -B -m tools.check_text_reranker
```

Nếu chọn chạy benchmark thật, lập run/version mới, development trước, cùng
candidate pool, ghi model/config/source hashes và paired wins/losses. Không dùng
human-v3 để chọn pool/chunk/aggregation. Chưa có holdout độc lập mới để promote.

## API/diagnostics

- `/api/status`: text_reranker config, loaded, load_error; xem status không nạp model.
- `/api/search`: `text_reranker`, `stage_seconds.text_reranker`.
- Từng result: `retrieval_score`, `text_ranking_score`, `text_logit`, `text_rerank_status`.
- Trace top-pool: frame identity, retrieval_rank, text_rank, raw logit, score gốc,
  score ranking. Không nhúng caption dài hoặc traceback vào diagnostics.
- Status phân biệt disabled/outside_pool/missing_query/missing_text/partial path,
  pair budget, evidence_error/inference_error/invalid_score/scored.
- Batch lỗi giữ fallback cho candidate bị ảnh hưởng; batch khác vẫn được dùng.
- Bất kỳ chunk score lỗi trong một candidate: candidate đó giữ fallback, không
  dùng max trên subset score thuận lợi rồi giả vờ toàn evidence đã được chấm.

## Packaging và artifact cũ

Giữ hai profile Kaggle **full** và **keyframes-caption**. Allowlist mới chứa
`kis/text_reranker.py` vì config import dataclass từ file này; chỉ mang code,
không weights và không kích hoạt text inference khi ingestion/caption.
Caption backend InternVL và batch16/OOM splitting không đổi.

Allowlist packet VLM mới cũng thêm dependency này. **Không tạo lại/ghi đè** ZIP,
notebook hay score đã công bố trong dist. Việc nâng code config/ranking/solver
làm source fingerprints khác bản benchmark cũ: replay cũ có kiểm tra source sẽ
cố ý từ chối current sources. Không sửa hash cho qua; dùng snapshot phiên bản cũ
hoặc tạo run mới. Nhãn, private bindings và scores cũ không bị sửa.

## Giới hạn và phần còn lại của outline

Đã hoàn thiện module text ranking/diagnostics và đo thật, **chưa có gain ổn định
đủ để bật mặc định**. Nó không giải trực tiếp gate/pool recall đã thấy ở thử nghiệm multi-path. Những phần còn
lại cần triển khai/đánh giá riêng: OCR độc lập, ASR timestamped, VLM rubric và
feedback/UI. Không đánh dấu cả outline đã hoàn tất hoặc bật mọi model cùng lúc.

## Kiểm thử triển khai (03/11/2026)

- Full suite: **362 tests,348 đạt,14 skipped**, không failure/error.
- Targeted text/solver/ranking/runtime/API/temporal: **86/86 đạt**.
- 22 tests mới kiểm tra config, chunk tail/budgets, raw negative logits, invalid
  scores, missing captions, batch/load errors, lazy/cache-only safetensors CPU
  adapter, scored-slot ranking, diagnostics-only mode, support-count isolation,
  VLM handoff, temporal/API integration và cache check không tải model.
- Unit tests dùng fake scorer/model hoặc tensor nhỏ, **không nạp checkpoint thật**.
- Hai profile Kaggle vẫn được kiểm tra riêng, caption-only boot trong isolation;
  benchmark ZIP giả lập trong test nằm ở thư mục tạm, không ghi đè dist.
- Targeted tests ban đầu lộ phụ thuộc thứ tự khởi tạo Windows/OpenSSL khi fixture
  xóa toàn bộ environment. Đã khởi tạo TLS context trước fixture cô lập credential
  trong test; không đổi TLS production hoặc tắt xác minh chứng chỉ/hostname.
- Không sửa qrels, index, embedding hoặc caption backend; không commit/push.
