# Notebook OCR tiếng Việt mới — Paddle detector + VietOCR

File nên dùng: **`notebooks/kaggle_ocr_vietnamese_gpu.ipynb`**.

## Bản hiện tại: v3.1 — batch16 + long-line segmentation trên Kaggle

File `.ipynb` trên đã được build lại với **VietOCR tensor batch tối đa16 crop**,
không phải batch của Qwen/laptop và không chỉ thay một biến cấu hình:

- Full/preview gom tối đa16 frame; Paddle detector vẫn xử lý từng frame một.
  Crop chữ và các đoạn v3 từ các frame được gom chung để VietOCR đọc theo batch.
- Chia bucket theo **đúng chiều rộng sau resize**. Không thêm width padding,
  không resize mọi crop thành một width chung, không bật FP16/AMP. Điều này giữ
  cách chuẩn bị input của v3; các bucket ít phần tử/batch cuối có thể nhỏ hơn16.
- Từng crop có EOS, độ dài và xác suất riêng. Crop xong được loại khỏi decoder
  active batch; kết quả được trả lại đúng frame/line/segment, không thêm slot giả.
- Giữ nguyên detector/weights, threshold, chia dòng dài, full-line baseline,
  public bảy trường và runtime Paddle/Torch tách biệt.
- Nếu OOM thì dừng, giữ checkpoint đã hoàn tất; không tự giảm batch hoặc fallback
  CPU. Một cửa sổ chưa hoàn tất có thể cần tính lại tối đa16 frame khi resume.
- `profile.json` pin `recognition_batch_size=16`, `recognition_frame_window=16`,
  batching policy và worker hash. `run.json.recognition_batches` ghi histogram
  **batch thực tế**, tổng crop và số lần encoder batch. Technical lines/full_line/
  segments ghi `inference_batch_size`, width và policy. Không suy tốc độ từ batch cap.

**Import lại toàn bộ notebook**, kiểm tra tiêu đề **v3.1 — batch16 + long-line segmentation**.
Điền INPUT_ROOT thật; batch16 đã nhúng sẵn, không phải tự thêm biến vào cell đầu.
Chạy preview, so với v3 scalar và ảnh gốc trước khi full. Sai khác số học khi batch
trên GPU có thể làm text/confidence thay đổi; chưa chứng minh output bit-identical.

Nếu muốn xuất ZIP, đặt `RUN_FULL=True` trong cell cấu hình rồi chạy **Full OCR →
Download**, hoặc Save & Run All. `RUN_FULL=False` chỉ preview, chủ động không xuất.
Không chạy lại cấu hình/preview giữa Full và Download vì RUN_OK sẽ bị reset;
không tự đặt RUN_OK=True để bỏ gate.

Output mới, không đè ZIP scalar v3:

```text
/kaggle/working/ocr_vietnamese_v3_batch16.zip
/kaggle/working/ocr_vietnamese_v3_batch16_diagnostics.zip
```

Scratch mới `ocr-vietnamese-lines-v3-batch16-<input-root-hash>` giữ nguyên runtime/
checkpoints cũ. Không sửa profile hash để ép resume v3 scalar vào batch16.
**93 tests liên quan pass**,36,992s: CPU tensors thật với toy logits chứng minh
37 crop chạy16+16+5, exact-width buckets, scalar/batch parity trên toy model,
EOS độc lập, restore identity, frame windows/resume, decode errors, fail-closed
backend/OOM và export/schema. Đây không phải chạy OCR weights hoặc benchmark T4.
Receipt/backups: `benchmark_results/kaggle_ocr_batch16_20261116_run01/`.
Full regression batch16 hoàn tất: **1.014 tests — 998 đạt, 16 bỏ qua, không lỗi**,
187,498s. Receipt: `benchmark_results/kaggle_ocr_batch16_20261116_run01/FULL_TESTS.json`.

**Chưa chạy bản batch16 trên Kaggle GPU, chưa đo speedup/VRAM hoặc accuracy.**
Không thay OCR live, school worker/web hay Qwen batch12; không cài thêm dependency
hoặc tải weights vào laptop/server.

## Cơ chế giữ nguyên từ v3 — nhận dạng dòng dài theo khoảng trống

Diagnostics L01 đã xác nhận các ticker có box đủ nhưng VietOCR ra EOS sớm:
`truncated=false`, confidence cao, raw text vẫn mất đuôi. Bản v3 giữ nguyên detector,
source/weights và dependencies; thay cách đưa **crop dài** vào nhận dạng:

- Khi width chuẩn hóa vượt **512**, tìm khoảng trống bằng Otsu luminance và
  polarity nền/sáng; chia thành các đoạn width chuẩn hóa tối đa **480**.
- Không overlap; các đoạn liền nhau phủ đủ cột pixels gốc. Không cắt cố định xuyên
  glyph khi không có khoảng trống phù hợp, không fuzzy dedup làm mất từ lặp.
  Đây là heuristic, không phải bảo đảm mọi điểm chia đều giữa hai từ.
- Ghép text đoạn theo thứ tự trái → phải, chỉ thêm khoảng trắng ở biên. Không
  sửa chính tả, thêm dấu, đoán từ hoặc chọn output vì confidence lớn hơn.
- Giữ **full_line baseline + từng segment**, vị trí x trong crop, SHA-256 bytes
  BGR từng đoạn, kích thước, EOS/truncation và review reasons trong diagnostics.
- Không tìm được khoảng trống, hoặc một đoạn blank/low-confidence/token-capped:
  giữ raw nhưng **loại cả dòng dài khỏi public**, không âm thầm xuất phần đầu.
  Coverage có thể giảm; cần xem các dòng đỏ/bị loại chứ không chỉ dòng đẹp.
- `confidence` của dòng ghép là **min(mean-character-softmax của từng đoạn)**,
  không phải xác suất dòng đúng. `completeness=not_verified` kể cả khi EOS thành công.
- Dòng ngắn giữ cách nhận dạng cũ. Không tự sửa lỗi logo/detector trong bản này.

**Output mới:** `ocr_vietnamese_v3.zip` và `ocr_vietnamese_v3_diagnostics.zip`.
Scratch mới `ocr-vietnamese-lines-v3-<input-root-hash>` không dùng đè checkpoints
v2.1; các ZIP/runtime cũ giữ nguyên. Không ép sửa profile hash để reuse dữ liệu cũ.
Không thêm dependency hoặc cài OCR/weights lên laptop/server.

### Chạy preview đúng những lỗi đã phát hiện

Import lại **toàn bộ** notebook; tiêu đề có **v3 — long-line word-gap segmentation**.
Bật Internet + T4 GPU và điền INPUT_ROOT thật chứa các video L01, sau đó:

```python
RUN_FULL = False
PREVIEW_COUNT = 8
PREVIEW_IDENTITIES = [
    'L01_V005:69', 'L01_V013:170', 'L01_V015:196',
    'L01_V020:29', 'L01_V030:67',
]
PUBLIC_MIN_CONFIDENCE = 0.5
```

`PREVIEW_IDENTITIES` là danh sách chính xác, theo thứ tự yêu cầu; nếu thiếu frame,
trùng identity hoặc sai format thì dừng trước model, không lấy ảnh khác bù vào.
Đặt `[]` để trở lại lấy mẫu cách đều theo PREVIEW_COUNT. Ngoài các ca đã dùng chỉnh
pipeline, cần thêm sample chưa dùng trước khi quyết định chạy full.

Preview in **text mới, full-line baseline và từng đoạn**. Kiểm tra đã khôi phục
đuôi câu, không lặp/mất từ tại biên, tên riêng và phần tiếng Việt cũ vẫn đúng.
Sau khi ổn mới đặt RUN_FULL=True rồi chạy full/download; với Save & Run All,
preview vẫn thực thi trước full. RUN_OK/EXPORT_OK do notebook tự quản lý.

Đã qua **82 kiểm thử liên quan** (34,638s), gồm EOS sớm mô phỏng, đủ pixels/không
overlap, cả hai polarity, giữ từ lặp, fail-closed đoạn lỗi, raw metadata qua
checkpoint/export và giữ archive cũ. Kiểm tra hình học trên crop dựng lại từ
polygon của 5 ca L01 cho ra 2 đoạn/ca, mỗi đoạn không quá480; không phải crop
bytes được xuất từ detector và không phải OCR inference. Receipt/ảnh phân đoạn:
`benchmark_results/kaggle_ocr_long_line_fix_20261115_run01/`.

**Chưa chạy model v3 trên Kaggle GPU, chưa chứng minh accuracy hoặc tốc độ tốt hơn.**
Full regression v3 hoàn tất: **1.003 tests — 987 đạt, 16 bỏ qua, không lỗi**,
186,815s. Receipt: `benchmark_results/kaggle_ocr_long_line_fix_20261115_run01/FULL_TESTS.json`.
Đọc cả dòng baseline rồi đọc các đoạn sẽ tốn thêm inference cho dòng dài; chưa
benchmark thời gian/VRAM. Không chạy full chỉ vì unit tests pass.

## Lịch sử: hotfix v2.1 — box-list serialization

Log Kaggle đã tái hiện lỗi `AttributeError: 'list' object has no attribute 'tolist'`:
PaddleX `SortQuadBoxes` trả về list, không nhất thiết là NumPy array. Bản v2.1
chuẩn hóa kết quả sau sort bằng `np.asarray(...)` trước khi crop và xuất JSON;
không đổi thứ tự box/crop, model, CUDA dependencies hoặc runtime isolation.

Đã tái hiện đúng traceback trước sửa bằng vòng `detector_worker` thật trong một
subprocess, chỉ stub GPU/model. Sau sửa cùng test qua list, array và empty;
native logs không lẫn JSON, crop/box giữ đúng thứ tự. **72tests liên quan pass**,
20,681s. Không coi kiểm thử stub là chạy lại model thực trên Kaggle.

Ở bản v2.1 trước đây, notebook có tiêu đề **split runtime v2.1 — box-list fix**, giữ INPUT_ROOT
thật/RUN_FULL như mong muốn và chạy lại từ setup → preview → full/export, hoặc
Save & Run All. Không cần xóa venv đã cài đúng hay đổi model. Setup ghi lại worker
mới; preview/profile cũ không được dùng thay cho preview của worker mới.

Bản gốc`notebooks/kaggle_ocr_simple_gpu.ipynb` được giữ nguyên SHA-256
`0202aadb6ff938858d66f6eead3e7b9880dc0f025bd6bb8799a952e5c9be6e25`.
Không thay OCR dataset, school worker, web hoặc Qwen batch12. Không cài package
hay tải weights vào laptop/server; các bước cài/tải mới chỉ chạy trong notebook
Kaggle khi người dùng thực thi, sau khi đã đồng ý dùng VietOCR.

## Lịch sử: bản sửa Save Version tách dependency Paddle/Torch

Log chạy Kaggle đã xác nhận bản runtime chung trước đây không hợp lệ:

| Dependency NVIDIA | Paddle3.2.2 cần | Torch2.8 cài trong log |
| --- | --- | --- |
| nvidia-cudnn-cu12 | 9.5.1.17 | 9.10.2.21 |
| nvidia-cusparselt-cu12 | 0.6.3 | 0.7.1 |
| nvidia-nccl-cu12 | 2.25.1 | 2.27.3 |

Notebook hiện đã sửa thành **hai venv và hai process**:

```text
runtime-paddle → Paddle GPU + detector + crop gốc
                  │ JSON pipe, numpy uint8 bytes trong RAM
runtime       → Torch GPU + VietOCR → checkpoint / public OCR
```

Không cài Torch vào venv Paddle hoặc ngược lại, không share site-packages/NVIDIA
wheel directories. Detector resident chỉ khởi động một lần cho mỗi preview/full
worker, không reload model cho từng ảnh. Không ghi file ảnh/crop RPC ra disk;
polygons và crop giữ đúng thứ tự. Native/Python detector logs ra stderr, không
làm lẫn JSON protocol. Child failure dừng bulk, không giả thành OCR empty.

Mỗi venv cài một dependency graph riêng và chạy **pip check riêng**. Không bỏ
gate, `--no-deps`, ép CUDA library sai, hoặc cài `wrapt` vào kernel để che vấn đề.
Python launcher dùng `-I -S`, thêm chỉ site-packages của venv đích, nạp riêng
package pip đã có mà không bật global site-packages/sitecustomize của Kaggle.
Cách này tránh hook báo`ModuleNotFoundError: wrapt` trong subprocess cũ. Pip
resolver/check vẫn chạy thật và báo đầy đủ lỗi nếu còn dependency hỏng.

Scratch mới `ocr-vietnamese-split-v2-<input-root-hash>` tự tránh runtime chung
đã cài lỗi. Venv/checkpoints/ZIP cũ được giữ nguyên, **không yêu cầu xóa thư mục**.
Nếu output ZIP cùng tên đã có nhưng thuộc scratch cũ, gate chống ghi đè sẽ dừng;
hãy giữ/move bản ZIP cũ sang tên khác có chủ đích trước khi publish bản mới.

**Cần upload/import lại toàn bộ notebook mới**, không chỉ copy dòng pip check:
setup, launcher và detector child protocol đều đã thay đổi. Để Save Version
chạy full: điền INPUT_ROOT thật, đặt RUN_FULL=True, bật Internet + GPU T4 rồi
Save Version → Save & Run All. Preview vẫn chạy trước full. Không sửa RUN_OK/
EXPORT_OK thànhTrue; notebook tự cập nhật sau thành công.

Đã có kiểm thử subprocess/venv thật (không GPU/package download): pip cài một
wheel fixture vào đúng venv, pip check thấy conflict cuDNN mô phỏng và trả lỗi,
sau khi graph fixture đúng thì pass; global sitecustomize bị loại; hai request
RPC qua cùng detector child giữ nguyên bytes. Đây không phải thử model trên
Kaggle, nên không cam kết mọi lỗi runtime khác đã được loại bỏ.
**71tests liên quan bản split-runtime pass**,23,454s. Notebook đã build lại,
worker/launcher nhúng trùng source và bản simple gốc giữ nguyên hash. Receipt:
`benchmark_results/kaggle_ocr_split_runtime_fix_20261115_run01/VERIFICATION.json`.
Full regression sau bản sửa: **992tests — 976pass,16skip**,168,704s, không lỗi.
Kết quả đầy đủ ở`benchmark_results/kaggle_ocr_split_runtime_fix_20261115_run01/FULL_TESTS.json`;
đây vẫn không phải chạy model thực tế trên GPU Kaggle.

## Vì sao thay model?

Recognizer Latin cũ thiếu nhiều ký tự tiếng Việt, khớp các lỗi`Quảng→Qung`,
`tấn→tn`, `chứa→cha`. Tăng resolution không chữa được bảng ký tự thiếu.
[Báo cáo nguyên nhân](OCR_NOTEBOOK_DIAGNOSIS.md).

Bản mới dùng:

-**Detector:** `PP-OCRv5_server_det`, chính thức PaddlePaddle, revision
`ca867c897ecbca8873081573a802ad70d499cb94`. Giữ local model_dir sau kiểm hash;
không tải recognizer Latin và không fallback sang nó.
-**Recognizer:** VietOCR `vgg_transformer`, upstream pbcquoc, source commit
`fe8c3a7fc714aec57ab81cec844eb3adf0c1636c`. Weights được mirror bởi
`doanhm/vgg_transformer`, revision`4de5b0429954fdac3dee3173978c062b3bc4d534`,
SHA-256`380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59`.
Đây là **mirror bên thứ ba**, không gọi là publisher chính thức. Notebook kiểm
hash, dùng `torch.load(weights_only=True)` và `load_state_dict(strict=True)`;
không có unsafe-pickle hoặc model khác thay thế nếu không load được.
-Kiểm tra alphabet đầy đủ các nguyên âm Việt có dấu, chữđ và chữ hoa; kiểm
encode/decode roundtrip, số class recognition head và checkpoint shapes.
**Alphabet đầy đủ là điều kiện cần, không phải bằng chứng accuracy tăng.**

## Cách dùng trên Kaggle

1.Import file notebook mới, gắn dataset keyframes, bật **Internet** và **T4 GPU**
(T4x2 vẫn chỉ dùng physicalGPU0). Kernel **Linux Python3.13**.
2.Đổi `INPUT_ROOT`, ví dụ`/kaggle/input/my-keyframes/L01` hoặc một folder video
`.../L01/L01_V001`. Chỉ subtree đó được đọc. Không tự quét nhóm bên cạnh.
3.Giữ`RUN_FULL=False`, chạy **Run All**. Runtime được chuẩn bị và preview8ảnh
thật rải đều theo inventory; bulk/export sẽ chủ động bỏ qua, không tạo link cũ.
4.Xem ảnh có box và raw text/confidence bên dưới. Xanh: được đưa vào public;
đỏ: confidence thấp, token-truncate hoặc segmentation không đủ điều kiện; vẫn giữ trong diagnostics. Xanh
không đồng nghĩa đúng, confidence không phải xác suất correctness đã calibration.
5.Nếu ổn, đặt`RUN_FULL=True`, chạy lại cell cấu hình rồi cell **Full OCR** và
cell **Download**. Nếu đổi input hoặc confidence thì phải chạy lại setup/preview;
guard sẽ chặn dùng settings cũ. Full run yêu cầu profile model/config đúng với
preview gần nhất, không được dùng preview từ model cũ.
6.Tải cả hai ZIP dưới đây. Đánh giá một nhóm nhỏ trước khi chạy cả dataset.

`PREVIEW_COUNT=8` chỉnh được trong1–50; nếu dataset ít hơn thì dùng đủ số ảnh.
`PREVIEW_IDENTITIES=[]` lấy mẫu cách đều; danh sách không rỗng chọn chính xác frame.
`PUBLIC_MIN_CONFIDENCE=0.5` là mức lọc khởi đầu **chưa hiệu chỉnh**. Có thể đổi
để đánh giá trade-off; luôn giữ raw lines. Đổi mức lọc làm profile thay đổi nên
không tái dùng checkpoint cũ một cách âm thầm.

## Những thay đổi giúp đọc chữ nhỏ/dòng dài

| Phần | Notebook cũ | Notebook mới |
| --- | --- | --- |
| Recognizer | Latin PP-OCRv5 thiếu nhiều chữ Việt | VietOCR VGG-Transformer + alphabet/head gate |
| Detector max side | 960 | **1600** |
| Input recognizer | Paddle internal | Crop từ raster gốc, rectified theo polygon |
| Dòng dài | VietOCR upstream thường max width512 | V3: width>512 chia gap, mỗi đoạn ≤480; baseline cap2048 giữ trong diagnostics |
| Giới hạn decode | — | Greedy tối đa256token, flag nếu chưa EOS |
| Confidence | Public giữ mọi dòng nonblank | Threshold tường minh, raw giữ đủ ở ZIP chẩn đoán |
| Kiểm tra trước bulk | CUDA/runtime smoke | CUDA hai backend + hashes + alphabet/head + preview |

Wrapper vẫn chỉ pad trắng32px mỗi cạnh, không resize/rotate toàn ảnh gốc. Detector
có thể resize nội bộ; crop nhận dạng lấy pixels từ raster gốc đã pad, không từ
ảnh detector thu nhỏ. Recognizer crop được resize LANCZOS đúng tỉ lệ trong giới
hạn; `width_capped` ghi nếu buộc đạt maxwidth, không coi đó là khôi phục chi tiết.
`truncated` ghi nếu decoder chưa ra EOS trước token cap; dòng này không được đưa
vào public để tránh presenting text cụt như hoàn chỉnh.

Không super-resolution, tự thêm dấu, dịch, sửa chính tả bằng LLM, hoặc đoán chữ
trên ảnh mờ. VietOCR vẫn có thể đọc nhầm/hallucinate, nhất là logo, chữ nghiêng
trên bao bì và background không phải chữ. T4 mạnh hơn không chữa thông tin pixels
đã mất. V3.1 dùng crop batch tối đa16; chưa benchmark tốc độ thực tế so với scalar v3.

## Runtime và tương thích

-Giữ Paddle GPU3.2.2 / PaddleOCR3.2.0 / PaddleX3.2.0 / NumPy2.2.6 / Pillow11.3.
-Thêm Torch2.8.0+cu126, torchvision0.23.0+cu126, einops0.8.1 và PyYAML6.0.2 trong
**venv nhận dạng tách biệt hoàn toàn với Paddle**, không notebook kernel. Đã xác minh có wheel cp313 Linux CUDA12.6
cho Torch và torchvision; chưa chứng minh cả stack chạy được trên T4 Kaggle.
-VietOCR PyPI0.3.13 pin Pillow10.2 và augmentation stack cũ, không phù hợp
Python3.13. Notebook **không** ép `pip install vietocr --no-deps`; chỉ lấy allowlist
source inference từ commit cố định, kiểm Git blob hashes, giữ LICENSE. Không cần
imgaug/albumentations/gdown hay training loader.
-Pip dùng core constraints, binary wheels, `pip check`; không tự bỏ pin, compile
package, hạ phiên bản Python/Pillow hoặc fallback CPU nếu dependency/GPU lỗi.
-Cả Paddle và Torch phải thấy GPU0, có CUDA tensor smoke riêng. Torch pretrained
ImageNet bị tắt; chỉ load checkpoint recognition đầy đủ đã pin. Model/API/GPU/
alphabet/state-dict lỗi dừng trước bulk, không đổi thành status=empty.
-Paddle dùng auto_growth để tránh giữ trước gần hết GPU khi hai backend cùng
chạy. Tốc độ và memory thực tế vẫn cần kiểm chứng trên Kaggle.

## Output và bảo toàn dữ liệu

### `ocr_vietnamese_v3_batch16.zip` — giữ contract tích hợp cũ

```text
L01/L01_V001/001.json
L01/L01_V001/002.json
...
```

Mỗi file là array một record đúng7field:

```json
[{"video_id":"L01_V001","frame_id":null,"keyframe_n":1,
  "image_path":"data/L01/L01_V001/001.jpg","ocr_text":"Quảng Ngãi",
  "retrieval_text":"Quảng Ngãi","status":"ok"}]
```

Không đổi stem/padding/ID; n là keyframe number, không frame_idx hoặc n−1.
`retrieval_text==ocr_text`; giữ nguyên Unicode model trả. Dòng low-confidence/
truncated hoặc segmentation chưa đủ điều kiện có thể khiến publicempty: nghĩa unknown, không phải không có chữ.
Public vẫn không tự có confidence/engine/hash provenance; không tự đánh verified.

### `ocr_vietnamese_v3_batch16_diagnostics.zip` — cần giữ để kiểm tra chất lượng

-`profile.json`: model/source revisions, SHA-256 model assets và source, alphabet
test, core versions, inference options và confidence interpretation.
-`run.json`: inventory, invocation, kết quả/error counts.
-`checkpoints/Lxx/Lxx_Vxxx/NNN.json`: hash bytes ảnh gốc, raw text của **mọi** dòng,
confidence, polygons padded/original/clipped, crop/input sizes, flags và quyết
định đưa vào public. Không chứa ảnh gốc/model weights hay credentials.
-V3 thêm full_line baseline, segments và x_range_crop/SHA BGR, segmentation plan,
  decoder_eos/completeness/public_eligible/review_reasons. Không suy completeness từ EOS.

Như vậy có thể phân biệt detector crop mất dấu với recognizer đọc sai; không
phải suy confidence hay model từ public text như trước. Technical ZIP không
được trộn vào caption hoặc import như public JSON 7field.

Tên output và scratch/venv mới, không ghi đè`ocr.zip`, dataset gốc hoặc runtime
notebook cũ. Resume cùng phiên Kaggle yêu cầu hash ảnh/profile/checkpoint khớp;
session reset mất transient scratch, không có cross-session resume. Archive
không có ownership hoặc bị thay bytes bị từ chối, không xóa dữ liệu người dùng.
Không tự upload/replace OCR trên CPU trường hoặc đổi web đang dùng.

## Kiểm thử nền tảng trước v3 và giới hạn

-65tests liên quan sau chỉnh sửa cuối pass,30.479s: notebook compile/nbformat, full alphabet guard,
pinned download/cache mutation, strict public schema, raw Unicode/geometry,
confidence/truncation, hai ZIP tách biệt, chống ghi đè, preview không bulk,
decoder EOS/length/width caps và stale kernel state.
-Full regression:986tests,970pass/16skip,168.177s. Generated worker được đối chiếu
trùng source builder và notebook gốc giữ nguyên hash. Receipt:
`benchmark_results/kaggle_ocr_vietnamese_notebook_20261115_run01/FINAL_TESTS.json`.
-Chưa chạy thực tế detector+VietOCR weights trên Kaggle; **không công bố % tăng
accuracy** chỉ từ unit tests. Preview/gates được thiết kế fail-closed cho bước đó.
-Nên chọn20–50ảnh có bản chép chữ thủ công, chia ticker rõ / ticker nhỏ / chữ
vật thể mờ, đánh giá CER và khớp cụm chữ trước khi OCR lại hàng chục nghìn frame.

Build lại (không network/không model load):
`python -m tools.build_kaggle_ocr_vietnamese_notebook`.
Nguồn: `tools/kaggle_ocr_vietnamese_extension.py`; helper IO được nhúng từ notebook
cũ đã kiểm tra, nên file mới chạy độc lập trên Kaggle, không cần repository.
Setup tách runtime: `tools/kaggle_ocr_split_setup.py`; launcher loại global hook:
`tools/kaggle_ocr_isolated_launcher.py` (đều được nhúng sẵn trong notebook).
