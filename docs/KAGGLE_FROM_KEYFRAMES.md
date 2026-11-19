# Kaggle: tạo artifacts từ keyframes đã có, không xử lý video

Notebook mới: `notebooks/keyframesprocessing.ipynb`, được tạo theo workflow của
`notebooks/videoprocessing.ipynb`. Notebook gốc được giữ nguyên.

## 1. Chuẩn bị

- Bật GPU và Internet trên Kaggle.
- Upload/add notebook mới và gói code `dist/kis-kaggle-from-keyframes.zip` thành
  Input đã giải nén, hoặc giải nén gói code vào `/kaggle/working`.
- Gói full mới chứa `tools/kaggle_prepare_keyframes.py`; không dùng gói cũ thiếu
  runner này. Tạo lại gói tại project root bằng:

```powershell
python -m tools.package_kaggle --profile full --output dist/kis-kaggle-from-keyframes.zip
```

Input ảnh JPEG (JPG/JPEG, tên số) có thể lồng nhiều cấp:

```text
keyframes/
  L03/
    L03_V001/
      001.jpg
      002.jpg
mapping/
  L03/
    L03_V001.csv
```

CSV phải là mapping gốc của bộ keyframes đó, header đúng thứ tự
`n,pts_time,fps,frame_idx`, UTF-8 không BOM. Số n liên tiếp từ1, frame_idx không
giảm, FPS dương, timestamp không giảm. Tên `0001.JPEG` được copy nguyên
bytes thành `001.jpg` để phù hợp loader, không lấy mẫu hay nén ảnh lại.

### Mapping L21 có frame_idx trùng

Nếu `001.jpg` và `002.jpg` cùng `frame_idx=0`, runner giữ **cả hai ảnh và hai
dòng CSV**, không bỏ ảnh đen hay đổi số n/frame_idx. Caption, objects, features
và caption index được tạo theo từng `(video_id, keyframe_n)`; hai vector vẫn có
thể có cùng frame_id gốc nhưng image_path/keyframe_n khác nhau.

Preflight in WARNING, lưu danh sách `frame_idx_aliases` trong preparation từng
video và `keyframes_run.json`. Checkpoint chỉ bật `allow_frame_idx_aliases`
cho import keyframes. Workflow trích xuất video vẫn bắt frame_idx tăng nghiêm
ngặt. Các lỗi n trùng/thiếu, frame_idx giảm/âm, FPS không hợp lệ, time giảm và
ảnh thiếu vẫn bị chặn. Phải cập nhật cả notebook và toàn bộ ZIP code.

**Đây là chế độ bảo toàn mapping chưa xác minh, không phải sửa frame ID đúng.**
Khi nhập artifacts vào hệ thống retrieval/submission sau này, không deduplicate
hay chọn ảnh chỉ bằng `(video_id, frame_id)` cho các alias. Pipeline retrieval
hiện tại có thể loại hoặc gộp những frame ambiguous; cần xử lý riêng trước khi
dùng chúng làm GT/submission. Notebook này không tự thay production retrieval.

Bản sao có hậu tố số trong ngoặc như `171 (1).jpg`, `171 (2).JPEG` hoặc
`L21_V001 171 (1).jpg` trong folder `L21_V001` được **bỏ qua** và ghi số lượng
trong log. Không xóa file nguồn, không chép bản sao vào output hay đổi mapping.
Nếu chỉ còn bản `(1)` mà thiếu ảnh gốc tương ứng CSV, preflight vẫn báo thiếu
coverage. Tên lỗi khác và nhiều ảnh gốc cùng số (`171.jpg` + `0171.jpeg`) vẫn
bị chặn để tránh chọn nhầm dữ liệu. Cần cập nhật cả notebook và gói code ZIP.

**Không có CSV thì không thể suy ra frame_idx/thời gian chính xác từ001.jpg.**
Notebook fail preflight và yêu cầu bổ sung mapping, không tạo identity giả.
PNG không tự đổi lossy sang JPEG; bộ input cần JPEG như pipeline hiện tại.
Nếu các Input vẫn là ZIP chưa giải nén, giải nén vào một thư mục riêng rồi đặt
ROOT trỏ đến thư mục đó. Không giải nén dữ liệu vào CODE_DIR hoặc OUTPUT_ROOT.

## 2. Sửa cell cấu hình, rồi Run All

- `CODE_DIR`: thư mục có `kis/`, `tools/`, `config/`, `requirements-kaggle.txt`.
- `KEYFRAMES_ROOT`: thư mục input ảnh; có thể trỏ cả group hoặc cả nhiều group.
- `MAPPING_ROOT`: thư mục CSV mapping (có thể cùng dataset với ảnh).
- `OUTPUT_ROOT`: `/kaggle/working/kis_from_keyframes`, tách biệt các input.
- `VIDEO_IDS = None`: tất cả; hoặc `['L03_V001']` để thử trước một video.
- Batch mặc định: visual8,object8,caption16; InternVL là caption backend mặc định.
- `INCLUDE_IMAGES_IN_ZIP = False`: ZIP chỉ artifacts, không tải lại ảnh đã có.

Preflight kiểm toàn bộ input trước khi nạp GPU model. Các ảnh và CSV sau đó được
copy sang Working, hash/checkpoint để resume an toàn. Cần đủ dung lượng Working
cho **một bản sao ảnh input và toàn bộ artifacts**; không sửa Kaggle Input.

## 3. Stage giữ lại và output

```text
Import JPEG + mapping gốc
  → SigLIP features (float32)
  → object detection
  → InternVL captions
  → sentence-transformer embeddings + FAISS text index
  → validate schema/count/hash/identity → ZIP
```

Không mở video, không decode, không chạy random-gap sampling, không tạo lại
keyframe. Model/revision/format dùng lại code pipeline đang có, không thay bằng
caption model khác. Runner chạy model lần lượt, release giữa stage.

Output dưới `OUTPUT_ROOT`:

```text
data/                 # bản copy JPEG để chạy, không kèm ZIP mặc định
mapping/              # CSV gốc, giữ bytes
clip-features-32/     # visual embeddings .npy
objects/              # object JSON
caption_generator/    # caption JSON
caption_mapping/      # join caption/index với frame identity
index/                # FAISS .index + .meta.json
preparation/          # checkpoint/provenance
keyframes_run.json    # trạng thái toàn lượt
```

ZIP hoàn tất: `/kaggle/working/kis_from_keyframes-from-keyframes.zip`.
Nếu stage lỗi, cố lưu ZIP `-partial` và giữ checkpoint. Không kết luận artifacts
đã hoàn tất khi manifest còn `partial`.

Resume: cùng output/input/config thì tái dùng artifacts qua validation. Không
ghi đè ảnh/mapping khác hash. Khi đổi session, restore toàn bộ Working cũ (gồm
data/preparation) để resume, không chỉ ZIP mặc định vốn không chứa ảnh. Nếu không
có đầy đủ checkpoint, dùng output root mới. `RESTART_CAPTIONS=True` chỉ dùng khi
cố ý archive caption trước đó và tạo lại; không đổi numbering keyframes.

Để nhập về local, giải nén artifacts vào folder staging riêng và dùng quy trình
`tools.ingest` hiện có. Không tự chép đè production dataset hoặc restart web từ
notebook này. Notebook không chạy OCR, planner, Qwen rerank hoặc DRES.

## Kiểm chứng

Unit tests gồm: mapping không đoán frame_idx, duplicate/missing/corrupt input,
copy giữ nguyên bytes, resume và bảo vệ dữ liệu khác hash, archive có/không ảnh,
partial checkpoint, stage orchestration không gọi extractor và gói code chạy
preflight độc lập ngoài repository. Chưa chạy toàn bộ ML trên Kaggle trong session
phát triển này; GPU/model download/runtime vẫn phải xác nhận ở Kaggle.
