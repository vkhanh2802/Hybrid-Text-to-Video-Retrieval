# Nộp kết quả KIS lên DRES v2

Planner text: Groq chính, có thể bật OpenCode fallback qua cấu hình local.
Xem [QUERY_PLANNER_FALLBACK.md](QUERY_PLANNER_FALLBACK.md). Chỉ fallback bước
cấu trúc query; không tự tìm/nộp DRES và không đổi corpus hoặc budget VLM.

## Kết nối

Giao diện dùng **duy nhất đề text/video chính thức từ DRES**, không có input
thay thế đề hoặc tải video thủ công. Với đề video có ô text bổ sung tùy chọn,
trộn 70% text + 30% video khi truy hồi.
Cấu hình đích trên server, không nhận URL tùy ý từ browser:

```powershell
$env:KIS_DRES_BASE_URL = 'https://aiclub.uit.edu.vn/dres'
$env:KIS_DRES_RUN_ID = '<evaluation-run-uuid>'
.\start_kis.ps1
```

Hoặc lưu **chỉ hai giá trị không bí mật** vào `config/dres.local.json` (được Git
bỏ qua). Env ưu tiên cao hơn file local:

```json
{"base_url":"https://aiclub.uit.edu.vn/dres","evaluation_id":"<evaluation-run-uuid>"}
```

`base_url` là gốc ứng dụng `/dres`, **không phải** link `/evaluation/viewer/...`.
HTTPS bắt buộc trừ localhost. API đối chiếu với OpenAPI 2.0.5 công khai tại:
<https://aiclub.uit.edu.vn/dres/openapi.json>.

1. Nhập Run ID (tự điền từ cấu hình nếu có).
2. Chọn **Session ID** hoặc **Username / password**, không dùng cả hai.
3. Bấm **Kết nối DRES**. Chỉ xác thực và đọc run/task, chưa gửi đáp án.
4. Sau khi xác thực (kể cả khôi phục session), khối **Kết nối & nộp bài** tự ẩn.
   Phần **Đề hiện tại** thay thế, có **Làm mới đề**, link viewer và **Đổi tài khoản**.
   Chọn frame từ card và dialog xác nhận vẫn được giữ, không tự nộp bài.

## Hiển thị đề text/video hiện tại

Frontend **không tự cập nhật theo chu kỳ**: đọc đề khi kết nối, khôi phục session,
bấm **Làm mới đề**, hoặc bấm **Tìm theo đề DRES**. Mỗi lần tìm sẽ đọc lại hint
mới nhất và dùng toàn bộ các câu đã phát trong task, không cần reset, reload
hay kết nối lại. Nút làm mới chỉ cập nhật hiển thị, không tự chạy model.
Backend dùng trạng thái để chọn đúng
`taskTemplateId`, chỉ gọi API **hint** của task đang RUNNING, không gọi API
target/đáp án hoặc tải toàn bộ evaluation. Chỉ dùng hint có offset (giây) đã đến
theo `timeElapsed`; không phát gợi ý trước countdown.

- Text: **cộng dồn mọi câu đã phát** theo offset và hiển thị đầy đủ ở frontend.
  Phần tử TEXT rỗng bắt đầu chuỗi mới (xóa các câu trước đó). Nhiều hint TEXT
  được ghép thành một query chính thức, không chỉ lấy câu đầu.
- Text hiển thị nguyên văn bằng `textContent`, không render HTML từ upstream.
- **Tìm theo đề DRES** gọi POST `/api/dres/search` với `task_id`, và tùy chọn
  `video_description` (tối đa 2000 ký tự, chỉ cho đề video). Backend tự lấy
  text/video đang được phát, không nhận `query`, URL hoặc video thay thế đề.
  Chưa đăng nhập/chưa có đề phù hợp/task dừng thì nút bị khóa.
- Trong khối task có tùy chọn **Tự nhập query thay đề hiện tại**. Bật tùy chọn
  này thì nút Tìm gọi POST `/api/search` với query tự nhập, không đọc hint DRES,
  không đổi đề và không tự nộp. Tắt tùy chọn để quay lại đề chính thức. Kết quả
  tự nhập không phải bằng chứng khớp đề; chỉ chọn nộp frame sau khi tự kiểm tra,
  và preview/submit vẫn dùng task DRES đang hiển thị.
- Text gọi pipeline Groq/retrieval/Qwen hiện có với toàn bộ câu đã phát. Video
  tải từ media hint đã xác thực; với task visual ưu tiên video, còn lại ưu tiên
  text nếu DRES phát cả hai.
- Video chính thức: 0,1–60 giây, 24 MiB, tối đa 6 frame trải trên clip bằng
  adaptive FPS (tối đa 2 FPS). Không cắt ngầm về 3 giây; clip quá giới hạn báo
  lỗi. Nhánh API upload cũ vẫn 1–3 giây/20 MiB cho tooling, không hiện ở frontend.
- Sau inference kiểm tra lại task và fingerprint hint; nếu đổi/dừng task hoặc
  phát hint mới, bỏ kết quả cũ với 409. Cả trước và sau inference đều đọc lại
  sequence, không dùng cache cũ để xác nhận. Frontend cập nhật đề khi nhận 409,
  không tự retry model; bấm Tìm lại, không cần reset. Frontend xóa card cũ khi đổi đề.
  Chọn nộp frame kèm task ID đang hiển thị, không dựng preview cho task mới
  bằng frame người dùng vừa chọn từ task cũ.
- Video/ảnh đề dùng fetch có header cùng-origin rồi Blob URL. Backend chỉ lấy
  media hint thuộc đúng evaluation/task hiện tại; không proxy URL tùy ý.
- Đổi/dừng task, ngắt kết nối, hết session hoặc lỗi đồng bộ: xóa đề/media cũ và
  preview chưa gửi; thu hồi Blob URL. Không tự logout session upstream.
- Nếu ban tổ chức chặn quyền xem hint hoặc media quá lớn (hint JSON 32 MiB,
  media 24 MiB), giao diện báo rõ và giữ link **Mở trang DRES**, không suy đề từ
  tên task. Khởi động lại backend yêu cầu kết nối lại vì session nằm trong RAM.

## 200 frame và budget LLM

Mỗi response tìm kiếm thành công trả **đúng 200 frame khác nhau** để chọn nộp,
kể cả đề có chuỗi sự kiện. Thiếu ứng viên sẽ mở rộng độ sâu truy hồi 200 → 400
→ 800 → … → 6400, luôn trong cùng scope và cùng query. Không nhân bản frame,
không bổ sung frame tùy ý từ corpus. Hết budget mà vẫn không đủ thì trả 503
với số frame tìm được, không trả vài frame như một lượt thành công. Đây là hợp
đồng số identity frame từ index, **không đảm bảo 200 JPEG đều đọc được từ nguồn ảnh**;
ảnh không truy cập được vẫn báo thiếu thật.

Với text, chỉ
**20 ứng viên đầu trước khi xếp hạng lại** được đưa vào Qwen LLM/VLM hiện có,
kể cả profile `verified40`. Không bù thêm ảnh ngoài top 20 khi thiếu ảnh/lỗi LLM.
Card hiển thị điểm LLM thật hoặc trạng thái chưa chấm/lỗi; thứ hạng cuối có thể
đổi sau rerank, không khẳng định mọi card top 20 cuối cùng đã được LLM chấm.

Nhánh video **không có text kèm theo** chỉ dùng SigLIP (không chấm LLM). Khi có
text (câu DRES đã phát và/hoặc text người dùng nhập), phần text chạy pipeline
text đầy đủ, rồi fusion hai danh sách **70% text + 30% video** bằng weighted RRF
(k=60). VLM chỉ xét 20 ứng viên đầu sau fusion, không chấm lặp ở nhánh text.
Temporal hiển thị riêng ở **Chuỗi sự kiện bổ sung**, trong response
`temporal_results`, với `temporal.presentation="auxiliary"`. Chuỗi không còn
thay thế danh sách frame hoặc làm kết quả chính giảm xuống 5–12 card. Temporal giữ
các chuỗi bằng chứng thật tìm được, không tạo thêm chuỗi/frame giả và
không dùng điểm ảnh đơn để khẳng định đã xác minh cả chuỗi. Ảnh kết quả được tải
khi gần viewport, tối đa 4 request đồng thời, không tải 200 ảnh cùng lúc.

Giới hạn web dùng chung là `WEB_RESULT_LIMIT=200` trong `kis/result_pool.py`.
CPU worker phải báo cùng `result_limit=200` khi handshake; worker phiên bản cũ
không được dùng âm thầm. Chế độ tự nhập query cũng trả 200 frame, không gọi DRES.
Các helper/benchmark có thể vẫn yêu cầu số frame riêng; không thay đổi kết quả
benchmark lịch sử bằng cách đổi số 100 trong artifact cũ.

### Text bổ sung cho video DRES

Khi input chính thức là video, ô **Text bổ sung cho query video (không bắt buộc)**
xuất hiện bên dưới phần nguồn đề. Ví dụ: “Tập trung vào người áo đỏ bên phải,
đang cầm một chiếc cốc”. Ô này không thay thế video, không là đáp án QA và không
được gửi sang DRES như bài nộp.

- Không có text DRES lẫn text bổ sung: query chỉ dùng hình ảnh video; SigLIP trả
  đủ 200 frame hoặc báo lỗi thiếu dữ liệu, không gọi LLM.
- Có text: ghép mọi câu text DRES đã phát với text bổ sung, chạy **cùng pipeline
  với đề text**: Groq cấu trúc query → SigLIP text-to-image + BM25 + semantic
  caption → fusion nội bộ + object evidence → OCR/text reranker/temporal theo
  cấu hình. VLM được hoãn đến sau khi hợp nhất với video.
- Nhánh video vẫn dùng clip chính thức → SigLIP image encoder → mean pool + L2
  normalize → truy hồi 200 frame, không trộn text vào vector video.
- Hợp nhất hai danh sách bằng **weighted RRF**, không cộng điểm thô khác thang đo:
  `score(frame) = 0,7 × 61/(60 + rank_text) + 0,3 × 61/(60 + rank_video)`.
  Frame không có trong nhánh nào nhận đóng góp 0 từ nhánh đó. Loại frame trùng,
  giữ metadata/bằng chứng nhánh text, chọn 200 frame trước VLM.
- Tổng text đã ghép tối đa **8000 ký tự**; riêng ô bổ sung tối đa **2000 ký tự**.
  Vượt tổng giới hạn thì báo lỗi để rút ngắn text bổ sung, không cắt ngầm query.
- Sau truy hồi hybrid, Qwen LLM/VLM hiện có so **ảnh ứng viên với query text**
  cho 20 ứng viên đầu trước rerank. Không đổi sang mô hình khác, không chấm bù
  ứng viên thứ 21 trở đi khi ảnh thiếu/lỗi; phần còn lại giữ điểm truy hồi.
- Response trả `hint_text`, `combined_text_query`, `supplemental_description`,
  `video_text_guidance.text_fusion` (trọng số thật), `llm_scoring_limit` và
  `llm_scored_count`; UI hiển thị đúng text và trọng số đã dùng.
- Đây không phải LLM đọc toàn video hoặc xác minh chuyển động/chuỗi. Khi tắt
  VLM chấm ảnh, text vẫn chạy pipeline đầy đủ và tham gia fusion. Nhánh text có
  temporal vẫn tìm danh sách frame thông thường; VLM chấm top 20 frame, **không
  chấm chuỗi temporal bổ sung**, không tuyên bố đã xác minh chuỗi. Điểm ảnh lỗi/thiếu không bị giả.
- Nếu planner text lỗi/thiếu key, báo lỗi thay vì âm thầm bỏ text và chỉ tìm video.
- Text được giữ khi làm mới cùng task; xóa khi đổi/dừng task, chuyển sang đề
  text hoặc ngắt kết nối. Không lưu localStorage/Git; đang tìm thì khóa ô để
  không đổi text giữa request. Nhấn tìm lại để áp dụng text mới.
- Câu mới trong cùng task không tải lại video đang phát nếu media không đổi.
  Tổng text DRES vượt 8000 ký tự báo lỗi tìm kiếm, không âm thầm chỉ dùng đoạn đầu.

Thông tin đăng nhập không nằm trong mã nguồn, HTML hay localStorage. Password
chỉ dùng lúc login; session DRES nằm trong RAM backend. Browser chỉ nhận cookie
local ngẫu nhiên `HttpOnly`, `SameSite=Strict`, có hạn 8 giờ. Khởi động lại server
hoặc **Ngắt kết nối** xóa khả năng dùng phiên local; không logout phiên DRES ở
các ứng dụng khác. UI xóa ô password/session sau lần kết nối.

Chỉ bind local theo cấu hình web mặc định. Nếu mở server cho nhiều máy/người dùng,
cần thêm xác thực truy cập ứng dụng và HTTPS ở tầng triển khai; đây không phải
một portal đăng nhập công khai.

## Nộp một frame

1. Bấm **Tìm theo đề DRES** khi task đang phát đề; kiểm tra kết quả.
2. Bấm **Chọn frame để nộp DRES** trên card. Với kết quả temporal có thể chọn
   riêng frame của một event; không tự nộp toàn bộ path.
3. Backend đối chiếu `video_id + image_number + frame_id` với mapping CSV của
   scope hiện tại. Lấy **`pts_time`**, chuyển giây sang mili-giây; không dùng số
   ảnh JPEG, `frame_id` hoặc FPS để đoán thời gian.
4. Kiểm tra dialog: video, IMG, FRAME, timestamp, run, task và JSON thực tế.
5. Bấm **Xác nhận gửi bài thật**. **Hủy** hoặc Escape không gửi đáp án.

Payload là temporal point (`start=end`) theo schema DRES v2, ví dụ minh họa:

```json
{
  "answerSets": [{
    "taskId": "<current-task-run-id>",
    "answers": [{"mediaItemName":"L01_V001","start":6000,"end":6000}]
  }]
}
```

Tên item hiện gửi nguyên `video_id` (`L01_V001`, không thêm `.mp4`). Collection
để DRES suy từ task theo schema của server. Cần đúng naming/corpus mà ban tổ
chức cấu hình. Nhánh frame không tự chọn media collection khác và không tự submit
một evidence span như thể nó là clip boundary đã xác minh.

## Đáp án chữ cho QA

Khi nhãn task type của DRES cho biết `Question & Answer`/`QA`, giao diện hiển thị
ô **Đáp án chữ cho câu QA hiện tại** và chặn chuẩn bị payload frame. Nhập đáp án
**thủ công** đúng định dạng yêu cầu của đề, bấm **Xem trước đáp án QA**, kiểm tra
JSON rồi xác nhận gửi. API dùng `answers: [{"text":"..."}]`, chỉ bỏ khoảng trắng
đầu/cuối; không tự thêm tên video, frame hay nội dung từ model. Task type là nhãn
client API, DRES vẫn quyết định payload có hợp lệ theo submission filters thực tế.

Model hiện chỉ gợi ý frame, **không tự sinh đáp án QA**. Khi làm mới và task ID
đổi, ô đáp án cũ và preview cũ được xóa để tránh nộp câu trước vào task mới.

Preview có hạn **60 giây**, chỉ dùng một lần. Trước khi gửi backend đọc lại trạng
thái, yêu cầu run ACTIVE/task RUNNING, cùng task ID và cùng mapping (với frame). Payload
không nhận tự do từ browser; server gửi đúng preview đã giữ trong RAM.

## Kết quả và lỗi

- HTTP 200/202 là phản hồi của DRES, **không mặc định là đáp án đúng**. UI hiển
  thị riêng `CORRECT`, `WRONG`, `INDETERMINATE`, v.v.
- HTTP 202 hoặc `INDETERMINATE`: đang chờ chấm. Xem trang DRES để biết verdict
  cập nhật; không tự poll toàn bộ bài nộp của các đội khác.
- 401: session hết hạn hoặc credentials không hợp lệ, hãy kết nối lại.
- 404: run không tồn tại hoặc tài khoản không tham gia/không được thấy run.
- 409 local: preview hết hạn/đã dùng, task thay đổi/dừng hoặc mapping đổi; chưa
  gửi bài. Chọn và kiểm tra lại.
- 412 DRES: bị submission filter từ chối (trùng, giới hạn hoặc quy tắc task).
- Timeout/5xx/JSON lỗi khi POST: **trạng thái ghi nhận có thể không rõ**. Token
  xác nhận đã bị tiêu thụ, không tự retry. Kiểm tra trang DRES trước khi chủ động
  nộp lại để tránh mất lượt vì gửi trùng.

## API local và an toàn

Các route `/api/dres/*` yêu cầu header `X-KIS-DRES: 1`, không cho Origin khác
origin ứng dụng. Không chuyển hướng request mang session đến host khác; upstream
dùng cookie `SESSIONID` thay vì đưa token lên URL. Response cấu hình/trạng thái
không trả session/password. Lỗi validation không echo request input chứa secret.

| Route | Tác dụng |
| --- | --- |
| GET `/api/dres/config` | URL/run không bí mật do server cấu hình |
| POST `/api/dres/connect` | Session hoặc login; xác thực và đọc trạng thái |
| GET `/api/dres/status` | Đọc trạng thái run/task của phiên browser |
| GET `/api/dres/task` | Đề text/media hint đang được phát; `?refresh=1` đọc lại danh sách hint khi người dùng bấm Làm mới |
| POST `/api/dres/search` | Đề chính thức: `task_id`, tùy chọn `video_description` cho video; gộp mọi câu text đã phát; không tự nộp bài |
| GET `/api/dres/task/media/{task_id}/{index}` | Media hint đã được phát của task hiện tại |
| POST `/api/dres/disconnect` | Xóa phiên local trong RAM |
| POST `/api/dres/preview` | Kiểm tra mapping, dựng JSON, chưa gửi |
| POST `/api/dres/preview/text` | Dựng JSON đáp án QA thủ công, chưa gửi |
| POST `/api/dres/submit` | `preview_id` + `confirmed: true`, gửi một lần |

Kiểm thử bằng fake DRES (không tiêu lượt thi thật):

```powershell
python -m unittest tests.test_dres tests.test_dres_task tests.test_video_query tests.test_input_frontend -b
node --check web/app.js
```

Kiểm tra kết nối/status live không chứng minh DRES sẽ chấm một đáp án là đúng.
Không thực hiện “test submit” lên cuộc thi thật nếu chưa có xác nhận đáp án của
người dùng.
