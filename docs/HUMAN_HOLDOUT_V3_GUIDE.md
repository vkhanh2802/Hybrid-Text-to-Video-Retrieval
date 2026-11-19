# Bạn cần làm gì với bộ holdout mới?

## Đã chuẩn bị xong

- **12 nhóm × 5 keyframe = 60 ảnh**, đọc Drive thành công toàn bộ.
- Seed `2026110203`, chỉ split `holdout`, không có development trong gói này.
- Loại **303 video** đã có trong nhãn/mẫu cũ hoặc các pool được chọn/chấm cục bộ.
- Bộ quy tắc `garment_attributes_v2`, cấu hình và ngưỡng production đã được lưu
  hash/snapshot **trước khi tải ảnh mới**. Không đổi quy tắc theo nhãn holdout.
- Không đọc caption corpus, không chạy retrieval và không để AI tự điền nhãn
  cho các ảnh mới. Trang HTML ban đầu không có câu query hoặc judgment nào.

**Giới hạn cần hiểu đúng:** seed hits của các lượt tìm kiếm trước từng bao phủ
phần lớn corpus; preflight cũng liệt kê toàn bộ 535 video. Không thể nói corpus
chưa từng được truy cập. Bộ này kiểm tra **câu hỏi/nhãn mới sau khi khóa quy
tắc**, loại target và local scoring pool cũ, không loại mọi vector từng có mặt
trong một kết quả search thô. Video-disjoint cũng không bảo đảm nội dung không
phát lại hoặc gần trùng. Nếu nhận ra nội dung đã biết, ghi vào ô Ghi chú.

Lượt chuẩn bị đầu tiên với tiêu chí “loại mọi ID xuất hiện trong JSON” không
còn đủ video và đã dừng **trước khi tải ảnh**. Thư mục thất bại được giữ lại để
không che lịch sử; **chỉ dùng gói `_v2` dưới đây**. Không đổi seed để chọn ảnh đẹp.

## 1. Mở trang gán nhãn

Mở File Explorer đến:

```text
C:\Users\Khanh\Desktop\KISBranche\benchmark_results\human_holdout_v3_seed2026110203_v2
```

Nhấp đúp **`review.html`**, mở bằng **Chrome hoặc Edge**. Giữ file HTML cùng
thư mục `images` bên cạnh; không kéo riêng HTML sang chỗ khác.

Hoặc chạy PowerShell:

```powershell
Start-Process 'C:\Users\Khanh\Desktop\KISBranche\benchmark_results\human_holdout_v3_seed2026110203_v2\review.html'
```

Trang chạy trực tiếp trên máy, không cần server hoặc mạng khi gán nhãn. Không
cần terminal, Python, Node hay Kaggle cho phần của bạn. Nếu khung xem HTML trong
OpenCode chặn JavaScript hoặc tải file, mở bằng Chrome/Edge như trên.

## 2. Điền tên hoặc bí danh

- Không cần email, tên thật hoặc tài khoản mới.
- Đọc ô xác nhận “Tôi tự xem ảnh để gán nhãn…”. Chỉ tích khi đúng với cách bạn làm.
- Không mở caption tự sinh hoặc kết quả retrieval để đoán đáp án.

## 3. Làm lần lượt 12 nhóm ảnh

Mỗi nhóm có 5 ô; dưới ảnh ghi `n=...` (số keyframe) và `t=...` (giây).

### A. Viết một mô tả cho event 1

Viết **tiếng Việt**, chỉ gồm chi tiết nhìn thấy rõ. Ví dụ minh họa, **không phải
gợi ý nội dung ảnh thật của bộ mới**:

> Một người đàn ông mặc áo sơ mi xanh trơn, đeo cà vạt đỏ, đứng trong trường quay.

- Nếu màu/họa tiết trang phục nhìn rõ thì có thể ghi; nếu cảnh không có người,
  mô tả cảnh/đồ vật bình thường, **không ép thành câu về trang phục**.
- Không suy danh tính, địa điểm, nguyên nhân hoặc thao tác liên tục từ ảnh tĩnh.
- Có thể để trống tiếng Anh. Tôi sẽ dịch từ mô tả của bạn, không xem caption/
  scores; sau đó gửi bạn duyệt bản dịch trước khi khóa nhãn.
- Nếu tự viết tiếng Anh, kiểm tra lại rồi tích xác nhận tiếng Anh của event đó.
  Sửa câu tiếng Anh sẽ tự bỏ tích để tránh duyệt nhầm bản cũ.

### B. Chọn nhãn cho từng ảnh đối với câu vừa viết

| Lựa chọn | Khi nào dùng? |
|---|---|
| **Đúng đầy đủ (positive)** | Ảnh cho thấy đủ các điều kiện đã viết. |
| **Sai rõ một điều kiện (negative)** | Có mâu thuẫn nhìn thấy rõ, ví dụ áo trắng thay vì áo xanh. |
| **Không đủ rõ (unknown)** | Mờ, quá nhỏ, bị che, không thể kiểm tra một chi tiết. |
| **Chưa xem / chưa quyết định** | Chỉ là trạng thái nháp; phải chọn lại trước khi xuất nhãn hoàn tất. |

**Không nhìn thấy ≠ chứng minh sai.** Ví dụ không phân biệt được màu cà vạt do
ảnh nhỏ thì chọn `unknown`, không chọn `negative`.

- Mỗi event cần ít nhất một positive ngay trong nhóm của nó.
- Nếu cả 5 ảnh đều đúng, cứ gán cả 5 là positive; **không tạo negative giả**.
- Tùy chọn “thêm ảnh đối chiếu” cho phép dùng ảnh nhóm khác. Hãy chuyển qua nhóm
  đó xem ảnh bằng mắt trước khi gán nhãn; không suy từ video ID/tên nhóm.
- Các ảnh contact sheet là thumbnail: nếu phóng to vẫn không đủ rõ, dùng unknown.

### C. Chỉ thêm event 2–3 khi có thứ tự cảnh rõ

Ví dụ minh họa: đầu tiên thấy mặt ngoài tòa nhà, sau đó một người đang nói trong
phòng. Mỗi event có câu riêng và **một bảng nhãn riêng cho 5 ảnh**.

- Positive của event sau phải có `t` lớn hơn event trước, khoảng cách mỗi bước
  không quá **60 giây** theo cấu hình đang khóa.
- Không dùng cùng timestamp để xác nhận hai bước liên tiếp.
- Nếu chỉ lặp một cảnh, giữ **một event**. Không bắt buộc mỗi nhóm có chuỗi.
- Nếu không có chuỗi phù hợp, ghi lý do ngắn ở ô Ghi chú; ít chuỗi là giới hạn
  của bộ thử, không phải lỗi cần bạn “điền cho đủ”.

### D. Đánh dấu trạng thái nhóm

Chọn **“Đã xem và gán nhãn”** khi hoàn thành nhóm.

Nếu không thể viết câu nào tin cậy: chọn **“Bỏ qua…”**, ghi lý do vào ô tương
ứng. Không tự thay video khác. Nhóm bỏ qua không bị biến thành negative.

## 4. Lưu và gửi lại cho tôi

- Có thể làm trước **3–4 nhóm**, bấm **“Tải bản nháp JSON”**, làm tiếp sau bằng
  ô “Mở lại file JSON đã lưu”. Bạn cũng có thể gửi bản nháp để tôi kiểm tra định
  dạng; chưa chạy đánh giá từ nhãn chưa hoàn tất.
- Lưu trình duyệt là tiện ích phụ; `file://` hoặc chế độ riêng tư có thể không
  hỗ trợ. **File JSON tải xuống mới là bản sao lưu bạn giữ.**
- Hoàn tất 12 nhóm (hoặc ghi rõ nhóm bỏ qua), bấm **“Xuất nhãn hoàn tất”**.
- File tải về thường nằm ở **Downloads**, tên **`human_annotations.json`**.
  Nếu trình duyệt thêm `(1)`, cứ gửi đúng file mới tải.
- **Đính kèm file JSON đó trong cuộc trò chuyện này.** Không cần gửi 60 ảnh,
  caption, mật khẩu, token hoặc Drive ID.

Ước lượng **20–40 phút**, có thể lâu hơn nếu thêm nhiều event/ảnh đối chiếu.

## 5. Phần tôi làm sau khi bạn gửi file

1. Kiểm tra file thuộc đúng gói, các ảnh/nhãn đã xem đầy đủ, không gán cho ảnh
   lỗi, không lẫn `unreviewed`, và các positive có thể ghép đúng thứ tự.
2. Nếu cần, dịch câu tiếng Việt sang tiếng Anh **chỉ từ câu bạn viết**, giữ nguyên
   ảnh và judgments; gửi lại bạn duyệt nghĩa. Chưa chấm retrieval ở bước dịch.
3. Chỉ sau khi bạn duyệt câu tiếng Anh: freeze file nhãn mới, lưu hash, không
   sửa lại nhãn theo kết quả model. Nhãn human có schema/provenance riêng, không
   đổi tên thành nhãn AI để vượt guard của công cụ cũ.
4. Đánh giá hai chế độ đã khóa `off` và `semantic_guard` trên cùng scores/pool;
   báo riêng frame/path metrics, số query đủ cả hai lớp và số `unknown`.
5. Nếu không đủ trường hợp áp dụng hoặc không cải thiện, báo **chưa đủ bằng
   chứng**; không tune theo holdout hoặc tự bật mặc định.

Tiêu chí đã lưu trước khi lấy mẫu: tối thiểu 8 event queries có cả positive và
negative, 4 chuỗi được gán nhãn; FPR giảm mà TPR không giảm, không regression
known-frame/path. Đây chỉ là điều kiện xem xét, **không tự động promote** và
không bảo đảm ý nghĩa thống kê trên bộ nhỏ. Bạn không phải tạo nhãn để đạt chỉ tiêu.

**Chưa có việc model nặng cần bạn chạy.** Nếu sau này thật sự cần VLM hoặc sinh
lại caption, tôi sẽ dừng trước khi chạy và cung cấp notebook Kaggle, đầu vào,
cấu hình GPU và file đầu ra cần gửi lại.

## Công cụ kỹ thuật (bạn không bắt buộc chạy)

Kiểm tra nhãn tiếng Việt/draft đã hoàn tất, chưa yêu cầu tiếng Anh:

```powershell
python -B -m tools.human_holdout validate --packet benchmark_results/human_holdout_v3_seed2026110203_v2 --annotations PATH_TO_HUMAN_JSON
```

Chỉ sau khi toàn bộ tiếng Anh đã được người gán nhãn duyệt:

```powershell
python -B -m tools.human_holdout freeze --packet benchmark_results/human_holdout_v3_seed2026110203_v2 --annotations PATH_TO_APPROVED_JSON --output benchmark_results/human_holdout_v3_seed2026110203_v2/frozen_human_labels_v1.json
```

Freeze từ chối ghi đè nhãn và **không chạy model**. Công cụ chấm development cũ
không nhận nhãn holdout mới này; không bỏ các split/provenance guards để chạy.

## Kiểm tra bàn giao

- Full suite sau khi thêm gói human holdout: **296 tests, 282 đạt, 14 skipped**.
  Các dòng publish/staging trong unit tests là mock/thư mục tạm, không rebuild
  Milvus thật.
- Browser smoke test dùng riêng ảnh hình chữ nhật giả lập trong thư mục tạm,
  **không mở ảnh holdout thật**: xuất JSON, giữ nhãn unknown, khôi phục dữ liệu
  khi chuyển nhóm, chặn file thuộc gói khác, chặn hai event cùng timestamp, chấp
  nhận thứ tự hợp lệ và tự bỏ duyệt tiếng Anh khi câu được sửa đều đạt.
- Gói thật được kiểm tra metadata: 12/12 nhóm chỉ là holdout; 60/60 ảnh có hash;
  không trùng 303 video đã loại; mọi event/nhãn vẫn trống; hash inference rules
  vẫn khớp bản đã khóa. Chưa đánh giá trên nhãn mới vì đang chờ người dùng.
- `git diff --check` đạt, chỉ có cảnh báo line endings trên các file có sẵn.
  Không đổi ngưỡng production, tải model, commit hoặc push.

## Trạng thái sau khi nhận nhãn của người dùng

Đã nhận `human_annotations.json`, kiểm tra cấu trúc, hash bộ ảnh và inference
sources khớp protocol. **Không sửa file người dùng nộp**:

- 11 nhóm được gán nhãn, 1 nhóm (`drive_02`) bỏ qua vì ảnh không rõ.
- 26 event queries có cả positive/negative, gồm **42 positive, 86 negative,
  2 unknown**; không có nhãn `unreviewed` trong các event được nhận.
- 10 nhóm có nhiều event, mỗi nhóm có ít nhất một path positive đúng thứ tự và
  nằm trong giới hạn gap đã khóa. Đây là kiểm tra nhãn, không phải retrieval hit.
- Cả 26 câu tiếng Anh đều trống; có một ô approval được tích nhưng không có câu
  tiếng Anh đi kèm, nên không coi đó là phê duyệt một bản dịch mới.

Đã chuẩn bị bản dịch **chỉ từ câu tiếng Việt người dùng viết**, chưa xem caption,
scores hoặc ảnh để bổ sung nghĩa. Bản review hiện hành:

`benchmark_results/human_holdout_v3_seed2026110203_v2/translation_review_v2/REVIEW_VI_EN.md`.

Có 19 câu đề xuất rõ nghĩa và 7 event cần người gán nhãn xác nhận cách gắn
thuộc tính/vị trí hoặc phân biệt “xanh” là blue/green. Lượt nháp v1 không được
duyệt và được thay bằng v2 để không tự chọn blue/green khi tiếng Việt chưa nói
rõ; giữ cả hai lượt để truy vết. Không có lượt chấm retrieval nào từ các bản nháp.

Mọi `english_approved` trong bản dịch nháp đều là false. Chờ người dùng duyệt
bản dịch và giải nghĩa các mục đánh dấu trước khi freeze/chấm holdout. Không
thay positive/negative/unknown theo bản dịch hoặc theo dự đoán model.

Người dùng đã xác nhận đủ 7 mục cần giải nghĩa. Đã lưu lựa chọn và câu tiếng
Anh tương ứng tại `translation_confirmations_v1/` trong gói, đánh dấu đúng 7
event đó được duyệt, giữ nguyên toàn bộ nhãn và câu tiếng Việt. **19 bản dịch
không đánh dấu vẫn chờ xác nhận chung**, chưa freeze hoặc chạy holdout. Chỉ cần
người dùng trả lời đồng ý với 19 bản dịch còn lại; không phải sửa hoặc gửi lại JSON.

Người dùng sau đó đã trả lời **“Duyệt, bắt đầu thực hiện”**. Đã ghi receipt
phê duyệt còn lại, giữ nguyên judgments, kiểm tra toàn bộ26 event được duyệt và
freeze `frozen_human_labels_v1.json` (SHA-256
`2449a46318bdb2925efea39146b70f43b152097eecbe29a8ae4ebe35a1f9af77`).
Đánh giá độc lập human/holdout nằm ở
[HUMAN_HOLDOUT_V3_RESULTS.md](HUMAN_HOLDOUT_V3_RESULTS.md); không cần người dùng
gán nhãn hoặc phê duyệt thêm để chạy lượt đã khóa này.

**Lượt đã hoàn tất:** baseline và guard cùng known-frame Hit@20=6/26,
known-path Hit@20=2/10, macro FPR7,69% và TPR49,36%. Guard không cải thiện
metric chính; giữ mặc định `off`, không sửa nhãn hoặc retune trên holdout này.
Chi tiết và chẩn đoán theo giai đoạn nằm trong báo cáo kết quả liên kết trên.
