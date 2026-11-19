# Vì sao OCR mất chữ tiếng Việt? Kiểm tra notebook và pixels

Notebook được kiểm tra: `notebooks/kaggle_ocr_simple_gpu.ipynb`.
Không sửa notebook, không cài/chạy Paddle, không đổi OCR nguồn hoặc web runtime.

## Kết luận

**Không thể quy tất cả cho ảnh mờ. Bằng chứng mạnh nhất là bảng ký tự recognizer
không bao phủ đầy đủ tiếng Việt ở cấu hình model chính thức đã kiểm tra.**

Notebook chọn detector`PP-OCRv5_server_det` và recognizer
`latin_PP-OCRv5_mobile_rec`. Cấu hình inference chính thức của recognizer tại
revision`ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57` có773entry ký tự duy nhất kể cả
space, **không có `ả/ấ/ứ/ừ/ế/ộ/ề/ắ/ụ/ệ/ị`** và không có Unicode combining marks
để ghép chúng bằng dạng decomposed. CTC decoder chỉ chọn trong bảng ký tự;
ảnh rõ hơn không tự thêm được các nhãn ký tự còn thiếu.

Mẫu lỗi khớp trực tiếp với chữ không biểu diễn được:

| Chữ đọc từ ảnh | OCR nguồn | Ký tự thiếu trong alphabet đã kiểm tra |
| --- | --- | --- |
| `Quảng Ngãi` | `Qung Ngãi` | `ả` |
| `tấn` | `tn` | `ấ` |
| `chứa` | `cha` | `ứ` |
| `Thừa Thiên-Huế` | `Tha Thiên-Hu` | `ừ`, `ế` |
| `Hà Nội` | `Hà Ni` | `ộ` |
| `MÌ TRỨNG` | `MITOONG` | `Ứ`; còn nhầm hình chữ do ảnh bao bì khó |

Đây không chỉ là mất dấu rồi còn chữ gốc: nhiều nguyên âm ghép dấu bị **bỏ hẳn**.
Chữ`ã/à/á` có trong alphabet nên vẫn xuất hiện, tạo cảm giác OCR hỗ trợ một phần
tiếng Việt trong khi thiếu nhiều tổ hợp dấu quan trọng.

## Kiểm tra chéo toàn bộ OCR được cung cấp

Đọc67.882record L01–L09, tổng **7.111.781 ký tự**:

-16ký tự kiểm tra (`ảấứừếộềắụệịỈỰỨẤẢ`) đều xuất hiện **0lần**.
-Không có ký tự non-ASCII nào trong output nằm ngoài alphabet inference đã
  kiểm tra. Các chữ có dấu được xuất ra đều là ký tự model có thể biểu diễn.
-Các nguồn nguyên bản và snapshot không sửa; scan chỉ đếm, không nhận dạng lại.

Kết hợp với mẫu lỗi và kiểm tra export bên dưới, đây là bằng chứng rất mạnh cho
giới hạn recognizer/alphabet, không phải chỉ ảnh nén/mờ ngẫu nhiên.

**Giới hạn xác nhận:** public ZIP không có profile, model hash hoặc hash ảnh;
notebook đính kèm có0execution counts và0output cells. Chưa có`inference.yml`
và checkpoint thực tế từ phiên Kaggle của người dùng, nên không khẳng định đã
chứng minh chuỗi provenance của chính lần chạy đó. Notebook chỉ pin phiên bản
package và model name, không pin nội dung/revision file model được tải. Cần đối
chiếu cached inference.yml/profile gốc để xác nhận hoàn toàn; không bịa provenance.

## Vai trò của ảnh: có, nhưng khác nhau theo loại chữ

Ba ảnh gốc đều1280×720, đã mở bản gốc và crop không chỉnh nét:

-`L09_V011/120`: ticker trắng nền đỏ, chữ`Quảng Ngãi` và dấu vẫn đọc được.
  Không có cơ sở coi ảnh này mờ đến mức mọi recognizer đều không thể đọc.
-`L02_V015/181`: ticker`3,2 tấn cá khoai chứa formol` đọc rõ hơn nhiều so với
  output mất các nguyên âm có dấu. Model/alphabet là nghi vấn chính.
-`L08_V002/050`: bao mì chuyển động, chữ nghiêng/cong trên vật liệu phản sáng,
  tương phản không đều và nén/mờ thấy được. Có **cả chất lượng ảnh lẫn giới hạn
  alphabet/model**; khó tách phần trăm đóng góp khi chưa OCR lại có đối chứng.

Crop so sánh ở
`benchmark_results/school_ocr_notebook_diagnosis_20261115_run01/*_compare.png`.
Chúng là manual diagnostic crops, **không phải bounding boxes detector đã chạy**.
Phóng nearest-neighbor để xem pixel, không super-resolution hay tái tạo chữ.

## Cấu hình notebook còn làm gì?

### 1. Padding không sửa mờ và không làm mất dấu ở bước export

`decode()` chỉ RGB và padding trắng32px mỗi cạnh. Test trên3ảnh xác nhận toàn
bộ raster pixels gốc giữ nguyên trong vùng giữa. Chữ không tự rõ hơn nhờ padding.

`public_record()` nối nguyên văn OCR lines bằng newline; không có strip dấu,
ASCII folding, sửa chính tả hay Unicode normalization. `.strip()` chỉ kiểm tra
dòng trống, không thay chuỗi được lưu. Synthetic roundtrip với`Quảng Ngãi`,
`MÌ TRỨNG`,`3,2 tấn cá khoai chứa formol` giữ nguyên Unicode qua UTF-8 JSON.
Vì vậy không thấy lỗi export biến các chữ đó thành`Qung/tn/cha`.

### 2. Detector vẫn resize, dù wrapper ghi wrapper_resize=False

Notebook đặt`text_det_limit_side_len=960`,`text_det_limit_type='max'`.
Theo PaddleX3.2.0,1280×720 được pad thành1344×784 rồi detector resize/làm tròn
stride32 thành khoảng **960×576**. Chữ nhỏ/dấu có ít pixel hơn khi tìm box, có
thể ảnh hưởng bỏ sót/cắt box. Đây là hình học mô phỏng từ source, không ảnh
detector thực tế của lần Kaggle.

**Không được hiểu sai:** pipeline dùng polygon tìm được để crop từ raster gốc
đã padding, không nhận dạng trực tiếp toàn ảnh960×576. Recognizer resize crop
riêng. Tăng detector resolution có thể giúp box, không chữa alphabet thiếu.

### 3. Chọn model theo tên, không phải thiếu một lang='vi'

Official PaddleOCR3.2.0 map`vi` đến chính`latin_PP-OCRv5_mobile_rec`; khi đã chỉ
định model name như notebook,`lang` còn bị bỏ qua. Vì vậy chỉ thêm`lang='vi'`
không phải cách sửa đã chứng minh. Cần kiểm tra **alphabet thực tế và weights
tương ứng**, không dựa tên “Latin” hoặc dòng mô tả “including Vietnamese”.

Không sửa tay dictionary để thêm`ấ/ứ/...` vào checkpoint cũ: thứ tự nhãn/class
phải khớp recognition head và weights đã train. Dùng checkpoint có đủ alphabet
và được huấn luyện tương ứng, hoặc fine-tune đúng quy trình.

### 4. Threshold0 giữ cả chữ nhiễu; status=ok không có nghĩa đúng

`text_rec_score_thresh=0.0` giữ mọi dòng không trống, kể cả confidence thấp.
`status='ok'` nghĩa có text, không phải được xác minh chính xác. Nâng threshold
có thể bỏ bớt nhiễu, không phục hồi ký tự model không biểu diễn được. Confidence
và boxes có trong private checkpoints nhưng bị loại khỏi public ZIP theo thiết kế;
không thể suy lại confidence cho JSON hiện có.

GPU T4 hay dùng một/hai GPU không bổ sung nhãn ký tự còn thiếu; đây không phải
vấn đề GPU không đủ mạnh. Không có bằng chứng hiện tại về lỗi decode/runtime
trong các record lỗi chữ này; chúng đều là nhận dạng thành công nhưng sai text.

## Việc nên làm tiếp (chưa tự thay đổi)

1.Lấy private`state/profile.json`, cached recognition`inference.yml` và vài line
checkpoints của phiên Kaggle nếu còn. So alphabet/hashes/boxes/confidence thực tế.
2.Chọn recognizer/checkpoint đã kiểm tra đủ bảng chữ Việt, không chỉ đổi sang
một“server_rec” bất kỳ vì model đó có thể nhắm ngôn ngữ khác.
3.Test20–50frame: ticker rõ, ticker nhỏ và chữ vật thể mờ/nghiêng. Freeze bản
chép chữ thủ công trước OCR. So full-frame hiện tại, detector-resolution cao hơn,
crop gốc và recognizer mới trên cùng ảnh; đo CER/khớp cụm chữ và thời gian.
4.Giữ profile, input SHA, polygons, raw text vàconfidence trong báo cáo kỹ thuật
riêng, không phá bảy-field public schema. Xuất phiên bản OCR mới sang root mới;
không ghi đè OCR hiện tại/caption. Chỉ triển khai sau khi đánh giá có lợi.

Không tự cài engine, tải weights, chạy lại67.882ảnh, sửa notebook hoặc đổi cấu
hình production trong lần chẩn đoán này.53tests notebook/diagnostic pass,21.877s.

## Nguồn kiểm chứng

-[Pinned model inference config](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec/blob/ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57/inference.yml).
-[PaddleOCR3.2.0 API: language/model mapping](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.2.0/paddleocr/_pipelines/ocr.py).
-[PaddleX3.2.0 detector resize](https://github.com/PaddlePaddle/PaddleX/blob/v3.2.0/paddlex/inference/models/text_detection/processors.py).
-[PaddleX3.2.0 crop/recognition pipeline](https://github.com/PaddlePaddle/PaddleX/blob/v3.2.0/paddlex/inference/pipelines/ocr/pipeline.py).
-[PaddleX3.2.0 CTC decoder and recognizer resize](https://github.com/PaddlePaddle/PaddleX/blob/v3.2.0/paddlex/inference/models/text_recognition/processors.py).

Lưu bản metadata/source và SHA-256 ở
`benchmark_results/school_ocr_notebook_diagnosis_20261115_run01/alphabet_run02/`.
`REPORT.json` tại root ghi notebook/worker hashes, code excerpts, crop geometry,
Unicode roundtrip và bảo toàn pixels. Runalphabet đầu chỉ fail vì Windows CA
store lỗiASN1 trước tải metadata; run02 dùng certifi CA bundle có xác minh TLS,
không vô hiệu hóa certificate verification.
