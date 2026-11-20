# Bộ dữ liệu AIC và artifact của KIS

## Nguồn và phạm vi

Dữ liệu video/keyframe được lấy từ bộ dữ liệu **cuộc thi AI Challenge (AIC)**,
không phải bộ dữ liệu do KIS Framefinder tự thu thập.
Tham khảo thông tin cuộc thi tại <https://aichallenge.vn/> và sử dụng theo điều
khoản/quyền truy cập của bộ dữ liệu được cung cấp cho bạn.

Repo không gắn một niên khóa AIC cụ thể khi chưa có metadata nguồn đủ để xác nhận.
Các nhóm L01–L24 là phạm vi cấu hình của dự án, không phải cam kết toàn bộ dữ
liệu cuộc thi đã được tải hoặc kiểm chứng đủ trên mọi máy.

Ảnh/video và mapping nguồn khác với artifact suy ra bằng mô hình. Caption,
metadata đối tượng, OCR, vector và FAISS index không mặc định là ground truth.
Không tái phân phối dữ liệu nguồn chỉ vì mã nguồn dự án có thể được chia sẻ.

## Quy ước thư mục

```text
dataset/<component>/<group>/<video>/<file>
```

Ví dụ với `L01_V001`:

| Đường dẫn chuẩn | Nội dung |
| --- | --- |
| `data/L01/L01_V001/001.jpg` | Ảnh keyframe số 1, giữ nguyên bytes nguồn |
| `mapping/L01/L01_V001/L01_V001.csv` | Mapping keyframe → frame/video time |
| `siglip2-features/L01/L01_V001/L01_V001.npy` | Vector SigLIP2 theo thứ tự CSV |
| `caption_generator/L01/L01_V001/L01_V001.json` | Caption theo video |
| `caption_mapping/L01/L01_V001/L01_V001.json` | Mapping các hàng index caption |
| `index/L01/L01_V001/L01_V001.index` | FAISS caption index |
| `index/L01/L01_V001/L01_V001.meta.json` | Manifest caption/index |
| `objects/L01/L01_V001/001.json` | Metadata đối tượng nếu có |

`ocr/` có thể chứa ZIP nhập liệu hoặc artifact/index đã chuẩn bị. Chỉ có ZIP
không đồng nghĩa nhánh OCR đã sẵn sàng: cần nhập đúng backend, cấu hình và index.
`siglip2-bundle/` là sản phẩm đóng gói/phục vụ embedding, không thay thế ảnh gốc.

Loader còn đọc bố cục rút gọn như `mapping/L01/L01_V001.csv`,
`caption_generator/L01/L01_V001.json` hoặc artifact flat tương thích.
Không tạo thêm bản chuẩn hóa ngay cạnh bản cũ: duplicate cùng video có thể
làm loader từ chối. Việc dọn repo không di chuyển dataset đang hoạt động.

## Định danh và mapping

- **Group:** `L01`, `L02`, …
- **Video ID:** `L01_V001`.
- **`n` / `keyframe_n`:** số thứ tự keyframe, ví dụ ảnh `001.jpg` là `n=1`.
- **`frame_idx` / `frame_id`:** chỉ số frame trong video gốc, không phải số ảnh.
- **`pts_time`:** thời gian theo giây từ mapping nguồn.
- **`fps`:** tốc độ khung hình ghi trong mapping.

DRES sử dụng video ID và thời gian; hệ thống lấy timestamp từ mapping gốc và
chuyển giây sang mili-giây theo quy trình nộp. Không suy frame/time từ tên ảnh
hoặc caption, không thay CSV để khớp một kết quả dự đoán.

## Ràng buộc nhất quán

1. Mỗi video chỉ có một CSV, một file vector và một bộ caption/index được chọn.
2. Số hàng vector ảnh phải bằng số dòng CSV; thứ tự hàng phải giữ nguyên.
3. Vector hiện tại là SigLIP2-384, 768 chiều, L2-normalized; model/revision nằm
   trong `config/models.json`. Không trộn embedding từ SigLIP/CLIP khác.
4. Caption index dùng MiniLM 384 chiều; index, mapping và manifest phải cùng bộ.
5. Giữ nguyên ảnh và mã video. Không dùng ảnh tải trùng có tên như `001(1).jpg`
   thay cho ảnh số gốc và không renumber để bù ảnh thiếu.
6. Thiếu ảnh, mapping, index hoặc model phải được xử lý rõ ràng; không tạo frame
   giả, không xem điểm caption/ảnh đơn là xác minh một chuỗi hành động.

## Dữ liệu local và Git

`dataset/`, `volumes/`, `dist/`, model weights và file vector được bỏ qua bởi Git.
`config/*.local.json` dành cho cấu hình riêng; không đưa token, mật khẩu hoặc
Session ID vào tài liệu, notebook, archive mã nguồn hay manifest công khai.

Không xóa ZIP nguồn, CSV, NPY, index, ảnh hoặc Docker volume để dọn cache.
