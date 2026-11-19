# SigLIP2-384 embedding

> Đã embed toàn bộ L01–L30 bằng GPU laptop với `tools/embed_siglip2.py` (đọc thẳng zip/bundle,
> resume theo video); notebook Kaggle bên dưới là phương án dự phòng khi có GPU Kaggle.

Mục tiêu: embed lại 161.722 keyframe L01–L21 bằng `timm/ViT-B-16-SigLIP2-384`
(revision `369333c629e1ab4eb90806c8e13df3054c3c1e63`) để A/B với SigLIP hiện tại.
Đã nối vào pipeline: `config/models.json` (`features_dir: siglip2-features`, collection `siglip2_keyframes`).

## 1. Đóng gói ảnh trên CPU trường

SigLIP2-384 tự resize ảnh về 384×384 (không crop), nên bundle lưu sẵn kích thước đó.
JPEG q95, chroma 4:4:4 (q90 4:2:0 làm vector lệch, cos ~0,955 so với ảnh gốc; q95 4:4:4 cos ≥0,995),
khoảng 67 KB/ảnh → L01–L09 4,5 GB, L10–L21 6,3 GB. Smoke 2 video: 469 ảnh trong 3,2 giây,
~20 phút với 8 process.

```bash
cd ~/siglip2-bundle-20261119
nohup nice -n 10 ~/KISBranche/.venv/bin/python prepare_siglip2_bundle.py \
  --dataset ~/kis-l21-data-20261118T051746Z/dataset \
  --output siglip2-bundle-l01-l21.zip --workers 6 \
  --groups L01 L02 L03 L04 L05 L06 L07 L08 L09 L10 L11 L12 L13 L14 L15 L16 L17 L18 L19 L20 L21 \
  > bundle.log 2>&1 &
```

Script ghi `.partial` trong lúc chạy, chỉ đổi tên khi xong; ảnh thiếu được liệt kê trong
`manifest.json`, không tạo ảnh thay thế.

## 2. Đưa bundle lên Kaggle

1. Tải về laptop (qua VPN):
   `scp <user>@<school-host>:~/siglip2-bundle-20261119/siglip2-bundle-l01-l21.zip .`
2. Kaggle → Datasets → New Dataset → upload zip (Kaggle tự giải nén), để Private.

## 3. Chạy notebook

1. Import `notebooks/kaggle_siglip2_embed.ipynb` vào Kaggle.
2. Add Input: dataset vừa tạo. Settings: **GPU T4 x1**, **Internet ON**.
3. Run All. Cell cuối in đường dẫn `siglip2-features.zip` (~250 MB) và số dòng thiếu.
4. Tải `/kaggle/working/siglip2-features.zip` về (tab Output).

Notebook đã chạy thử trên CPU laptop với bundle 2 video: 469/469 vector, `(rows, 768)`
float16, norm 1, đúng thứ tự mapping CSV; probe text→image trả ảnh hợp nghĩa.
Chưa chạy trên GPU Kaggle; tốc độ T4 ước tính 100–200 ảnh/giây.

## 4. Sau khi có vector

Gửi lại zip; bước tiếp theo (chưa làm) là thêm encoder SigLIP2 cho text query trong worker
và chạy A/B trên bộ 17 query ở `benchmark_results/ab_visual_vs_rrf_20261119_run01/`.
