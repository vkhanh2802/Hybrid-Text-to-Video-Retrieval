# Paper, GitHub và outline nâng cấp KIS theo HCMC AI Challenge

> 04/11/2026: [benchmark cross-encoder CPU thật đã hoàn tất](TEXT_RERANKER_CPU_RESULTS_20261104.md).
> Kết quả chưa đủ nhất quán để bật mặc định; giữ code opt-in và không retune regression.

> 04/11/2026: [OCR độc lập đã có code retrieval/API và pilot Kaggle CPU](OCR_RETRIEVAL.md).
> Gói 12 ảnh/12 video đã qua kiểm tra integrity; OCR vẫn off, chưa inference thật
> hoặc đo chất lượng. ASR và rubric verification vẫn là các bước riêng chưa hoàn tất.

> Cập nhật code 03/11/2026: event planner/temporal DP đã triển khai;
> [module reranker nhẹ](TEXT_RERANKER.md) đã nối vào pipeline/API, opt-in và chưa
> có benchmark model thật. OCR/ASR/rubric là các bước riêng chưa hoàn tất.
> Nội dung nghiên cứu/đối chiếu ban đầu dưới đây được giữ làm lịch sử thiết kế.

Ngày tra cứu: **01/11/2026**. Phạm vi: HCMC AI Challenge video-event retrieval,
Known-Item Search (KIS), Temporal Retrieval and Alignment of Key Events (TRAKE),
cùng các nghiên cứu/module có thể áp dụng vào project hiện tại.

## 1. Kết luận để quyết định

**Nên nâng cấp cách truy hồi và ghép bằng chứng trước khi tiếp tục thay model caption.**
Các nguồn gần bài toán nhất là DANTE [H3], cross-modal temporal retrieval của
EEIoT_newbie [H4] và EVISTA/TEMPO-P [H6]. Cả ba hướng tới việc tìm nhiều sự kiện
trong một video theo thứ tự, thay vì coi cả query dài như mô tả một ảnh.

Đề xuất cho project, theo thứ tự:

1. Benchmark/corpus snapshot và lưu rank từng stage.
2. **Event retrieval + temporal alignment** trên embeddings/mapping hiện có.
3. **OCR độc lập**, ưu tiên bảng bài giảng, bảng lớp, tên nguyên liệu.
4. **ASR theo timestamp** cho lời nói/bài giảng/nội dung nấu ăn.
5. Reranker nhẹ rồi kiểm chứng một nhóm frame bằng VLM.
6. UI timeline, tìm ảnh tương tự, feedback và mở rộng frame lân cận.
7. Shot-aware sampling/embedding mới chỉ sau khi xác định thiếu recall.

Đây là **đề xuất kỹ thuật**, không phải kết quả benchmark mới của project.
Các paper/repo được liệt kê không chứng minh trực tiếp hiệu quả của trọng số,
model hoặc top-k đang dùng trong KISBranche.

## 2. Cách tìm và xác minh

- Từ khóa: `Ho Chi Minh City AI Challenge video retrieval`, `HCMC AI Challenge
  KIS TRAKE`, `SOICT 2024 video retrieval`, `temporal event retrieval`, `VBS
  interactive video search`, `OCR ASR video retrieval`.
- Nguồn kiểm tra: arXiv abstract/full text, Springer proceedings, CVPR/PMLR,
  GitHub README và liên kết code từ paper/project page; metadata DOI/Crossref
  được dùng đối chiếu khi cần.
- Ưu tiên nguồn tác giả, code được tác giả liên kết, hoặc repo tự xác định rõ
  tham gia HCMC. Không gán repo cho paper chỉ vì có tên hoặc pipeline giống nhau.
- Loại khỏi phạm vi **AI City Challenge về giao thông/smart-city**, là sự kiện
  khác với HCMC AI Challenge về truy hồi video đang xét.
- Năm cuộc thi, năm arXiv và năm proceedings có thể khác nhau. Năm trong danh
  sách arXiv dưới đây là năm đăng bản được đọc, không suy ra thứ hạng cuộc thi.
- Đây là targeted literature/repository scan, không phải systematic review
  exhaustive. Chưa chạy code bên ngoài hoặc tái lập kết quả paper.

## 3. Paper liên quan trực tiếp đến HCMC AI Challenge

### [H1] Event Retrieval from Large Video Collection in Ho Chi Minh City AI Challenge 2024

- **Tác giả:** Trong-Le Do và cộng sự.
- **Năm:** cuộc thi/SOICT 2024; xuất bản Springer **2025**.
- **Paper:** https://doi.org/10.1007/978-981-96-4291-5_1
- **Xác minh:** trang publisher, abstract và danh mục tài liệu tham khảo;
  nội dung chapter đầy đủ bị paywall.
- **Nội dung:** tổng quan cuộc thi; abstract ghi 1.471 video/328 giờ, các query
  textual/visual KIS, QA và nhu cầu hiểu sự kiện theo thời gian.
- **Học được:** xây benchmark đúng bài toán cuộc thi và phân biệt tìm đúng
  video, đúng frame, đúng đoạn; không chỉ đo image-text similarity.
- **Giới hạn:** đây là overview, không phải bằng chứng một module cụ thể làm
  top-1 tốt hơn. Không có system repository được xác minh cho overview này.

### [H2] A Comprehensive Video Event Retrieval System for Vietnamese News: Integrating CLIP ViT, TASK-Former, Transcripts, and OCR

- **Tác giả:** Thinh-Phat Vo, Quang-Thang Duong, Quoc-Thang Nguyen,
  Dang-Khoa Mai, Nguyen-Khang Ly.
- **Năm:** SOICT 2024; xuất bản **2025**.
- **Paper:** https://doi.org/10.1007/978-981-96-4291-5_19
- **Xác minh:** publisher abstract; chapter đầy đủ bị paywall. Được dẫn trong
  overview HCMC 2024 [H1].
- **Cách làm:** quick search, temporal search, sketch + text, transcript search,
  OCR search; CLIP ViT-L, TASK-former; similar-image exploration và clip preview.
- **Có thể áp dụng:** OCR và ASR thành hai nguồn search riêng; xem frame lân cận
  và preview đoạn video ngay ở kết quả.
- **Giới hạn:** abstract không đủ để xác nhận chi tiết fusion hoặc gain riêng
  của OCR/ASR; chưa xác minh repository của chính hệ thống.

### [H3] Integrated Semantic and Temporal Alignment for Interactive Video Retrieval — DANTE

- **Tác giả:** Thanh-Danh Luu và cộng sự, hệ thống AIO_Owlgorithms.
- **Năm bản đọc:** **2025**, HCMC AI Challenge 2025.
- **Paper:** https://arxiv.org/abs/2512.13169
- **Full text/pseudocode:** https://arxiv.org/html/2512.13169v1
- **Cách làm:** TransNetV2; BEiT-3/Milvus; OCR/Elasticsearch; QUEST cho query
  rewriting/image exemplars; **DANTE dynamic programming** để tìm chuỗi event.
- **Bằng chứng kỹ thuật:** §4.3, Algorithm 1 yêu cầu match trước nằm ở keyframe
  sớm hơn và trừ penalty khoảng cách; có running-max để giảm chi phí alignment.
- **Có thể áp dụng:** tái sử dụng SigLIP hiện tại, tính score mỗi event cho các
  frame trong candidate video, rồi dùng DP tìm đường đi đúng thứ tự trên CPU.
- **Điều chỉnh riêng cho project:** dùng timestamp `pts_time` thay keyframe-index
  distance vì sampling không đều theo thời gian.
- **Giới hạn:** paper đưa ví dụ minh họa và mô tả hệ thống; chưa phải controlled
  ablation trên KISBranche. Chưa xác minh official repository cho toàn hệ thống.

### [H4] Enhanced Multimodal Video Retrieval System: Integrating Query Expansion and Cross-modal Temporal Event Retrieval

- **Tác giả:** Van-Thinh Vo và cộng sự, team EEIoT_newbie.
- **Năm bản đọc:** **2025**, HCMC AI Challenge 2025.
- **Paper:** https://arxiv.org/abs/2512.06334
- **Full text:** https://arxiv.org/html/2512.06334v1
- **Project page:** https://huyrua27.github.io/eeiotnewbie-AIC2025
- **Code liên kết từ project:** https://github.com/ThinhVan27/AIC-2025
- **Cách làm:** semantic CLIP/BEiT-3, object/color/OCR metadata, query augmentation,
  RRF và temporal reranking; từng scene trong chuỗi có thể dùng modality khác nhau.
- **Ví dụ rất gần query nấu ăn:** §4.2 mô tả semantic search chưa tìm đúng bước
  cắt củ năng; thay một event bằng chữ OCR `cu nang` giúp tìm cảnh liên quan.
- **Có thể áp dụng:** e1 tìm bằng visual, e2 bằng OCR tên nguyên liệu, e3 bằng
  caption/ASR; sau đó ghép cùng video và đúng thứ tự.
- **Giới hạn:** temporal pseudocode dùng danh sách top-k và cửa sổ theo frame;
  không nên copy Cartesian-product implementation khi pool lớn. Các ví dụ thành
  công không xác lập gain trung bình hoặc độ phù hợp của mọi model trong repo.

### [H5] RAPID: Retrieval-Augmented Parallel Inference Drafting for Text-Based Video Event Retrieval

- **Tác giả:** Long Nguyen và cộng sự.
- **Năm:** **2025**, hệ thống tham gia HCMC AI Challenge 2024.
- **Paper:** https://arxiv.org/abs/2501.16303
- **Full text:** https://arxiv.org/html/2501.16303v1
- **Proceedings:** https://doi.org/10.1007/978-981-96-4291-5_15
- **Cách làm:** LLM tạo contextual drafts, parallel retrieval, sau đó đánh giá
  candidate theo mức khớp **query gốc**; có lựa chọn của người dùng.
- **Có thể áp dụng:** giữ raw query và bản dịch trung thành như anchor, dùng
  variants bổ trợ; kiểm tra model có thêm nguyên liệu/màu/địa điểm không có trong query.
- **Giới hạn:** không coi LLM expansion luôn có lợi. Paper báo trên custom dataset;
  không dùng các con số đó như kỳ vọng accuracy của corpus hiện tại. Chưa xác minh code.

### [H6] EVISTA: Unified Multimodal Video Search with Evidence-Aware Temporal Path Optimization

- **Tác giả:** Minh-Huy Tran và cộng sự.
- **Năm:** **2026**, arXiv đăng 10/10/2026, hệ thống HCMC AI Challenge 2026.
- **Paper:** https://arxiv.org/abs/2610.14510
- **Trạng thái kiểm tra:** đã đọc abstract/metadata; **preprint**, chưa xác minh
  peer-reviewed publication hoặc official code trong lần tra cứu này.
- **Cách làm theo abstract:** hợp nhất visual/OCR/ASR/object/metadata; TEMPO-P
  gom candidate videos từ nhiều events/modalities, chấm joint evidence và chọn
  đường đi đúng thứ tự, gọn trong thời gian bằng DP.
- **Có thể áp dụng:** không bắt mọi event phải tự lọt top-k độc lập mới xét video;
  lấy union video rồi bổ sung evidence cục bộ cho những event đang yếu.
- **Giới hạn quan trọng:** abstract báo 5 organizer queries + 36 additional queries;
  visual-only và proposed alignment có cùng R@1=0,42 trên visual events. Đây là
  bằng chứng sơ bộ cho text-defined events; abstract không tách riêng tác động
  OCR, ASR, candidate expansion và alignment. Không có kết luận luôn thắng visual.
  H4/H6 có nhiều tác giả chung, không coi là hai independent replications.

### [H7] Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval

- **Tác giả:** Duc-Tho Nguyen và cộng sự, team FocusOnFun.
- **Năm bản đọc:** **2026**, hệ thống HCMC AI Challenge 2025.
- **Paper:** https://arxiv.org/abs/2606.19682
- **Full text:** https://arxiv.org/html/2606.19682v1
- **Cách làm:** CLIP + SigLIP2/RRF, Milvus + Elasticsearch, Qwen metadata,
  timestamped ASR, Rocchio relevance feedback và before/now/after search.
- **Có thể áp dụng:** kiến trúc gần project hiện tại; tham khảo cách giữ các
  evidence riêng và cho người dùng feedback/tìm ảnh tương tự.
- **Giới hạn:** pseudocode temporal trong bản đọc lấy before/after tốt nhất cùng
  video nhưng không kiểm tra timestamp order; không dùng nó thay strict ordered
  alignment. Điểm giữa các vòng với query khác nhau không phải ablation. Qwen
  tạo metadata không chứng minh hiệu quả của pointwise reranker Qwen top-20.
- **Code:** chưa xác minh repo liên kết chính thức cho hệ thống này.

## 4. GitHub nên đọc trước

| ID | Repository | Liên quan và module đáng học | Giới hạn đã thấy |
|---|---|---|---|
| R1 | [ThinhVan27/AIC-2025](https://github.com/ThinhVan27/AIC-2025) | HCMC 2025, đi kèm [H4]; semantic/metadata/temporal API và UI | Không phải drop-in cho project; BEiT3/FAISS và metadata riêng |
| R2 | [tb-tian/AIC2025-VideoRetrieval-float97](https://github.com/tb-tian/AIC2025-VideoRetrieval-float97) | HCMC 2025; SigLIP2, TransNetV2, DINOv3, WhisperX, timestamped transcript search | README ghi rõ OCR/caption Vintern còn WIP; phần mô tả CLIP/512D không thống nhất với architecture chính |
| R3 | [Nhathuy1305/BetterDay-Tool](https://github.com/Nhathuy1305/BetterDay-Tool) | Bản giản lược hệ thống HCMC 2023; feedback, rescore, SOM browsing, submission | README có mô tả license/backend không thống nhất; cần kiểm tra trước khi tái sử dụng code |
| R4 | [aimh-lab/visione](https://github.com/aimh-lab/visione) | **VBS**, không phải HCMC; UI temporal “first this, then that”, object/color/position, similar image, video playback | Học interaction thay vì bê nguyên stack; Docker/GPU requirements khác máy hiện tại |

R4 liên kết paper **VISIONE 5.0: Enhanced User Interface and AI Models for VBS2024**,
Giuseppe Amato và cộng sự (2024), DOI:
https://doi.org/10.1007/978-3-031-53302-0_29.
Các feature UI ở bảng được xác nhận từ README, không suy thành kết quả user study.

**Thứ tự đọc code:** R1 cho temporal/multimodal flow -> R2 cho transcript processing
-> R4 cho UI -> R3 cho relevance feedback. Repo có feature trong README chưa đủ
để xác nhận chất lượng, dependency compatibility hay khả năng chạy GPU 6GB.

## 5. Paper/module bổ trợ có code

### [G1] PP-OCR: A Practical Ultra Lightweight OCR System — 2020

- Yuning Du và cộng sự.
- Paper: https://arxiv.org/abs/2009.09941
- Official code: https://github.com/PaddlePaddle/PaddleOCR
- Dùng cho: OCR bảng Tiếng Anh, số lớp, chữ nguyên liệu; lưu text+bbox+confidence.
- Phân biệt: hỗ trợ tiếng Việt trong implementation multilingual sau này không
  đồng nghĩa original paper đã benchmark tiếng Việt. Pin OCR release/checkpoint,
  không mặc định tải document VLM lớn của repo hiện tại.

### [G2] Robust Speech Recognition via Large-Scale Weak Supervision — ICML 2023

- Alec Radford và cộng sự; preprint năm 2022.
- Paper: https://proceedings.mlr.press/v202/radford23a.html
- Official code: https://github.com/openai/whisper
- Dùng cho: transcript theo timestamp; audio có thể nêu tên nguyên liệu/công thức.
- Thực thi tối ưu thay thế: https://github.com/SYSTRAN/faster-whisper
  là implementation CTranslate2, **không phải repo gốc của paper Whisper**.
- Bắt đầu thử multilingual `small`, chạy offline riêng; không giữ đồng thời
  mọi model trong GPU. ASR có thể hallucinate, cần audio evidence và confidence/flags.

### [G3] Passage Re-ranking with BERT — 2019 preprint

- Rodrigo Nogueira, Kyunghyun Cho.
- Paper: https://arxiv.org/abs/1901.04085
- Original code: https://github.com/nyu-dl/dl4marco-bert
- Ý tưởng: query và passage được đọc joint để chấm relevance ở second stage.
- Ứng viên nhỏ để thử: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
  — đây là model card khác, không phải checkpoint gốc BERT của paper.
- Dùng cho: rerank caption/OCR/ASR của top candidates; không thể tự xác nhận pixels.
  Checkpoint này English-oriented, cần query tiếng Anh hoặc benchmark ngôn ngữ riêng.

### [G4] QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries — NeurIPS 2021

- Jie Lei, Tamara L. Berg, Mohit Bansal; model tên **Moment-DETR**.
- Paper: https://arxiv.org/abs/2107.09609
- Official code: https://github.com/jayleicn/moment_detr
- Dùng cho: dự đoán start/end trong candidate clip sau retrieval.
- Caveat: demo CLIP-only và cấu hình train SlowFast+CLIP là hai pipeline khác nhau;
  không nạp feature SigLIP 768D hiện có vào checkpoint mà chưa thích nghi. Demo có
  giới hạn 75 clips × 2 giây = 150 giây; video dài cần windowing và timestamp offset.

### [G5] Query-Dependent Video Representation for Moment Retrieval and Highlight Detection — CVPR 2023

- WonJun Moon và cộng sự; model **QD-DETR**.
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Query-Dependent_Video_Representation_for_Moment_Retrieval_and_Highlight_Detection_CVPR_2023_paper.html
- Official code: https://github.com/wjun0830/QD-DETR
- Ý tưởng: query-conditioned video representation và saliency; thử khi cần
  localize chính xác hơn trong candidate window.
- Caveat: feature/checkpoint compatibility và links chuẩn bị dữ liệu là chi phí
  tích hợp. Không tự giải A -> B -> C chỉ nhờ moment-localization head.

### [G6] TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection — 2020 preprint

- Tomáš Souček, Jakub Lokoč.
- Paper: https://arxiv.org/abs/2008.04838
- Tác giả/code: https://github.com/soCzech/TransNetV2
- Dùng cho: thêm shot boundaries và chọn representative frame, hạn chế bỏ shot ngắn.
- Caveat: không đổi numbering/mapping corpus cũ tại chỗ; tạo artifact version
  mới nếu re-extract. Shot detection không bảo đảm bắt mọi thao tác trong một shot dài.

### Nguồn đọc thêm, chưa ưu tiên triển khai

- **A Lightweight Moment Retrieval System with Global Re-Ranking and Robust
  Adaptive Bidirectional Temporal Search** (2025): https://arxiv.org/abs/2504.09298.
- **Towards Efficient and Robust Moment Retrieval System: A Unified Framework
  for Multi-Granularity Models and Temporal Reranking** (2025):
  https://arxiv.org/abs/2504.08384.

Đã xác nhận abstract/metadata hai nguồn này; không gán official code hoặc thành
tích HCMC khi chưa xác minh từ nguồn tương ứng.

## 6. Đối chiếu với pipeline hiện tại

Theo `config/pipeline.json`, `kis/solve_kis.py`, `kis/ranking.py` và `kis/text.py`:

| Hiện trạng trong project | Khoảng trống | Nguồn hướng dẫn |
|---|---|---|
| Có events/temporal_constraints nhưng retrieval dùng stream queries | Chưa thực thi ordered-event search | H3, H4, H6 |
| BM25 và MiniLM tìm cùng caption InternVL | Thiếu text độc lập từ ảnh/audio; lỗi caption có thể được cả hai stream ủng hộ | H2, H4, H7, G1, G2 |
| Weighted RRF có sẵn, object boost theo label | Cần đo fusion/coverage, chưa giải quan hệ và thứ tự | H4, H7; ablation riêng của project |
| Qwen pointwise top 20 | Frame đúng ngoài top-20 không được kiểm chứng; ảnh đơn không xác nhận chuỗi | H3, H6; clip verification là đề xuất tích hợp |
| Top grid + ảnh preview | Chưa có ordered-event filmstrip, feedback và sửa từng event | R1, R3, R4 |
| Mapping có timestamp và features sẵn | Có thể temporal-rescore trước khi re-embed/sinh caption lại | H3, H6 |

Các nhận xét về thiếu module dựa trên code đang đọc, không phải lỗi runtime đã
được tái hiện ở từng query. Chưa nạp/rebuild index hoặc chạy benchmark trong lượt nghiên cứu này.

## 7. Outline triển khai theo module

### P0 — Evidence/benchmark snapshot

**Files đề xuất:** `tools/corpus_health.py`, mở rộng `tools/evaluation.py`.

- Xác minh scope runtime, row counts/live Milvus, caption/index identity, ảnh
  local/Drive có thể đọc; phiên bản corpus/model/config có fingerprint.
- Lưu raw query, planner output, raw/rank từng stream, union candidate, sau fusion,
  sau object, trước/sau VLM; tách lỗi fetch ảnh khỏi lỗi inference.
- Giữ 32 query cũ làm regression; bổ sung query bốn nhóm visual/OCR/ASR/sequence
  từ video khác, gán ground truth video+đoạn+frame/event bằng ảnh/video.
- Split theo video/nội dung trùng; không tune trên holdout. Nguồn [H1] định hướng
  loại task, quy trình benchmark này là thiết kế đề xuất riêng.
- Phân loại lỗi trước khi chốt thứ tự đầu tư: evidence không có trong keyframes;
  video đúng không vào union; có evidence nhưng rank sai; sai thứ tự; thiếu OCR/ASR.
  Nếu sampling làm mất frame cần thiết thì đưa sampling lên ưu tiên sớm, không
  bắt buộc đợi P3. Roadmap dưới đây là giả thuyết kỹ thuật cần qua decision gates.

### P1 — Event planner + temporal alignment

**Nguồn:** H3, H4, H6. **Files:** `kis/query_plan.py`, `kis/temporal.py`.

1. Tách query thành global context và các event, cho người dùng sửa từng event.
2. Truy hồi mỗi event qua visual/caption và OCR/ASR khi đã có.
3. Lấy **union candidate videos** của các event, không yêu cầu giao top-k cứng.
4. Rescore local frames trong video ứng viên cho mỗi event, không chỉ các hit
   đã lọt top-k độc lập. Cách này giảm phụ thuộc vào việc từng event tự vào top-k
   [H6], nhưng không cứu được video không lọt bất kỳ seed list nào. Đặt quota mỗi
   event/modality, deduplicate video và giới hạn số local frames tải/rescore;
   cần vectors/metadata của các frame đó, không chỉ ANN hits ban đầu.
5. DP giữ thứ tự và phạt khoảng cách thời gian; tune penalty trên development.
6. Trả representative frame, start/end, event frames, coverage và unknown evidence.

Recurrence đề xuất, thích nghi từ DANTE với timestamp thực:

```text
DP[event, frame] = event_score(event, frame)
    + max(DP[event-1, previous] - lambda * (time(frame)-time(previous)))
      with same_video and time(previous) < time(frame)
```

Chấm thời gian bằng `pts_time`, không dùng số thứ tự ảnh như số giây. Nếu không
đủ bằng chứng, báo coverage thấp; không tự khẳng định cả chuỗi từ một ảnh nồi súp.
Không áp dụng cùng penalty cho query khác duration mà chưa đo. Không ghép các
phần tin không liên quan chỉ vì cùng video dài.

**DP này là compact ordered-path baseline, không phải xác nhận event đã xảy ra.**
Score cộng có thể để event mạnh bù event không khớp. Cần định nghĩa event support
riêng, `unknown`, partial/complete status và kiểm tra query cố tình thiếu một event.
Với lambda cố định, tổng gap penalty bằng lambda × (thời gian cuối - đầu), nên
chỉ khuyến khích span gọn; không kiểm tra độ hợp lý từng bước. Thêm segment
constraints/gap bounds khi có cơ sở; không giữ running-max optimization nếu
transition rule mới không cho phép. Sort timestamp, khởi tạo impossible states,
tie policy và việc cho phép event đồng thời phải được quy định trước.

Start/end ở bước này là **evidence span**, chưa phải boundary sự kiện được model
localization xác định. Đo hai loại output riêng.

**Rebuild:** timestamp/segment metadata; giữ SigLIP/Milvus hiện tại.

### P1 — OCR candidate generation độc lập

**Nguồn:** H2, H4, H7, G1. **Files:** `tools/build_ocr.py`, `kis/ocr.py`.

- Build OCR text+bbox+confidence+timestamp+model/image checksum.
- Tách vùng nội dung/bảng và TV overlay/ticker; giữ text gốc, không xóa vĩnh viễn.
- Exact/phrase/fuzzy retrieval, giữ số và ký hiệu công thức.
- OCR công thức là yêu cầu chưa được xác lập chỉ từ PP-OCR: pilot riêng chữ
  viết tay, text nhỏ, chỉ số và toán tử; lưu ký hiệu không có nghĩa đã nhận dạng đúng.
- Thêm OCR hits vào candidate union **trước rerank**, không chỉ cộng điểm cho
  những frame đã được visual tìm thấy.
- Cho phép event 1 dùng visual, event 2 dùng OCR [H4]; có UI override khi planner sai.

**Rebuild:** OCR artifacts/index; không cần sinh lại mọi caption/features.

### P1/P2 — ASR retrieval theo timestamp

**Nguồn:** H2, H7, R2, G2. **Files:** `tools/build_asr.py`, `kis/asr.py`.

- Whisper/faster-whisper multilingual nhỏ trên video nguồn, chạy offline theo batch/video.
- Lưu segment start/end, transcript gốc, ngôn ngữ; giữ dịch thành field riêng nếu có.
- Search transcript theo phrase/semantic, join tới frame bằng time interval.
- Lời nói về một thao tác không chứng minh thao tác đang xuất hiện; giữ modality
  provenance và dùng visual verification khi query đòi bằng chứng nhìn thấy.

**Rebuild:** cần audio/video, thêm transcript/index; không re-embed visual.

### P2 — Reranker nhẹ và VLM theo đoạn

**Nguồn:** G3 cho text reranking; H3/H6 cho ordered evidence.
**Files:** `kis/text_reranker.py`, `kis/evidence_verifier.py`.

- Rerank 100–300 textual candidates bằng cross-encoder nhỏ trên query và caption/
  OCR/ASR; chọn pool size bằng benchmark. Không gán score=0 khi candidate thiếu text.
- Caption+OCR+ASR dài cần event-aware chunks, không chỉ truncate phần đầu.
- English query chưa đủ nếu OCR/ASR vẫn tiếng Việt. So sánh multilingual reranker
  với query/evidence cùng được dịch, giữ bản gốc; logits không phải xác suất
  điều kiện đã được xác nhận hoặc calibrated confidence.
- Chỉ đưa đoạn/ảnh shortlist vào VLM; xem 4–8 frame theo timestamp cho sequence query.
- Rubric `matched/contradicted/unknown` từng điều kiện, không chỉ một score 0–10.
- Đây là đề xuất chuyển giao; không nguồn nào ở trên xác nhận chính công thức
  Qwen top-20/final score hiện tại của project đã tối ưu.

**Rebuild:** không đổi embedding; thêm online model và metadata.

### P2 — UI timeline, similar-image search và relevance feedback

**Nguồn:** R1, R3, R4, H7. **Files:** API trong `kis/web_app.py`, UI `web/`.

- Filmstrip trước/sau, timeline có event markers, mở video đúng timestamp.
- Click frame phù hợp -> tìm ảnh tương tự bằng SigLIP image encoder đã có.
- Đánh dấu phù hợp/không phù hợp -> thử Rocchio feedback; giữ negative rõ ràng,
  tránh query drift từ một frame gần giống nhưng sai hành động.
- Tách các event thành ô query để sửa đúng phần đang gây nhiễu.
- Hiển thị OCR/ASR gốc và score contribution để kiểm tra lý do top đầu.

**Rebuild:** không bắt buộc; đa phần online/UI. Đo thời gian người dùng tìm đúng
đáp án, không dùng metric tự động để đại diện mọi lợi ích UI.

### P3 — Shot-aware sampling và moment localization

**Nguồn:** H3, H4, G4–G6.

- Nếu frame đúng không tồn tại: shot boundary + chọn nhiều vị trí hoặc adaptive
  sampling trong shot dài; tạo version keyframes/mapping/features mới.
- Moment-DETR/QD-DETR chỉ chạy trên candidate windows để localize start/end.
- Kiểm tra feature/checkpoint compatibility trước khi tích hợp; không coi là
  drop-in head cho SigLIP hiện tại.
- Multi-encoder ensemble như H4/H7 chỉ thử sau khi recall diagnostics chứng minh
  cần kênh visual khác; tránh re-embed toàn corpus ngay từ đầu.

## 8. Áp dụng vào bốn query thực tế của bạn

| Query | Nâng cấp chính | Bằng chứng cần kiểm tra |
|---|---|---|
| p2-1: đổ trứng, cắt đậu hũ, cho vào nồi, khuấy | Event DP + local neighboring frames; ASR/OCR bổ trợ nguyên liệu | Có đúng thao tác và thứ tự, không chỉ ảnh món súp |
| p2-2: học sinh xếp hàng, bảng lớp, đi vào trường, hành lang/cầu thang | Temporal segment ranking + OCR bảng lớp | Chuỗi có đủ sân trường -> bên trong; bảng số có thể đọc thật |
| p2-3: lần lượt cho rau củ/nguyên liệu/thịt vào nồi | Cross-modal per-event retrieval + ASR tên nguyên liệu | Màu/hình dạng, thứ tự đổ và khoảng thời gian cùng đoạn |
| p2-4: ví dụ Tiếng Anh -> giải thích -> công thức | OCR bảng theo thời gian + slide-change detection | Đúng câu chữ và quá trình thay đổi nội dung, không chỉ cùng giáo viên |

Không ép mọi query dùng mọi modality. Với query không cần audio, ASR không phải
điều kiện bắt buộc; OCR/ASR thiếu không đồng nghĩa visual candidate sai.

## 9. Thứ tự làm và thiết kế A/B

| Bước | Thí nghiệm | Điều kiện so sánh |
|---|---|---|
| 0 | Corpus snapshot + baseline rrf hiện tại | Cố định corpus, raw query, planner variants, ground truth |
| 1 | +Event decomposition; +DP temporal là hai ablation | Giữ embedding và ngân sách candidate/API rõ ràng |
| 2 | +OCR; sau đó +ASR | Đo riêng visual, text-critical, speech-critical, multi-event |
| 3 | +Cross-encoder; sau đó +VLM clip verification | Cùng candidate set và cùng ngân sách ảnh khi so thuật toán |
| 4 | +Feedback/UI; +diversity | Tách time-to-answer khỏi Hit@K tự động |
| 5 | Sampling/localization/encoder mới nếu cần | Đo offline time, storage, VRAM và recall gain |

Các control bắt buộc cho temporal pilot:

- Event decomposition + **unordered aggregation** để tách lợi ích nhiều mô tả
  khỏi lợi ích ordering.
- Ordered DP với lambda=0 để tách thứ tự khỏi compactness penalty.
- Đảo/shuffle thứ tự và thêm event không có thật; đo sequence/path correctness
  riêng với video Hit@K để tránh cùng video vẫn được tính là đã hiểu đúng chuỗi.

### Ngân sách tài nguyên pilot

- Tách lịch offline OCR/ASR khỏi model online; pin runtime CUDA/cuDNN và model
  revisions. Faster-whisper có yêu cầu runtime riêng, không suy khả năng tương
  thích chỉ từ việc PyTorch đã nhận CUDA [G2].
- Đo image-fetch/decode time, OCR frames/s, ASR real-time factor, RAM/VRAM đỉnh,
  storage và latency cold/warm trên phần dữ liệu nhỏ trước khi chạy toàn corpus.
- 20 candidate × 4–8 frame = **80–160 ảnh đầu vào**, không còn cùng budget với
  Qwen pointwise 20 ảnh. Chọn số segment nhỏ hơn hoặc báo riêng cost/quality curve.
- Vector 768D FP32 tốn 3.072 bytes/frame; một triệu vector khoảng 3,072 GB raw,
  chưa gồm index/metadata. Đây là phép tính dung lượng, không phải đo RAM process.
  Tránh nhân đôi toàn feature matrix chỉ để temporal rescore.

Metric: Hit@1/5/20, MRR@20, candidate known-target coverage@100/300,
ordered-event correctness, segment overlap/IoU khi có interval ground truth,
VLM valid-score rate, image-fetch rate, P50/P95 latency và thời gian tìm đúng
đáp án có tương tác. nDCG chỉ dùng khi có graded relevance đủ tốt.

Giữ RRF làm baseline nhưng A/B với weighted-sum có sẵn; RRF xuất hiện trong
paper [H4,H7] không chứng minh weight/k/object boost của project là tối ưu.
Không tăng rerank budget rồi gán toàn bộ gain cho model/thuật toán mới.

## 10. Giới hạn, nguồn chưa xác minh và trạng thái thực thi

- Không có khẳng định team đứng nhất/đạt huy chương nếu chưa có nguồn ban tổ chức.
- Publisher abstract không đủ để tái lập toàn bộ hệ thống; code repo chưa được
  chạy trong lượt này. Repo WIP được ghi đúng trạng thái, không coi là module hoàn thiện.
- H6 mới được kiểm tra abstract, chưa đọc full PDF; đánh giá TEMPO-P ở đây giới
  hạn ở thông tin abstract. Các đề xuất implementation là phân tích kỹ thuật riêng.
- Khả năng chạy GPU 6GB, tổng thời gian OCR/ASR và mức tăng top-1 của các module
  chưa được benchmark trên máy/corpus này.
- Các mốc công bố được tách theo bản đọc; paper hệ thống có thể thuộc proceedings
  xuất bản sau năm cuộc thi. Khi cần bibliography chuẩn, lấy citation publisher
  thay vì suy năm từ tên cuộc thi.
- Tài liệu được tổng hợp với AI hỗ trợ tìm kiếm/xác minh nguồn; không tạo số đo
  benchmark hoặc gán kết quả nghiên cứu ngoài thành kết quả của project.

**Đề xuất bắt đầu:** triển khai **event planner + timestamped DP** theo H3/H6 và
**OCR độc lập** theo H2/H4. Hai hướng này sát query của bạn nhất và tận dụng được
caption/features hiện có. Sau khi đo lỗi còn lại, mới quyết định ASR/reranker/
sampling cần đầu tư tới mức nào. Trước full rollout, yêu cầu pilot chứng minh
order sensitivity, đủ event support và chi phí fetch/inference chấp nhận được.
