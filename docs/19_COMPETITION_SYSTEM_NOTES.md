kế hoạch hiện tại vẫn đang nghiêng về pipeline AI kỹ thuật hơn là một hệ thống hoàn chỉnh để thi đấu. Những solution đạt điểm cao cho thấy chất lượng model chỉ là một phần; hệ thống thắng còn cần:

Giao diện thao tác rất nhanh.
Search nhiều cách và chuyển đổi linh hoạt giữa chúng.
Cho phép người dùng sửa query parser.
Điều hướng temporal trước/sau một kết quả.
Chọn, gom và nộp kết quả nhanh.
Hỗ trợ tìm kiếm tương tác nhiều vòng.
Có fallback khi một nhánh retrieval hoặc model lỗi.
Ghi lại toàn bộ session để đánh giá và cải tiến.

MEMORIA đạt hạng nhất tổng thể tại LSC 2025, đồng thời đứng đầu KIS, QA và Ad-hoc. Điểm cải thiện nổi bật nhất của họ là bổ sung visual embedding và Milvus; query parser có ích, nhưng việc rút query thành một “main topic” trước khi tìm bằng vector đôi khi không cải thiện, thậm chí làm kết quả tệ hơn. Đây là cảnh báo rất quan trọng cho kế hoạch hiện tại: không được bắt mọi query đi qua một bản tóm tắt duy nhất rồi mới retrieve.

1. Những bài học trực tiếp từ các solution mạnh
1.1. Bài học từ MEMORIA — hệ thống thắng LSC 2025

MEMORIA dùng hai dòng dữ liệu song song:

Ảnh
├── Annotation
│   ├── YOLOv11 objects
│   ├── PaddleOCR
│   ├── BLIP2 + ClipCap captions
│   ├── Places365 scene
│   ├── time/location metadata
│   └── PostgreSQL
│
└── CLIP visual embedding
    └── Milvus

Sơ đồ ở trang 3 của paper thể hiện rất rõ hai nhánh này: annotation có thể giải thích và lọc; embedding phục vụ tìm kiếm visual trực tiếp.

Các chức năng đáng học:

Text-to-image retrieval.
Image-to-image retrieval.
Event retrieval.
Query parser tự động.
Cho phép bật/tắt từng entity đã parse.
Trộn kết quả textual và vector.
Event segmentation.
Xem event trước và sau theo thời gian.
Chọn nhiều ảnh trong cùng event.
Gửi nhiều kết quả lên server thi đấu.
Album và RAG chatbot, dù đây không phải chức năng thi đấu cốt lõi.

Điểm quan trọng nhất là interface và retrieval hỗ trợ lẫn nhau. Query parser không phải black box; entity được đưa lên UI để người dùng bật hoặc tắt.

1.2. Bài học từ sparse lexical retrieval

Paper sparse lexical biến hình ảnh thành captions và tags bằng M-LLM, sau đó index bằng BM25. Nhánh này đặc biệt mạnh với keyword rõ ràng, tên, vật thể và các điều kiện cụ thể. Paper cũng cho thấy việc người dùng thêm keyword qua nhiều vòng có thể cải thiện retrieval.

Ở trang 5, pipeline gồm:

Ảnh → M-LLM sinh tags/captions → sparse encoding → BM25
Query keyword → sparse encoding → tìm ảnh

Ở trang 6–8, tác giả crop ảnh thành nhiều vùng, caption từng crop rồi nối lại để mở rộng lexical features. Với MS-COCO, hiệu quả tăng dần tới 17 crop và bão hòa khi lên 40 crop. Điều này có nghĩa:

Nên thử crop metadata.
Không nên mặc định chạy 17 crop cho mọi keyframe.
Chỉ nên giữ kỹ thuật này nếu retrieval ablation trên dữ liệu AIC cho thấy có lợi.
Có thể kích hoạt crop cho frame phức tạp, nhiều vùng hoặc có vật thể nhỏ.
1.3. Bài học từ CoLLM

CoLLM cho thấy composed query không nên xử lý bằng cách cộng đơn giản image score và text score. Họ dùng LLM để tạo joint embedding giữa ảnh tham chiếu và modification text.

Từ đó, hệ thống AIC nên hỗ trợ:

Text query.
Image query.
Image + text modification.
“Tìm giống ảnh này nhưng đổi bối cảnh”.
“Giữ người/vật thể này, nhưng hành động khác”.
“Giống kết quả này hơn”.
Negative feedback từ ảnh không phù hợp.

Tuy nhiên, không nên đưa CoLLM full vào baseline ngay. Đây là nhánh nghiên cứu bổ sung sau khi text-only KIS, VQA và AVS đã ổn định.

CoLLM cũng cảnh báo benchmark có thể mơ hồ: nhiều kết quả đúng nhưng chỉ một kết quả được đánh dấu. Vì vậy, gold set nội bộ phải chứa near-positive, hard negative và nhiều interval hợp lệ khi cần.

2. Đánh giá từng phần trong kế hoạch hiện tại
Thành phần hiện tại	Đánh giá	Điều chỉnh cần thiết
Scene detection	Đúng	Thêm fallback uniform window và event merging
Keyframe extraction	Đúng	Thêm neighbor frames và exact-frame selection
Caption	Đúng nhưng chưa đủ	Caption theo frame, multi-frame, region và event
OCR	Đúng	Thêm text normalization, exact/partial/fuzzy search
ASR	Đúng	Cần timestamp word/segment và seek trực tiếp
Object/action metadata	Hợp lý	Không nên chạy detector nặng cho mọi query online
Dense retrieval	Bắt buộc	Không dùng một model duy nhất nếu chưa benchmark
Sparse retrieval	Bắt buộc	Nên index field riêng, không nối tất cả text thành một trường
Query decomposition	Có ích	Không thay thế query gốc; phải search cả original query
Query expansion	Có tiềm năng	Cần kiểm soát drift và cho người dùng tắt variant
Modality routing	Hợp lý	Rule-based trước, LLM chỉ hỗ trợ
Fusion	Bắt buộc	Bắt đầu bằng RRF, sau đó mới học weight
Reranking	Cần thiết	Chỉ rerank tập nhỏ; không để LLM rerank hàng nghìn candidate
Temporal reasoning	Đang còn chung chung	Cần event/neighbor retrieval và sequence scorer cụ thể
VQA	Chưa đủ hoàn thiện	Cần evidence selection, answer generator, verifier, abstention
AVS	Chưa đủ hoàn thiện	Cần relevance grade, dedup, diversity và threshold
UI	Chưa đủ để thi	Thiếu interactive query editing, exact-frame selection, submission workflow
Evaluation	Khá tốt	Thêm interaction latency, time-to-first-correct-result và submission error rate

Problem definition hiện tại đã mô tả đúng KIS, VQA và AVS, cũng như yêu cầu evidence, temporal reasoning và task-specific metric. Nhưng để trở thành hệ thống thi đấu, cần mở rộng từ “search API” thành interactive competition platform.

3. Kiến trúc chức năng hoàn chỉnh
AIC 2026 Competition System
│
├── A. Data & Index Management
├── B. Query Understanding
├── C. Multi-branch Retrieval
├── D. Fusion & Reranking
├── E. Temporal & Event Reasoning
├── F. KIS Workspace
├── G. VQA Workspace
├── H. AVS Workspace
├── I. Interactive Search & Feedback
├── J. Evidence Viewer
├── K. Submission Management
├── L. Competition Operations
├── M. Evaluation & Experimentation
└── N. Administration & Reliability
4. A. Data và Index Management

Đây là chức năng online quản lý dữ liệu đã được offline pipeline tạo ra.

4.1. Dataset browser
Xem danh sách collection, batch, video.
Tìm theo video_id.
Xem duration, fps, resolution, frame count.
Xem trạng thái đã xử lý:
scene;
keyframe;
visual embedding;
caption;
OCR;
ASR;
object/action;
event.
Xem lỗi và phần metadata còn thiếu.
Reprocess một video hoặc một scene.
4.2. Index registry

Quản lý:

Dense index theo model.
Sparse index.
OCR exact index.
ASR index.
Object/action index.
Event index.
Clip/video embedding index.
Region/crop index nếu có.

Mỗi index phải có:

{
  "index_id": "siglip_scene_v4",
  "model": "siglip-so400m",
  "granularity": "scene",
  "dataset_version": "aic2026_v7",
  "num_items": 4200000,
  "created_at": "...",
  "status": "ready"
}
4.3. Version compatibility check

Hệ thống phải chặn hoặc cảnh báo khi:

Query encoder khác index encoder.
Metadata và vector index khác dataset version.
Frame mapping không tồn tại.
Event index dùng scene version cũ.
OCR index chưa hoàn tất.
4.4. Index selector

Trong chế độ nghiên cứu:

Chọn visual model.
Chọn sparse backend.
Chọn granularity: frame, scene, event, clip.
Chọn fusion profile.

Trong chế độ thi đấu, chỉ hiển thị preset đã kiểm thử.

5. B. Query Understanding đầy đủ
5.1. Các loại input

Hệ thống nên nhận:

Text tiếng Việt.
Text tiếng Anh.
Text hỗn hợp.
Nhiều câu.
Query được cung cấp dần theo thời gian.
Ảnh tham chiếu.
Ảnh + modification text.
Frame đã chọn từ kết quả trước.
VQA question.
Bộ inclusion/exclusion criteria cho AVS.
5.2. Task router

Cho phép:

Người dùng chọn KIS/VQA/AVS.
Auto-detect task.
Hiển thị confidence.
Người dùng override task.

Không nên dựa hoàn toàn vào auto-routing.

5.3. Query decomposition

Tách thành:

Objects.
People.
Clothing/attributes.
Actions.
Scene.
Spatial relations.
OCR phrases.
ASR phrases.
Named entities.
Numbers.
Colors.
Camera/view.
Temporal sequence.
Negative conditions.
Must-match.
Nice-to-have.
Answer type.
Evidence type.

Schema hiện tại đã có nền tảng đúng cho phần này.

5.4. Query preservation

Luôn tạo ít nhất ba loại query:

Q0: Query gốc, giữ nguyên.
Q1: Query chuẩn hóa nhẹ.
Q2...Qn: Query mở rộng hoặc query con.

Không thay Q0 bằng một bản tóm tắt LLM. Kết quả MEMORIA cho thấy main-topic extraction đôi khi làm Milvus trả kết quả tệ hơn query nguyên bản.

5.5. Multilingual normalization
Giữ bản tiếng Việt.
Sinh bản tiếng Anh.
Giữ proper noun nguyên gốc.
Sinh không dấu cho OCR/ASR fuzzy search.
Chuẩn hóa viết tắt.
Map synonym Việt–Anh.
Không dịch số, mã hoặc tên riêng một cách tùy tiện.
5.6. Query editor tương tác

Người dùng có thể:

Bật/tắt entity.
Đổi must-match/nice-to-have.
Thêm exact phrase.
Đánh dấu OCR hoặc ASR phrase.
Chỉnh temporal order.
Thêm negative constraint.
Chỉnh modality weight.
Khôi phục query gốc.

Đây là điểm nên học trực tiếp từ MEMORIA: entity được parse không chỉ nằm trong backend mà có thể toggle trên UI.

5.7. Progressive query

KIS chung kết có thể cung cấp mô tả dần. Hệ thống cần:

Lưu query state từng mốc.
Append clue mới.
Rerun chỉ các nhánh bị ảnh hưởng.
So sánh rank trước và sau clue.
Highlight candidate tăng/giảm rank.
Giữ pinned candidate qua các vòng.
6. C. Multi-branch Retrieval đầy đủ
6.1. Dense text-to-frame search
Query text với keyframe embedding.
Search nhiều encoder.
Search theo frame và scene.
Top-K riêng cho từng model.
Có ANN index.
Có exact-search mode cho tập nhỏ/debug.
6.2. Dense text-to-clip search

Cần cho:

Actions.
Motion.
Chuỗi sự kiện ngắn.
Query không thể thể hiện bằng một frame.

Index clip nên tách biệt với frame index.

6.3. Caption search

Search trên các field:

short caption;
dense caption;
scene caption;
event caption;
action caption;
relation caption;
region caption.

Không nên nối tất cả vào một chuỗi vì sẽ mất khả năng weight từng field.

6.4. OCR retrieval

Các mode:

Exact phrase.
AND keyword.
OR keyword.
Fuzzy match.
Number match.
Vietnamese normalized match.
Bounding-box confidence filter.
Search theo neighboring frames.
6.5. ASR retrieval
Exact phrase.
Keyword search.
Semantic transcript search.
Speaker-aware search nếu có diarization.
Time-window expansion.
Seek trực tiếp tới timestamp của từ/câu.
6.6. Object/action search
Closed vocabulary tags.
Open vocabulary tags.
Object + attribute.
Object + action.
Object + relative position.
Count filter.
Presence/absence condition.
6.7. Scene/environment search
Indoor/outdoor.
Kitchen, road, classroom, field, hospital.
Day/night.
Weather.
Camera/view.
Dominant color.
Location/time metadata nếu dataset hỗ trợ.

MEMORIA cho thấy scene recognition, time và location vẫn có giá trị như filter bổ sung.

6.8. Image-to-image search

Nguồn ảnh có thể là:

Upload.
Clipboard.
Screenshot.
Một result card.
Current video frame.
Crop do người dùng chọn.

Search:

Global image similarity.
Person/object crop similarity.
Same scene.
Same event.
Similar visual layout.
6.9. Composed search
Reference image + text modification

Ví dụ:

“Giống ảnh này nhưng ban đêm.”
“Giữ người áo đỏ, nhưng đang chạy.”
“Cùng căn phòng này nhưng không có người.”
“Giống cảnh này, phía sau có biển chữ.”

Đây là extension phù hợp với hướng CoLLM.

6.10. Event retrieval
Search event caption.
Search event representative frame.
Xem toàn bộ ảnh/frame trong event.
Đi tới event trước/sau.
Search related events.
Search “A rồi sau đó B”.

MEMORIA cho thấy event-centered retrieval giảm clutter và phù hợp cách người dùng nhớ một chuỗi hoạt động hơn image-centered retrieval.

7. D. Fusion và Reranking
7.1. Candidate normalization

Mỗi nhánh trả về score khác scale. Cần:

rank normalization;
min-max/z-score khi hợp lý;
percentile score;
reciprocal rank.
7.2. Candidate deduplication

Gộp candidate khi:

Cùng scene.
Frame gần nhau.
Cùng event.
Embedding quá giống.
Cùng video_id và timestamp gần nhau.

Vẫn giữ best frame và danh sách supporting frames.

7.3. Fusion methods

Nên hỗ trợ:

RRF.
Weighted score.
Query-adaptive weights.
Learning-to-rank.
Rule-based exact-match boosting.

Baseline nên là RRF vì ít phụ thuộc calibration.

7.4. Must-match filtering

Không hard-filter ngay từ đầu, trừ điều kiện chắc chắn như:

collection;
video;
thời gian;
exact OCR rất rõ.

Với visual attributes, nên dùng penalty/boost vì metadata có thể sai.

7.5. Lightweight reranking

Top 100–300:

Text-text cross encoder trên caption/metadata.
Late interaction.
Query-candidate feature scorer.
Rule-based coverage scorer.
7.6. Deep reranking

Top 5–30:

VLM xem keyframes.
Multi-frame VLM.
LLM evidence scorer.
Temporal verifier.
OCR/ASR recheck.
7.7. Evidence coverage score

Reranker cần trả về không chỉ score mà cả:

{
  "must_match_coverage": 0.8,
  "visual_match": 0.73,
  "ocr_match": 1.0,
  "asr_match": 0.0,
  "temporal_match": 0.65,
  "contradictions": ["indoor instead of outdoor"]
}
7.8. Hard-negative comparison

Candidate tốt thường chỉ khác target một chi tiết. Nên cho reranker so sánh:

target-like candidate;
nearest visual distractors;
candidate có cùng object nhưng sai action;
candidate có cùng scene nhưng sai text.

CoLLM cho thấy hard negatives và benchmark ambiguity là yếu tố quan trọng trong đánh giá retrieval.

8. E. Temporal và Event Reasoning
8.1. Neighbor navigation

Từ một result:

Previous frame.
Next frame.
Previous scene.
Next scene.
Previous event.
Next event.
±2, ±5, ±10, ±30 giây.
8.2. Ordered sub-query retrieval

Với query gồm A → B → C:

Retrieve candidates cho A.
Retrieve candidates cho B.
Retrieve candidates cho C.
Join theo video.
Kiểm tra thời gian tăng dần.
Tính gap phù hợp.
Rank sequence.
8.3. Conditional event search

Ví dụ:

Tìm cảnh ăn tại nhà hàng sau khi lái xe dưới trời mưa.

Pipeline:

Retrieve “driving in rain”
→ lấy các khoảng thời gian
→ tìm event restaurant sau đó trong window
→ rank theo temporal distance và semantic match

Đây chính là chức năng MEMORIA nêu trong future work và rất phù hợp với query phức tạp của AIC.

8.4. Temporal storyboard

Hiển thị chuỗi:

[A: cào muối] → [B: vẫy tay sau bảng] → [C: đứng trước căn nhà]

Mỗi step:

frame;
score;
timestamp;
evidence;
button replace.
8.5. Event graph

Chức năng nâng cao:

Person/object appears.
Action.
Location.
Before/after.
Same event.
Same track.
Related OCR/ASR evidence.

Không cần làm graph database ngay trong MVP; có thể biểu diễn bằng relational/event table trước.

9. F. Chức năng KIS hoàn chỉnh
9.1. Progressive clue mode
Nhập clue từng phần.
Đồng hồ countdown.
Search tự động hoặc manual.
Candidate rank history.
Pin candidate.
Lock result.
Nộp sớm.
9.2. Known-item signature

Tự động trích:

Unique object.
Unique action.
Unique scene.
Exact text.
Temporal cue.
Rare conjunction.
9.3. Multi-view results
Grid.
Detailed list.
Video-grouped.
Timeline.
Event-grouped.
Storyboard.
9.4. Exact moment refinement
Xem video loop.
Chọn scene.
Chọn frame.
Chọn current frame.
Fine seek theo frame.
Nhập frame index.
Verify timestamp ↔ frame.
9.5. Similarity refinement
More like this.
Less like this.
Same person/object.
Same scene.
Same action.
Exclude this visual style.
Search from crop.
9.6. Fast submission
Add top candidate.
Replace candidate.
Bulk select.
Validate.
Send to competition server.
Record server response.
Prevent duplicate submission.
Undo trước khi hết thời gian nếu luật cho phép.

MEMORIA bổ sung các nút chọn nhiều ảnh trong cùng event và gửi hàng loạt để tăng tốc khi thi.

10. G. Chức năng VQA hoàn chỉnh
10.1. Question parser

Xác định:

answer type;
target entity;
evidence modality;
reasoning type;
temporal scope;
count/tracking requirement.
10.2. Evidence retrieval
Visual evidence.
OCR evidence.
ASR evidence.
Temporal neighbor.
Event context.
Multiple segments.
10.3. Evidence pack builder

Chỉ đưa vào model:

best keyframes;
relevant crops;
ASR windows;
OCR lines;
previous/next context;
score và uncertainty.

Không đưa toàn bộ top-100 vào LLM.

10.4. Answer engines

Theo answer type:

Rule extraction cho exact OCR/ASR.
Count model/tool.
VLM reasoning.
LLM trên evidence text.
Hybrid rule + VLM.
Multiple-choice scorer nếu task có lựa chọn.
10.5. Counting
Candidate frame selection.
Object detector.
Deduplicate boxes.
Multi-frame consistency.
Manual correction trong UI.
10.6. Tracking
Chọn target.
Theo dõi qua frames.
Kiểm tra hành động tiếp theo.
Gộp identity qua scene gần nhau khi khả thi.
10.7. Answer verifier

Kiểm tra:

Answer có được evidence hỗ trợ không.
Có contradiction không.
Numeric answer có nhất quán không.
OCR/ASR có confidence đủ không.
Có nhiều answer khả dĩ không.
10.8. Abstention

Trạng thái:

Supported.
Partially supported.
Insufficient evidence.
Contradicted.

Không đủ evidence thì không đoán.

10.9. Human-assisted VQA

Cho phép người dùng:

Chọn lại evidence.
Sửa OCR.
Sửa count.
Chọn answer type.
Viết answer thủ công.
Chạy verifier lại.
Nộp answer.
11. H. Chức năng AVS hoàn chỉnh
11.1. Inclusion/exclusion editor

Ví dụ:

INCLUDE
- adult
- child
- garden
- teaching OR watering

EXCLUDE
- indoor
- artificial plants
11.2. Relevance grading
3: full match.
2: mostly match.
1: weak.
0: irrelevant.
11.3. Threshold control
Strict.
Balanced.
Broad recall.
Custom score threshold.
11.4. Diversity
MMR.
Visual clustering.
Event clustering.
Maximum per video.
Minimum temporal distance.
Best per cluster.
11.5. Coverage overview

Hiển thị:

số unique video;
số event;
số result đã chọn;
cluster coverage;
duplicate rate;
estimated precision.
11.6. Bulk selection
Select all grade ≥2.
Select best per event.
Select best per video.
Exclude cluster.
Manually add/remove.
11.7. Result basket

AVS basket khác KIS:

chứa nhiều segment;
reorder;
deduplicate;
validate max count;
export hoặc submit hàng loạt.
12. I. Interactive Search và Relevance Feedback

Đây là phần kế hoạch hiện tại còn thiếu rõ nhất.

12.1. Positive feedback
Candidate này đúng.
Candidate này gần đúng.
Giống candidate này hơn.
Giữ object này.
Giữ scene này.
Giữ action này.
12.2. Negative feedback
Sai object.
Sai action.
Sai scene.
Sai thời gian.
Sai text.
Trùng lặp.
Đúng video nhưng sai moment.
12.3. Keyword refinement
Add keyword.
Remove keyword.
Promote to exact phrase.
AND/OR toggle.
Exclude keyword.

Paper sparse lexical cho thấy iterative keyword incorporation là một cơ chế thực tế để query dần phản ánh đúng ý định người dùng.

12.4. Search history tree
Query 1
├── + “biển chữ”
│   ├── + exact phrase
│   └── exclude indoor
└── More like result #4

Cho phép quay lại branch trước.

12.5. Session memory

Ghi:

query;
parsed query;
configuration;
results;
clicked candidates;
watched segments;
selected frame;
submitted item;
response của competition server.
13. J. Evidence Viewer hoàn chỉnh
13.1. Video player
Segment autoplay.
Loop.
Fine seek.
Frame-by-frame.
Speed.
Fullscreen.
Keyboard shortcuts.
Current frame index.
Copy timestamp/frame.
13.2. Frame strip
Neighbor thumbnails.
Keyframe marker.
OCR marker.
Motion marker.
Blur/quality.
Selected frame.
13.3. Metadata tabs
Caption.
OCR.
ASR.
Objects.
Actions.
Scene.
Event.
Temporal.
Scores.
Raw metadata.
13.4. Overlays
OCR boxes.
Object boxes.
Selected crop.
Track path.
Click-to-search region.
13.5. Score explanation
Branch score.
Fusion rank.
Rerank score.
Must-match coverage.
Contradiction.
Index/model version.
13.6. Event context

Hiển thị:

current event;
previous event;
next event;
event description;
all frames in event.
14. K. Submission Management
14.1. Submission formatter

Tách riêng theo task:

KIS formatter.
VQA formatter.
AVS formatter.

Không dùng trực tiếp result object nội bộ.

14.2. Validation

Trước khi gửi:

Task đúng.
Video ID tồn tại.
Frame hợp lệ.
Timestamp hợp lệ.
Answer không rỗng.
Số lượng kết quả đúng luật.
Không duplicate.
Không dùng stale dataset/index.
14.3. Competition server connector
Endpoint config.
API key.
Test connection.
Submit.
Retry có kiểm soát.
Record response.
Prevent accidental double submit.
Offline queue khi mạng lỗi.
14.4. Submission log
Time | Query | Task | Payload | Server response | Status
14.5. Team coordination

Khi nhiều thành viên:

Shared candidate board.
Claimed query.
Reviewer.
Approved result.
Submission owner.
Lock để tránh hai người nộp trùng.
15. L. Competition Operations
15.1. Competition mode
UI tối giản.
Model/index cố định.
Không hiển thị config nguy hiểm.
Keyboard-first.
Countdown.
Auto-save.
Quick submission.
15.2. Research mode
Full query plan.
Model selector.
Score details.
Compare runs.
Raw metadata.
Feedback labels.
15.3. Practice mode
Replay query.
Reveal ground truth sau khi làm.
Measure time-to-correct.
Compare human/team performance.
Error analysis.
15.4. Emergency fallback

Khi GPU/VLM lỗi:

Dense CPU index.
Sparse BM25.
OCR/ASR exact search.
Cached recent queries.
Manual metadata filters.
Local video playback.
15.5. Warm-up

Trước vòng thi:

Load model.
Verify indexes.
Test competition endpoint.
Check video paths.
Check GPU memory.
Run fixed smoke queries.
Cache common data.
16. M. Evaluation và Experimentation
16.1. Retrieval metrics

KIS:

R@1/5/20/50/100.
MRR.
Hit in interval.
Time-to-first-correct-result.

AVS:

mAP.
nDCG.
Precision/Recall.
Unique relevant events.
Redundancy.

VQA:

Exact Match/F1.
Numeric accuracy.
Evidence hit.
Grounded answer rate.
Hallucination rate.

Các metric task-specific này đã được định hướng đúng trong tài liệu hiện tại.

16.2. Interactive metrics

Cần thêm:

Search latency.
First result latency.
Rerank latency.
Number of interactions.
Time to submission.
Submission error rate.
Number of query refinements.
Human success rate.
16.3. Ablation
Visual only
+ Caption
+ OCR
+ ASR
+ Object/action
+ Sparse retrieval
+ Query expansion
+ Fusion
+ Rerank
+ Temporal
+ Feedback
16.4. Query group evaluation
Visual-only.
OCR-heavy.
ASR-heavy.
Action-heavy.
Temporal.
Negative constraint.
Counting.
Ambiguous.
Vietnamese.
Mixed-language.
16.5. Compare runs
Side-by-side ranking.
Rank movement.
Common/missing candidates.
Latency.
Cost.
Configuration snapshot.
16.6. Benchmark refinement
Hard-negative review.
Multiple valid intervals.
Ambiguity flag.
Reviewer agreement.
Remove unusable queries.
Add near positives.

CoLLM cho thấy benchmark refinement có thể làm rõ chênh lệch giữa model thực sự hiểu composed query và model chỉ khớp chung chung.

17. N. Administration và Reliability
17.1. Health dashboard
API.
GPU.
Models.
Dense index.
Sparse index.
Metadata DB.
Video storage.
Competition server.
17.2. Request trace
query
→ parse
→ branch retrieval
→ fusion
→ rerank
→ evidence
→ answer/submission
17.3. Partial failure

Response phải ghi rõ:

ASR timeout.
OCR unavailable.
Reranker skipped.
VLM out-of-memory.
Index stale.

Không silent fallback.

17.4. Cache
Query embedding.
Search result.
Evidence pack.
Thumbnail.
Video segment.
VQA result.
17.5. Security
Secrets không nằm trong frontend bundle.
Competition API token phía backend.
Role-based access.
Audit log.
Rate limiting.
Input validation.
17.6. Reproducibility

Mỗi result lưu:

model version;
index version;
prompt version;
fusion config;
timestamp;
request ID.


19. Thiết kế giao diện chốt
┌───────────────────────────────────────────────────────────────────────────────┐
│ AIC 2026   KIS | VQA | AVS   Competition Mode   API● GPU● Index●  08:42      │
├─────────────────┬──────────────────────────────────┬──────────────────────────┤
│ QUERY           │ RESULTS                          │ EVIDENCE                 │
│                 │                                  │                          │
│ Raw query       │ Grid / List / Timeline / Event   │ Video player             │
│ Progressive     │                                  │ Frame strip              │
│ clue history    │ #1 Scene card                    │ Previous / Next event     │
│                 │ #2 Scene card                    │                          │
│ Parsed entities │ #3 Scene card                    │ Caption / OCR / ASR       │
│ [toggle chips]  │                                  │ Objects / scores          │
│                 │                                  │                          │
│ Must match      │                                  │ Select exact frame        │
│ Nice to have    │                                  │ Search from image/crop    │
│ Negative        │                                  │                          │
│                 │                                  │                          │
│ Search / Refine │                                  │                          │
├─────────────────┴──────────────────────────────────┴──────────────────────────┤
│ RESULT BASKET / SUBMISSION: Query 07 | V003 F03610 | Validate | Send | Log   │
└───────────────────────────────────────────────────────────────────────────────┘

Các drawer phụ:

Query Studio.
Compare Runs.
Event Explorer.
AVS Basket.
VQA Evidence Table.
Submission Log.
System Health.
20. Kiến trúc triển khai nên chốt
LOCAL COMPETITION CLIENT
React + TypeScript
│
├── Search workspace
├── Video player
├── Submission manager
├── Local session/cache
└── SSE/WebSocket client
        │ HTTPS
        ▼
REMOTE FASTAPI BACKEND
│
├── Query Service
├── Retrieval Orchestrator
├── Dense Search Service
├── Sparse Search Service
├── Fusion Service
├── Rerank Service
├── Evidence Service
├── VQA Service
├── Video/Frame Service
├── Submission Proxy
└── Monitoring
        │
        ├── FAISS/Milvus/Qdrant
        ├── Elasticsearch/OpenSearch
        ├── PostgreSQL
        ├── Redis
        ├── Object storage
        └── GPU workers

Backend server nên giữ competition API token. Frontend local không nên gửi token chính thức trực tiếp đến server của ban tổ chức.