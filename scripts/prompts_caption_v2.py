"""Bộ prompt caption v2 — thiết kế ngược từ 55 câu hỏi thi thật.

Nguồn: 25 câu vòng sơ tuyển P1 + 30 câu P2 nhóm A (xem docs/41). Đo trên 55
câu đó, tần suất loại manh mối là:

    đếm / thứ tự            67,3%
    chuỗi nhiều cảnh        65,5%
    màu sắc                 61,8%
    vị trí không gian       41,8%
    trang phục / phụ kiện   40,0%
    chữ trên màn hình       36,4%
    phỏng vấn / người nói   18,2%
    danh từ riêng / văn hoá 18,2%
    máy quay / loại cảnh    16,4%

Caption hiện tại của `competition_pack_v3` là MỘT câu ≤35 từ (prompt
`export_caption_v1`). Nó tả được chủ thể + bối cảnh, nhưng đo trên 168.414
caption thì `quay chậm` xuất hiện 0 lần, `zoom` 3 lần, `sau đó` 16 lần
(0,01%) — tức không có chỗ nào lưu chuyển động máy quay lẫn diễn biến giữa
các cảnh. Bốn prompt dưới đây lấp đúng những chỗ đó.

Ba tầng, chạy độc lập được:

    T3  video rollup   —  873 lời gọi, TEXT-ONLY, dùng caption ĐÃ CÓ.
    T2  shot window    —  ~88.000 lời gọi, 3 keyframe liên tiếp mỗi lời gọi.
    T1  keyframe card  —  176.707 lời gọi, thay caption 35 từ hiện tại.

Thứ tự đáng chạy là T3 -> T2 -> T1, không phải ngược lại: T3 không tốn GPU
mà đánh trúng 65,5% câu hỏi dạng chuỗi cảnh.
"""

from __future__ import annotations

from datetime import datetime, timezone

# --------------------------------------------------------------------------
# T1 — thẻ keyframe. Thay cho `export_caption_v1` (1 câu, 35 từ).
# --------------------------------------------------------------------------

PROMPT_KEYFRAME_CARD = """\
Bạn đang tạo METADATA TÌM KIẾM cho MỘT khung hình video.

Người tìm kiếm sẽ mô tả khung hình này bằng lời, từ trí nhớ, sau khi đã xem
video. Họ thường nhớ: có bao nhiêu người, ai mặc màu gì, ai đứng bên nào, chữ
gì hiện trên màn hình, và chi tiết nhỏ lạ nhất. Metadata phải ghi đúng những
thứ đó.

Trả về ĐÚNG 10 dòng theo nhãn dưới đây, đúng thứ tự, mỗi nhãn một dòng.
Không Markdown. Không đánh số. Không lời dẫn. Không giải thích.
Dòng nào không có gì để ghi thì viết đúng một chữ: none
Không lặp lại một từ hay một cụm quá hai lần trong cùng một dòng.

MOTA:
DEM:
NGUOI:
VATTHE:
CHITIET_NHO:
CHU:
NGHE:
KHUNGHINH:
DACTRUNG:
TUKHOA:

Cách điền từng dòng:

MOTA — 2 đến 4 câu tiếng Việt, TỐI THIỂU 40 từ. Bắt buộc nêu: cỡ cảnh và góc
máy, ai hoặc vật gì ở bên trái / ở giữa / bên phải, màu sắc chính, và hành
động đang diễn ra. Dưới 40 từ là SAI. Không lặp lại cùng một chi tiết hai lần.

DEM — chỉ liệt kê những nhóm THẬT SỰ có trong ảnh, dạng "tên nhóm=số", ngăn
cách bằng dấu chấm phẩy. Tên nhóm viết tiếng Việt có dấu, đầy đủ, ví dụ dạng
"người=3" hoặc "người đội nón đỏ=2". Chỉ ghi số khi đếm được từng cái một;
đông hơn 10 hoặc phải ước lượng thì ghi "=nhiều". Tối đa 5 nhóm, chọn nhóm
đáng nhớ nhất. Nhóm nào không có trong ảnh thì BỎ HẲN nhóm đó, không được
viết "=none" cho từng nhóm. Cả ảnh không có gì đếm được thì cả dòng ghi: none

NGUOI — tối đa 5 người nổi bật, mỗi người một cụm ngăn cách bằng " || ".
Mỗi cụm: vị trí trong khung, giới và độ tuổi ước lượng, trang phục KÈM MÀU,
phụ kiện, tư thế hoặc hành động.

VATTHE — 5 đến 15 vật thể nhìn rõ, ngăn cách bằng dấu phẩy. MỖI vật bắt buộc
kèm màu. Chỉ dùng tiếng Việt. Không chắc đó là món gì hay loài gì thì tả hình
dạng và màu thay vì đoán tên.

CHITIET_NHO — vật nhỏ dễ bỏ qua nhưng dễ nhớ: dây buộc, nhãn dán, thẻ tên,
hoa văn, huy hiệu, phụ kiện, vết bẩn, hoạ tiết in trên áo. Mỗi thứ kèm màu và
vị trí. Đây là dòng quyết định phân biệt khung hình, hãy nhìn kỹ trước khi ghi.

CHU — MỌI chuỗi chữ hoặc số đọc được, chép NGUYÊN VĂN từng ký tự, giữ đúng
dấu tiếng Việt, ký hiệu tiền tệ và chữ nước ngoài. Mỗi chuỗi một cụm dạng
"nguyên văn" @vitri (loai). Trường loai phải là ĐÚNG MỘT trong các giá trị
sau, chép y nguyên, không dấu: bienhieu banner chyron phude bang_gia bien_bao
the_ten slide man_hinh bao_chi bien_so khac.
Giữ lại MỌI chuỗi bạn đọc được, kể cả khi nó khác với gợi ý OCR. Chỉ bỏ chuỗi
mà bạn CHẮC CHẮN không có trong ảnh. Đọc được lờ mờ thì vẫn ghi và thêm dấu ?
ngay sau dấu ngoặc kép.
KHÔNG ghi vào đây: tên kênh (HTV, HTV7, HTV9, HTV Online), đồng hồ giờ, tên
chương trình lặp ở mọi khung hình (60 giây, Món Ngon Mỗi Ngày), và dòng chữ
chạy ở đáy màn hình.

NGHE — chỉ điền khi phần LỜI THUYẾT MINH được cung cấp ở cuối prompt. Ghi tên
riêng, tên loài, địa danh, chức danh, số liệu kèm đơn vị nghe được VÀ có liên
quan tới thứ đang hiện trong khung hình. Không chép lại cả câu thuyết minh.
Không có thì ghi: none

KHUNGHINH — đúng 3 giá trị ngăn cách bằng dấu phẩy, mỗi nhóm chọn ĐÚNG MỘT:
  cỡ cảnh:  dai_canh | toan | trung | can | dac_ta
  góc máy:  ngang_tam_mat | tu_tren_cao | tu_duoi_len | flycam | nghieng
  kiểu cảnh: phong_van | canh_quay_thuc_te | do_hoa | slide_bai_giang |
             camera_an_ninh | anh_tinh
Góc máy phải KHỚP với câu đầu của MOTA: MOTA nói "máy đặt thấp nhìn lên" thì ở
đây phải là tu_duoi_len, nói "nhìn từ trên xuống" thì là tu_tren_cao. Hai dòng
mâu thuẫn nhau là SAI. Không chắc nhóm nào thì ghi khong_ro cho nhóm đó.

DACTRUNG — ĐÚNG một câu, tối đa 25 từ, kết thúc bằng dấu chấm: chi tiết dị
biệt nhất giúp phân biệt khung hình này với những khung hình gần giống. Phải
nêu kèm màu hoặc số lượng. Không viết hoa nhấn mạnh. Không được là câu chung
chung kiểu "nồi inox sáng bóng" hay "tay cầm kéo".

TUKHOA — 8 đến 15 từ khoá tiếng Việt ngăn cách bằng dấu phẩy. Gồm tên riêng,
tên loài, tên trang phục truyền thống, tên lễ hội — chỉ khi nhận ra CHẮC CHẮN.

VÍ DỤ ĐỊNH DẠNG. Ảnh trong ví dụ là một cảnh trượt tuyết trên núi Alps, KHÔNG
liên quan gì tới ảnh bạn đang xem. Chỉ học cách trình bày từng dòng; tuyệt đối
không mang bất kỳ chi tiết nào của ví dụ sang thẻ của bạn.
MOTA: Cảnh toàn, máy đặt cao nhìn xuống, ghi lại một sườn núi phủ tuyết trắng dưới trời xanh. Ở giữa khung, hai người trượt tuyết mặc áo khoác đỏ đang đổ dốc theo hàng dọc. Bên trái là hàng cột cáp treo màu vàng, bên phải là rừng thông xanh sẫm phủ tuyết.
DEM: người trượt tuyết=2; cột cáp treo=4
NGUOI: giữa khung, người lớn, áo khoác đỏ, đeo kính trượt tuyết, đang đổ dốc || sau lưng người thứ nhất, người lớn, áo khoác đỏ, cầm gậy trượt
VATTHE: tuyết trắng phủ kín sườn dốc, áo khoác đỏ, cáp treo màu vàng, rừng thông xanh sẫm, bầu trời xanh nhạt
CHITIET_NHO: số hiệu màu đen in trên cabin cáp treo, vệt trượt hình chữ S trên nền tuyết
CHU: none
NGHE: none
KHUNGHINH: toan, tu_tren_cao, canh_quay_thuc_te
DACTRUNG: Hai người áo khoác đỏ đổ dốc thành hàng dọc ngay dưới bốn cột cáp treo vàng.
TUKHOA: trượt tuyết, sườn núi, tuyết trắng, áo khoác đỏ, cáp treo, rừng thông, thể thao mùa đông, Alps

Quy tắc bắt buộc:
- Chỉ ghi thứ NHÌN THẤY trong chính ảnh này. Không suy đoán nguyên nhân, cảm xúc, danh tính, địa danh hay sự kiện nằm ngoài ảnh.
- Không chắc thì bỏ, đừng đoán. Nếu gần chắc, viết "có thể" ngay trước cụm đó.
- Đếm sai tệ hơn không đếm. Đoán sai tên món ăn hay tên loài tệ hơn tả hình dạng.
- Chép sai một ký tự trong CHU là hỏng cả dòng: đọc lại từng ký tự trước khi ghi.\
"""

# Nối vào cuối PROMPT_KEYFRAME_CARD khi đã có OCR sidecar cho keyframe đó.
# 80,2% keyframe của pack có sẵn OCR — đưa vào làm mồi giúp model khỏi phải
# đọc lại từ đầu, nhưng phải kèm quyền phủ nhận, nếu không nó sẽ chép mù.
OCR_HINT_SUFFIX = """

GỢI Ý OCR (máy trích sẵn, CÓ THỂ SAI hoặc THIẾU):
{ocr_hint}

Đối chiếu danh sách trên với ảnh: giữ chuỗi nào thật sự nhìn thấy, sửa chuỗi
sai, BỎ chuỗi không có trong ảnh, và bổ sung chuỗi mà máy đã bỏ sót.\
"""

# Nối vào cuối PROMPT_KEYFRAME_CARD khi scene chứa keyframe có ASR.
# 92,0% scene của pack có ASR. Giá trị của nó là GỌI ĐÚNG TÊN: tên loài, tên
# địa danh, tên người, số liệu — đúng những thứ mà câu hỏi thi hay dùng
# (P1-17 tên đèo, P2-9 "cá nhám 211kg", P2-24 loài cá nguy cấp). Nhưng nó cũng
# là nguồn bịa lớn nhất: lời dẫn nói về thứ không có trong khung hình. Vì vậy
# ASR bị nhốt vào đúng một dòng NGHE và bị cấm rò sang các dòng thị giác.
ASR_CONTEXT_SUFFIX = """

LỜI THUYẾT MINH nghe được quanh khoảnh khắc này (KHÔNG phải thứ nhìn thấy):
{asr}

Lời thuyết minh chỉ được dùng cho ĐÚNG MỘT việc: gọi đúng tên riêng, tên loài,
tên địa danh, chức danh và số liệu. Tuyệt đối không đưa nội dung chỉ nghe thấy
vào các dòng MOTA, NGUOI, VATTHE, CHITIET_NHO — bốn dòng đó chỉ tả thứ NHÌN
THẤY trong ảnh. Thông tin lấy từ lời nói ghi vào dòng NGHE, và chỉ ghi phần
liên quan tới thứ đang hiện trong khung hình.\
"""

# Thể loại suy được từ mã nhóm video, KHÔNG cần hỏi model. Đo trên pack: model
# mặc định trả "ban_tin" cho cả cảnh hái nho lẫn cảnh đua xe, nên hỏi nó là tự
# rước nhiễu vào một trường mà metadata đã biết chắc.
GROUP_GENRE = {
    "L21": "bản tin thời sự 60 giây của HTV9",
    "L22": "bản tin thời sự của HTV7",
    "L23": "tường thuật giải đua xe đạp Cúp Truyền hình",
    "L24": "tường thuật giải lân sư rồng",
    "L25": "bài giảng ôn thi THPT, phần lớn khung hình là slide",
    "L26": "chương trình dạy nấu ăn Món Ngon Mỗi Ngày",
    "L27": "phóng sự du lịch Đi là ghiền",
    "L28": "phóng sự Tản Mạn Mekong",
    "L29": "phóng sự Đôi Mắt Mekong",
    "L30": "phóng sự truyền hình Tuổi Trẻ TV",
}

GENRE_SUFFIX = """

THỂ LOẠI CHƯƠNG TRÌNH (đã biết chắc, không cần đoán): {genre}
Dùng nó để hiểu bối cảnh, nhưng đừng chép câu này vào bất kỳ dòng nào."""


# --------------------------------------------------------------------------
# T1-SPLIT — cùng 10 trường, nhưng chia làm HAI lời gọi.
#
# Lý do tồn tại, đo trên FPT/Qwen2.5-VL-7B ngày 2026-08-25: hợp đồng 10 trường
# trong MỘT lời gọi làm model tràn trường này sang trường kia. Bằng chứng cụ
# thể của bản v2.3 một-lời-gọi:
#   - CHU trả "ĐỊCH @tu_tren_cao @slide_bai_giang" — nó bốc giá trị enum của
#     dòng KHUNGHINH làm vị trí và loại.
#   - KHUNGHINH ghi tu_tren_cao trong khi chính MOTA của nó nói "máy đặt thấp
#     nhìn lên", dù prompt có luật bắt hai dòng phải khớp.
#   - DEM vẫn đẻ ra "người=none; vật thể=none" dù bị cấm thẳng.
# Mỗi luật thêm vào chữa được một chỗ và làm hỏng một chỗ khác. Đó là dấu hiệu
# hết chỗ chú ý, không phải dấu hiệu prompt viết chưa đủ rõ.
# --------------------------------------------------------------------------

PROMPT_SPLIT_A = """Mô tả MỘT khung hình video cho hệ thống tìm kiếm. Chỉ tả thứ NHÌN THẤY.

Trả về ĐÚNG 4 dòng, đúng thứ tự, mỗi nhãn một dòng. Không Markdown, không lời
dẫn, không giải thích. Dòng trống thì ghi: none

MOTA: 2-4 câu tiếng Việt, tối thiểu 40 từ. Mở đầu bằng cỡ cảnh và góc máy.
      Nêu rõ ai hoặc vật gì ở bên trái, ở giữa, bên phải. Nêu màu sắc chính
      và hành động đang diễn ra.
NGUOI: tối đa 5 người nổi bật, ngăn cách bằng " || ". Mỗi người: vị trí trong
      khung, giới và tuổi ước lượng, trang phục kèm màu, phụ kiện, tư thế.
VATTHE: 5-15 vật thể nhìn rõ, ngăn cách bằng dấu phẩy, MỖI vật kèm màu. Chỉ
      tiếng Việt. Không chắc là món gì hay loài gì thì tả hình dạng, đừng đoán tên.
KHUNGHINH: đúng 3 giá trị ngăn cách bằng dấu phẩy.
      Giá trị 1 chọn một trong: dai_canh toan trung can dac_ta
      Giá trị 2 chọn một trong: ngang_tam_mat tu_tren_cao tu_duoi_len flycam nghieng
      Giá trị 3 chọn một trong: phong_van canh_quay_thuc_te do_hoa slide_bai_giang camera_an_ninh anh_tinh
      Giá trị 2 phải khớp với góc máy bạn vừa viết ở MOTA.

Không suy đoán nguyên nhân, cảm xúc, danh tính hay địa điểm."""

PROMPT_SPLIT_B = """Bạn nhìn MỘT khung hình video. Nhiệm vụ KHÔNG phải mô tả toàn cảnh, mà là ghi
lại đúng bốn thứ giúp phân biệt khung hình này với hàng nghìn khung hình khác:
số lượng, chi tiết nhỏ, chữ, và tên riêng.

Trả về ĐÚNG 5 dòng, đúng thứ tự, mỗi nhãn một dòng. Không Markdown, không lời
dẫn. Dòng nào không có gì thì ghi đúng một chữ: none

DEM: đếm các nhóm đáng nhớ, dạng "tên nhóm=số", ngăn cách bằng dấu chấm phẩy,
     tối đa 5 nhóm. Tên nhóm tiếng Việt có dấu. Chỉ ghi số khi đếm được từng
     cái; đông hơn 10 thì ghi "=nhiều". Nhóm không có trong ảnh thì bỏ hẳn,
     không viết "=none".
CHITIET_NHO: vật nhỏ dễ bỏ qua nhưng dễ nhớ — dây buộc, nhãn dán, thẻ tên, hoa
     văn, huy hiệu, hoạ tiết in trên áo, vết bẩn. Mỗi thứ kèm màu và vị trí.
     Nhìn kỹ các góc và các vật nhỏ trước khi trả lời.
CHU: mọi chuỗi chữ hoặc số đọc được, chép nguyên văn từng ký tự, giữ dấu tiếng
     Việt và chữ nước ngoài, các chuỗi ngăn cách bằng dấu chấm phẩy. Đọc lờ mờ
     thì vẫn ghi kèm dấu ? ở cuối. Bỏ qua tên kênh, đồng hồ giờ và dòng chữ
     chạy ở đáy màn hình.
NGHE: chỉ điền nếu có phần LỜI THUYẾT MINH ở cuối prompt. Ghi tên riêng, tên
     loài, địa danh, chức danh, số liệu kèm đơn vị — chỉ những cái liên quan
     tới thứ đang thấy trong ảnh. Không chép lại cả câu.
DACTRUNG: đúng một câu, tối đa 25 từ: chi tiết dị biệt nhất của khung hình
     này, bắt buộc kèm màu hoặc số lượng.

Đếm sai tệ hơn không đếm. Bịa chữ tệ hơn để trống."""

FIELDS_SPLIT_A = ("MOTA", "NGUOI", "VATTHE", "KHUNGHINH")
FIELDS_SPLIT_B = ("DEM", "CHITIET_NHO", "CHU", "NGHE", "DACTRUNG")


# --------------------------------------------------------------------------
# T1-SLIDE — biến thể cho slide bài giảng / đồ hoạ / infographic.
# Dùng cho nhóm L25 (88 video, 37.083 keyframe, OCR mới phủ 16,3%) và cho mọi
# khung hình mà KHUNGHINH ở pass trước rơi vào slide_bai_giang|do_hoa.
# Đích nhắm: P1-q23 (sơ đồ 3 tầng), P2-q14 (lưới icon 5x10), P2-q21 (bài tập).
# --------------------------------------------------------------------------

PROMPT_SLIDE_CARD = """\
Bạn đang số hoá MỘT khung hình là slide bài giảng, bảng biểu hoặc đồ hoạ
thông tin. Người tìm kiếm sẽ hỏi lại bằng nội dung chữ và bằng CẤU TRÚC hình
vẽ trên slide, nên cả hai đều phải được ghi lại.

Trả về ĐÚNG 8 dòng theo nhãn dưới đây, đúng thứ tự, mỗi nhãn một dòng.
Không Markdown. Không lời dẫn. Mục nào không có thì viết: none

TOANVAN: <chép TOÀN BỘ chữ trên slide theo đúng thứ tự đọc từ trên xuống, từ trái sang phải. Giữ nguyên văn từng ký tự và dấu tiếng Việt. Ngăn cách các dòng bằng " / ". Không tóm tắt, không diễn giải.>
TIEUDE: <tiêu đề slide, nguyên văn>
SODO: <nếu có sơ đồ hoặc lưu đồ: số tầng, số khối mỗi tầng, MÀU từng khối, hướng và màu mũi tên, khối nào nối với khối nào. Ví dụ: 3 tầng; tầng 1 có 2 khối trắng bao trong 1 khung cam; tầng 2 có 1 khối xanh dương đậm ở giữa; tầng 3 có 2 khối bao trong khung xanh lá; mũi tên xanh ngọc trỏ xuống nối 3 tầng>
BANG: <nếu có bảng: mỗi hàng một cụm, các ô ngăn cách bằng " | ", các hàng ngăn cách bằng " ; ">
BIEUDO: <nếu có biểu đồ hoặc lưới biểu tượng: loại biểu đồ, trục, đơn vị, và CÁCH SẮP XẾP kèm SỐ LƯỢNG. Ví dụ: lưới icon hình não xếp 5 hàng x 10 cột, hàng cuối dư thêm 2 icon>
SOLIEU: <mọi con số, đơn vị, phần trăm, năm, tiền tệ xuất hiện trên slide, nguyên văn, ngăn cách bằng dấu phẩy>
HINHANH: <mô tả người hoặc ảnh minh hoạ trên slide: vị trí, trang phục và màu, phông nền>
CHUDE: <môn học hoặc chủ đề, chỉ khi ghi rõ trên slide>

Quy tắc bắt buộc:
- TOANVAN là mục quan trọng nhất. Thiếu chữ còn tệ hơn thừa dòng.
- Không dịch, không sửa chính tả, không chuẩn hoá số. Chép đúng như hiện trên màn hình.
- Không suy luận nội dung bài học ngoài phần chữ nhìn thấy.\
"""

# --------------------------------------------------------------------------
# T2 — thẻ cửa sổ cảnh. Đầu vào là 3 keyframe LIÊN TIẾP (stride 2).
# Đây là tầng DUY NHẤT sinh ra được chuyển động và diễn biến; caption một
# khung hình không bao giờ có. 47,5% scene chỉ có 1 keyframe nên cửa sổ phải
# trượt theo chuỗi keyframe của cả video, không cắt theo ranh giới scene.
# --------------------------------------------------------------------------

PROMPT_SHOT_WINDOW = """\
Bạn nhận {n} khung hình LIÊN TIẾP của cùng một video, xếp theo thứ tự thời
gian (khung 1 sớm nhất). Nhiệm vụ là mô tả đoạn này như một CẢNH ĐỘNG, nhấn
vào cái THAY ĐỔI giữa các khung — đó chính là thứ mà một khung hình đơn lẻ
không nói được.

Trả về ĐÚNG 7 dòng theo nhãn dưới đây, đúng thứ tự, mỗi nhãn một dòng.
Không Markdown. Không lời dẫn. Mục nào không có thì viết: none

TOMTAT: <1-2 câu tiếng Việt cho cả đoạn: ai làm gì ở đâu>
DIENBIEN: <thay đổi theo thứ tự, dạng "1->2: ...; 2->3: ...". Chỉ ghi thay đổi THẬT SỰ quan sát được. Nếu ba khung gần như giống nhau, ghi: gần như không đổi>
HANHDONG: <chuỗi cụm động từ theo đúng thứ tự xảy ra, ngăn cách bằng " > ". Ví dụ: cầm kéo > cắt cuống nho > nhấc chùm nho khỏi giàn>
CHUYENDONG: <chủ thể di chuyển ra sao trong khung: trái sang phải, phải sang trái, tiến về phía máy quay, lùi xa, đứng yên>
MAYQUAY: <chọn một hoặc nhiều, ngăn cách bằng dấu phẩy: tinh, lia_trai_phai, lia_phai_trai, lia_doc, zoom_vao, zoom_ra, may_di_theo, cam_tay_rung, quay_cham, tua_nhanh, cat_canh_giua_chung>
CHU_THAYDOI: <chữ mới xuất hiện hoặc biến mất giữa các khung, chép nguyên văn. Bỏ qua logo đài và đồng hồ.>
DACTRUNG: <MỘT câu: điều khiến đoạn này khác mọi đoạn tương tự trong cùng thể loại>

Quy tắc bắt buộc:
- Chỉ nói về những gì thấy trong {n} khung này. Không kể tiếp câu chuyện.
- MAYQUAY suy từ sự dịch chuyển của TOÀN khung hình giữa các khung, không phải từ chủ thể. Nếu nền tĩnh mà chủ thể động thì máy quay là "tinh".
- Nếu ba khung thuộc hai cảnh khác hẳn nhau, ghi rõ ở DIENBIEN là "cắt sang cảnh khác" và mô tả cả hai.\
"""

# --------------------------------------------------------------------------
# T3 — tài liệu tra cứu cấp VIDEO. TEXT-ONLY, không cần GPU thị giác.
# Chạy được ngay hôm nay trên caption đã có trong pack.
# Đây là tầng đánh trúng 65,5% câu hỏi dạng "mở đầu... sau đó... kết thúc".
# --------------------------------------------------------------------------

PROMPT_VIDEO_ROLLUP = """\
Bạn nhận DANH SÁCH các cảnh của MỘT video, theo thứ tự thời gian, mỗi cảnh
gồm mốc thời gian và phần mô tả đã có sẵn.

Nhiệm vụ: viết tài liệu tra cứu cho video này. Người tìm kiếm hầu như luôn mô
tả video bằng CHUỖI cảnh nối tiếp — "mở đầu là ..., sau đó ..., kết thúc bằng
..." — nên tài liệu phải chứa sẵn những chuỗi viết theo đúng văn phong ấy.

Trả về ĐÚNG 6 dòng theo nhãn dưới đây, đúng thứ tự.
Không Markdown. Không lời dẫn. Mục nào không có thì viết: none

TOMTAT_VIDEO: <3-5 câu: video nói về chuyện gì, quay ở đâu, có những ai, thuộc thể loại nào>
CHUDE: <3-8 từ khoá chủ đề, ngăn cách bằng dấu phẩy>
DANHSACH_CANH: <mỗi cảnh một cụm, ngăn cách bằng " ;; ", theo mẫu: mm:ss-mm:ss | một câu mô tả | chữ nổi bật nếu có>
CHUOI_SUKIEN: <3-8 cụm, ngăn cách bằng " ;; ". Mỗi cụm gộp 2-4 cảnh LIỀN NHAU thành một câu kể theo đúng văn phong: "Mở đầu ..., sau đó ..., kết thúc bằng ...". Chỉ gộp các cảnh thật sự liền nhau về thời gian và nội dung.>
THUCTHE: <tên người, tổ chức, địa danh, tên món ăn, tên loài, tên sản phẩm, số liệu kèm đơn vị — chỉ lấy từ dữ liệu đầu vào>
CHU_QUANTRONG: <các chuỗi chữ đáng nhớ nhất của cả video, nguyên văn, ngăn cách bằng dấu phẩy>

Quy tắc bắt buộc:
- Không bịa thêm bất kỳ chi tiết nào không có trong danh sách đầu vào.
- CHUOI_SUKIEN phải dùng đúng các từ nối "mở đầu", "sau đó", "tiếp theo", "kết thúc bằng" — đó là văn phong của câu hỏi sẽ tới.
- Giữ nguyên văn mọi chuỗi chữ; không dịch, không chuẩn hoá.\
"""


# --------------------------------------------------------------------------
# Parser + ghi ra JSONL
# --------------------------------------------------------------------------

FIELDS_KEYFRAME = (
    "MOTA", "DEM", "NGUOI", "VATTHE", "CHITIET_NHO",
    "CHU", "NGHE", "KHUNGHINH", "DACTRUNG", "TUKHOA",
)
FIELDS_SLIDE = (
    "TOANVAN", "TIEUDE", "SODO", "BANG", "BIEUDO", "SOLIEU", "HINHANH", "CHUDE",
)
FIELDS_SHOT = (
    "TOMTAT", "DIENBIEN", "HANHDONG", "CHUYENDONG",
    "MAYQUAY", "CHU_THAYDOI", "DACTRUNG",
)
FIELDS_ROLLUP = (
    "TOMTAT_VIDEO", "CHUDE", "DANHSACH_CANH",
    "CHUOI_SUKIEN", "THUCTHE", "CHU_QUANTRONG",
)

PROMPT_VERSION = "caption_card_v2_3"


def parse_card(raw: str, fields: tuple[str, ...]) -> dict[str, str]:
    """Tách output dạng "NHÃN: giá trị" thành dict.

    Model 4B thỉnh thoảng trả thừa dòng, thiếu nhãn, hoặc xuống dòng giữa
    chừng một giá trị. Hàm này gom mọi dòng không mang nhãn vào nhãn ĐANG mở,
    nên một lần xuống dòng không làm mất cả trường. Nhãn thiếu -> chuỗi rỗng,
    và người gọi đếm được bao nhiêu trường rỗng để quyết định chạy lại.
    """

    out: dict[str, list[str]] = {f: [] for f in fields}
    current: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        head, sep, tail = line.partition(":")
        key = head.strip().upper()
        if sep and key in out:
            current = key
            if tail.strip():
                out[current].append(tail.strip())
            continue
        if current is not None:
            out[current].append(line)
    clean: dict[str, str] = {}
    for field, parts in out.items():
        value = " ".join(parts).strip()
        clean[field] = "" if value.lower() in {"none", "n/a", "-", ""} else value
    return clean


LIST_FIELDS = ("DEM", "VATTHE", "CHITIET_NHO", "TUKHOA", "CHU")


def dedupe_list_fields(card: dict[str, str]) -> dict[str, str]:
    """Bỏ phần tử trùng trong các trường dạng danh sách.

    Model 7B thỉnh thoảng rơi vào vòng lặp: "số màu xanh lá cây, số màu xanh
    dương, số màu xanh lá cây, số màu xanh dương..." lặp bốn lần trong cùng
    một dòng. Nhắc trong prompt không chặn được; lọc sau là chắc chắn và
    không tốn thêm lời gọi nào.
    """

    for field in LIST_FIELDS:
        value = card.get(field)
        if not value:
            continue
        sep = ";" if field == "DEM" else ","
        seen: list[str] = []
        for item in value.split(sep):
            item = item.strip()
            if item and item.lower() not in {s.lower() for s in seen}:
                seen.append(item)
        card[field] = f"{sep} ".join(seen)
    return card


def missing_fields(card: dict[str, str], required: tuple[str, ...]) -> list[str]:
    """Trường bắt buộc còn rỗng — dùng làm cổng chạy lại trước khi ghi."""

    return [f for f in required if not card.get(f)]


def _provenance(model_name: str, prompt_version: str = PROMPT_VERSION) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": "kaggle-t4x2",
        "model_name": model_name,
        "model_revision": prompt_version,
        "parameters": {},
        "pipeline_version": "aic-v1.0.0",
        "prompt_version": prompt_version,
    }


def keyframe_caption_record(text: str, model_name: str) -> dict:
    """Caption ở tầng KEYFRAME.

    Schema khác tầng scene và đã từng làm hỏng cả một đợt nạp: `caption_type`
    là enum `short|detailed|tags|crop` (KHÔNG có "visual"), có `crop_bbox`, và
    KHÔNG nhận `evidence_keyframe_ids`. Xem EVAL-MULTIVIDEO-01 trong docs/20.
    """

    return {
        "caption_type": "detailed",
        "confidence": None,
        "crop_bbox": None,
        "language": "vi",
        "text": text,
        "provenance": _provenance(model_name),
    }


def scene_caption_record(text: str, model_name: str, evidence: list[str] | None = None) -> dict:
    """Caption ở tầng SCENE — ngược lại: nhận "visual" và evidence_keyframe_ids."""

    return {
        "caption_type": "visual",
        "confidence": None,
        "language": "vi",
        "text": text,
        "evidence_keyframe_ids": list(evidence or []),
        "provenance": _provenance(model_name),
    }


# Trọng số khi ghép các trường thành MỘT chuỗi cho BM25. Chỉ dùng khi không
# thể index theo trường riêng: caption đi từ 35 lên ~200 từ nên nếu nhét tất
# cả vào một field thì `b` của BM25 phải chỉnh lại, và các trường loãng
# (MOTA, BOICANH) sẽ dìm các trường sắc (CHU, DACTRUNG, TUKHOA).
BM25_FIELD_REPEAT = {
    "DACTRUNG": 3,
    "TUKHOA": 3,
    "CHITIET_NHO": 2,
    "CHU": 2,
    "NGHE": 2,
    "DEM": 2,
    "NGUOI": 1,
    "VATTHE": 1,
    "MOTA": 1,
    "KHUNGHINH": 1,
}


def flatten_for_bm25(card: dict[str, str]) -> str:
    """Ghép thẻ thành một chuỗi, lặp lại các trường sắc nét.

    Đây là giải pháp tạm. Cách đúng là index theo trường riêng và chỉnh weight
    ở tầng fusion — xem docs/41 mục 5.
    """

    parts: list[str] = []
    for field, repeat in BM25_FIELD_REPEAT.items():
        value = card.get(field)
        if value:
            parts.extend([value] * repeat)
    return " ".join(parts)


__all__ = [
    "PROMPT_KEYFRAME_CARD", "PROMPT_SLIDE_CARD", "PROMPT_SHOT_WINDOW",
    "PROMPT_VIDEO_ROLLUP", "OCR_HINT_SUFFIX", "ASR_CONTEXT_SUFFIX",
    "GENRE_SUFFIX", "GROUP_GENRE", "PROMPT_VERSION",
    "PROMPT_SPLIT_A", "PROMPT_SPLIT_B", "FIELDS_SPLIT_A", "FIELDS_SPLIT_B",
    "FIELDS_KEYFRAME", "FIELDS_SLIDE", "FIELDS_SHOT", "FIELDS_ROLLUP",
    "parse_card", "missing_fields", "dedupe_list_fields", "flatten_for_bm25",
    "keyframe_caption_record", "scene_caption_record",
]
