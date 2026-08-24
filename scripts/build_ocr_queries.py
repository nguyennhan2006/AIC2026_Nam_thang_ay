# -*- coding: utf-8 -*-
"""Sinh lai examples/AIC2026_ocr_queries_v1.jsonl — bo gold OCR-first.

Moi frame trong file nay lay tu mot chuoi OCR co that trong
`storage/ocr/ocr_sidecar_merged.jsonl`, va duoc DOI CHIEU lai voi
`storage/exports_competition/keyframes.jsonl` truoc khi ghi: frame khong phai
keyframe thi nop bai duoc 0 diem du dung khoanh khac (xem docs/39).

    python -m scripts.build_ocr_queries

Xem docs/39_OCR_QUERY_EXPLOITATION.md de biet vi sao chon tung truy van.
"""
import json, re, sys, collections

KEYFRAMES = "storage/exports_competition/keyframes.jsonl"

# video_id -> {frame_idx (str) -> [timestamp_sec, scene_id]}
KF = collections.defaultdict(dict)
_VID_RE = re.compile(r'"video_id":\s*"([^"]+)"')
for _line in open(KEYFRAMES, encoding="utf-8"):
    _m = _VID_RE.search(_line[:200])
    if not _m:
        continue
    _d = json.loads(_line)
    KF[_d["video_id"]][str(_d["frame_idx"])] = [_d.get("timestamp_sec"),
                                                _d.get("scene_id")]


def ts(v, f):
    rec = KF.get(v, {}).get(str(f))
    return None if rec is None else rec[0]


def iv(v, a, b):
    return {"start_frame": a, "end_frame": b,
            "start_sec": round(ts(v, a), 3), "end_sec": round(ts(v, b), 3)}


Q = []


def kis(qid, diff, video, frames, rep, vi, en, dense, sparse, hardneg, tags, ev):
    Q.append({
        "query_id": qid, "task": "KIS", "difficulty": diff,
        "query_vi": vi, "query_en": en, "dense_query_en": dense,
        "sparse_terms": sparse,
        "target_video": video,
        "target_intervals": [iv(video, frames[0], frames[-1])],
        "representative_frame": rep,
        "hard_negative": hardneg,
        "required_modalities": ["OCR", "visual"],
        "diagnostic_tags": tags,
        "ocr_evidence": ev,
        "ocr_evidence_frames": frames,
        "quality_note": ("GT neo theo chuoi OCR co that trong ocr_sidecar_merged.jsonl; "
                         "frame da doi chieu keyframes.jsonl. CHUA soi anh bang mat."),
    })


def vqa(qid, diff, video, frames, rep, event, qvi, qen, ans, accepted, rtype,
        hardneg, tags, ev):
    Q.append({
        "query_id": qid, "task": "VQA", "difficulty": diff,
        "event_description_vi": event, "question_vi": qvi, "question_en": qen,
        "answer_canonical": ans, "accepted_answers": accepted,
        "target_video": video,
        "target_intervals": [iv(video, frames[0], frames[-1])],
        "representative_frame": rep,
        "reasoning_type": rtype,
        "required_modalities": ["OCR"],
        "hard_negative": hardneg,
        "diagnostic_tags": tags,
        "ocr_evidence": ev,
        "ocr_evidence_frames": frames,
        "quality_note": "Dap an doc truc tiep tu chuoi OCR; khong dung ASR.",
    })


def trake(qid, diff, video, vi, events, tags, note):
    Q.append({
        "query_id": qid, "task": "TRAKE", "difficulty": diff,
        "query_vi": vi, "target_video": video, "event_count": len(events),
        "events": [{"event_order": i + 1, "description_vi": d,
                    "representative_frame": f,
                    "gt_start_frame": max(0, f - 4), "gt_end_frame": f + 4,
                    "semantic_time_sec": round(ts(video, f), 3),
                    "ocr_evidence": ev}
                   for i, (d, f, ev) in enumerate(events)],
        "required_modalities": ["OCR", "temporal"],
        "diagnostic_tags": tags, "quality_note": note,
    })


def avs(qid, diff, video, vi, criteria, intervals, tags, note):
    Q.append({
        "query_id": qid, "task": "AVS", "difficulty": diff,
        "query_vi": vi, "criteria": criteria, "target_video": video,
        "relevant_intervals": [
            {"event_id": e, "start_frame": a, "end_frame": b,
             "start_sec": round(ts(video, a), 3), "end_sec": round(ts(video, b), 3),
             "relevance_grade": g, "reason": r, "ocr_evidence": ev}
            for e, a, b, g, r, ev in intervals],
        "required_modalities": ["OCR"],
        "diagnostic_tags": tags, "quality_note": note,
    })


# ---------------- KIS: chu lop phu (chyron/lower-third) ----------------
kis("OCR_KIS_01", "Easy", "L21_V001", [3932, 3932], 3932,
    'Tìm khoảnh khắc bản tin chạy dòng tiêu đề "CẦN THƠ THIỆT HẠI HƠN 14,5 TỶ ĐỒNG DO SẠT LỞ BỜ SÔNG".',
    'Find the moment where the news chyron reads "Can Tho suffered over 14.5 billion VND in riverbank erosion damage".',
    "news lower-third headline about riverbank erosion damage in Can Tho",
    "cần thơ thiệt hại 14,5 tỷ đồng sạt lở bờ sông",
    "Cùng video còn 2 chyron sạt lở khác (frame 2036, 2130) không có con số tiền.",
    ["chyron exact phrase", "numeric in text", "overlay text"],
    ["CẦN THƠ THIỆT HẠI HƠN 14,5 TỶ ĐỒNG DO SẠT LỞ BỜ SÔNG"])

kis("OCR_KIS_02", "Medium", "L21_V001", [2130, 2217], 2130,
    'Tìm cảnh có tấm biển đặt ngoài hiện trường ghi "CẢNH BÁO SẠT LỞ NGUY HIỂM — TẠM DỪNG LƯU THÔNG ĐỐI VỚI XE 3 BÁNH TRỞ LÊN".',
    'Find the scene with an on-site warning board reading "DANGEROUS LANDSLIDE WARNING - TRAFFIC SUSPENDED FOR 3-WHEELED VEHICLES AND LARGER".',
    "a red-and-yellow roadside warning board beside a collapsed riverbank",
    "cảnh báo sạt lở nguy hiểm tạm dừng lưu thông xe 3 bánh",
    "Chyron 'SỤT LÚN Ở ĐBSCL...' (frame 2036) ngay trước đó là CHỮ LỚP PHỦ, không phải biển trong cảnh.",
    ["scene text vs overlay text", "multi-line sign", "hard negative cùng chủ đề"],
    ["CẢNH BÁO", "SẠT LỞ NGUY HIỂM", "TẠM DỪNG LƯU THÔNG",
     "ĐỐI VỚI XE 3 BÁNH TRỞ LÊN", "NGƯỜI DÂN ĐI LẠI CHÚ Ý QUAN SÁT"])

# ---------------- KIS: the nguyen lieu / meo bep (L26 = 57% pack) ----------------
kis("OCR_KIS_03", "Easy", "L26_V056", [7748, 7936], 7748,
    'Tìm thẻ nguyên liệu ghi "ĐẬU HỦ TRẮNG 300g", "CỦ HỦ DỪA 100g", "RAU RỪNG 100g".',
    'Find the ingredient card listing "white tofu 300g", "coconut heart 100g", "wild greens 100g".',
    "cooking show ingredient list card with tofu and coconut heart",
    "đậu hủ trắng 300g củ hủ dừa 100g rau rừng 100g",
    "Thẻ nguyên liệu đầu video (frame 768) cùng món nhưng chữ thường, ít dòng hơn.",
    ["ingredient + quantity", "L26 dominant group", "near-duplicate trong cùng video"],
    ["NGUYÊN LIỆU", "ĐẬU HỦ TRẮNG 300g", "CỦ HỦ DƯA 100g", "RAU RỪNG 100g"])

kis("OCR_KIS_04", "Medium", "L26_V037", [4107, 4107], 4107,
    'Tìm thẻ MẸO ghi "CHIÊN DA HEO VỚI LỬA NHỎ VÀ THÊM ÍT BỘT NĂNG ĐỂ HẠN CHẾ VĂNG DẦU".',
    'Find the TIP card reading "fry pork skin on low heat and add a little tapioca starch to reduce oil splatter".',
    "cooking tip overlay card about frying pork skin on low heat",
    "mẹo chiên da heo lửa nhỏ bột năng hạn chế văng dầu",
    "Cùng video có 2 thẻ MẸO khác về da heo (frame 2493 'LUỘC DA HEO VỚI ÍT MUỐI', frame 4493 'LUÔN GIỮ DA HEO NGẬP TRONG DẦU').",
    ["tip card", "3 hard negative cùng chủ ngữ", "phân biệt cụm dài"],
    ["MẸO", "CHIÊN DA HEO VỚI LỬA NHỎ", "VÀ THÊM ÍT BỘT NĂNG", "ĐỂ HẠN CHẾ VẮNG DẦU"])

kis("OCR_KIS_05", "Medium", "L26_V343", [1536, 1536], 1536,
    'Tìm thẻ nguyên liệu món dê có "Thịt đùi dê: 400g" và "Cà bát: 2 quả".',
    'Find the ingredient card with "goat leg meat: 400g" and "eggplant: 2 fruits".',
    "ingredient card for a goat meat dish with eggplant",
    "thịt đùi dê 400g cà bát 2 quả",
    "Nhiều video L26 khác cũng có thẻ 'NGUYÊN LIỆU' — chỉ định lượng + tên nguyên liệu mới phân biệt được.",
    ["ingredient + quantity", "cross-video boilerplate trap"],
    ["Thịt đùi dê: 400g", "Cà bát: 2 quả", "NGUYÊN LIỆU"])

kis("OCR_KIS_06", "Hard", "L26_V033", [7748, 7830], 7748,
    'Tìm bảng "THỰC ĐƠN HỖ TRỢ THẬN KHỎE MẠNH" do TS.BS Đào Thị Yến Phi tư vấn.',
    'Find the "menu supporting kidney health" board advised by nutrition expert Dr Dao Thi Yen Phi.',
    "nutrition menu board advised by a dietitian",
    "thực đơn hỗ trợ thận khỏe mạnh Đào Thị Yến Phi chuyên gia dinh dưỡng",
    "Khung 'THỰC ĐƠN DINH DƯỠNG / ĐƯỢC TƯ VẤN BỞI CHUYÊN GIA DINH DƯỠNG / TS.BS Đào Thị Yến Phi' là boilerplate ở rất nhiều video L26; chỉ dòng chủ đề thực đơn là phân biệt.",
    ["boilerplate vs discriminative", "named entity", "L26 cross-video confusion"],
    ["THỰC ĐƠN HỖ TRỢ THẬN KHỎE MẠNH", "TS.BS Đào Thị Yến Phi - Chuyên gia dinh dưỡng"])

# ---------------- KIS: bien hieu / chu trong canh ----------------
kis("OCR_KIS_07", "Medium", "L28_V002", [23355, 26189], 24283,
    'Tìm biển hiệu "CƠ SỞ DỆT THỔ CẨM TRUYỀN THỐNG LÀNG CHĂM CHÂU PHONG".',
    'Find the shop sign "traditional Cham brocade weaving workshop, Chau Phong village".',
    "a shop signboard of a traditional Cham brocade weaving workshop",
    "cơ sở dệt thổ cẩm truyền thống làng Chăm Châu Phong",
    "OCR đọc sai thành 'LÀNG CHÂM CHÂU PHONG' (mất dấu ă). NHÓM ĐỐI CHỨNG: cụm còn 6 token đúng nên bm25_ocr được kỳ vọng VẪN giải được — dùng để phân biệt 'fuzzy có ích' với 'fuzzy thừa'.",
    ["diacritic error", "control group cho ocr_fuzzy", "scene text"],
    ["CƠ SỞ DỆT THỔ CẨM TRUYỀN THỐNG LÀNG CHÂM CHÂU PHONG"])

kis("OCR_KIS_08", "Hard", "L29_V005", [9990, 14472], 11880,
    'Tìm băng-rôn "CHÀO MỪNG LỄ HỘI NGHINH ÔNG SÔNG ĐỐC NĂM 2024".',
    'Find the banner "welcome to the Nghinh Ong Song Doc festival 2024".',
    "a festival welcome banner at a coastal town ceremony",
    "chào mừng lễ hội nghinh ông sông đốc năm 2024",
    "OCR ghi 'SÔNG ĐỌC' (sai dấu) ở mọi frame; cùng video còn 'LỄ HỘI NGHINH ÔNG SÔNG ĐỌC' (bảng di sản) không phải băng-rôn chào mừng. NHÓM ĐỐI CHỨNG: cụm dài nên bm25_ocr được kỳ vọng vẫn giải được.",
    ["diacritic error", "control group cho ocr_fuzzy", "banner"],
    ["CHÀO MỪNG LỄ HỘI NGHINH ỐNG SÔNG ĐỌC NĂM 2024", "LỄ HỘI NGHINH ÔNG SÔNG ĐỌC"])

kis("OCR_KIS_09", "Medium", "L27_V007", [7559, 7568], 7559,
    'Tìm thẻ địa danh ghi "Thạch Động Thôn Vân, phường Mỹ Đức, TP Hà Tiên, tỉnh Kiên Giang".',
    'Find the location caption "Thach Dong Thon Van, My Duc ward, Ha Tien city, Kien Giang province".',
    "a location caption card over a limestone cave landscape",
    "Thạch Động Thôn Vân phường Mỹ Đức Hà Tiên Kiên Giang",
    "Frame 8505 chỉ hiện 'Thạch Động' cụt, không đủ chuỗi địa chỉ đầy đủ.",
    ["địa danh hành chính", "chuỗi dài", "partial vs full match"],
    ["Thạch Động Thôn Vân, phường Mỹ Đức, TP Hà Tiên, tỉnh Kiên Giang"])

kis("OCR_KIS_10", "Hard", "L21_V009", [1853, 1853], 1853,
    'Tìm khung liên hệ ghi "ĐT: 0888 090 711 | E: info@btba.vn | W: www.btba.vn" của Hội Doanh nghiệp Bình Thạnh.',
    'Find the contact card with phone 0888 090 711, email info@btba.vn and site www.btba.vn.',
    "a contact information overlay with phone, email and website",
    "0888 090 711 info@btba.vn www.btba.vn hội doanh nghiệp bình thạnh",
    "Nhiều video khác cũng có 'ĐT:' + số — chỉ đúng cụm số/email mới phân biệt.",
    ["numeric/URL token", "tokenizer stress test", "contact overlay"],
    ["ĐT: 0888 090 711 | E: info@btba.vn | W: www.btba.vn", "HỘI DOANH NGHIỆP", "BÌNH THẠNH"])

kis("OCR_KIS_11", "Hard", "L21_V013", [3954, 4551], 3954,
    'Tìm chiếc xe của Công ty Công viên Cây xanh mang biển số 51E-025.18.',
    'Find the parks-and-greenery company truck with licence plate 51E-025.18.',
    "a municipal green-space company truck parked on a street",
    "51E-025.18 công ty công viên cây xanh",
    "OCR đọc biển thành 4 biến thể qua các frame: '51E-025.18', '51C 02518', '51E 02518', và '51E-02518' ở L21_V012 frame 638.",
    ["licence plate", "OCR variant explosion", "cross-video near-duplicate"],
    ["51E-025.18", "CÔNG TY CÔNG VIÊN CÂY XANH"])

kis("OCR_KIS_12", "Hard", "L21_V021", [13511, 13961], 13811,
    'Tìm ảnh chụp bài đăng Facebook tìm trẻ lạc có số điện thoại mẹ 0394164789.',
    'Find the screenshot of a Facebook missing-child post showing the mother phone number 0394164789.',
    "a screenshot of a social media post about a lost child",
    "0394164789 tìm trẻ lạc SĐT mẹ Nguyễn Thị Phương",
    "Cùng video còn số 0965938187 trong một bài đăng khác cũng về trẻ đi lạc (frame 14411).",
    ["screenshot text", "phone number", "two similar posts"],
    ["0394164789", "SĐT mẹ 0394164789", "KHẨN CẤP NHỚ CHIA SẺ (tim trẻ lạc)"])

kis("OCR_KIS_13", "Medium", "L30_V014", [413, 413], 413,
    "Tìm khung tiêu đề phóng sự \"Tiệm sách miễn phí '3 không' ở Sài Gòn\".",
    "Find the story title card \"the free '3 no' bookstore in Saigon\".",
    "a title card about a free bookstore in Saigon",
    "tiệm sách miễn phí 3 không Sài Gòn",
    "Video chỉ có 31 keyframe; chữ tiêu đề là tín hiệu OCR gần như duy nhất.",
    ["title card", "video ít keyframe", "OCR-only cue"],
    ["Tiệm sách miễn phí '3 không' ở Sài Gòn"])

kis("OCR_KIS_14", "Medium", "L27_V016", [7850, 7994], 7850,
    'Tìm cảnh có website đặc sản "cakhobotuhung.vn" kèm chứng nhận an toàn thực phẩm.',
    'Find the scene showing the speciality-food website cakhobotuhung.vn next to a food-safety certificate.',
    "a local speciality food brand website shown with a certificate",
    "cakhobotuhung.vn chứng nhận điều kiện đảm bảo an toàn thực phẩm Cà Mau",
    "Cùng video có 'GIẤY CHỨNG NHẬN' rời không kèm tên miền.",
    ["domain token", "chứng nhận", "compound OCR cue"],
    ["cakhobotuhung.vn", "Chứng nhận: Điều kiện đảm bảo an toàn thực phẩm"])

kis("OCR_KIS_15", "Hard", "L26_V037", [7695, 7695], 7695,
    'Tìm frame hiện đồng thời logo "Món Ngon Mỗi Ngày" và địa chỉ web monngonmoingay.com.',
    'Find the frame showing both the "Mon Ngon Moi Ngay" logo and the site monngonmoingay.com.',
    "cooking show closing logo with website address",
    "món ngon mỗi ngày monngonmoingay.com",
    "BẪY BOILERPLATE: 'Món Ngon Mỗi Ngày' xuất hiện ở 372/498 video L26. Chỉ tên miền monngonmoingay.com mới thu hẹp được.",
    ["boilerplate trap", "overlay DF filter test", "high-DF string"],
    ["Món Ngon Mỗi Ngày", "monngonmoingay.com"])

# ---------------- VQA: doc so / ten tren man hinh ----------------
vqa("OCR_VQA_01", "Easy", "L21_V011", [17400, 17400], 17400,
    "Bản tin về người dân đảo Sicilia ứng phó nắng nóng.",
    "Theo dòng chữ trên màn hình, người dân đảo Sicilia phải ứng phó với cái nóng lên đến bao nhiêu độ C?",
    "According to the on-screen text, what temperature must the people of Sicily cope with?",
    "40 độ C", ["40 độ C", "40 do C", "40°C", "40 độ", "40 degrees Celsius"],
    "OCR numeric lookup",
    "Cùng video có 'Hà Nội nắng nóng hơn 39 độ C' — con số gần giống, khác địa danh.",
    ["numeric lookup", "hard negative lệch 1 đơn vị"],
    ["ỨNG PHÓ VỚI CÁI NÓNG LÊN ĐẾN 40 ĐỘ C", "ITALIA: NGƯỜI DÂN ĐẢO SICILIA"])

vqa("OCR_VQA_02", "Medium", "L21_V009", [14460, 14460], 14460,
    "Bản tin nhóm thanh niên ở Đắk Nông lừa đảo trên mạng.",
    "Theo tiêu đề bản tin, nhóm thanh niên ở Đắk Nông chiếm đoạt hơn bao nhiêu tiền?",
    "According to the headline, how much money did the Dak Nong group appropriate?",
    "Hơn 700 triệu đồng",
    ["700 triệu đồng", "hơn 700 triệu đồng", "700 trieu dong", "over 700 million VND"],
    "OCR numeric lookup",
    "Cùng video có chyron 'TẠM GIỮ GẦN 130 TRIỆU ĐỒNG' (frame 16110) — cũng là số tiền, khác vụ án.",
    ["numeric lookup", "chyron 2 dòng", "same-video money confusion"],
    ["ĐẮK NÔNG: NHÓM THANH NIÊN LỪA ĐẢO HÀNG TRĂM NGƯỜI",
     "TRÊN MẠNG CHIẾM ĐỌAT HƠN 700 TRIỆU ĐỒNG"])

vqa("OCR_VQA_03", "Easy", "L23_V010", [317, 500], 375,
    "Bảng thông số vận động viên trong chặng đua xe đạp.",
    "Theo bảng thông số hiện trên màn hình, vận động viên nặng bao nhiêu ki-lô-gam?",
    "According to the on-screen stats panel, how much does the rider weigh?",
    "62 kg", ["62kg", "62 kg", "62"],
    "OCR numeric lookup",
    "Cùng bảng còn 'Chiều cao: 1m72' và 'Năm sinh: 2004' — ba số dễ lẫn.",
    ["stats panel", "3 số cùng khung", "chọn đúng trường"],
    ["Năm sinh: 2004", "Chiều cao: 1m72", "Cân nặng: 62kg"])

vqa("OCR_VQA_04", "Medium", "L23_V016", [0, 754], 0,
    "Bản tin thể thao truyền hình trực tiếp cuộc đua xe đạp.",
    "Đoạn đua đang phát là chặng thứ mấy?",
    "Which stage of the race is being broadcast?",
    "Chặng 16", ["chặng 16", "chang 16", "16", "stage 16"],
    "OCR numeric lookup",
    "OCR đọc sai thành 'CHẠNG 16' / 'CHƯƠNG 16' ở phần lớn frame; chỉ vài frame đúng 'CHẶNG 16'.",
    ["diacritic error", "overlay bền vững cả video", "video-level answer"],
    ["CHẶNG 16", "CHẠNG 16", "CHƯƠNG 16"])

vqa("OCR_VQA_05", "Hard", "L24_V004", [22819, 23168], 22912,
    "Bảng điểm chấm của 5 trọng tài cho một đoàn lân sư rồng.",
    "Trọng tài số 2 chấm đoàn dự thi bao nhiêu điểm?",
    "What score did referee number 2 give the competing team?",
    "8,70", ["8,70", "8.70", "8,7", "8.7"],
    "OCR bảng — đọc theo cặp nhãn/giá trị",
    "Cùng bảng có 5 điểm gần nhau: 8,65 / 8,70 / 8,35 / 8,60 / 8,61 — trả lời sai chỉ vì lệch một dòng.",
    ["table reading", "label-value pairing", "5 giá trị gần nhau"],
    ["TRỌNG TÀI 1", "8,65", "TRỌNG TÀI 2", "8,70", "TRỌNG TÀI 3", "8,35",
     "TRỌNG TÀI 4", "8,60", "TRỌNG TÀI 5", "8,61"])

vqa("OCR_VQA_06", "Medium", "L24_V004", [22819, 23168], 22819,
    "Bảng thông tin đơn vị dự thi giải lân sư rồng.",
    "Đoàn dự thi trong bảng điểm đến từ quận nào?",
    "Which district is the competing team from?",
    "Quận 5, TP.HCM", ["quận 5", "Q.5", "quan 5", "District 5"],
    "OCR entity lookup",
    "Video L24_V003 có bảng cùng bố cục nhưng đoàn 'LSR Nam Hoa - TP. Phan Thiết - Tỉnh Bình Thuận'.",
    ["entity lookup", "cross-video same-layout", "địa danh"],
    ["Đơn vị dự thi:", "Đoàn LSR Hải Nam Liên Hữu - Quận 5 - TPHCM",
     "Chủ đề: VƯƠN CAO CON RỒNG ĐẤT VIỆT"])

vqa("OCR_VQA_07", "Easy", "L26_V056", [7748, 7936], 7748,
    "Thẻ nguyên liệu món đậu hủ rau rừng.",
    "Món ăn dùng bao nhiêu gam đậu hủ trắng?",
    "How many grams of white tofu does the dish use?",
    "300g", ["300g", "300 g", "300 gam", "300"],
    "OCR numeric lookup",
    "Cùng thẻ có 'CỦ HỦ DƯA 100g' và 'RAU RỪNG 100g' — hai giá trị 100g dễ bị chọn nhầm.",
    ["ingredient quantity", "3 số cùng thẻ"],
    ["NGUYÊN LIỆU", "ĐẬU HỦ TRẮNG 300g", "CỦ HỦ DƯA 100g", "RAU RỪNG 100g"])

vqa("OCR_VQA_08", "Medium", "L27_V001", [1215, 1215], 1215,
    "Màn hình giới thiệu tập của chương trình du lịch.",
    "Đây là tập bao nhiêu của chương trình?",
    "Which episode number is this?",
    "Tập 99", ["99", "tập 99", "tap 99", "episode 99"],
    "OCR numeric lookup",
    "Cuối video có 'GIỚI THIỆU TẬP 100' — số tập của tập SAU, không phải tập đang xem.",
    ["episode number", "bẫy tập kế tiếp", "temporal semantics"],
    ["Tập 99", "VIẾNG MIÊU BÀ NÚI SAM VÀ", "THƯỞNG THỨC ẨM THỰC CHÂU ĐỐC"])

vqa("OCR_VQA_09", "Medium", "L29_V001", [6150, 23400], 6150,
    "Phóng sự về câu lạc bộ phụ nữ khuyết tật ở Cà Mau.",
    "Người phụ nữ trong khung chữ giới thiệu giữ chức vụ gì ở câu lạc bộ?",
    "What position does the woman in the name caption hold in the club?",
    "Chủ nhiệm CLB Phụ nữ khuyết tật xã Tân Bằng",
    ["chủ nhiệm", "chủ nhiệm CLB", "chủ nhiệm câu lạc bộ", "club president"],
    "OCR chức danh — phân biệt chức vụ",
    "Cùng video, ba frame khác dùng đúng bố cục đó nhưng ghi 'Thành viên CLB Phụ nữ khuyết tật...'.",
    ["role caption", "chủ nhiệm vs thành viên", "same-layout hard negative"],
    ["Chủ nhiệm CLB Phụ nữ khuyết tật xã Tân Bằng, huyện Thới Bình, tỉnh Cà Mau",
     "Thành viên CLB Phụ nữ khuyết tật xã Tân Bằng, huyện Thới Bình, tỉnh Cà Mau"])

vqa("OCR_VQA_10", "Hard", "L30_V083", [1525, 6756], 2261,
    "Phóng sự về tác giả tự truyện 'Màu của hy vọng'.",
    "Tác giả cuốn tự truyện 'Màu của hy vọng' sinh năm bao nhiêu?",
    "In which year was the author of the memoir 'Colour of Hope' born?",
    "1984", ["1984", "năm 1984"],
    "OCR entity + numeric lookup",
    "Tên tác giả bị OCR đọc thành 5 biến thể: 'Đỗ Hà Cừ', 'ĐỖ HÀ CƯ', 'ĐÔ HÀ CƯ', 'Đỗ Hà Cư', 'GIẢ ĐỖ HÀ CỪ'.",
    ["name variant explosion", "diacritic error", "phụ đề tư liệu"],
    ["Tên đầy đủ là Đỗ Hà Cừ, sinh năm 1984", "TỰ TRUYỆN", "Màu của", "HY VỌNG"])

# ---------------- TRAKE: chuoi trang thai doc tu OCR ----------------
trake("OCR_TRAKE_01", "Medium", "L26_V037",
      "Tìm video dạy nấu ăn và căn chỉnh 4 mốc theo thứ tự: (1) hiện tên món; (2) thẻ MẸO luộc da heo với ít muối; (3) thẻ MẸO chiên da heo với lửa nhỏ; (4) thẻ QUÉT MÃ QR ĐỂ XEM CÔNG THỨC.",
      [("Thẻ tên món 'DA HEO CHIÊN NƯỚC MẮM'", 405,
        ["DA HEO", "CHIÊN NƯỚC MẮM", "#NấuChuẩnĂnLành"]),
       ("Thẻ MẸO 'LUỘC DA HEO VỚI ÍT MUỐI ĐỂ DA HEO THẤM VỊ VÀ PHỒNG GIÒN HƠN'", 2493,
        ["MẸO", "LUỘC DA HEO VỚI ÍT MUỐI", "ĐỂ DA HEO THẤM VỊ", "VÀ PHỒNG GIÒN HƠN"]),
       ("Thẻ MẸO 'CHIÊN DA HEO VỚI LỬA NHỎ VÀ THÊM ÍT BỘT NĂNG'", 4107,
        ["MẸO", "CHIÊN DA HEO VỚI LỬA NHỎ", "VÀ THÊM ÍT BỘT NĂNG"]),
       ("Thẻ 'QUÉT MÃ QR ĐỂ XEM CÔNG THỨC'", 5893,
        ["QUÉT MÃ QR ĐỂ XEM CÔNG THỨC", "QR"])],
      ["OCR-only chain", "thẻ lớp phủ", "L26 template chung 498 video"],
      "Chuỗi CHỈ đọc được từ OCR: khung hình giữa các mốc gần như giống nhau (bàn bếp, tay người nấu).")

trake("OCR_TRAKE_02", "Hard", "L23_V016",
      "Tìm chặng đua xe đạp và căn chỉnh 4 mốc theo đồng hồ và cột số km còn lại hiện trên màn hình: (1) 6 Km; (2) 3 Km; (3) 1 Km; (4) về ĐÍCH.",
      [("Overlay hiện '6 Km' còn lại, đồng hồ 03:13:51", 0,
        ["CHẶNG 16", "03:13:51", "6 Km", "32°"]),
       ("Overlay hiện '3 Km' còn lại, đồng hồ 03:14:22", 754,
        ["CHẶNG 16", "03:14:22", "3 Km"]),
       ("Overlay hiện '1 Km' còn lại, đồng hồ 03:15:43", 2789,
        ["CHẶNG 16", "03:15:43", "1 Km"]),
       ("Hiện chữ 'ĐÍCH' và cúp truyền hình", 5980,
        ["ĐÍCH", "CÚP TRUYỀN HÌNH", "TON DONG A", "REPLAY"])],
      ["OCR-only chain", "đồng hồ tăng đơn điệu", "counter giảm dần",
       "diacritic error CHẶNG/CHẠNG"],
      "Thứ tự suy được từ đồng hồ đua và cột km còn lại — hình ảnh đoàn đua gần như bất biến, dense visual không phân biệt được.")

trake("OCR_TRAKE_03", "Medium", "L26_V056",
      "Tìm video dạy nấu ăn và căn chỉnh 4 mốc: (1) tên món ĐẬU HỦ RAU RỪNG; (2) thẻ CỦ HỦ DỪA CẮT SỢI; (3) thẻ ĐẬU HỦ CẮT SỢI; (4) bảng nguyên liệu đầy đủ cuối video.",
      [("Thẻ tên món 'ĐẬU HỦ RAU RỪNG' (kèm 'CUỐN CHAY')", 384,
        ["ĐẬU HỦ RAU RỪNG", "CUỐN CHAY"]),
       ("Thẻ thao tác 'CỦ HỦ DỪA CẮT SỢI'", 2048, ["CỦ HỦ DỪA CẮT SỢI"]),
       ("Thẻ thao tác 'ĐẬU HỦ CẮT SỢI'", 2432, ["ĐẬU HỦ CẮT SỢI"]),
       ("Bảng NGUYÊN LIỆU đầy đủ in hoa cuối video", 7748,
        ["NGUYÊN LIỆU", "ĐẬU HỦ TRẮNG 300g", "CỦ HỦ DƯA 100g", "RAU RỪNG 100g"])],
      ["OCR-only chain", "hai thẻ 'CẮT SỢI' liền kề", "thứ tự dễ đảo"],
      "Mốc 2 và 3 chỉ khác nhau ở tên nguyên liệu, cách nhau ~384 frame — kiểm tra hệ có giữ đúng thứ tự không.")

# ---------------- AVS: moi doan co mot LOP chu ----------------
avs("OCR_AVS_01", "Medium", "L26_V037",
    "Tìm mọi đoạn trong video có thẻ MẸO nấu ăn hiện trên màn hình.",
    "on-screen cooking TIP card is visible",
    [("E01_MEO_LUOC", 2493, 2565, 3,
      "Thẻ MẸO luộc da heo với ít muối, chữ đầy đủ 3 dòng",
      ["MẸO", "LUỘC DA HEO VỚI ÍT MUỐI", "ĐỂ DA HEO THẤM VỊ", "VÀ PHỒNG GIÒN HƠN"]),
     ("E02_MEO_CHIEN", 4050, 4107, 3, "Thẻ MẸO chiên da heo với lửa nhỏ",
      ["MẸO", "CHIÊN DA HEO VỚI LỬA NHỎ", "VÀ THÊM ÍT BỘT NĂNG"]),
     ("E03_MEO_NGAP_DAU", 4493, 4590, 3, "Thẻ MẸO luôn giữ da heo ngập trong dầu",
      ["MẸO", "LUÔN GIỮ DA HEO NGẬP TRONG DẦU", "ĐỂ DA HEO KHÔNG BỊ CHAI"])],
    ["class-of-text retrieval", "3 sự kiện cùng template", "dedup theo event"],
    "Ba thẻ dùng CHUNG một template đồ hoạ — bài toán là phủ đủ 3 sự kiện chứ không phải xếp hạng 1 cái.")

avs("OCR_AVS_02", "Hard", "L29_V019",
    "Tìm mọi đoạn có biển hiệu hoặc khung liên hệ hiện SỐ ĐIỆN THOẠI.",
    "an on-screen phone number appears on a sign or contact overlay",
    [("E01_UBND", 734, 734, 2, "Khung UBND tỉnh Cà Mau kèm ĐT/FAX cơ quan",
      ["ĐT: 02963 910 020 - FAX: 02963 910 040"]),
     ("E02_VUA_MUC_KHO", 11565, 11610, 3, "Biển vựa mực khô kèm 2 số di động",
      ["ĐT: 0916.745766 (ANH -)", "ĐT: 0916.74.57.66 | ANH: 0919.84.88.66"]),
     ("E03_HAI_SAN", 17010, 17010, 2, "Khung hải sản Cà Mau hiện số 0919848866",
      ["0919848866"]),
     ("E04_KIEU_NUONG", 23214, 23760, 3, "Biển quán nướng kèm dòng mời liên hệ",
      ["ĐT: 0913848866", "ĐT: 0919848866", "SỐ SOT LIÊN HỆ BEN EM 091 984 8866"])],
    ["pattern-class retrieval", "OCR đọc sai chữ số", "4 sự kiện trong 1 video"],
    "Cùng một số bị OCR đọc thành 0919848866 / 0913848866 / 091 984 8866 / 0919.84.88.66 — kiểm tra khả năng gom biến thể.")

avs("OCR_AVS_03", "Medium", "L21_V001",
    "Tìm mọi đoạn bản tin hiện thẻ điều hướng chuyên mục 'TIN CHÍNH' hoặc 'TIẾP THEO'.",
    "a news section card reading TIN CHINH or TIEP THEO is on screen",
    [("E01_TIN_CHINH", 531, 711, 3, "Thẻ TIN CHÍNH ở đầu bản tin",
      ["TIN CHÍNH", "60 giây"]),
     ("E02_TIEP_THEO_A", 12630, 12962, 3, "Thẻ TIẾP THEO trước 2 tin pháp luật",
      ["TIẾP THEO", "60 giây"]),
     ("E03_TIEP_THEO_B", 22142, 22584, 3, "Thẻ TIẾP THEO trước cụm tin quốc tế",
      ["TIẾP THEO", "60 giây"])],
    ["overlay class", "chuỗi ngắn tần suất cao", "kiểm tra overlay-DF filter"],
    "Đây là chuỗi OCR NGẮN và lặp — dùng để đo tác động của ocr_overlay_df: lọc quá tay là mất hẳn truy vấn này.")

# ---------------- Verify + ghi ----------------
bad = []
for q in Q:
    v = q["target_video"]
    frames = []
    for it in q.get("target_intervals", []) + q.get("relevant_intervals", []):
        frames += [it["start_frame"], it["end_frame"]]
    for e in q.get("events", []):
        frames.append(e["representative_frame"])
    if "representative_frame" in q:
        frames.append(q["representative_frame"])
    for f in frames:
        if str(f) not in KF.get(v, {}):
            bad.append((q["query_id"], v, f))
if bad:
    print("!! FRAME KHONG PHAI KEYFRAME:", bad)
    sys.exit(1)

out = "examples/AIC2026_ocr_queries_v1.jsonl"
with open(out, "w", encoding="utf-8") as fh:
    for q in Q:
        fh.write(json.dumps(q, ensure_ascii=False) + "\n")

c = collections.Counter(q["task"] for q in Q)
d = collections.Counter(q["difficulty"] for q in Q)
vids = sorted({q["target_video"] for q in Q})
print("OK: %d query -> %s" % (len(Q), out))
print("  theo task:", dict(c))
print("  theo do kho:", dict(d))
print("  %d video: %s" % (len(vids), vids))
print("  %d nhom: %s" % (len({v.split("_")[0] for v in vids}),
                         sorted({v.split("_")[0] for v in vids})))
