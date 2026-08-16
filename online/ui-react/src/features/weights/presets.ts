// Preset lưu client-side (localStorage) — KHÔNG có endpoint GET /v1/search/
// presets ở backend (đã audit trước khi build UI), nên toàn bộ vòng đời
// preset (save/load/duplicate/rename/delete/export/import) chỉ chạy ở đây.

import type { SearchOptions, TaskType } from "../../types";

export interface SearchPreset {
  id: string;
  name: string;
  task: TaskType;
  version: string;
  searchOptions: SearchOptions;
  createdAt: string;
  source: "built_in" | "user";
  /** Preset làm gì. Hiện ngay dưới ô chọn để không phải đoán. */
  description?: string;
  /** Khi nào NÊN chọn nó — phần người chưa quen cần nhất. */
  whenToUse?: string;
  /** Số đo thật, hoặc ghi rõ là chưa đo. Không có bằng chứng thì phải nói. */
  evidence?: string;
}

const STORAGE_KEY = "aic_search_presets_v2";
const PRESET_VERSION = "1";

// Preset dựng sẵn — MỖI cái phải nói được nó làm gì, khi nào dùng, và có số
// đo hay không. Bộ cũ chỉ là chỗ giữ chỗ (`Balanced` = `{}`, "Visual Heavy" =
// weight 3 không rõ từ đâu ra) nên người chưa quen chọn xong vẫn không biết
// mình vừa đổi gì.
//
// Số đo dưới đây lấy từ 36 truy vấn KIS gold, fusion `norm_max`, ngày 13/08.

const BUILT_IN_BASE = { version: PRESET_VERSION, createdAt: "1970-01-01T00:00:00Z", source: "built_in" as const };

export const BUILT_IN_PRESETS: SearchPreset[] = [
  {
    ...BUILT_IN_BASE,
    id: "builtin_default",
    name: "Mặc định — dùng cái này trước",
    task: "TEXTUAL_KIS",
    searchOptions: {},
    description:
      "Không ghi đè gì; dùng đúng cấu hình server đã được đo và chốt.",
    whenToUse:
      "Luôn bắt đầu từ đây. Chỉ đổi khi đã tìm một lượt mà chưa thấy đáp án.",
    evidence: "KIS R@1 0.750 · R@20 1.000 · MRR 0.852 (36 truy vấn gold).",
  },
  {
    ...BUILT_IN_BASE,
    id: "builtin_onscreen_text",
    name: "Tìm chữ hiện trên màn hình",
    task: "TEXTUAL_KIS",
    searchOptions: { branches: { bm25_ocr: { weight: 5 } } },
    description:
      "Đẩy mạnh nhánh OCR để chữ trên khung hình thắng các tín hiệu khác.",
    whenToUse:
      "Khi bạn tra TÊN NGƯỜI, tên cơ quan, biển hiệu, con số, hoặc tiêu đề bản tin — thứ đọc được bằng mắt trên video.",
    evidence:
      "Truy vấn tên 'Ông NGUYỄN TRANG SƯ…': trọng số 0.25 không vào nổi top-20, 1.0 lên hạng 1, 5.0 vẫn hạng 1 và bỏ xa hơn.",
  },
  {
    ...BUILT_IN_BASE,
    id: "builtin_spoken",
    name: "Tìm theo lời nói (ASR)",
    task: "TEXTUAL_KIS",
    searchOptions: { branches: { bm25_asr: { weight: 3 } } },
    description: "Đẩy mạnh nhánh lời thoại đã chuyển thành chữ.",
    whenToUse:
      "Khi thứ bạn tìm được NHẮC TỚI chứ không nhìn thấy — địa danh, tên chương trình, con số người dẫn đọc lên.",
    evidence:
      "Có ca thật: 'Đồng Tháp' không có trong caption lẫn OCR của phóng sự cá tra, chỉ ASR nói 'Ghi nhận sau tại Đồng Tháp'.",
  },
  {
    ...BUILT_IN_BASE,
    id: "builtin_visual",
    name: "Tìm theo hình ảnh",
    task: "TEXTUAL_KIS",
    searchOptions: { branches: { dense_visual: { weight: 3 } } },
    description: "Đẩy mạnh nhánh CLIP so khớp thẳng nội dung hình.",
    whenToUse:
      "Khi bạn mô tả CẢNH chứ không mô tả chữ — 'đàn cá quẫy trên mặt nước', 'người đội mũ bảo hiểm đi xe đạp'.",
    evidence:
      "Nhánh mạnh nhất của hệ; đã cứu được một keyframe mà caption mô tả sai hoàn toàn (đàn cá bị tả thành đám cháy).",
  },
  {
    ...BUILT_IN_BASE,
    id: "builtin_wide_net",
    name: "Quăng lưới rộng",
    task: "TEXTUAL_KIS",
    searchOptions: { fusion: { max_results_per_video: 40 }, results: {} },
    description: "Nới trần số kết quả mỗi video để top-20 đa dạng hơn.",
    whenToUse:
      "Khi kết quả dồn hết vào một video mà bạn ngờ đáp án nằm ở video khác.",
    evidence: "AVS: trần 20 cho nDCG 0.354, nới lên 40 cho 0.547 (bão hoà từ 40).",
  },
  {
    ...BUILT_IN_BASE,
    id: "builtin_qa_evidence",
    name: "QA — ưu tiên bằng chứng",
    task: "QA",
    searchOptions: { rerank: { text: { enabled: true } } },
    description: "Bật rerank văn bản để scene có bằng chứng rõ lên trước.",
    whenToUse: "Khi câu trả lời sai nhưng bạn thấy đúng đoạn video trong danh sách.",
    evidence:
      "QA giữ fusion `rrf` (không phải `norm_max`) vì QA cần ĐỒNG THUẬN nhiều nhánh chứ không cần một nhánh chắc chắn.",
  },
  {
    ...BUILT_IN_BASE,
    id: "builtin_trake_default",
    name: "TRAKE — mặc định",
    task: "TRAKE",
    searchOptions: {},
    description: "Cấu hình đã đo cho chuỗi nhiều bước.",
    whenToUse:
      "Luôn bắt đầu từ đây. Nhớ VIẾT ĐÚNG THỨ TỰ THỜI GIAN các bước — sai thứ tự thì chuỗi đúng không ghép được.",
    evidence:
      "video_recall@1 = 1.000 trên cả ba video. Đảo thứ tự hai bước đã từng biến 'không có trong top-20' thành 'hạng 1'.",
  },
  {
    ...BUILT_IN_BASE,
    id: "builtin_avs_diversity",
    name: "AVS — ưu tiên đa dạng",
    task: "AVS",
    searchOptions: { fusion: { dedup_scope: "event", max_results_per_video: 40 } },
    description: "Gom trùng theo sự kiện và nới trần mỗi video.",
    whenToUse: "Khi cần phủ nhiều đoạn khác nhau thay vì nhiều ảnh của cùng một đoạn.",
    evidence: "nDCG 0.547 · event_coverage 0.793 ở trần 40.",
  },
];

export function loadUserPresets(): SearchPreset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveUserPresets(presets: SearchPreset[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
}

export function upsertUserPreset(preset: SearchPreset): SearchPreset[] {
  const existing = loadUserPresets();
  const withoutOld = existing.filter((item) => item.id !== preset.id);
  const next = [...withoutOld, preset];
  saveUserPresets(next);
  return next;
}

export function deleteUserPreset(id: string): SearchPreset[] {
  const next = loadUserPresets().filter((item) => item.id !== id);
  saveUserPresets(next);
  return next;
}

export function allPresets(): SearchPreset[] {
  return [...BUILT_IN_PRESETS, ...loadUserPresets()];
}

export function newPresetId(): string {
  return `user_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
