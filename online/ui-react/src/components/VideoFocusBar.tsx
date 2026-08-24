import { useState } from "react";
import { Crosshair, Plus, X } from "lucide-react";
import { Button, IconButton } from "../ui";

export interface VideoFocusBarProps {
  videoIds: string[];
  onChange: (videoIds: string[]) => void;
  /** Chạy lại truy vấn hiện tại, đào sâu vào đúng các video đang khoanh. */
  onDeepDive: (videoIds: string[]) => void;
  /** Video của kết quả đang chọn — lối thêm nhanh nhất, không phải gõ id. */
  suggestion?: string | null;
  busy?: boolean;
}

/** Khoanh vùng tìm kiếm vào một vài video.
 *
 * Ra đời cho một tình huống cụ thể: người dùng đã nhìn kết quả và TIN là đáp
 * án nằm trong hai ba video này, nhưng frame/đáp án còn chưa đúng. Lúc đó
 * quăng lưới rộng ra cả 873 video là phí — cái cần là đào sâu trong đúng mấy
 * video đó.
 *
 * "Đào sâu" không chỉ là lọc: nó đồng thời nới trần số kết quả mỗi video (mặc
 * định KIS chỉ giữ 5/video) và tăng top-K. Chỉ lọc mà giữ nguyên trần thì kết
 * quả vẫn đúng 5 dòng như cũ — người dùng bấm xong không thấy gì thay đổi và
 * kết luận là tính năng hỏng.
 */
export function VideoFocusBar({
  videoIds, onChange, onDeepDive, suggestion, busy = false,
}: VideoFocusBarProps) {
  const [draft, setDraft] = useState("");

  function add(videoId: string) {
    const value = videoId.trim().toUpperCase();
    if (!value || videoIds.includes(value)) return;
    onChange([...videoIds, value]);
    setDraft("");
  }

  const canSuggest = suggestion != null && !videoIds.includes(suggestion);

  return (
    <div className={videoIds.length > 0 ? "video-focus is-active" : "video-focus"}>
      <Crosshair size={13} />
      <span className="eyebrow">Khoanh vùng</span>

      {videoIds.length === 0 && (
        <span className="muted small">
          Chưa khoanh — truy vấn chạy trên toàn bộ dataset.
        </span>
      )}

      {videoIds.map((videoId) => (
        <span key={videoId} className="video-focus-chip">
          <span className="tabular">{videoId}</span>
          <button
            type="button"
            className="video-focus-drop"
            aria-label={`Bỏ ${videoId} khỏi vùng khoanh`}
            title={`Bỏ ${videoId}`}
            onClick={() => onChange(videoIds.filter((item) => item !== videoId))}
          >
            <X size={10} />
          </button>
        </span>
      ))}

      {canSuggest && (
        <Button
          size="sm" variant="ghost" icon={<Plus size={12} />}
          onClick={() => add(suggestion!)}
        >
          {suggestion}
        </Button>
      )}

      <input
        type="text"
        className="video-focus-input tabular"
        value={draft}
        placeholder="L01_V001"
        aria-label="Thêm video vào vùng khoanh"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") add(draft);
        }}
      />

      <Button
        size="sm" variant="primary" icon={<Crosshair size={12} />}
        disabled={videoIds.length === 0 || busy}
        onClick={() => onDeepDive(videoIds)}
      >
        Đào sâu {videoIds.length || ""}
      </Button>

      {videoIds.length > 0 && (
        <IconButton
          icon={<X size={13} />}
          label="Bỏ khoanh vùng, tìm lại trên toàn dataset"
          size="sm"
          variant="control"
          onClick={() => onChange([])}
        />
      )}
    </div>
  );
}
