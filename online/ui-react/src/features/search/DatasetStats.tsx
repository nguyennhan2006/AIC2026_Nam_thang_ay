import { AudioLines, Film, Images, LayoutGrid } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { HealthResponse } from "../../types";
import { Skeleton } from "../../ui";

const CARDS: { key: keyof HealthResponse; label: string; icon: LucideIcon; tone: string }[] = [
  { key: "scene_count", label: "Scenes", icon: LayoutGrid, tone: "stat-scenes" },
  { key: "keyframe_count", label: "Keyframes", icon: Images, tone: "stat-keyframes" },
  { key: "video_count", label: "Videos", icon: Film, tone: "stat-videos" },
  { key: "asr_segment_count", label: "ASR", icon: AudioLines, tone: "stat-asr" },
];

export interface DatasetStatsProps {
  health: HealthResponse | null;
  /** Chưa có phản hồi health nào → hiện skeleton thay vì bốn ô "—" trơ. */
  loading?: boolean;
}

/** Bốn thẻ thống kê — đọc thẳng GET /v1/health. "—" khi giá trị null (vd
 * export không kèm dataset_manifest.json) thay vì hiển thị 0 sai lệch. */
export function DatasetStats({ health, loading = false }: DatasetStatsProps) {
  return (
    <div className="stat-grid">
      {CARDS.map((card) => {
        const Icon = card.icon;
        const raw = health?.[card.key];
        return (
          <div key={card.key} className={`stat-card ${card.tone}`} title={`Nguồn: GET /v1/health.${String(card.key)}`}>
            <span className="stat-card-head">
              <Icon size={12} aria-hidden="true" />
              <span className="stat-card-label">{card.label}</span>
            </span>
            {loading ? (
              <Skeleton width="60%" height={19} />
            ) : (
              <output className="stat-card-value">{typeof raw === "number" ? raw.toLocaleString("vi-VN") : "—"}</output>
            )}
          </div>
        );
      })}
    </div>
  );
}
