import type { HealthResponse } from "../../types";

const CARDS: { key: keyof HealthResponse; label: string; icon: string; tone: string }[] = [
  { key: "scene_count", label: "Scenes", icon: "📊", tone: "stat-blue" },
  { key: "keyframe_count", label: "Keyframes", icon: "🖼", tone: "stat-cyan" },
  { key: "video_count", label: "Videos", icon: "🎬", tone: "stat-purple" },
  { key: "asr_segment_count", label: "ASR segments", icon: "🔊", tone: "stat-pink" },
];

export interface DatasetStatsProps {
  health: HealthResponse | null;
}

/** Bốn thẻ thống kê dataset — đọc thẳng từ GET /v1/health, không hard-code số
 * nào. "—" khi giá trị null/undefined (vd export không kèm dataset_manifest.
 * json) thay vì hiển thị 0 sai lệch. */
export function DatasetStats({ health }: DatasetStatsProps) {
  return (
    <div className="dataset-stats">
      {CARDS.map((card) => {
        const raw = health?.[card.key];
        const value = typeof raw === "number" ? raw.toLocaleString("vi-VN") : "—";
        return (
          <div key={card.key} className={`stat-card ${card.tone}`} title={`Nguồn: GET /v1/health.${card.key}`}>
            <span className="stat-icon" aria-hidden="true">
              {card.icon}
            </span>
            <div className="stat-body">
              <span className="stat-label">{card.label}</span>
              <output className="stat-value">{value}</output>
            </div>
          </div>
        );
      })}
    </div>
  );
}
