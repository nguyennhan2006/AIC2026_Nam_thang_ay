import { useState } from "react";
import type { ApiClientConfig } from "../api";
import { getSceneDetail, mediaUrl } from "../api";
import type { SceneDocument, SearchHit } from "../types";

const FIELD_LABELS: Record<string, string> = { caption: "Caption", ocr: "OCR", asr: "ASR", keyword: "Keyword" };
const ICONS = [
  { field: "reason", icon: "\u{1F4CA}", title: "Vì sao khớp" },
  { field: "caption", icon: "\u{1F4DD}", title: "Caption" },
  { field: "ocr", icon: "\u{1F524}", title: "OCR" },
  { field: "asr", icon: "\u{1F3A4}", title: "ASR" },
  { field: "keyword", icon: "\u{1F3F7}️", title: "Keyword" },
  { field: "video", icon: "\u{1F3AC}", title: "Xem video" },
] as const;

function ReasonPanel({ hit }: { hit: SearchHit }) {
  const entries = Object.entries(hit.component_scores || {}).sort((a, b) => b[1] - a[1]);
  return (
    <>
      {entries.length > 0 && (
        <ul className="reason">
          {entries.map(([name, value]) => (
            <li key={name}>
              <span>{name}</span>
              <output>{value.toFixed(4)}</output>
            </li>
          ))}
        </ul>
      )}
      {hit.evidence.length > 0 ? (
        <ul className="evidence">
          {hit.evidence.map((item, i) => (
            <li key={i}>
              <strong>{item.modality}</strong>: {item.text}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">Không có evidence text.</p>
      )}
    </>
  );
}

function VideoPanel({ hit, apiConfig }: { hit: SearchHit; apiConfig: ApiClientConfig }) {
  if (!hit.video_path) return <p className="muted">Không có video nguồn.</p>;
  return (
    // eslint-disable-next-line jsx-a11y/media-has-caption
    <video className="video" controls preload="metadata" src={`${mediaUrl(apiConfig, hit.video_path)}#t=${hit.start_sec},${hit.end_sec}`} />
  );
}

function FieldListPanel({ field, detail }: { field: string; detail: SceneDocument }) {
  const values =
    ({ caption: detail.captions, ocr: detail.ocr_texts, asr: detail.asr_texts, keyword: detail.keywords } as Record<string, string[]>)[
      field
    ] || [];
  if (!values.length) return <p className="muted">Không có {FIELD_LABELS[field]}.</p>;
  return (
    <ul className="field-list">
      {values.map((v, i) => (
        <li key={i}>{v}</li>
      ))}
    </ul>
  );
}

export interface SceneCardProps {
  hit: SearchHit;
  index: number;
  selected: boolean;
  onToggleSelect: (hit: SearchHit, checked: boolean) => void;
  apiConfig: ApiClientConfig;
  sceneDetailCache: Map<string, Promise<SceneDocument>>;
}

export function SceneCard({ hit, index, selected, onToggleSelect, apiConfig, sceneDetailCache }: SceneCardProps) {
  const [openField, setOpenField] = useState<string | null>(null);
  const [detail, setDetail] = useState<SceneDocument | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loadingField, setLoadingField] = useState<string | null>(null);

  async function handleIconClick(field: string) {
    if (openField === field) {
      setOpenField(null);
      return;
    }
    setOpenField(field);
    setDetailError(null);
    if (field === "reason" || field === "video") return;

    setLoadingField(field);
    try {
      if (!sceneDetailCache.has(hit.scene_id)) {
        sceneDetailCache.set(hit.scene_id, getSceneDetail(apiConfig, hit.scene_id));
      }
      const result = await sceneDetailCache.get(hit.scene_id)!;
      setDetail(result);
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : "Không tải được");
    } finally {
      setLoadingField(null);
    }
  }

  const thumbnail = hit.best_keyframe_path ? (
    // eslint-disable-next-line jsx-a11y/alt-text
    <img className="thumb" loading="lazy" src={mediaUrl(apiConfig, hit.best_keyframe_path)} alt={hit.best_keyframe_id ?? ""} />
  ) : (
    <div className="thumb thumb-empty">Không có ảnh</div>
  );

  return (
    <article className={`card${selected ? " selected" : ""}`} data-scene-id={hit.scene_id}>
      <header className="card-head">
        <label className="select-label" title="Chọn kết quả này">
          <input
            type="checkbox"
            className="select-box"
            checked={selected}
            onChange={(e) => onToggleSelect(hit, e.target.checked)}
          />
          <span className="rank">#{index + 1}</span>
        </label>
        <output>{hit.score.toFixed(4)}</output>
      </header>
      <div className="thumb-wrap">{thumbnail}</div>
      <div className="card-meta">
        <strong>{hit.video_id}</strong>
        <span>
          {hit.start_sec.toFixed(1)}s–{hit.end_sec.toFixed(1)}s
          {hit.best_timestamp_sec != null ? ` · ${hit.best_timestamp_sec.toFixed(2)}s` : ""}
        </span>
        <div className="chips">
          {hit.matched_modalities.map((m) => (
            <span key={m}>{m}</span>
          ))}
        </div>
      </div>
      <div className="icon-bar">
        {ICONS.map((item) => (
          <button
            key={item.field}
            type="button"
            className={`icon-btn${openField === item.field ? " active" : ""}`}
            title={item.title}
            aria-label={item.title}
            onClick={() => handleIconClick(item.field)}
          >
            {item.icon}
          </button>
        ))}
      </div>
      {openField && (
        <div className="expand-panel">
          {openField === "reason" && <ReasonPanel hit={hit} />}
          {openField === "video" && <VideoPanel hit={hit} apiConfig={apiConfig} />}
          {openField !== "reason" && openField !== "video" && (
            <>
              {loadingField === openField && <p className="muted">Đang tải…</p>}
              {detailError && loadingField !== openField && <p className="muted">Không tải được: {detailError}</p>}
              {detail && !detailError && loadingField !== openField && <FieldListPanel field={openField} detail={detail} />}
            </>
          )}
        </div>
      )}
    </article>
  );
}
