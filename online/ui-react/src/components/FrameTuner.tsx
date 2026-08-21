import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, FileDown, FileUp, RotateCcw, Video } from "lucide-react";
import type { ApiClientConfig } from "../api";
import { listVideoFrames, listVideos, mediaUrl } from "../api";
import { downloadCsv } from "../exportCsv";
import type { TaskType, VideoFrame, VideoMeta } from "../types";
import { Badge, Button, EmptyState, InlineError } from "../ui";

/**
 * Tab chỉnh frame bằng tay cho submission đã lọc bằng hệ thống.
 *
 * Vì sao cần một tab riêng thay vì nhét vào bảng nộp bài: hai việc khác nhau về
 * bản chất. Bảng nộp bài trả lời *"chọn dòng nào, xếp thứ tự ra sao"*; tab này
 * trả lời *"dòng này đã trỏ đúng khoảnh khắc chưa"* — và câu hỏi thứ hai cần
 * tua được toàn bộ video chứ không chỉ cửa sổ của một scene.
 *
 * Quy đổi frame <-> giây dùng `fps` THẬT lấy từ `GET /v1/videos`. Đo trên corpus
 * hiện tại: V001/V002 chạy 30 fps nhưng **V003 chạy 25 fps** — hằng số 30 sẽ tua
 * lệch 20% trên V003, đúng loại lỗi khiến người chấm tưởng hệ thống chọn sai
 * frame trong khi thực ra chỉ là trình phát nhảy nhầm chỗ.
 */

export interface FrameTunerProps {
  apiConfig: ApiClientConfig;
  task: TaskType;
  /** Dòng dựng sẵn từ kết quả tìm kiếm hiện tại — nguồn nạp không ma sát. */
  seedRows: TunerRow[];
  /** Đổi giá trị này = nạp NGAY `seedRows`, không đợi bấm nút.
   *
   *  Dùng cho "Lưu & chỉnh frame" ở bảng nộp: người dùng đã ra lệnh chỉnh rồi,
   *  bắt bấm thêm một nút "nạp" nữa là thừa — và tệ hơn, tab mở ra rỗng trông
   *  y hệt như tính năng bị hỏng. `undefined` = giữ hành vi cũ (nạp thủ công). */
  autoLoadKey?: number;
}

export interface TunerRow {
  /** Ổn định qua mọi lần sửa; KHÔNG dùng frame làm khoá vì frame sẽ đổi. */
  id: string;
  videoId: string;
  /** Frame gốc do hệ thống chọn — giữ để so sánh và để hoàn tác. */
  originalFrame: number;
  frame: number;
  answer?: string;
  /** TRAKE: một dòng là một chuỗi, mỗi bước chỉnh riêng. */
  step?: number;
  chain?: number;
  /** Bước KHÔNG có bằng chứng riêng: frame chỉ là điểm giữa hai mốc lân cận
   *  (hoặc frame nội suy do backend lấp). Phải nhìn thấy được — đây chính là
   *  những bước đáng mở video ra chỉnh, và cũng là những bước dễ tưởng nhầm
   *  là hệ đã tìm ra. */
  placeholder?: boolean;
}

const NUDGES = [-300, -30, -5, -1, 1, 5, 30, 300];

function parseCsv(text: string): TunerRow[] {
  const rows: TunerRow[] = [];
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  lines.forEach((line, lineIndex) => {
    const cells = line.split(",").map((cell) => cell.trim().replace(/^"|"$/g, ""));
    if (cells.length < 2) return;
    const videoId = cells[0];
    // Bỏ dòng tiêu đề nếu có: cột 2 không phải số.
    if (!/^\d+$/.test(cells[1])) return;
    // KIS: video,frame | QA: video,frame,answer | TRAKE: video,f1,f2,...
    const numeric = cells.slice(1).filter((cell) => /^\d+$/.test(cell)).map(Number);
    const answer = cells.slice(1).find((cell) => !/^\d+$/.test(cell));
    if (numeric.length > 1) {
      numeric.forEach((frame, step) => {
        rows.push({
          id: `${lineIndex}-${step}`, videoId, originalFrame: frame, frame,
          chain: lineIndex + 1, step: step + 1,
        });
      });
      return;
    }
    rows.push({
      id: `${lineIndex}`, videoId, originalFrame: numeric[0], frame: numeric[0], answer,
    });
  });
  return rows;
}

function toCsv(task: TaskType, rows: TunerRow[]): string {
  if (task === "TRAKE") {
    const chains = new Map<number, TunerRow[]>();
    for (const row of rows) {
      const key = row.chain ?? 0;
      chains.set(key, [...(chains.get(key) ?? []), row]);
    }
    return [...chains.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, steps]) => [steps[0].videoId, ...steps.map((step) => step.frame)].join(","))
      .join("\n");
  }
  return rows
    .map((row) =>
      row.answer ? `${row.videoId},${row.frame},"${row.answer.replace(/"/g, '""')}"`
                 : `${row.videoId},${row.frame}`
    )
    .join("\n");
}

export function FrameTuner({ apiConfig, task, seedRows, autoLoadKey }: FrameTunerProps) {
  const [rows, setRows] = useState<TunerRow[]>([]);
  const [loadedFrom, setLoadedFrom] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [videos, setVideos] = useState<Record<string, VideoMeta>>({});
  const [videosError, setVideosError] = useState<string | null>(null);
  /** Keyframe theo video — ảnh thay thế khi thiếu mp4. Nạp lười theo video. */
  const [frames, setFrames] = useState<Record<string, VideoFrame[]>>({});
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  /** Chặn vòng lặp: seek tự động -> onTimeUpdate -> lại đổi frame -> seek... */
  const seekingRef = useRef(false);

  useEffect(() => {
    let alive = true;
    listVideos(apiConfig)
      .then((list) => {
        if (!alive) return;
        setVideos(Object.fromEntries(list.map((item) => [item.video_id, item])));
        setVideosError(null);
      })
      .catch((error) => alive && setVideosError(String(error?.message ?? error)));
    return () => { alive = false; };
  }, [apiConfig]);

  const selected = useMemo(
    () => rows.find((row) => row.id === selectedId) ?? null,
    [rows, selectedId]
  );
  const meta = selected ? videos[selected.videoId] : undefined;

  // Nạp keyframe của video đang chọn (một lần mỗi video). Cần cho CẢ video có
  // mp4: ảnh keyframe hiện ngay lập tức trong lúc video còn đang buffer, nên
  // người soát không phải nhìn khung đen.
  useEffect(() => {
    const videoId = selected?.videoId;
    if (!videoId || frames[videoId]) return;
    let alive = true;
    listVideoFrames(apiConfig, videoId)
      .then((list) => alive && setFrames((current) => ({ ...current, [videoId]: list })))
      .catch(() => alive && setFrames((current) => ({ ...current, [videoId]: [] })));
    return () => { alive = false; };
  }, [apiConfig, selected?.videoId, frames]);

  /** Keyframe gần `frame` nhất — tìm nhị phân trên danh sách đã sắp. */
  const nearestFrame = useMemo(() => {
    if (!selected) return null;
    const list = frames[selected.videoId];
    if (!list || list.length === 0) return null;
    let low = 0;
    let high = list.length - 1;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (list[mid].frame_idx < selected.frame) low = mid + 1;
      else high = mid;
    }
    const after = list[low];
    const before = list[Math.max(0, low - 1)];
    return Math.abs(before.frame_idx - selected.frame) <= Math.abs(after.frame_idx - selected.frame)
      ? before
      : after;
  }, [frames, selected]);

  /** Ruy băng ảnh: lấy ~36 keyframe rải đều thay vì cả 300 — đủ để nhận ra
   *  đang ở đoạn nào của video mà không tải hàng trăm ảnh. */
  const filmstrip = useMemo(() => {
    if (!selected) return [];
    const list = frames[selected.videoId];
    if (!list || list.length === 0) return [];
    const wanted = 36;
    if (list.length <= wanted) return list;
    const stride = list.length / wanted;
    return Array.from({ length: wanted }, (_, i) => list[Math.floor(i * stride)]);
  }, [frames, selected]);

  const setFrame = useCallback((id: string, frame: number) => {
    setRows((current) =>
      current.map((row) => {
        if (row.id !== id) return row;
        const ceiling = (videos[row.videoId]?.frame_count ?? Number.MAX_SAFE_INTEGER) - 1;
        return { ...row, frame: Math.max(0, Math.min(ceiling, Math.round(frame))) };
      })
    );
  }, [videos]);

  // Tua video mỗi khi frame đang chọn đổi. `seekingRef` chặn vòng lặp với
  // `onTimeUpdate` bên dưới — thiếu nó thì kéo thanh trượt xong video tự nhảy
  // về chỗ cũ vì sự kiện timeupdate ghi đè lại state.
  useEffect(() => {
    const element = videoRef.current;
    if (!element || !selected || !meta) return;
    const target = selected.frame / meta.fps;
    if (Math.abs(element.currentTime - target) < 0.5 / meta.fps) return;
    seekingRef.current = true;
    const apply = () => { element.currentTime = target; };
    if (element.readyState >= 1) apply();
    else element.addEventListener("loadedmetadata", apply, { once: true });
  }, [selected, meta]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!selected) return;
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const step = event.shiftKey ? 10 : 1;
      if (event.key === "ArrowLeft") { event.preventDefault(); setFrame(selected.id, selected.frame - step); }
      if (event.key === "ArrowRight") { event.preventDefault(); setFrame(selected.id, selected.frame + step); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, setFrame]);

  /** Kéo trên ruy băng -> đổi frame. Dùng pointer capture để con trỏ ra ngoài
   *  phần tử vẫn tiếp tục kéo, giống mọi timeline editor. */
  function scrubTo(event: React.PointerEvent<HTMLDivElement>, frameCount: number, id: string) {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    setFrame(id, ratio * (frameCount - 1));
  }

  function loadSeed() {
    setRows(seedRows.map((row) => ({ ...row })));
    setSelectedId(seedRows[0]?.id ?? null);
    setLoadedFrom(`kết quả hiện tại (${seedRows.length} dòng)`);
  }

  // Nạp thẳng khi được đẩy sang từ bảng nộp. Chỉ chạy khi `autoLoadKey` ĐỔI,
  // nên thao tác chỉnh tay sau đó không bị ghi đè ở mỗi lần render.
  const loadedKey = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (autoLoadKey === undefined || autoLoadKey === loadedKey.current) return;
    loadedKey.current = autoLoadKey;
    if (seedRows.length === 0) return;
    setRows(seedRows.map((row) => ({ ...row })));
    setSelectedId(seedRows[0]?.id ?? null);
    setLoadedFrom(`bảng nộp bài (${seedRows.length} dòng)`);
  }, [autoLoadKey, seedRows]);

  function loadFile(file: File) {
    file.text().then((text) => {
      const parsed = parseCsv(text);
      setRows(parsed);
      setSelectedId(parsed[0]?.id ?? null);
      setLoadedFrom(`${file.name} (${parsed.length} dòng)`);
    });
  }

  const changed = rows.filter((row) => row.frame !== row.originalFrame).length;

  if (rows.length === 0) {
    return (
      <div className="results-scroll scroll-y">
        <EmptyState
          icon={<Video size={20} />}
          title="Chưa nạp submission nào"
          description="Tab này để soát lại từng dòng đã lọc: tua video, kéo tới đúng khoảnh khắc, rồi xuất lại CSV."
          hints={[
            "Nạp từ kết quả tìm kiếm hiện tại, hoặc mở file CSV đã xuất trước đó",
            "Phím ← → dịch 1 frame, Shift + ← → dịch 10 frame",
          ]}
        />
        <div className="tuner-actions">
          <Button onClick={loadSeed} disabled={seedRows.length === 0}>
            Nạp từ kết quả hiện tại{seedRows.length ? ` (${seedRows.length})` : ""}
          </Button>
          <Button variant="ghost" onClick={() => fileRef.current?.click()}>
            <FileUp size={13} /> Mở file CSV
          </Button>
          <input
            ref={fileRef} type="file" accept=".csv,text/csv" hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) loadFile(file);
              event.target.value = "";
            }}
          />
        </div>
        {videosError && <InlineError message={`Không lấy được metadata video: ${videosError}`} />}
      </div>
    );
  }

  return (
    <div className="tuner scroll-y">
      <div className="tuner-bar">
        <span className="result-group-meta truncate">Nguồn: {loadedFrom}</span>
        <Badge tone={changed ? "warning" : "neutral"}>{changed} dòng đã chỉnh</Badge>
        <div className="tuner-actions">
          <Button variant="ghost" size="sm" onClick={() => fileRef.current?.click()}>
            <FileUp size={13} /> Đổi file
          </Button>
          <Button
            variant="ghost" size="sm"
            onClick={() => setRows((current) => current.map((row) => ({ ...row, frame: row.originalFrame })))}
            disabled={changed === 0}
          >
            <RotateCcw size={13} /> Hoàn tác tất cả
          </Button>
          <Button size="sm" onClick={() => downloadCsv(toCsv(task, rows), `submission_tuned_${task}.csv`)}>
            <FileDown size={13} /> Xuất CSV
          </Button>
        </div>
        <input
          ref={fileRef} type="file" accept=".csv,text/csv" hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) loadFile(file);
            event.target.value = "";
          }}
        />
      </div>

      {videosError && (
        <InlineError message={`Không lấy được metadata video: ${videosError}. Kiểm tra backend có đang chạy và địa chỉ API ở góc trên.`} />
      )}

      <div className="tuner-body">
        <div className="tuner-list scroll-y">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th><th>Video</th><th className="num">Frame</th><th className="num">Lệch</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => {
                const delta = row.frame - row.originalFrame;
                const info = videos[row.videoId];
                return (
                  <tr
                    key={row.id}
                    onClick={() => setSelectedId(row.id)}
                    className={row.id === selectedId ? "is-selected" : undefined}
                    style={{ cursor: "pointer" }}
                  >
                    <td className="muted-cell">
                      {row.chain ? `C${row.chain}·B${row.step}` : index + 1}
                      {row.placeholder && (
                        <span className="tuner-placeholder-dot" title="Bước chưa có bằng chứng — frame chỉ là điểm giữa hai mốc lân cận, cần chỉnh tay">
                          {" "}●
                        </span>
                      )}
                    </td>
                    <td className="cell-strong">
                      {row.videoId}
                      {info && !info.media_available && (
                        <span title="Dataset có video này nhưng thiếu file mp4 trên đĩa">
                          {" "}<Badge tone="warning">thiếu mp4</Badge>
                        </span>
                      )}
                    </td>
                    <td className="num tabular">{row.frame}</td>
                    <td className={`num tabular ${delta ? "verdict-warn" : "muted-cell"}`}>
                      {delta ? (delta > 0 ? `+${delta}` : delta) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="tuner-stage">
          {!selected ? (
            <EmptyState icon={<Video size={18} />} title="Chọn một dòng" description="Bấm một dòng ở bảng bên trái để soát." />
          ) : !meta ? (
            <InlineError message={`Không có metadata cho ${selected.videoId} — kiểm tra videos.jsonl`} />
          ) : meta.media_available ? (
            <video
              ref={videoRef}
              className="tuner-video"
              src={mediaUrl(apiConfig, meta.media_path)}
              controls
              preload="metadata"
              onTimeUpdate={(event) => {
                // Người dùng tự tua trên thanh của trình phát -> đồng bộ ngược
                // về số frame. Bỏ qua lần đầu ngay sau khi CHÍNH TA vừa seek.
                if (seekingRef.current) { seekingRef.current = false; return; }
                const element = event.currentTarget;
                if (element.paused) setFrame(selected.id, element.currentTime * meta.fps);
              }}
            />
          ) : nearestFrame ? (
            // Thiếu mp4 -> soát bằng ảnh keyframe. Không mượt như video nhưng
            // vẫn NHÌN ĐƯỢC nội dung, và đó là việc chính của tab này.
            <div className="tuner-fallback">
              <img
                className="tuner-video"
                src={mediaUrl(apiConfig, nearestFrame.image_path)}
                alt={`Keyframe ${nearestFrame.frame_idx}`}
              />
              <p className="result-sub">
                <AlertTriangle size={12} /> {selected.videoId} chưa có file mp4 — đang xem bằng
                ảnh <strong>keyframe gần nhất</strong>: frame{" "}
                <strong className="tabular">{nearestFrame.frame_idx}</strong>
                {nearestFrame.frame_idx !== selected.frame && (
                  <> (cách frame đang chọn{" "}
                    <strong className="tabular">
                      {Math.abs(nearestFrame.frame_idx - selected.frame)}
                    </strong>{" "}
                    frame ≈ {(Math.abs(nearestFrame.frame_idx - selected.frame) / meta.fps).toFixed(2)}s)</>
                )}
                . Đặt file vào <code>storage/{meta.media_path}</code> rồi tải lại trang để xem video thật.
              </p>
              <Button
                variant="ghost" size="sm"
                onClick={() => setFrame(selected.id, nearestFrame.frame_idx)}
                disabled={nearestFrame.frame_idx === selected.frame}
              >
                Nhảy tới keyframe {nearestFrame.frame_idx}
              </Button>
            </div>
          ) : (
            <div className="tuner-missing">
              <AlertTriangle size={18} />
              <div>
                <strong>{selected.videoId}: không có mp4 lẫn keyframe</strong>
                <p className="result-sub">
                  Dataset khai <code>{meta.media_path}</code> nhưng file không có trên đĩa, và
                  cũng không nạp được keyframe nào. Vẫn chỉnh được số frame bằng tay ở dưới.
                </p>
              </div>
            </div>
          )}

          {selected && meta && (
            <div className="tuner-controls">
              <div className="tuner-readout">
                <span className="tabular">frame <strong>{selected.frame}</strong></span>
                <span className="tabular muted-cell">
                  {(selected.frame / meta.fps).toFixed(3)}s · {meta.fps} fps · {meta.frame_count} frame
                </span>
                {selected.frame !== selected.originalFrame && (
                  <Button
                    variant="ghost" size="sm"
                    onClick={() => setFrame(selected.id, selected.originalFrame)}
                  >
                    <RotateCcw size={12} /> về {selected.originalFrame}
                  </Button>
                )}
              </div>

              {/* Ruy băng thời gian: ảnh keyframe trải theo trục, kèm đầu đọc
                  kéo được — bấm hoặc kéo bất kỳ đâu để nhảy tới frame đó. */}
              <div
                className="tuner-timeline"
                onPointerDown={(event) => {
                  event.currentTarget.setPointerCapture(event.pointerId);
                  scrubTo(event, meta.frame_count, selected.id);
                }}
                onPointerMove={(event) => {
                  if (event.buttons === 1) scrubTo(event, meta.frame_count, selected.id);
                }}
                role="slider"
                aria-label="Kéo để chọn frame"
                aria-valuemin={0}
                aria-valuemax={meta.frame_count - 1}
                aria-valuenow={selected.frame}
                tabIndex={0}
              >
                <div className="tuner-film">
                  {filmstrip.map((frame) => (
                    <img
                      key={frame.frame_idx}
                      src={mediaUrl(apiConfig, frame.image_path)}
                      alt=""
                      draggable={false}
                      loading="lazy"
                    />
                  ))}
                </div>
                <div
                  className="tuner-playhead"
                  style={{ left: `${(selected.frame / Math.max(1, meta.frame_count - 1)) * 100}%` }}
                />
              </div>

              <input
                className="tuner-slider"
                type="range"
                min={0}
                max={Math.max(0, meta.frame_count - 1)}
                step={1}
                value={selected.frame}
                onChange={(event) => setFrame(selected.id, Number(event.target.value))}
                aria-label="Thanh kéo frame (chính xác từng frame)"
              />

              <div className="tuner-nudges">
                {NUDGES.map((delta) => (
                  <Button
                    key={delta} variant="ghost" size="sm"
                    onClick={() => setFrame(selected.id, selected.frame + delta)}
                  >
                    {delta > 0 ? `+${delta}` : delta}
                  </Button>
                ))}
                <input
                  className="tuner-number tabular"
                  type="number"
                  min={0}
                  max={meta.frame_count - 1}
                  value={selected.frame}
                  onChange={(event) => setFrame(selected.id, Number(event.target.value))}
                  aria-label="Nhập số frame chính xác"
                />
              </div>
              <p className="result-sub">Phím ← → dịch 1 frame · Shift + ← → dịch 10 frame</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
