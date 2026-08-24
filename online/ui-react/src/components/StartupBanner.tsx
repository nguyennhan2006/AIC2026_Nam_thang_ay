import { useEffect, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import type { ApiClientConfig } from "../api";
import { startupState } from "../api";
import type { StartupState } from "../types";

/** Tên chặng của backend -> câu người đọc hiểu được.
 *
 * Chặng thô ("vectors", "encoder") không nói cho ai biết còn bao lâu; câu ở
 * đây nói ra thứ đang thực sự tốn thời gian, để người chờ biết là hệ đang
 * chạy chứ không phải đã treo. */
const PHASE_TEXT: Record<string, string> = {
  starting: "đang khởi động",
  metadata: "đọc metadata scene",
  lexical: "dựng chỉ mục BM25",
  vectors: "nạp ma trận vector ảnh",
  encoder: "nạp model text encoder",
  events: "nạp sự kiện + rerank",
  ready: "sẵn sàng",
  failed: "hỏng",
  unknown: "đang khởi động",
};

const POLL_MS = 2000;

export interface StartupBannerProps {
  apiConfig: ApiClientConfig;
  /** Gọi đúng MỘT lần khi server chuyển sang sẵn sàng. */
  onReady?: () => void;
}

/** Thanh báo "server đang nạp".
 *
 * Nạp corpus thi đấu mất ~4 phút. Trước đây uvicorn chưa mở cổng suốt quãng
 * đó nên trình duyệt chỉ báo "không kết nối được" — không ai phân biệt được
 * "đang nạp" với "đã chết", và cách duy nhất để biết là ngồi nhìn log.
 *
 * Không render gì khi server đã sẵn sàng: thanh này là trạng thái tạm, không
 * phải một widget thường trực.
 */
export function StartupBanner({ apiConfig, onReady }: StartupBannerProps) {
  const [state, setState] = useState<StartupState | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let notified = false;

    async function poll() {
      try {
        const next = await startupState(apiConfig);
        if (cancelled) return;
        setState(next);
        setUnreachable(false);
        if (next.status === "ready") {
          if (!notified) {
            notified = true;
            onReady?.();
          }
          return; // Sẵn sàng rồi thì thôi hỏi — không để một vòng lặp treo mãi.
        }
      } catch {
        if (cancelled) return;
        // Chưa gọi được /v1/startup nghĩa là tiến trình chưa lên (hoặc sai
        // API base) — KHÁC với "đang nạp", và phải nói khác đi.
        setUnreachable(true);
      }
      if (!cancelled) timer = setTimeout(poll, POLL_MS);
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiConfig]);

  if (state?.status === "ready") return null;
  if (!state && !unreachable) return null;

  if (state?.status === "failed") {
    return (
      <div className="startup-banner is-failed" role="alert">
        <AlertTriangle size={14} />
        <span>
          Server nạp thất bại — <strong>{state.error ?? "không rõ nguyên nhân"}</strong>. Sửa cấu
          hình rồi khởi động lại; bấm lại ở đây không cứu được.
        </span>
      </div>
    );
  }

  if (unreachable) {
    return (
      <div className="startup-banner is-failed" role="status" aria-live="polite">
        <AlertTriangle size={14} />
        <span>
          Chưa gọi được <code>{apiConfig.base}</code> — tiến trình server chưa lên, hoặc API base
          đang trỏ sai. Đây KHÔNG phải "đang nạp".
        </span>
      </div>
    );
  }

  const phase = PHASE_TEXT[state?.phase ?? "starting"] ?? state?.phase ?? "đang khởi động";
  return (
    <div className="startup-banner" role="status" aria-live="polite">
      <Loader2 size={14} className="startup-spinner" />
      <span>
        Server đang nạp: <strong>{phase}</strong>
        {" · "}
        <span className="tabular">{(state?.elapsed_sec ?? 0).toFixed(0)}s</span>
      </span>
      <span className="startup-note">
        Tìm kiếm bấm được ngay — truy vấn sẽ tự chạy khi nạp xong, không cần bấm lại.
      </span>
    </div>
  );
}
