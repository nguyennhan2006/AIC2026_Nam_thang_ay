import { SearchX } from "lucide-react";
import type { ApiClientConfig } from "../api";
import type { SearchHit, SequenceHit } from "../types";
import { EmptyState } from "../ui";
import { ResultCard } from "./ResultCard";

export interface ResultsExplorerProps {
  results: SearchHit[];
  sequences: SequenceHit[];
  apiConfig: ApiClientConfig;
  selectedCandidateId: string | null;
  onSelect: (candidateId: string) => void;
  /**
   * TRAKE: bấm vào một thẻ thì báo luôn thẻ đó thuộc CHUỖI nào, BƯỚC nào.
   *
   * Không có callback này thì lưới và panel xem chạy trên hai state rời nhau:
   * lưới đổi `selectedCandidateId`, còn panel đọc `trake[selectedSequenceIndex]`
   * mà `selectedSequenceIndex` chẳng ai đụng tới — nên bấm chuỗi nào panel cũng
   * hiện chuỗi 1. Dữ liệu backend vẫn khớp 10/10; lỗi hoàn toàn ở tầng UI.
   */
  onSelectSequenceStep?: (sequenceIndex: number, stepIndex: number) => void;
  /** Chưa từng search lần nào — khác với "đã search nhưng không ra gì". */
  pristine: boolean;
  topK: number;
  /** Người dùng đã tự đặt fusion.max_results_per_video chưa. */
  perVideoCapSet: boolean;
}

export function ResultsExplorer({
  results,
  sequences,
  apiConfig,
  selectedCandidateId,
  onSelect,
  onSelectSequenceStep,
  pristine,
  topK,
  perVideoCapSet,
}: ResultsExplorerProps) {
  if (sequences.length > 0) {
    return (
      <div className="results-scroll scroll-y">
        {sequences.map((sequence, index) => (
          <section className="result-group" key={`${sequence.video_id}-${index}`}>
            <header className="result-group-head">
              <span className="eyebrow">Chuỗi {index + 1}</span>
              <span className="result-group-meta truncate">
                {sequence.video_id} · frames [{sequence.frame_ids.join(", ")}]
              </span>
            </header>
            <div className="result-grid">
              {sequence.scenes.map((hit, stepIndex) => (
                <ResultCard
                  key={hit.candidate_id}
                  hit={hit}
                  apiConfig={apiConfig}
                  onInspect={(candidateId) => {
                    onSelect(candidateId);
                    // Vị trí trong `scenes` CHÍNH LÀ số thứ tự bước của chuỗi,
                    // nên bấm thẻ thứ 3 của chuỗi 5 là chọn đúng chuỗi 5 bước 3.
                    onSelectSequenceStep?.(index, stepIndex);
                  }}
                  selected={hit.candidate_id === selectedCandidateId}
                />
              ))}
            </div>
          </section>
        ))}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <EmptyState
        icon={<SearchX size={20} />}
        title={pristine ? "Chưa có tìm kiếm nào" : "Không có kết quả khớp"}
        description={
          pristine
            ? "Nhập truy vấn ở ô phía trên rồi bấm Tìm kiếm. Kết quả sẽ hiện ở đây kèm đóng góp của từng nhánh retrieval."
            : "Truy vấn chạy xong nhưng không nhánh nào trả về candidate nào."
        }
        hints={
          pristine
            ? ["Chọn task phù hợp: KIS tìm 1 khung hình, AVS tìm nhiều đoạn", "Ctrl+Enter để tìm nhanh"]
            : ["Thử bỏ bớt từ khoá quá cụ thể", "Tăng Top-K hoặc nới Max/video ở panel Trọng số", "Kiểm tra caption/OCR đã được sinh cho dataset chưa"]
        }
      />
    );
  }

  // Trả về ít hơn top_k thường KHÔNG phải vì hết candidate mà vì dedup giới
  // hạn số kết quả mỗi video (KIS mặc định 5). Nói thẳng ra chỗ này thay vì
  // để người dùng nhìn khoảng trống rồi tưởng search hỏng.
  const cappedByDedup = results.length < topK && !perVideoCapSet;

  return (
    <div className="results-scroll scroll-y">
      <div className="result-grid">
        {results.map((hit) => (
          <ResultCard
            key={hit.candidate_id}
            hit={hit}
            apiConfig={apiConfig}
            onInspect={onSelect}
            selected={hit.candidate_id === selectedCandidateId}
          />
        ))}
      </div>

      {results.length < topK && (
        <p className="results-note">
          Hiển thị <strong className="tabular">{results.length}</strong>/{topK} kết quả yêu cầu.
          {cappedByDedup && " Dedup theo task giới hạn số kết quả mỗi video — đặt “Max / video” ở panel Trọng số để nới."}
        </p>
      )}
    </div>
  );
}
