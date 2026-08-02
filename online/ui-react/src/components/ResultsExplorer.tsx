import { useState } from "react";
import type { ApiClientConfig } from "../api";
import type { SearchHit, SequenceHit } from "../types";
import { EvidenceInspector } from "./EvidenceInspector";
import { ResultCard } from "./ResultCard";

export interface ResultsExplorerProps {
  results: SearchHit[];
  sequences: SequenceHit[];
  apiConfig: ApiClientConfig;
}

/** Results Explorer — hiển thị chung cho mọi task, không riêng KIS/QA/TRAKE/
 * AVS (các workspace riêng nằm ở TaskWorkspaces.tsx). */
export function ResultsExplorer({ results, sequences, apiConfig }: ResultsExplorerProps) {
  const [inspecting, setInspecting] = useState<string | null>(null);

  if (sequences.length > 0) {
    return (
      <div className="explorer-layout">
        <div className="sequence-mode">
          {sequences.map((sequence, i) => (
            <section className="sequence" key={i}>
              <h3>
                Chuỗi {i + 1} · {sequence.video_id} · frame_ids [{sequence.frame_ids.join(", ")}]
              </h3>
              <div className="card-grid">
                {sequence.scenes.map((hit) => (
                  <ResultCard key={hit.candidate_id} hit={hit} onInspect={setInspecting} apiConfig={apiConfig} />
                ))}
              </div>
            </section>
          ))}
        </div>
        {inspecting && (
          <EvidenceInspector apiConfig={apiConfig} candidateId={inspecting} onClose={() => setInspecting(null)} />
        )}
      </div>
    );
  }

  if (results.length === 0) return <p className="muted">Chưa có kết quả — chạy tìm kiếm ở Query Studio.</p>;

  return (
    <div className="explorer-layout">
      <div className="card-grid">
        {results.map((hit) => (
          <ResultCard key={hit.candidate_id} hit={hit} onInspect={setInspecting} apiConfig={apiConfig} />
        ))}
      </div>
      {inspecting && (
        <EvidenceInspector apiConfig={apiConfig} candidateId={inspecting} onClose={() => setInspecting(null)} />
      )}
    </div>
  );
}
