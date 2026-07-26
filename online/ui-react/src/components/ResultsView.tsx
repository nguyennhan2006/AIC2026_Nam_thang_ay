import type { ApiClientConfig } from "../api";
import type { SceneDocument, SearchHit, SearchResponse, Task, VQAResponse } from "../types";
import { SceneCard } from "./SceneCard";

export interface ResultsViewProps {
  task: Task;
  searchResult: SearchResponse | null;
  vqaResult: VQAResponse | null;
  selection: Map<string, unknown>;
  onToggleSelect: (hit: SearchHit, checked: boolean) => void;
  apiConfig: ApiClientConfig;
  sceneDetailCache: Map<string, Promise<SceneDocument>>;
}

function Grid({
  hits,
  selection,
  onToggleSelect,
  apiConfig,
  sceneDetailCache,
}: {
  hits: SearchHit[];
  selection: Map<string, unknown>;
  onToggleSelect: (hit: SearchHit, checked: boolean) => void;
  apiConfig: ApiClientConfig;
  sceneDetailCache: Map<string, Promise<SceneDocument>>;
}) {
  return (
    <div className="card-grid">
      {hits.map((hit, index) => (
        <SceneCard
          key={hit.scene_id}
          hit={hit}
          index={index}
          selected={selection.has(hit.scene_id)}
          onToggleSelect={onToggleSelect}
          apiConfig={apiConfig}
          sceneDetailCache={sceneDetailCache}
        />
      ))}
    </div>
  );
}

export function ResultsView({ task, searchResult, vqaResult, selection, onToggleSelect, apiConfig, sceneDetailCache }: ResultsViewProps) {
  if (task === "vqa") {
    if (!vqaResult) return null;
    return (
      <div className="sequence-mode">
        <article className="answer">
          <h2>Trả lời</h2>
          <pre>{vqaResult.answer}</pre>
        </article>
        <Grid hits={vqaResult.evidence} selection={selection} onToggleSelect={onToggleSelect} apiConfig={apiConfig} sceneDetailCache={sceneDetailCache} />
      </div>
    );
  }

  if (!searchResult) return null;

  if (task === "sequence") {
    return (
      <div className="sequence-mode">
        {searchResult.sequences.map((sequence, i) => (
          <section className="sequence" key={i}>
            <h2>
              Chuỗi {i + 1} · {sequence.video_id}
            </h2>
            <Grid hits={sequence.scenes} selection={selection} onToggleSelect={onToggleSelect} apiConfig={apiConfig} sceneDetailCache={sceneDetailCache} />
          </section>
        ))}
      </div>
    );
  }

  return (
    <Grid hits={searchResult.results} selection={selection} onToggleSelect={onToggleSelect} apiConfig={apiConfig} sceneDetailCache={sceneDetailCache} />
  );
}
