import type { QueryPlan } from "../types";

export function DebugPanel({ queryPlan }: { queryPlan: QueryPlan | null }) {
  if (!queryPlan) return null;
  return (
    <details id="debug-panel">
      <summary>Query plan</summary>
      <pre id="debug-output">{JSON.stringify(queryPlan, null, 2)}</pre>
    </details>
  );
}
