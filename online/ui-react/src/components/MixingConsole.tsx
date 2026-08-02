import { useEffect, useState } from "react";
import type { ApiClientConfig } from "../api";
import { ApiError, getCapabilities } from "../api";
import type { BranchRuntimeOptions, CapabilitiesResponse, SearchOptions } from "../types";

export interface MixingConsoleProps {
  apiConfig: ApiClientConfig;
  searchOptions: SearchOptions;
  onChange: (options: SearchOptions) => void;
}

/**
 * Search Mixing Console — render HOÀN TOÀN từ GET /v1/search/capabilities,
 * không hard-code branch nào (nguyên tắc docs 01082026 §16.1 "UI đọc
 * capabilities"). Branch không có trong `capabilities.branches` thì không
 * render control nào cho nó — không có control giả.
 */
export function MixingConsole({ apiConfig, searchOptions, onChange }: MixingConsoleProps) {
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getCapabilities(apiConfig)
      .then((result) => {
        if (!cancelled) setCapabilities(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [apiConfig]);

  function updateBranch(branchId: string, patch: Partial<BranchRuntimeOptions>) {
    const branches = { ...(searchOptions.branches ?? {}) };
    branches[branchId] = { ...branches[branchId], ...patch };
    onChange({ ...searchOptions, branches });
  }

  if (error) return <p className="muted">Không tải được capabilities: {error}</p>;
  if (!capabilities) return <p className="muted">Đang tải danh sách branch…</p>;

  return (
    <div className="mixing-console">
      <table className="mixing-table">
        <thead>
          <tr>
            <th>Branch</th>
            <th>Backend</th>
            <th>Bật</th>
            <th>Weight</th>
            <th>Top-k</th>
            <th>Timeout (ms)</th>
          </tr>
        </thead>
        <tbody>
          {capabilities.branches.map((branch) => {
            const override = searchOptions.branches?.[branch.branch_id] ?? {};
            const controls = new Set(branch.supported_controls);
            return (
              <tr key={branch.branch_id} className={branch.degraded ? "row-degraded" : undefined}>
                <td>
                  <strong>{branch.branch_id}</strong>
                  {branch.degraded && (
                    <span className="degraded-badge" title={branch.degraded_reason ?? ""}>
                      degraded
                    </span>
                  )}
                  <div className="muted small">{branch.modality ?? "—"}</div>
                </td>
                <td>{branch.backend_kind}</td>
                <td>
                  <input
                    type="checkbox"
                    checked={override.enabled ?? true}
                    disabled={!controls.has("enabled")}
                    onChange={(e) => updateBranch(branch.branch_id, { enabled: e.target.checked })}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    step={0.1}
                    className="narrow"
                    value={override.weight ?? ""}
                    placeholder="mặc định"
                    disabled={!controls.has("weight")}
                    onChange={(e) =>
                      updateBranch(branch.branch_id, {
                        weight: e.target.value === "" ? undefined : Number(e.target.value),
                      })
                    }
                  />
                </td>
                <td>
                  <input
                    type="number"
                    min={1}
                    className="narrow"
                    value={override.top_k ?? ""}
                    placeholder="mặc định"
                    disabled={!controls.has("top_k")}
                    onChange={(e) =>
                      updateBranch(branch.branch_id, {
                        top_k: e.target.value === "" ? undefined : Number(e.target.value),
                      })
                    }
                  />
                </td>
                <td>
                  <input
                    type="number"
                    min={100}
                    step={100}
                    className="narrow"
                    value={override.timeout_ms ?? ""}
                    placeholder="3000"
                    disabled={!controls.has("timeout_ms")}
                    onChange={(e) =>
                      updateBranch(branch.branch_id, {
                        timeout_ms: e.target.value === "" ? undefined : Number(e.target.value),
                      })
                    }
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <details className="unsupported-options">
        <summary>{Object.keys(capabilities.unsupported_options).length} option chưa chạy thật (server sẽ trả 422 nếu bật)</summary>
        <ul>
          {Object.entries(capabilities.unsupported_options).map(([path, reason]) => (
            <li key={path}>
              <code>{path}</code>: {reason}
            </li>
          ))}
        </ul>
      </details>
      <p className="muted small">
        Rerank: rules={String(capabilities.rerank.rules)} · text={String(capabilities.rerank.text)} · vlm=
        {String(capabilities.rerank.vlm)}
      </p>
    </div>
  );
}
