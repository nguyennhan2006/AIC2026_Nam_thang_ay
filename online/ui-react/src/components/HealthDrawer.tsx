import { useEffect, useState } from "react";
import type { ApiClientConfig } from "../api";
import { ApiError, getCapabilities, health } from "../api";
import type { CapabilitiesResponse, HealthResponse } from "../types";

export function HealthDrawer({ apiConfig }: { apiConfig: ApiClientConfig }) {
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setError(null);
    try {
      const [h, c] = await Promise.all([health(apiConfig), getCapabilities(apiConfig)]);
      setHealthData(h);
      setCapabilities(c);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiConfig]);

  const degraded = capabilities?.branches.filter((item) => item.degraded) ?? [];

  return (
    <div className="health-drawer">
      <button type="button" onClick={refresh}>
        Làm mới
      </button>
      {error && <p className="muted">Không kết nối được: {error}</p>}
      {healthData && (
        <ul className="health-list">
          <li>status: {healthData.status}</li>
          <li>backend: {healthData.backend}</li>
          <li>scene_count: {healthData.scene_count}</li>
          <li>dataset: {healthData.dataset}</li>
        </ul>
      )}
      {capabilities && (
        <>
          <p>
            {capabilities.branches.length} branch đang đăng ký · rerank rules={String(capabilities.rerank.rules)} text=
            {String(capabilities.rerank.text)} vlm={String(capabilities.rerank.vlm)}
          </p>
          {degraded.length > 0 && (
            <div className="warning-text">
              <p>{degraded.length} branch degraded:</p>
              <ul>
                {degraded.map((item) => (
                  <li key={item.branch_id}>
                    {item.branch_id}: {item.degraded_reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="muted small">events_available: {String(capabilities.events_available)}</p>
        </>
      )}
    </div>
  );
}
