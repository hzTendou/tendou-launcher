import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE, studioApi } from '../api/client';
import type { StudioState, TelemetrySnapshot } from '../types/api';

export interface TelemetryHookState {
  studio: StudioState | null;
  telemetry: TelemetrySnapshot | null;
  connected: boolean;
  error: string | null;
  refreshState: () => Promise<void>;
}

export function useTelemetry(): TelemetryHookState {
  const [studio, setStudio] = useState<StudioState | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const retryTimer = useRef<number | null>(null);

  const refreshState = useCallback(async () => {
    try {
      const st = await studioApi.getState();
      setStudio(st);
      setTelemetry((prev) => prev ?? st.telemetry);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to reach Tendou Studio backend');
    }
  }, []);

  useEffect(() => {
    void refreshState();
    const stateInterval = window.setInterval(() => {
      void refreshState();
    }, 4000);
    return () => window.clearInterval(stateInterval);
  }, [refreshState]);

  useEffect(() => {
    let closed = false;
    let es: EventSource | null = null;

    const connect = () => {
      if (closed) return;
      es = new EventSource(`${API_BASE}/api/ui/telemetry/stream`);
      es.onopen = () => {
        setConnected(true);
        setError(null);
      };
      es.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data) as TelemetrySnapshot;
          setTelemetry(parsed);
          setConnected(true);
        } catch {
          // ignore malformed event
        }
      };
      es.onerror = () => {
        setConnected(false);
        if (es) {
          es.close();
          es = null;
        }
        if (!closed) {
          retryTimer.current = window.setTimeout(connect, 1500);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retryTimer.current !== null) window.clearTimeout(retryTimer.current);
      if (es) es.close();
    };
  }, []);

  return { studio, telemetry, connected, error, refreshState };
}
