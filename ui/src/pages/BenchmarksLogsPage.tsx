import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play, Search, Terminal, Trash2, Zap } from 'lucide-react';
import { API_BASE, studioApi } from '../api/client';
import type { BenchmarkHistoryVersion, LogEntry, QuickBenchmarkResult } from '../types/api';

const VERSION_COLORS: Record<string, string> = {
  'v0.14.0': 'var(--td-ink-3)',
  'v0.16.0': 'var(--td-ink-2)',
  'v0.16.1': 'var(--td-blue)',
  'v0.16.2': 'var(--td-green)',
  'v0.16.3': 'var(--td-accent)',
};

export const BenchmarksLogsPage: React.FC = () => {
  const [history, setHistory] = useState<BenchmarkHistoryVersion[]>([]);
  const [liveRun, setLiveRun] = useState<QuickBenchmarkResult | null>(null);
  const [runningBench, setRunningBench] = useState(false);
  const [benchError, setBenchError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('qwen3.6-35b-a3b');
  const [metric, setMetric] = useState<'tps' | 'ttft'>('tps');

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logSource, setLogSource] = useState<'all' | 'engine' | 'server'>('all');
  const [logFilter, setLogFilter] = useState('');
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    void studioApi.getBenchmarkHistory().then((res) => {
      setHistory(res.versions);
      if (res.last_live_run) setLiveRun(res.last_live_run);
    });
  }, []);

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/ui/logs/stream?replay=120`);
    es.onmessage = (ev) => {
      try {
        const entry = JSON.parse(ev.data) as LogEntry;
        setLogs((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].seq >= entry.seq) return prev;
          const next = [...prev, entry];
          return next.length > 500 ? next.slice(-500) : next;
        });
      } catch {
        // ignore keepalive
      }
    };
    return () => es.close();
  }, []);

  useEffect(() => {
    if (!paused) {
      logEndRef.current?.scrollIntoView({ behavior: 'auto' });
    }
  }, [logs, paused]);

  const handleRunSpeedTest = async () => {
    setRunningBench(true);
    setBenchError(null);
    try {
      const res = await studioApi.runBenchmark({ max_tokens: 64 });
      setLiveRun(res);
    } catch (err) {
      setBenchError(err instanceof Error ? err.message : 'Speed test failed');
    } finally {
      setRunningBench(false);
    }
  };

  const chartWorkloads = useMemo(() => {
    const ids = ['workload-1-refactor', 'workload-2-greenfield', 'workload-3-tool-calling'];
    const labels: Record<string, string> = {
      'workload-1-refactor': 'W1 · Code Refactor',
      'workload-2-greenfield': 'W2 · Greenfield',
      'workload-3-tool-calling': 'W3 · Tool Stream',
    };
    return ids.map((wid) => ({
      wid,
      label: labels[wid],
      bars: history.map((ver) => {
        const mod = ver.models[selectedModel];
        const wl = mod?.workloads.find((w) => w.workload_id === wid);
        const val =
          metric === 'tps'
            ? (wl?.claude_code?.best_decode_tps ?? wl?.opencode?.best_decode_tps ?? 0)
            : (wl?.opencode?.cold_ttft_ms ?? 0);
        return { version: ver.label, value: val };
      }),
    }));
  }, [history, selectedModel, metric]);

  const maxChartVal = Math.max(1, ...chartWorkloads.flatMap((w) => w.bars.map((b) => b.value)));

  const filteredLogs = useMemo(() => {
    const q = logFilter.trim().toLowerCase();
    return logs.filter((l) => {
      if (logSource !== 'all' && l.source !== logSource) return false;
      if (q && !l.line.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [logs, logSource, logFilter]);

  return (
    <div className="p-7 space-y-6 max-w-[1440px] mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-[24px] font-semibold font-display tracking-tight" style={{ color: 'var(--td-ink)' }}>
            Benchmarks &amp; Console
          </h1>
          <p className="text-[13px] mt-0.5" style={{ color: 'var(--td-ink-3)' }}>
            On-demand Cold/Warm speed verification, release progression, and live engine telemetry stream.
          </p>
        </div>

        <button
          type="button"
          onClick={() => void handleRunSpeedTest()}
          disabled={runningBench}
          className="td-btn-primary px-4 py-2 rounded-full text-xs font-medium inline-flex items-center gap-1.5 disabled:opacity-40 cursor-pointer"
        >
          <Zap className="w-3.5 h-3.5" />
          {runningBench ? 'Running Speed Test…' : 'Run Speed Test'}
        </button>
      </div>

      {benchError && (
        <div
          className="px-4 py-2.5 rounded-xl text-xs"
          style={{ background: 'var(--td-red-soft)', border: '1px solid var(--td-border)', color: 'var(--td-red)' }}
        >
          {benchError}
        </div>
      )}

      {liveRun && (
        <div
          className="td-card p-5 grid grid-cols-2 lg:grid-cols-5 gap-4 text-xs font-mono-num"
          style={{ borderColor: 'var(--td-accent)' }}
        >
          <div>
            <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Model</div>
            <div className="font-semibold mt-0.5 truncate" style={{ color: 'var(--td-ink)' }}>{liveRun.model_id}</div>
            <div className="text-[10.5px]" style={{ color: 'var(--td-accent)' }}>v{liveRun.version}</div>
          </div>
          <div>
            <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Cold TTFT</div>
            <div className="text-[18px] font-semibold mt-0.5" style={{ color: 'var(--td-blue)' }}>
              {liveRun.cold.ttft_ms.toFixed(0)} ms
            </div>
          </div>
          <div>
            <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Warm TTFT</div>
            <div className="text-[18px] font-semibold mt-0.5" style={{ color: 'var(--td-green)' }}>
              {liveRun.warm.ttft_ms.toFixed(0)} ms{' '}
              <span className="text-xs font-normal" style={{ color: 'var(--td-ink-3)' }}>
                ({liveRun.ttft_speedup_x}×)
              </span>
            </div>
          </div>
          <div>
            <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Warm Decode</div>
            <div className="text-[18px] font-semibold mt-0.5" style={{ color: 'var(--td-accent)' }}>
              {liveRun.warm.decode_tps.toFixed(1)} tok/s
            </div>
          </div>
          <div>
            <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>VRAM Peak</div>
            <div className="text-[18px] font-semibold mt-0.5" style={{ color: 'var(--td-ink)' }}>
              {(liveRun.vram_peak_mb / 1024).toFixed(2)} GiB
            </div>
          </div>
        </div>
      )}

      {/* Historical Version Comparison */}
      <div className="td-card p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-[14px] font-semibold font-display" style={{ color: 'var(--td-ink)' }}>
              Verified Release Progression
            </h2>
            <p className="text-[11.5px] mt-0.5" style={{ color: 'var(--td-ink-3)' }}>
              Hardware-verified reports from <code>docs/json/benchmark_*_kubuntu_report.json</code>
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="rounded-full px-3 py-1.5 text-xs font-mono-num focus:outline-none"
              style={{ background: 'var(--td-surface-2)', border: '1px solid var(--td-border)', color: 'var(--td-ink)' }}
            >
              <option value="qwen3.6-35b-a3b">Qwen3.6-35B-A3B (16.5 GiB)</option>
              <option value="qwen3.8-flash-next">Qwen3.8-Flash-Next (75.8 GB)</option>
            </select>
            <div
              className="inline-flex rounded-full p-0.5"
              style={{ background: 'var(--td-surface-2)', border: '1px solid var(--td-border)' }}
            >
              <button
                type="button"
                onClick={() => setMetric('tps')}
                className="px-3 py-1 rounded-full text-[11px] font-medium cursor-pointer transition-colors"
                style={{
                  background: metric === 'tps' ? 'var(--td-surface)' : 'transparent',
                  color: metric === 'tps' ? 'var(--td-ink)' : 'var(--td-ink-3)',
                }}
              >
                Decode TPS
              </button>
              <button
                type="button"
                onClick={() => setMetric('ttft')}
                className="px-3 py-1 rounded-full text-[11px] font-medium cursor-pointer transition-colors"
                style={{
                  background: metric === 'ttft' ? 'var(--td-surface)' : 'transparent',
                  color: metric === 'ttft' ? 'var(--td-ink)' : 'var(--td-ink-3)',
                }}
              >
                Cold TTFT
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {chartWorkloads.map((wl) => (
            <div key={wl.wid} className="td-card-subtle p-4 space-y-3">
              <div className="text-[12.5px] font-semibold" style={{ color: 'var(--td-ink)' }}>{wl.label}</div>
              {wl.bars.map((b) => {
                const pct = Math.min(100, (b.value / maxChartVal) * 100);
                return (
                  <div key={b.version} className="space-y-1">
                    <div className="flex justify-between text-[11px] font-mono-num">
                      <span style={{ color: 'var(--td-ink-3)' }}>{b.version}</span>
                      <span className="font-medium" style={{ color: 'var(--td-ink)' }}>
                        {metric === 'tps' ? `${b.value.toFixed(1)} tok/s` : `${b.value.toFixed(0)} ms`}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--td-surface-3)' }}>
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{ width: `${pct}%`, backgroundColor: VERSION_COLORS[b.version] ?? 'var(--td-accent)' }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Live SSE Console Logs */}
      <div className="td-card overflow-hidden">
        <div
          className="px-5 py-3 flex flex-wrap items-center justify-between gap-2"
          style={{ borderBottom: '1px solid var(--td-border)', background: 'var(--td-surface-2)' }}
        >
          <div className="flex items-center gap-2 text-[12.5px] font-semibold" style={{ color: 'var(--td-ink)' }}>
            <Terminal className="w-4 h-4" style={{ color: 'var(--td-accent)' }} />
            Live Engine &amp; Server Console ({filteredLogs.length})
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search
                className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
                style={{ color: 'var(--td-ink-3)' }}
              />
              <input
                value={logFilter}
                onChange={(e) => setLogFilter(e.target.value)}
                placeholder="Filter logs…"
                className="pl-8 pr-3 py-1 rounded-full text-xs focus:outline-none"
                style={{ background: 'var(--td-surface)', border: '1px solid var(--td-border)', color: 'var(--td-ink)' }}
              />
            </div>
            <select
              value={logSource}
              onChange={(e) => setLogSource(e.target.value as 'all' | 'engine' | 'server')}
              className="rounded-full px-2.5 py-1 text-xs font-mono-num focus:outline-none"
              style={{ background: 'var(--td-surface)', border: '1px solid var(--td-border)', color: 'var(--td-ink-2)' }}
            >
              <option value="all">All</option>
              <option value="engine">Engine</option>
              <option value="server">Server</option>
            </select>
            <button
              type="button"
              onClick={() => setPaused((p) => !p)}
              className="td-btn-secondary px-2.5 py-1 rounded-full text-xs inline-flex items-center gap-1 cursor-pointer"
            >
              {paused ? <Play className="w-3 h-3" style={{ color: 'var(--td-green)' }} /> : <Pause className="w-3 h-3" style={{ color: 'var(--td-accent)' }} />}
              {paused ? 'Resume' : 'Pause'}
            </button>
            <button
              type="button"
              onClick={() => setLogs([])}
              className="td-btn-secondary p-1.5 rounded-full cursor-pointer"
              title="Clear console view"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div
          className="h-72 overflow-y-auto p-4 font-mono-num text-[11px] space-y-1"
          style={{ background: 'var(--td-code-bg)' }}
        >
          {filteredLogs.length === 0 && (
            <div style={{ color: 'var(--td-ink-3)' }}>Waiting for engine &amp; server logs…</div>
          )}
          {filteredLogs.map((l) => (
            <div key={l.seq} className="flex gap-2.5 leading-snug">
              <span
                className="shrink-0 uppercase text-[9.5px] px-1.5 py-0.5 rounded-md font-medium"
                style={{
                  background: l.source === 'engine' ? 'var(--td-blue-soft)' : 'var(--td-accent-soft)',
                  color: l.source === 'engine' ? 'var(--td-blue)' : 'var(--td-accent)',
                }}
              >
                {l.source === 'engine' ? 'ENG' : 'API'}
              </span>
              <span className="break-all" style={{ color: 'var(--td-ink-2)' }}>{l.line}</span>
            </div>
          ))}
          <div ref={logEndRef} />
        </div>
      </div>
    </div>
  );
};
