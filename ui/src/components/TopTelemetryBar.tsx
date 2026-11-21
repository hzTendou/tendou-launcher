import React from 'react';
import { Play, RotateCw, Square } from 'lucide-react';
import type { EngineStatus, StudioState, TelemetrySnapshot } from '../types/api';

interface TopTelemetryBarProps {
  studio: StudioState | null;
  telemetry: TelemetrySnapshot | null;
  connected: boolean;
  busyAction: string | null;
  onStart: () => void;
  onRestart: () => void;
  onStop: () => void;
}

function HairlineSparkline({ points }: { points: Array<{ tps: number; generating: boolean }> }) {
  const width = 76;
  const height = 22;
  const slice = points.slice(-36);
  if (slice.length < 2) {
    return (
      <svg width={width} height={height} className="overflow-visible">
        <line
          x1={0}
          y1={height - 4}
          x2={width}
          y2={height - 4}
          stroke="var(--td-border-strong)"
          strokeWidth={1.25}
          strokeLinecap="round"
        />
      </svg>
    );
  }
  const maxVal = Math.max(20, ...slice.map((p) => p.tps));
  const stepX = width / Math.max(1, slice.length - 1);
  const coords = slice.map((p, i) => {
    const x = i * stepX;
    const y = height - 2 - (Math.min(p.tps, maxVal) / maxVal) * (height - 5);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        fill="none"
        stroke="var(--td-accent)"
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
        points={coords.join(' ')}
      />
    </svg>
  );
}

const STATUS_META: Record<EngineStatus, { label: string; dot: string; tone: string }> = {
  ready: { label: 'Online', dot: 'var(--td-green)', tone: 'var(--td-green)' },
  starting: { label: 'Starting', dot: 'var(--td-accent)', tone: 'var(--td-accent)' },
  restarting: { label: 'Hot-Swap', dot: 'var(--td-blue)', tone: 'var(--td-blue)' },
  stopped: { label: 'Standby', dot: 'var(--td-ink-3)', tone: 'var(--td-ink-3)' },
  error: { label: 'Error', dot: 'var(--td-red)', tone: 'var(--td-red)' },
};

export const TopTelemetryBar: React.FC<TopTelemetryBarProps> = ({
  studio,
  telemetry,
  connected,
  busyAction,
  onStart,
  onRestart,
  onStop,
}) => {
  const status: EngineStatus = telemetry?.engine_status ?? studio?.status ?? 'stopped';
  const meta = STATUS_META[status] ?? STATUS_META.stopped;

  const tps = telemetry?.tps;
  const isGen = Boolean(tps?.is_generating);
  const displayTps = isGen
    ? (tps?.live_decode_tps ?? 0)
    : (tps?.last_turn_decode_tps || tps?.ema_decode_tps || 0);

  const ttftMs = tps?.last_ttft_ms ?? 0;
  const vramUsedGb = (telemetry?.gpu.used_mb ?? 0) / 1024;
  const vramTotalGb = (telemetry?.gpu.total_mb ?? studio?.hardware.vram_total_mb ?? 8151) / 1024;
  const vramPct = vramTotalGb > 0 ? Math.min(100, (vramUsedGb / vramTotalGb) * 100) : 0;

  const sys = telemetry?.system;
  const hotRamGb = (sys?.hot_ram_mb ?? 0) / 1024;
  const ramTotalGb = sys?.ram_total_gb ?? studio?.hardware.ram_total_gb ?? 16;
  const ramDisplayGb = hotRamGb > 0.1 ? hotRamGb : (sys?.ram_used_gb ?? 0);
  const ramPct = ramTotalGb > 0 ? Math.min(100, (ramDisplayGb / ramTotalGb) * 100) : 0;
  const cpuDisplayPct = (sys?.engine_cpu_pct ?? 0) > 0.5 ? (sys?.engine_cpu_pct ?? 0) : (sys?.cpu_util_pct ?? 0);

  return (
    <header
      className="h-14 shrink-0 td-chrome px-5 flex items-center justify-between gap-4 select-none z-20"
      style={{ borderBottom: '1px solid var(--td-border)' }}
    >
      {/* Left: Quiet Precision Telemetry Strip */}
      <div className="flex items-center gap-5 min-w-0">
        {/* Velocity */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${isGen ? 'td-live-dot' : ''}`}
              style={{ background: isGen ? 'var(--td-accent)' : connected ? 'var(--td-green)' : 'var(--td-red)' }}
            />
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono-num text-[17px] font-semibold tracking-tight" style={{ color: 'var(--td-ink)' }}>
                {displayTps.toFixed(1)}
              </span>
              <span className="text-[11px] font-medium" style={{ color: 'var(--td-ink-3)' }}>
                tok/s
              </span>
            </div>
          </div>

          <div className="hidden sm:block">
            <HairlineSparkline points={tps?.tps_history ?? []} />
          </div>
        </div>

        <div className="hidden md:block h-5 w-px" style={{ background: 'var(--td-border)' }} />

        {/* TTFT */}
        <div className="hidden md:flex items-baseline gap-1.5 text-xs">
          <span style={{ color: 'var(--td-ink-3)' }}>TTFT</span>
          <span className="font-mono-num font-medium" style={{ color: 'var(--td-ink)' }}>
            {ttftMs >= 1000 ? `${(ttftMs / 1000).toFixed(2)}s` : `${ttftMs.toFixed(0)}ms`}
          </span>
        </div>

        {/* VRAM & Hot RAM Meters */}
        <div className="hidden xl:flex items-center gap-5">
          <div className="h-5 w-px" style={{ background: 'var(--td-border)' }} />

          <div className="w-32">
            <div className="flex justify-between text-[10.5px] font-mono-num mb-1">
              <span style={{ color: 'var(--td-ink-3)' }}>VRAM</span>
              <span style={{ color: 'var(--td-ink-2)' }}>
                {vramUsedGb.toFixed(1)}/{vramTotalGb.toFixed(1)}G
              </span>
            </div>
            <div className="h-[3px] rounded-full overflow-hidden" style={{ background: 'var(--td-surface-3)' }}>
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${vramPct}%`,
                  background: vramPct > 90 ? 'var(--td-red)' : 'var(--td-blue)',
                }}
              />
            </div>
          </div>

          <div className="w-36">
            <div className="flex justify-between text-[10.5px] font-mono-num mb-1">
              <span style={{ color: 'var(--td-ink-3)' }}>
                {hotRamGb > 0.1 ? 'Hot RAM' : 'RAM'}
              </span>
              <span style={{ color: 'var(--td-ink-2)' }}>
                {ramDisplayGb.toFixed(1)}/{ramTotalGb.toFixed(0)}G · {cpuDisplayPct.toFixed(0)}%
              </span>
            </div>
            <div className="h-[3px] rounded-full overflow-hidden" style={{ background: 'var(--td-surface-3)' }}>
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${ramPct}%`,
                  background: 'var(--td-accent)',
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Right: Status & Lifecycle Controls */}
      <div className="flex items-center gap-2 shrink-0">
        <div
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium"
          style={{
            background: 'var(--td-surface-2)',
            border: '1px solid var(--td-border)',
            color: meta.tone,
          }}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${status === 'starting' || status === 'restarting' ? 'td-live-dot' : ''}`}
            style={{ background: meta.dot }}
          />
          {meta.label}
        </div>

        <div className="h-4 w-px mx-1" style={{ background: 'var(--td-border)' }} />

        {status === 'stopped' || status === 'error' ? (
          <button
            type="button"
            disabled={busyAction !== null}
            onClick={onStart}
            className="td-btn-primary inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium disabled:opacity-40 cursor-pointer"
          >
            <Play className="w-3 h-3" />
            {busyAction === 'start' ? 'Starting…' : 'Start'}
          </button>
        ) : (
          <>
            <button
              type="button"
              disabled={busyAction !== null}
              onClick={onRestart}
              className="td-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium disabled:opacity-40 cursor-pointer"
              title="Gracefully drain and hot-swap engine configuration"
            >
              <RotateCw className={`w-3 h-3 ${busyAction === 'restart' ? 'animate-spin' : ''}`} />
              {busyAction === 'restart' ? 'Swapping…' : 'Hot-Swap'}
            </button>
            <button
              type="button"
              disabled={busyAction !== null}
              onClick={onStop}
              className="td-btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium disabled:opacity-40 cursor-pointer"
              style={{ color: 'var(--td-red)' }}
              title="Stop engine subprocess and release VRAM"
            >
              <Square className="w-3 h-3" />
              {busyAction === 'stop' ? 'Stopping…' : 'Stop'}
            </button>
          </>
        )}
      </div>
    </header>
  );
};
