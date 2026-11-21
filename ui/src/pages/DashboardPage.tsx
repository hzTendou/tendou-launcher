import React from 'react';
import { Cpu, Database, HardDrive, Layers, Thermometer, Zap } from 'lucide-react';
import type { StudioState, TelemetrySnapshot } from '../types/api';

interface DashboardPageProps {
  studio: StudioState | null;
  telemetry: TelemetrySnapshot | null;
}

function PrecisionArcGauge({ value, peak, isGenerating }: { value: number; peak: number; isGenerating: boolean }) {
  const maxScale = Math.max(100, Math.ceil(Math.max(peak, value, 100) / 20) * 20);
  const ratio = Math.min(1, Math.max(0, value / maxScale));
  const r = 74;
  const cx = 100;
  const cy = 94;
  const arcLen = Math.PI * r;
  const dashOffset = arcLen * (1 - ratio);

  return (
    <div className="relative flex flex-col items-center justify-center py-1">
      <svg viewBox="0 0 200 118" className="w-56 h-32">
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="var(--td-surface-3)"
          strokeWidth="8"
          strokeLinecap="round"
        />
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={isGenerating ? 'var(--td-accent)' : 'var(--td-ink)'}
          strokeOpacity={isGenerating ? 1 : 0.72}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={arcLen}
          strokeDashoffset={dashOffset}
          className="transition-all duration-300"
        />
        <text
          x={cx}
          y={cy - 18}
          textAnchor="middle"
          fill="var(--td-ink)"
          className="font-mono-num text-[30px] font-semibold"
        >
          {value.toFixed(1)}
        </text>
        <text
          x={cx}
          y={cy + 2}
          textAnchor="middle"
          fill="var(--td-ink-3)"
          className="text-[10px] font-medium tracking-[0.16em] uppercase"
        >
          {isGenerating ? 'Live tok/s' : 'Decode tok/s'}
        </text>
        <text x={cx - r} y={cy + 17} textAnchor="middle" fill="var(--td-ink-3)" className="font-mono-num text-[9px]">
          0
        </text>
        <text x={cx + r} y={cy + 17} textAnchor="middle" fill="var(--td-ink-3)" className="font-mono-num text-[9px]">
          {maxScale}
        </text>
      </svg>
    </div>
  );
}

function DualTelemetryChart({
  history,
  vramTotalMb,
}: {
  history: Array<{ ts: number; tps: number; vram_mb: number }>;
  vramTotalMb: number;
}) {
  const width = 560;
  const height = 160;
  const pad = { l: 36, r: 38, t: 12, b: 20 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;

  const pts = history.slice(-120);
  const maxTps = Math.max(40, ...pts.map((p) => p.tps));
  const maxVram = Math.max(vramTotalMb || 8151, ...pts.map((p) => p.vram_mb), 1024);

  const tpsCoords = pts.map((p, i) => {
    const x = pad.l + (i / Math.max(1, pts.length - 1)) * plotW;
    const y = pad.t + plotH - (p.tps / maxTps) * plotH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const vramCoords = pts.map((p, i) => {
    const x = pad.l + (i / Math.max(1, pts.length - 1)) * plotW;
    const y = pad.t + plotH - (p.vram_mb / maxVram) * plotH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const areaPoints =
    pts.length >= 2
      ? `${pad.l},${pad.t + plotH} ${tpsCoords.join(' ')} ${pad.l + plotW},${pad.t + plotH}`
      : '';

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-44">
      <defs>
        <linearGradient id="tdTpsArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--td-accent)" stopOpacity="0.22" />
          <stop offset="100%" stopColor="var(--td-accent)" stopOpacity="0.0" />
        </linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => {
        const y = pad.t + plotH * (1 - g);
        return (
          <g key={g}>
            <line x1={pad.l} y1={y} x2={pad.l + plotW} y2={y} stroke="var(--td-border)" strokeDasharray="2 4" />
            <text x={pad.l - 6} y={y + 3} textAnchor="end" fill="var(--td-ink-3)" className="font-mono-num text-[9px]">
              {(maxTps * g).toFixed(0)}
            </text>
            <text x={pad.l + plotW + 6} y={y + 3} textAnchor="start" fill="var(--td-ink-3)" className="font-mono-num text-[9px]">
              {((maxVram * g) / 1024).toFixed(1)}G
            </text>
          </g>
        );
      })}
      {areaPoints && <polygon points={areaPoints} fill="url(#tdTpsArea)" />}
      {pts.length >= 2 && (
        <>
          <polyline
            fill="none"
            stroke="var(--td-blue)"
            strokeOpacity="0.7"
            strokeWidth="1.35"
            strokeDasharray="4 3"
            points={vramCoords.join(' ')}
          />
          <polyline
            fill="none"
            stroke="var(--td-accent)"
            strokeWidth="1.9"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={tpsCoords.join(' ')}
          />
        </>
      )}
      {pts.length < 2 && (
        <text x={width / 2} y={height / 2} textAnchor="middle" fill="var(--td-ink-3)" className="text-[11px]">
          Streaming 4 Hz telemetry… Send a prompt in Chats to observe live decode throughput
        </text>
      )}
    </svg>
  );
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ studio, telemetry }) => {
  const tps = telemetry?.tps;
  const gpu = telemetry?.gpu;
  const sys = telemetry?.system;
  const moe = telemetry?.moe_cache;
  const prefix = telemetry?.prefix_cache;
  const plan = (studio?.autotune_plan ?? null) as Record<string, unknown> | null;
  const rationale = (plan?.rationale ?? {}) as Record<string, unknown>;

  const liveOrLastTps = tps?.is_generating ? tps.live_decode_tps : tps?.last_turn_decode_tps || tps?.ema_decode_tps || 0;
  const hotRamGb = (sys?.hot_ram_mb ?? 0) / 1024;
  const lockedGb = (sys?.engine_locked_mb ?? 0) / 1024;
  const cachedGb = sys?.ram_cached_gb ?? 0;

  return (
    <div className="p-7 space-y-6 max-w-[1440px] mx-auto">
      {/* Page Title */}
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-[24px] font-semibold font-display tracking-tight" style={{ color: 'var(--td-ink)' }}>
            System Overview
          </h1>
          <p className="text-[13px] mt-0.5" style={{ color: 'var(--td-ink-3)' }}>
            Live 4 Hz inference telemetry, 3-tier memory residency, and hardware utilization.
          </p>
        </div>
        <span
          className="px-3 py-1 rounded-full text-[11px] font-mono-num"
          style={{
            background: 'var(--td-surface)',
            border: '1px solid var(--td-border)',
            color: 'var(--td-ink-2)',
          }}
        >
          {studio?.active_model.id ?? 'No model'} · v{studio?.version ?? '0.16.3-stable'}
        </span>
      </div>

      {/* Hero Row: Velocity Arc + Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-4 td-card p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <div>
              <div className="text-[13px] font-semibold" style={{ color: 'var(--td-ink)' }}>
                Inference Velocity
              </div>
              <div className="text-[11px]" style={{ color: 'var(--td-ink-3)' }}>
                500 ms sliding window
              </div>
            </div>
            <span
              className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10.5px] font-medium"
              style={{
                background: tps?.is_generating ? 'var(--td-accent-soft)' : 'var(--td-surface-2)',
                color: tps?.is_generating ? 'var(--td-accent)' : 'var(--td-ink-3)',
                border: '1px solid var(--td-border)',
              }}
            >
              {tps?.is_generating && (
                <span className="w-1.5 h-1.5 rounded-full td-live-dot" style={{ background: 'var(--td-accent)' }} />
              )}
              {tps?.is_generating ? 'Generating' : 'Idle'}
            </span>
          </div>

          <PrecisionArcGauge
            value={liveOrLastTps}
            peak={tps?.peak_decode_tps ?? 0}
            isGenerating={Boolean(tps?.is_generating)}
          />

          <div className="grid grid-cols-3 gap-3 mt-3 pt-4" style={{ borderTop: '1px solid var(--td-border)' }}>
            <div>
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>EMA</div>
              <div className="font-mono-num text-[14px] font-semibold mt-0.5" style={{ color: 'var(--td-ink)' }}>
                {(tps?.ema_decode_tps ?? 0).toFixed(1)}
              </div>
            </div>
            <div>
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Peak</div>
              <div className="font-mono-num text-[14px] font-semibold mt-0.5" style={{ color: 'var(--td-accent)' }}>
                {(tps?.peak_decode_tps ?? 0).toFixed(1)}
              </div>
            </div>
            <div>
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Session Tok</div>
              <div className="font-mono-num text-[14px] font-semibold mt-0.5" style={{ color: 'var(--td-ink)' }}>
                {(tps?.tokens_session_total ?? 0).toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-8 td-card p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="text-[13px] font-semibold" style={{ color: 'var(--td-ink)' }}>
                Throughput &amp; Memory Timeline
              </div>
              <div className="text-[11px]" style={{ color: 'var(--td-ink-3)' }}>
                30-second synchronized decode velocity vs GPU VRAM occupancy
              </div>
            </div>
            <div className="flex items-center gap-4 text-[11px] font-mono-num">
              <span className="flex items-center gap-1.5" style={{ color: 'var(--td-accent)' }}>
                <span className="w-2.5 h-[2px] inline-block rounded" style={{ background: 'var(--td-accent)' }} /> TPS
              </span>
              <span className="flex items-center gap-1.5" style={{ color: 'var(--td-blue)' }}>
                <span className="w-2.5 h-[2px] inline-block rounded" style={{ background: 'var(--td-blue)' }} /> VRAM
              </span>
            </div>
          </div>
          <DualTelemetryChart
            history={tps?.tps_history ?? []}
            vramTotalMb={gpu?.total_mb ?? studio?.hardware.vram_total_mb ?? 8151}
          />
        </div>
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="td-card p-5">
          <div className="flex items-center justify-between text-[12px] mb-2" style={{ color: 'var(--td-ink-3)' }}>
            <span>Time to First Token</span>
            <Zap className="w-3.5 h-3.5" style={{ color: 'var(--td-accent)' }} />
          </div>
          <div className="font-mono-num text-[24px] font-semibold tracking-tight" style={{ color: 'var(--td-ink)' }}>
            {(tps?.last_ttft_ms ?? 0) >= 1000
              ? `${((tps?.last_ttft_ms ?? 0) / 1000).toFixed(2)} s`
              : `${(tps?.last_ttft_ms ?? 0).toFixed(0)} ms`}
          </div>
          <div className="text-[11px] font-mono-num mt-1.5" style={{ color: 'var(--td-ink-3)' }}>
            Prefill {(tps?.last_prefill_tps ?? 0).toFixed(0)} tok/s · {tps?.last_prefix_fork_kind ?? 'cold'}
          </div>
        </div>

        <div className="td-card p-5">
          <div className="flex items-center justify-between text-[12px] mb-2" style={{ color: 'var(--td-ink-3)' }}>
            <span>GPU VRAM</span>
            <Thermometer className="w-3.5 h-3.5" style={{ color: 'var(--td-blue)' }} />
          </div>
          <div className="font-mono-num text-[24px] font-semibold tracking-tight" style={{ color: 'var(--td-ink)' }}>
            {((gpu?.used_mb ?? 0) / 1024).toFixed(2)}
            <span className="text-[13px] font-normal ml-1" style={{ color: 'var(--td-ink-3)' }}>
              / {((gpu?.total_mb ?? 8151) / 1024).toFixed(1)} GiB
            </span>
          </div>
          <div className="text-[11px] font-mono-num mt-1.5" style={{ color: 'var(--td-ink-3)' }}>
            Peak {((gpu?.peak_mb ?? 0) / 1024).toFixed(2)}G · {(gpu?.util_pct ?? 0).toFixed(0)}% · {(gpu?.temp_c ?? 0).toFixed(0)}°C
          </div>
        </div>

        <div className="td-card p-5">
          <div className="flex items-center justify-between text-[12px] mb-2" style={{ color: 'var(--td-ink-3)' }}>
            <span>Hot RAM &amp; CPU</span>
            <Cpu className="w-3.5 h-3.5" style={{ color: 'var(--td-accent)' }} />
          </div>
          <div className="font-mono-num text-[24px] font-semibold tracking-tight" style={{ color: 'var(--td-ink)' }}>
            {(hotRamGb > 0.1 ? hotRamGb : (sys?.ram_used_gb ?? 0)).toFixed(2)}
            <span className="text-[13px] font-normal ml-1" style={{ color: 'var(--td-ink-3)' }}>
              / {(sys?.ram_total_gb ?? 16).toFixed(1)} GiB
            </span>
          </div>
          <div className="text-[11px] font-mono-num mt-1.5" style={{ color: 'var(--td-ink-3)' }}>
            {lockedGb > 0.05
              ? `Pinned ${lockedGb.toFixed(1)}G`
              : `Cache ${cachedGb.toFixed(1)}G`}{' '}
            · CPU {(sys?.engine_cpu_pct || sys?.cpu_util_pct || 0).toFixed(0)}%
            {(sys?.engine_threads ?? 0) > 0 ? ` (${sys!.engine_threads}T)` : ''}
          </div>
        </div>

        <div className="td-card p-5">
          <div className="flex items-center justify-between text-[12px] mb-2" style={{ color: 'var(--td-ink-3)' }}>
            <span>PrefixCache KV Tree</span>
            <Layers className="w-3.5 h-3.5" style={{ color: 'var(--td-green)' }} />
          </div>
          <div className="font-mono-num text-[24px] font-semibold tracking-tight" style={{ color: 'var(--td-ink)' }}>
            {(prefix?.hit_rate_pct ?? 0).toFixed(1)}%
          </div>
          <div className="text-[11px] font-mono-num mt-1.5" style={{ color: 'var(--td-ink-3)' }}>
            Slots {prefix?.slots_used ?? 0}/{prefix?.max_slots ?? 16} · {tps?.requests_total ?? 0} turns
          </div>
        </div>
      </div>

      {/* Bottom Row: 3-Tier Memory Residency + Active AutoTune Plan */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-6 td-card p-6">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4" style={{ color: 'var(--td-accent)' }} />
              <h2 className="text-[14px] font-semibold font-display" style={{ color: 'var(--td-ink)' }}>
                3-Tier Hybrid MoE Residency
              </h2>
            </div>
            <span className="font-mono-num text-[11.5px]" style={{ color: 'var(--td-ink-3)' }}>
              Dynamic-K:{' '}
              <strong style={{ color: 'var(--td-ink)' }}>
                {(moe?.avg_dynamic_k ?? 0) > 0 ? moe!.avg_dynamic_k.toFixed(2) : 'Boost L2'}
              </strong>
            </span>
          </div>

          <div className="space-y-4">
            {[
              {
                label: 'Tier 0 · CUDA GDDR6 VRAM (Hot Anchors + Reasoning Band)',
                pct: moe?.tier0_vram_pct ?? 0,
                hits: moe?.tier0_vram_hits ?? 0,
                barColor: 'var(--td-blue)',
              },
              {
                label: 'Tier 1 · Pinned Zero-Copy Hot RAM (THP + AVX-512 / AVX2 x4)',
                pct: moe?.tier1_ram_pct ?? 0,
                hits: moe?.tier1_ram_hits ?? 0,
                barColor: 'var(--td-accent)',
              },
              {
                label: 'Tier 2 · NVMe SSD Direct Streaming (io_uring / DirectStorage)',
                pct: moe?.tier2_nvme_pct ?? 0,
                hits: moe?.tier2_nvme_misses ?? 0,
                barColor: 'var(--td-ink-3)',
              },
            ].map((tier) => (
              <div key={tier.label}>
                <div className="flex items-center justify-between text-[12px] mb-1.5">
                  <span style={{ color: 'var(--td-ink-2)' }}>{tier.label}</span>
                  <span className="font-mono-num" style={{ color: 'var(--td-ink)' }}>
                    {tier.pct.toFixed(1)}%{' '}
                    <span style={{ color: 'var(--td-ink-3)' }}>({tier.hits})</span>
                  </span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--td-surface-3)' }}>
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${tier.pct}%`, background: tier.barColor }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div
            className="mt-5 pt-4 flex items-center justify-between text-[11px]"
            style={{ borderTop: '1px solid var(--td-border)', color: 'var(--td-ink-3)' }}
          >
            <span>Kernels: IQ4_XS_x4 · IQ3_S_x4 · IQ3_XXS_x4 (AVX-512 VNNI)</span>
            <span className="font-mono-num" style={{ color: 'var(--td-green)' }}>
              0-ULP Verified
            </span>
          </div>
        </div>

        <div className="lg:col-span-6 td-card p-6">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <HardDrive className="w-4 h-4" style={{ color: 'var(--td-accent)' }} />
              <h2 className="text-[14px] font-semibold font-display" style={{ color: 'var(--td-ink)' }}>
                Active AutoTune Plan
              </h2>
            </div>
            <span
              className="px-2.5 py-0.5 rounded-full text-[10.5px] font-mono-num"
              style={{
                background: 'var(--td-surface-2)',
                border: '1px solid var(--td-border)',
                color: 'var(--td-ink-2)',
              }}
            >
              MTP {String((studio?.settings?.mtp as string) ?? 'off').toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="td-card-subtle p-3.5">
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Model</div>
              <div className="font-mono-num font-medium mt-1 truncate" style={{ color: 'var(--td-ink)' }}>
                {studio?.active_model.id ?? '—'}
              </div>
            </div>
            <div className="td-card-subtle p-3.5">
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>GPU Placement</div>
              <div className="font-mono-num font-medium mt-1" style={{ color: 'var(--td-ink)' }}>
                {plan ? `${String(plan.gpu_layers)} Layers (${String(rationale.gpu_layer_policy ?? 'auto')})` : 'AutoTune Active'}
              </div>
            </div>
            <div className="td-card-subtle p-3.5">
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>KV Quantization</div>
              <div className="font-mono-num font-medium mt-1" style={{ color: 'var(--td-ink)' }}>
                {plan ? `${String(plan.cache_type_k)} / ${String(plan.cache_type_v)}` : 'Adaptive q8_0 / FP8'}
              </div>
            </div>
            <div className="td-card-subtle p-3.5">
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Hot RAM Policy</div>
              <div className="font-mono-num font-medium mt-1 truncate" style={{ color: 'var(--td-accent)' }}>
                {sys?.hot_ram_mode ?? String(plan?.ram_lock_policy ?? 'auto')}
                {lockedGb > 0.05 ? ` · ${lockedGb.toFixed(1)} GiB pinned` : ''}
              </div>
            </div>
          </div>

          <div
            className="mt-3 p-3 rounded-xl font-mono-num text-[11px] truncate"
            style={{
              background: 'var(--td-surface-2)',
              border: '1px solid var(--td-border)',
              color: 'var(--td-ink-3)',
            }}
          >
            Engine: <span style={{ color: 'var(--td-ink-2)' }}>{studio?.exe_path ?? 'releases/tendou-v0.16.3-stable/bin/tendou-engine'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
