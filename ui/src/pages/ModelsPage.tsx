import React, { useCallback, useEffect, useState } from 'react';
import { FolderSearch, Layers, Play, RefreshCw, Zap } from 'lucide-react';
import { studioApi } from '../api/client';
import type { LocalModelEntry } from '../types/api';

interface ModelsPageProps {
  ctxSize?: number;
  onModelLoaded: () => void;
}

export const ModelsPage: React.FC<ModelsPageProps> = ({ ctxSize = 65536, onModelLoaded }) => {
  const [models, setModels] = useState<LocalModelEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [scanDir, setScanDir] = useState('');
  const [swappingPath, setSwappingPath] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await studioApi.getModels(ctxSize);
      setModels(res.models);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to scan models');
    } finally {
      setLoading(false);
    }
  }, [ctxSize]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const handleScanDir = async () => {
    if (!scanDir.trim()) return;
    setError(null);
    setNotice(null);
    try {
      const res = await studioApi.scanModels(scanDir.trim());
      setModels(res.models);
      setNotice(`Scanned ${res.scanned} — ${res.count} model(s) discovered.`);
      setScanDir('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Directory scan failed');
    }
  };

  const handleHotSwap = async (m: LocalModelEntry) => {
    setSwappingPath(m.path);
    setError(null);
    setNotice(null);
    try {
      await studioApi.restartEngine({
        model_path: m.path,
        settings: {
          mock: false,
          gpu_layers: null,
          vram_layer_cap: m.autotune.vram_layer_cap,
          boost_level: m.autotune.boost_level ?? 2,
          mtp: 'off',
        },
        save_profile: true,
      });
      setNotice(`Loaded ${m.name} with AutoTune (${m.autotune.gpu_layers} GPU layers @ ${ctxSize} ctx, Zero-Copy Hot-RAM active).`);
      await loadModels();
      onModelLoaded();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model hot-swap failed');
    } finally {
      setSwappingPath(null);
    }
  };

  return (
    <div className="p-7 space-y-6 max-w-[1440px] mx-auto">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-[24px] font-semibold font-display tracking-tight" style={{ color: 'var(--td-ink)' }}>
            Models &amp; AutoTune
          </h1>
          <p className="text-[13px] mt-0.5" style={{ color: 'var(--td-ink-3)' }}>
            Local GGUF discovery with automatic VRAM layer placement, Zero-Copy Hot-RAM planning, and hot-swap loading.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <input
            value={scanDir}
            onChange={(e) => setScanDir(e.target.value)}
            placeholder="Scan folder for .gguf models…"
            className="w-64 rounded-full px-4 py-1.5 text-xs focus:outline-none"
            style={{
              background: 'var(--td-surface)',
              border: '1px solid var(--td-border)',
              color: 'var(--td-ink)',
            }}
          />
          <button
            type="button"
            onClick={() => void handleScanDir()}
            className="td-btn-secondary px-3.5 py-1.5 rounded-full text-xs font-medium inline-flex items-center gap-1.5 cursor-pointer"
          >
            <FolderSearch className="w-3.5 h-3.5" /> Scan
          </button>
          <button
            type="button"
            onClick={() => void loadModels()}
            className="td-btn-secondary w-8 h-8 rounded-full flex items-center justify-center cursor-pointer"
            title="Refresh list"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {notice && (
        <div
          className="px-4 py-2.5 rounded-xl text-xs"
          style={{
            background: 'var(--td-green-soft)',
            border: '1px solid var(--td-border)',
            color: 'var(--td-green)',
          }}
        >
          {notice}
        </div>
      )}
      {error && (
        <div
          className="px-4 py-2.5 rounded-xl text-xs"
          style={{
            background: 'var(--td-red-soft)',
            border: '1px solid var(--td-border)',
            color: 'var(--td-red)',
          }}
        >
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {models.map((m) => (
          <div
            key={m.path}
            className="td-card p-6 transition-all"
            style={{
              borderColor: m.active ? 'var(--td-accent)' : 'var(--td-border)',
            }}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-[15px] font-semibold font-display truncate" style={{ color: 'var(--td-ink)' }}>
                    {m.name}
                  </h3>
                  {m.active && (
                    <span
                      className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                      style={{ background: 'var(--td-green-soft)', color: 'var(--td-green)' }}
                    >
                      Active
                    </span>
                  )}
                  {m.is_mtp_head && (
                    <span
                      className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                      style={{ background: 'var(--td-accent-soft)', color: 'var(--td-accent)' }}
                    >
                      MTP Draft
                    </span>
                  )}
                </div>
                <div className="text-[11px] font-mono-num truncate mt-1" style={{ color: 'var(--td-ink-3)' }} title={m.path}>
                  {m.path}
                </div>
              </div>

              {!m.is_mtp_head && (
                <button
                  type="button"
                  onClick={() => void handleHotSwap(m)}
                  disabled={swappingPath !== null}
                  className={`${m.active ? 'td-btn-secondary' : 'td-btn-primary'} shrink-0 px-3.5 py-1.5 rounded-full text-xs font-medium inline-flex items-center gap-1.5 disabled:opacity-40 cursor-pointer`}
                >
                  <Play className="w-3 h-3" />
                  {swappingPath === m.path ? 'Loading…' : m.active ? 'Reload' : 'Load'}
                </button>
              )}
            </div>

            {m.is_mtp_head ? (
              <>
                <div className="grid grid-cols-3 gap-2.5 mt-5 text-xs">
                  <div className="td-card-subtle p-3">
                    <div className="text-[10px]" style={{ color: 'var(--td-ink-3)' }}>Size</div>
                    <div className="font-mono-num font-semibold mt-0.5" style={{ color: 'var(--td-ink)' }}>
                      {m.file_size_gb.toFixed(1)} GB
                    </div>
                  </div>
                  <div className="td-card-subtle p-3">
                    <div className="text-[10px]" style={{ color: 'var(--td-ink-3)' }}>Quant</div>
                    <div className="font-mono-num font-semibold mt-0.5" style={{ color: 'var(--td-blue)' }}>
                      {m.quant_type}
                    </div>
                  </div>
                  <div className="td-card-subtle p-3">
                    <div className="text-[10px]" style={{ color: 'var(--td-ink-3)' }}>Role</div>
                    <div className="font-mono-num font-semibold mt-0.5" style={{ color: 'var(--td-accent)' }}>
                      Speculative Head
                    </div>
                  </div>
                </div>
                <div
                  className="mt-4 pt-3.5 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono-num"
                  style={{ borderTop: '1px solid var(--td-border)', color: 'var(--td-ink-3)' }}
                >
                  <span>
                    Opt-in draft head · enable via <strong style={{ color: 'var(--td-ink-2)' }}>--mtp hybrid / vram</strong>
                  </span>
                  <span>Default OFF</span>
                </div>
              </>
            ) : (
              <>
                <div className="grid grid-cols-4 gap-2.5 mt-5 text-xs">
                  <div className="td-card-subtle p-3">
                    <div className="text-[10px]" style={{ color: 'var(--td-ink-3)' }}>Size</div>
                    <div className="font-mono-num font-semibold mt-0.5" style={{ color: 'var(--td-ink)' }}>
                      {m.file_size_gb.toFixed(1)} GB{m.shards > 1 ? ` · ${m.shards}×` : ''}
                    </div>
                  </div>
                  <div className="td-card-subtle p-3">
                    <div className="text-[10px]" style={{ color: 'var(--td-ink-3)' }}>Topology</div>
                    <div className="font-mono-num font-semibold mt-0.5" style={{ color: 'var(--td-ink)' }}>
                      {m.block_count}L × {m.expert_count}E
                    </div>
                  </div>
                  <div className="td-card-subtle p-3">
                    <div className="text-[10px]" style={{ color: 'var(--td-ink-3)' }}>Quant</div>
                    <div className="font-mono-num font-semibold mt-0.5" style={{ color: 'var(--td-blue)' }}>
                      {m.quant_type}
                    </div>
                  </div>
                  <div className="td-card-subtle p-3">
                    <div className="text-[10px]" style={{ color: 'var(--td-ink-3)' }}>Est. TPS</div>
                    <div className="font-mono-num font-semibold mt-0.5 flex items-center gap-1" style={{ color: 'var(--td-accent)' }}>
                      <Zap className="w-3 h-3" /> ~{m.autotune.estimated_decode_tps}
                    </div>
                  </div>
                </div>

                <div
                  className="mt-4 pt-3.5 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono-num"
                  style={{ borderTop: '1px solid var(--td-border)', color: 'var(--td-ink-3)' }}
                >
                  <span>
                    AutoTune: <strong style={{ color: 'var(--td-ink)' }}>{m.autotune.gpu_layers} GPU Layers</strong> · KV{' '}
                    <strong style={{ color: 'var(--td-ink-2)' }}>{m.autotune.cache_type_k}</strong>
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <Layers className="w-3 h-3" style={{ color: 'var(--td-blue)' }} />
                    Est. VRAM <strong style={{ color: 'var(--td-ink)' }}>{(m.autotune.estimated_vram_peak_mb / 1024).toFixed(2)} GiB</strong>
                  </span>
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {!loading && models.length === 0 && (
        <div className="td-card p-12 text-center text-sm" style={{ color: 'var(--td-ink-3)' }}>
          No local <code>.gguf</code> models found. Place GGUF files in <code>models/</code> or scan a folder above.
        </div>
      )}
    </div>
  );
};
