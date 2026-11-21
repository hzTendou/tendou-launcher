import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Check,
  Copy,
  RotateCw,
  Save,
  Terminal,
} from 'lucide-react';
import { studioApi } from '../api/client';
import type { ConfigPreset, ConfigResponse, SettingFieldSchema, SettingsSchemaResponse } from '../types/api';

interface SettingsPageProps {
  onApplied: () => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ onApplied }) => {
  const [schema, setSchema] = useState<SettingsSchemaResponse | null>(null);
  const [presets, setPresets] = useState<ConfigPreset[]>([]);
  const [configRes, setConfigRes] = useState<ConfigResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, unknown>>({});
  const [activeGroup, setActiveGroup] = useState<string>('core');
  const [activePreset, setActivePreset] = useState<string>('custom');
  const [cliPlatform, setCliPlatform] = useState<'linux' | 'windows'>('linux');
  const [saving, setSaving] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [copiedCli, setCopiedCli] = useState(false);
  const [toast, setToast] = useState<{ kind: 'ok' | 'err'; msg: string } | null>(null);

  const loadAll = useCallback(async () => {
    try {
      const [sc, pr, cfg] = await Promise.all([
        studioApi.getConfigSchema(),
        studioApi.getPresets(),
        studioApi.getConfig(),
      ]);
      setSchema(sc);
      setPresets(pr.presets);
      setConfigRes(cfg);
      setDraft(cfg.settings);
    } catch (err) {
      setToast({ kind: 'err', msg: err instanceof Error ? err.message : 'Failed to load settings' });
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  const setField = (key: string, val: unknown) => {
    setActivePreset('custom');
    setDraft((prev) => ({ ...prev, [key]: val }));
  };

  const applyPreset = (p: ConfigPreset) => {
    setActivePreset(p.id);
    setDraft((prev) => ({ ...prev, ...p.overrides }));
    setToast({ kind: 'ok', msg: `Preset "${p.name}" staged. Save or Save & Restart to apply.` });
  };

  const liveCliCommand = useMemo(() => {
    const parts: string[] = [
      '--host', String(draft.host ?? '127.0.0.1'),
      '--port', String(draft.port ?? 8000),
    ];
    if (draft.force) parts.push('--force');
    if (draft.mock) parts.push('--mock');
    if (draft.exe_path) parts.push('--exe-path', String(draft.exe_path));
    if (draft.model) parts.push('--model', String(draft.model));
    else if (draft.model_path) parts.push('--model-path', String(draft.model_path));

    parts.push(
      '--threads', String(draft.threads ?? 8),
      '--ctx-size', String(draft.ctx_size ?? 65536),
      '--prompt-cache-mb', String(draft.prompt_cache_mb ?? 768),
      '--atlas-k', String(draft.expert_k ?? 5),
      draft.boost === false ? '--no-boost' : '--boost',
      '--boost-level', String(draft.boost_level ?? 2),
    );
    if (draft.gpu === false) {
      parts.push('--cpu-only');
    } else {
      parts.push('--gpu');
      if (draft.gpu_layers !== null && draft.gpu_layers !== undefined && draft.gpu_layers !== '') {
        parts.push('--gpu-layers', String(draft.gpu_layers));
      }
      parts.push(
        '--ngl', String(draft.ngl ?? 99),
        '--vram-cache', String(draft.vram_cache ?? 3200),
        '--cuda-sched', String(draft.cuda_sched ?? 'yield'),
      );
    }
    if (draft.fp8_kv === true) parts.push('--fp8-kv');
    else if (draft.fp8_kv === false) parts.push('--no-fp8-kv');
    if (draft.cache_type_k && draft.cache_type_k !== 'auto') parts.push('--cache-type-k', String(draft.cache_type_k));
    if (draft.cache_type_v && draft.cache_type_v !== 'auto') parts.push('--cache-type-v', String(draft.cache_type_v));
    if (draft.vram_layer_cap) parts.push('--vram-layer-cap', String(draft.vram_layer_cap));
    if (draft.ram_lock_policy) parts.push('--ram-lock-policy', String(draft.ram_lock_policy));

    parts.push(
      '--moe-cache-mib', String(draft.moe_cache_mib ?? 2048),
      '--min-p', String(draft.min_p ?? 0.0),
      draft.no_sync_cache === false ? '--no-zero-sync-cache' : '--zero-sync-cache',
      draft.cascading_migration === false ? '--no-cascading-migration' : '--cascading-migration',
    );
    if (draft.moe_cache_cpu_share) parts.push('--moe-cache-cpu-share', String(draft.moe_cache_cpu_share));
    if (draft.ffn_chunk_size !== null && draft.ffn_chunk_size !== undefined && draft.ffn_chunk_size !== '') {
      parts.push('--ffn-chunk-size', String(draft.ffn_chunk_size));
    }

    parts.push(
      '--odmoe-lead', String(draft.odmoe_lead ?? 2),
      draft.odmoe_quick_evict === false ? '--no-odmoe-quick-evict' : '--odmoe-quick-evict',
      '--spice-conf-high', String(draft.spice_conf_high ?? 0.65),
      '--spice-conf-mid', String(draft.spice_conf_mid ?? 0.3),
      draft.tutti_async_io ? '--tutti-async-io' : '--no-tutti-async-io',
      '--readback-interval', String(draft.readback_interval ?? 8),
      '--prefetch-candidates', String(draft.prefetch_candidates ?? 10),
      '--tutti-queue-depth', String(draft.tutti_queue_depth ?? 64),
    );
    if (draft.aggressive_prefetch) parts.push('--aggressive-prefetch');
    if (draft.p2_gpu_binding) {
      parts.push('--p2-gpu-binding');
      if (Number(draft.p2_vram_budget_mib ?? 0) > 0) {
        parts.push('--p2-vram-budget-mib', String(draft.p2_vram_budget_mib));
      }
    }
    if (draft.p3_async_transfer) {
      parts.push(
        '--p3-async-transfer',
        '--p3-staging-slots', String(draft.p3_staging_slots ?? 4),
        '--p3-queue-depth', String(draft.p3_queue_depth ?? 8),
      );
    }

    const effMtp = String(draft.mtp_mode || draft.mtp || 'off');
    parts.push('--mtp', effMtp);
    if (effMtp !== 'off') {
      parts.push('--mtp-draft-n', String(draft.mtp_draft_n ?? 2));
      if (draft.mtp_loc) parts.push('--mtp-loc', String(draft.mtp_loc));
      if (draft.mtp_path) parts.push('--mtp-path', String(draft.mtp_path));
    }
    if (draft.tlep) {
      parts.push(
        '--tlep',
        '--tlep-depth', String(draft.tlep_depth ?? 4),
        '--tlep-guard', String(draft.tlep_guard ?? 1.5),
        '--tlep-bw', String(draft.tlep_bw ?? 128),
      );
    }

    const prefix = cliPlatform === 'windows' ? 'start_server.cmd ' : './start_server.sh ';
    const quoteArg = (p: string) =>
      cliPlatform === 'windows'
        ? p.includes(' ') || p.includes('\\')
          ? `"${p}"`
          : p
        : /[\s"'\\$`]/.test(p)
          ? `'${p.replace(/'/g, "'\\''")}'`
          : p;
    return prefix + parts.map(quoteArg).join(' ');
  }, [draft, cliPlatform]);

  const stagedDiff = useMemo(() => {
    const running = configRes?.running_settings ?? configRes?.settings ?? {};
    const diffs: Array<{ key: string; oldVal: unknown; newVal: unknown }> = [];
    for (const k of Object.keys(draft)) {
      if (JSON.stringify(running[k]) !== JSON.stringify(draft[k])) {
        diffs.push({ key: k, oldVal: running[k], newVal: draft[k] });
      }
    }
    return diffs;
  }, [configRes, draft]);

  const handleSave = async () => {
    setSaving(true);
    setToast(null);
    try {
      const updated = await studioApi.saveConfig(draft);
      setConfigRes(updated);
      setDraft(updated.settings);
      setToast({ kind: 'ok', msg: 'Profile saved to config/studio_profile.json.' });
      onApplied();
    } catch (err) {
      setToast({ kind: 'err', msg: err instanceof Error ? err.message : 'Save failed' });
    } finally {
      setSaving(false);
    }
  };

  const handleSaveAndRestart = async () => {
    setRestarting(true);
    setToast(null);
    try {
      await studioApi.saveConfig(draft);
      const res = await studioApi.restartEngine({ settings: draft, save_profile: true });
      setToast({
        kind: 'ok',
        msg: `Hot-Swap complete — engine online (${res.applied_diff?.length ?? 0} setting(s) updated).`,
      });
      await loadAll();
      onApplied();
    } catch (err) {
      setToast({ kind: 'err', msg: err instanceof Error ? err.message : 'Hot-Swap failed' });
    } finally {
      setRestarting(false);
    }
  };

  const renderControl = (field: SettingFieldSchema) => {
    const val = draft[field.key];
    const showMtpDanger =
      (field.key === 'mtp' || field.key === 'mtp_mode') &&
      field.safety_warning &&
      Boolean(val) &&
      val !== 'off' &&
      val !== 'pld';

    return (
      <div key={field.key} className="td-card p-4 flex flex-col justify-between gap-3">
        <div>
          <div className="flex items-start justify-between gap-2">
            <label className="text-[12.5px] font-semibold" style={{ color: 'var(--td-ink)' }}>
              {field.label}
            </label>
            <code
              className="text-[10px] font-mono-num px-2 py-0.5 rounded-md shrink-0"
              style={{
                background: 'var(--td-surface-2)',
                border: '1px solid var(--td-border)',
                color: 'var(--td-ink-2)',
              }}
            >
              {field.cli_flag}
            </code>
          </div>
          <p className="text-[11.5px] mt-1 leading-relaxed" style={{ color: 'var(--td-ink-3)' }}>
            {field.description}
          </p>
        </div>

        {showMtpDanger && (
          <div
            className="p-2.5 rounded-xl text-[11px] flex items-start gap-2"
            style={{
              background: 'var(--td-amber-soft)',
              border: '1px solid var(--td-border)',
              color: 'var(--td-amber)',
            }}
          >
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{field.safety_warning}</span>
          </div>
        )}

        <div>
          {field.type === 'boolean' && (
            <button
              type="button"
              onClick={() => setField(field.key, !Boolean(val))}
              className="inline-flex items-center gap-2.5 cursor-pointer"
            >
              <span
                className="w-10 h-6 rounded-full p-0.5 transition-colors inline-block"
                style={{ background: val ? 'var(--td-green)' : 'var(--td-surface-3)' }}
              >
                <span
                  className={`block w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    val ? 'translate-x-4' : 'translate-x-0'
                  }`}
                />
              </span>
              <span className="font-mono-num text-xs" style={{ color: 'var(--td-ink-2)' }}>
                {val ? 'Enabled' : 'Disabled'}
              </span>
            </button>
          )}

          {field.type === 'tribool' && (
            <div
              className="inline-flex rounded-full p-0.5"
              style={{ background: 'var(--td-surface-2)', border: '1px solid var(--td-border)' }}
            >
              {[
                { label: 'Auto', v: null },
                { label: 'On', v: true },
                { label: 'Off', v: false },
              ].map((opt) => (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => setField(field.key, opt.v)}
                  className="px-3 py-1 rounded-full text-[11px] font-medium cursor-pointer transition-colors"
                  style={{
                    background: val === opt.v ? 'var(--td-cta-bg)' : 'transparent',
                    color: val === opt.v ? 'var(--td-cta-ink)' : 'var(--td-ink-2)',
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}

          {field.type === 'select' && (
            <select
              value={val === null || val === undefined ? (field.choices?.includes('auto') ? 'auto' : '') : String(val)}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === '' && field.nullable) setField(field.key, null);
                else if (field.choices?.every((c) => typeof c === 'number')) setField(field.key, Number(raw));
                else setField(field.key, raw);
              }}
              className="w-full rounded-xl px-3 py-1.5 text-xs font-mono-num focus:outline-none"
              style={{
                background: 'var(--td-surface-2)',
                border: '1px solid var(--td-border)',
                color: 'var(--td-ink)',
              }}
            >
              {(field.choices ?? []).map((ch) => (
                <option key={String(ch)} value={String(ch)}>
                  {ch === '' ? '(Default)' : String(ch)}
                </option>
              ))}
            </select>
          )}

          {field.type === 'slider' && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs font-mono-num">
                <span style={{ color: 'var(--td-ink-3)' }}>{field.min}</span>
                <span className="font-semibold" style={{ color: 'var(--td-accent)' }}>
                  {String(val ?? field.min)}
                </span>
                <span style={{ color: 'var(--td-ink-3)' }}>{field.max}</span>
              </div>
              <input
                type="range"
                min={field.min}
                max={field.max}
                step={field.step ?? 1}
                value={Number(val ?? field.min ?? 0)}
                onChange={(e) =>
                  setField(field.key, (field.step ?? 1) < 1 ? parseFloat(e.target.value) : parseInt(e.target.value, 10))
                }
                className="w-full"
              />
            </div>
          )}

          {field.type === 'number' && (
            <div className="space-y-2">
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.step ?? 1}
                placeholder={field.placeholder ?? ''}
                value={val === null || val === undefined ? '' : String(val)}
                onChange={(e) => {
                  const s = e.target.value;
                  if (s === '' && field.nullable) setField(field.key, null);
                  else if (s !== '') setField(field.key, Number(s));
                }}
                className="w-full rounded-xl px-3 py-1.5 text-xs font-mono-num focus:outline-none"
                style={{
                  background: 'var(--td-surface-2)',
                  border: '1px solid var(--td-border)',
                  color: 'var(--td-ink)',
                }}
              />
              {field.quick_values && (
                <div className="flex flex-wrap gap-1.5">
                  {field.quick_values.map((qv) => (
                    <button
                      key={String(qv)}
                      type="button"
                      onClick={() => setField(field.key, qv)}
                      className="px-2.5 py-0.5 rounded-full text-[10.5px] font-mono-num cursor-pointer transition-colors"
                      style={{
                        background: val === qv ? 'var(--td-cta-bg)' : 'var(--td-surface-2)',
                        color: val === qv ? 'var(--td-cta-ink)' : 'var(--td-ink-2)',
                        border: '1px solid var(--td-border)',
                      }}
                    >
                      {typeof qv === 'number' && qv >= 1024 ? `${qv / 1024}K` : String(qv)}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {field.type === 'string' && (
            <div className="space-y-2">
              <input
                type="text"
                placeholder={field.placeholder ?? ''}
                value={val === null || val === undefined ? '' : String(val)}
                onChange={(e) => setField(field.key, e.target.value === '' && field.nullable ? null : e.target.value)}
                className="w-full rounded-xl px-3 py-1.5 text-xs font-mono-num focus:outline-none"
                style={{
                  background: 'var(--td-surface-2)',
                  border: '1px solid var(--td-border)',
                  color: 'var(--td-ink)',
                }}
              />
              {field.quick_values && (
                <div className="flex flex-wrap gap-1.5">
                  {field.quick_values.map((qv) => (
                    <button
                      key={String(qv)}
                      type="button"
                      onClick={() => setField(field.key, qv)}
                      className="px-2.5 py-0.5 rounded-full text-[10.5px] font-mono-num cursor-pointer transition-colors"
                      style={{
                        background: String(val) === String(qv) ? 'var(--td-cta-bg)' : 'var(--td-surface-2)',
                        color: String(val) === String(qv) ? 'var(--td-cta-ink)' : 'var(--td-ink-2)',
                        border: '1px solid var(--td-border)',
                      }}
                    >
                      {String(qv)}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const visibleFields = (schema?.fields ?? []).filter((f) => f.group === activeGroup);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-7 space-y-6 max-w-[1440px] mx-auto w-full">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
          <div>
            <h1 className="text-[24px] font-semibold font-display tracking-tight" style={{ color: 'var(--td-ink)' }}>
              Engine Configuration
            </h1>
            <p className="text-[13px] mt-0.5" style={{ color: 'var(--td-ink-3)' }}>
              Complete {schema?.total_flags ?? 52}-flag parameter matrix with live CLI preview and zero-zombie Hot-Swap.
            </p>
          </div>
          {stagedDiff.length > 0 && (
            <span
              className="px-3 py-1 rounded-full font-mono-num text-xs"
              style={{
                background: 'var(--td-accent-soft)',
                border: '1px solid var(--td-border)',
                color: 'var(--td-accent)',
              }}
            >
              {stagedDiff.length} pending change(s)
            </span>
          )}
        </div>

        {/* Presets */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3.5">
          {presets.map((p) => {
            const selected = activePreset === p.id;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => applyPreset(p)}
                className="td-card p-4 text-left transition-all cursor-pointer"
                style={{
                  borderColor: selected ? 'var(--td-accent)' : 'var(--td-border)',
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[13px] font-semibold" style={{ color: 'var(--td-ink)' }}>
                    {p.name}
                  </span>
                  <span
                    className="text-[10px] font-mono-num px-2 py-0.5 rounded-full"
                    style={{ background: 'var(--td-surface-2)', color: 'var(--td-accent)' }}
                  >
                    {p.badge}
                  </span>
                </div>
                <p className="text-[11.5px] mt-1.5 leading-snug" style={{ color: 'var(--td-ink-3)' }}>
                  {p.description}
                </p>
              </button>
            );
          })}
        </div>

        {toast && (
          <div
            className="px-4 py-2.5 rounded-xl text-xs"
            style={{
              background: toast.kind === 'ok' ? 'var(--td-green-soft)' : 'var(--td-red-soft)',
              border: '1px solid var(--td-border)',
              color: toast.kind === 'ok' ? 'var(--td-green)' : 'var(--td-red)',
            }}
          >
            {toast.msg}
          </div>
        )}

        {/* Segmented Category Tabs */}
        <div className="flex flex-wrap gap-1.5 pb-3" style={{ borderBottom: '1px solid var(--td-border)' }}>
          {(schema?.groups ?? []).map((g) => (
            <button
              key={g.id}
              type="button"
              onClick={() => setActiveGroup(g.id)}
              className="px-3.5 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer"
              style={{
                background: activeGroup === g.id ? 'var(--td-cta-bg)' : 'var(--td-surface)',
                color: activeGroup === g.id ? 'var(--td-cta-ink)' : 'var(--td-ink-2)',
                border: '1px solid var(--td-border)',
              }}
            >
              {g.label}
            </button>
          ))}
        </div>

        {/* Active Tab Content */}
        {activeGroup === 'cli_profiles' ? (
          <div className="td-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-[14px] font-semibold font-display flex items-center gap-2" style={{ color: 'var(--td-ink)' }}>
                <Terminal className="w-4 h-4" style={{ color: 'var(--td-accent)' }} /> Pending Hot-Swap Diff
              </div>
              <span className="font-mono-num text-xs" style={{ color: 'var(--td-ink-3)' }}>
                {configRes?.profile_path ?? 'config/studio_profile.json'}
              </span>
            </div>
            {stagedDiff.length === 0 ? (
              <div className="text-xs" style={{ color: 'var(--td-ink-3)' }}>
                All settings match the currently running engine.
              </div>
            ) : (
              <div className="rounded-xl overflow-hidden divide-y" style={{ border: '1px solid var(--td-border)' }}>
                {stagedDiff.map((d) => (
                  <div
                    key={d.key}
                    className="px-4 py-2.5 flex items-center justify-between text-xs font-mono-num"
                    style={{ background: 'var(--td-surface-2)', borderColor: 'var(--td-border)' }}
                  >
                    <span style={{ color: 'var(--td-ink)' }}>{d.key}</span>
                    <div className="flex items-center gap-2">
                      <span className="line-through opacity-60" style={{ color: 'var(--td-red)' }}>
                        {String(d.oldVal)}
                      </span>
                      <span style={{ color: 'var(--td-ink-3)' }}>→</span>
                      <span className="font-semibold" style={{ color: 'var(--td-accent)' }}>
                        {String(d.newVal)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{visibleFields.map(renderControl)}</div>
        )}
      </div>

      {/* Frosted Bottom CLI Command Bar */}
      <div
        className="shrink-0 td-chrome px-6 py-3.5"
        style={{ borderTop: '1px solid var(--td-border)' }}
      >
        <div className="max-w-[1440px] mx-auto flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <div
            className="flex-1 min-w-0 flex items-center gap-2.5 rounded-full px-3 py-1.5"
            style={{ background: 'var(--td-surface)', border: '1px solid var(--td-border)' }}
          >
            <div
              className="flex rounded-full p-0.5 shrink-0"
              style={{ background: 'var(--td-surface-2)' }}
            >
              {(['linux', 'windows'] as const).map((osKey) => (
                <button
                  key={osKey}
                  type="button"
                  onClick={() => setCliPlatform(osKey)}
                  className="px-2 py-0.5 rounded-full text-[10px] font-mono-num uppercase cursor-pointer"
                  style={{
                    background: cliPlatform === osKey ? 'var(--td-surface)' : 'transparent',
                    color: cliPlatform === osKey ? 'var(--td-ink)' : 'var(--td-ink-3)',
                  }}
                >
                  {osKey}
                </button>
              ))}
            </div>
            <code
              className="text-[11px] font-mono-num truncate flex-1"
              style={{ color: 'var(--td-ink-2)' }}
              title={liveCliCommand}
            >
              {liveCliCommand}
            </code>
            <button
              type="button"
              onClick={() => {
                void navigator.clipboard.writeText(liveCliCommand);
                setCopiedCli(true);
                window.setTimeout(() => setCopiedCli(false), 1500);
              }}
              className="shrink-0 cursor-pointer transition-opacity hover:opacity-80 pr-1"
              style={{ color: copiedCli ? 'var(--td-green)' : 'var(--td-ink-3)' }}
              title="Copy CLI command"
            >
              {copiedCli ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => void handleSave()}
              disabled={saving || restarting}
              className="td-btn-secondary px-4 py-2 rounded-full text-xs font-medium inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-40"
            >
              <Save className="w-3.5 h-3.5" /> {saving ? 'Saving…' : 'Save Profile'}
            </button>
            <button
              type="button"
              onClick={() => void handleSaveAndRestart()}
              disabled={saving || restarting}
              className="td-btn-primary px-4 py-2 rounded-full text-xs font-medium inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-40"
            >
              <RotateCw className={`w-3.5 h-3.5 ${restarting ? 'animate-spin' : ''}`} />
              {restarting ? 'Hot-Swapping…' : 'Save & Restart'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
