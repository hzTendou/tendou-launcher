import React, { useEffect, useState } from 'react';
import { Check, Copy, GitBranch, Layers, Users } from 'lucide-react';
import { studioApi } from '../api/client';
import type { AgentConfigsResponse, TelemetrySnapshot } from '../types/api';

interface AgentHubPageProps {
  telemetry: TelemetrySnapshot | null;
}

function CopySnippetCard({ title, badge, code }: { title: string; badge: string; code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="td-card overflow-hidden flex flex-col">
      <div
        className="px-4 py-3 flex items-center justify-between"
        style={{ borderBottom: '1px solid var(--td-border)', background: 'var(--td-surface-2)' }}
      >
        <div className="flex items-center gap-2">
          <span className="text-[12.5px] font-semibold" style={{ color: 'var(--td-ink)' }}>{title}</span>
          <span
            className="px-2 py-0.5 rounded-full font-mono-num text-[10px]"
            style={{ background: 'var(--td-surface)', border: '1px solid var(--td-border)', color: 'var(--td-ink-3)' }}
          >
            {badge}
          </span>
        </div>
        <button
          type="button"
          onClick={() => {
            void navigator.clipboard.writeText(code);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          }}
          className="text-xs inline-flex items-center gap-1 cursor-pointer transition-opacity hover:opacity-80"
          style={{ color: copied ? 'var(--td-green)' : 'var(--td-ink-2)' }}
        >
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre
        className="p-4 text-[11.5px] leading-relaxed font-mono-num overflow-x-auto flex-1"
        style={{ background: 'var(--td-code-bg)', color: 'var(--td-ink-2)' }}
      >
        {code}
      </pre>
    </div>
  );
}

export const AgentHubPage: React.FC<AgentHubPageProps> = ({ telemetry }) => {
  const [configs, setConfigs] = useState<AgentConfigsResponse | null>(null);
  const [claudeOs, setClaudeOs] = useState<'linux' | 'windows'>('linux');

  useEffect(() => {
    void studioApi.getAgentConfigs().then(setConfigs).catch(() => {});
  }, []);

  const swarm = telemetry?.swarm;
  const prefix = telemetry?.prefix_cache;
  const maxSlots = prefix?.max_slots ?? 16;
  const usedSlots = prefix?.slots_used ?? 0;

  return (
    <div className="p-7 space-y-6 max-w-[1440px] mx-auto">
      <div>
        <h1 className="text-[24px] font-semibold font-display tracking-tight" style={{ color: 'var(--td-ink)' }}>
          Agents &amp; PrefixCache Swarm
        </h1>
        <p className="text-[13px] mt-0.5" style={{ color: 'var(--td-ink-3)' }}>
          Ready-to-paste client configurations for Claude Code, OpenCode, and Cline, plus live KV tree slot telemetry.
        </p>
      </div>

      {/* Live Swarm Queue & 16-Slot PrefixCache */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-5 td-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[14px] font-semibold font-display flex items-center gap-2" style={{ color: 'var(--td-ink)' }}>
              <Users className="w-4 h-4" style={{ color: 'var(--td-accent)' }} /> Priority Engine Gate
            </h2>
            <span className="font-mono-num text-xs" style={{ color: 'var(--td-ink-3)' }}>
              Queue {swarm?.queue_depth ?? 0}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono-num">
            <div className="td-card-subtle p-3.5">
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Active HTTP</div>
              <div className="text-[20px] font-semibold mt-0.5" style={{ color: 'var(--td-ink)' }}>
                {swarm?.active_http_requests ?? 0}
              </div>
            </div>
            <div className="td-card-subtle p-3.5">
              <div className="text-[10.5px]" style={{ color: 'var(--td-ink-3)' }}>Session</div>
              <div className="text-xs font-medium mt-2 truncate" style={{ color: 'var(--td-accent)' }}>
                {swarm?.active_session_id ?? 'Idle'}
              </div>
            </div>
          </div>

          <div className="text-[12px] space-y-1.5" style={{ color: 'var(--td-ink-2)' }}>
            <div><strong style={{ color: 'var(--td-ink)' }}>P0 Fast-Lane:</strong> Interactive short prompts &amp; titles</div>
            <div><strong style={{ color: 'var(--td-ink)' }}>P1 Warm Affinity:</strong> Same session reusing hot KV prefix</div>
            <div><strong style={{ color: 'var(--td-ink)' }}>P2 Starvation Guard:</strong> 15s automatic aging promotion</div>
          </div>
        </div>

        <div className="lg:col-span-7 td-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[14px] font-semibold font-display flex items-center gap-2" style={{ color: 'var(--td-ink)' }}>
              <GitBranch className="w-4 h-4" style={{ color: 'var(--td-accent)' }} /> 16-Slot PrefixCache KV Tree
            </h2>
            <span className="font-mono-num text-xs" style={{ color: 'var(--td-ink-2)' }}>
              Hit {(prefix?.hit_rate_pct ?? 0).toFixed(1)}% · {usedSlots}/{maxSlots}
            </span>
          </div>

          <div className="grid grid-cols-8 gap-2">
            {Array.from({ length: maxSlots }).map((_, idx) => {
              const isRoot = idx === 0 && usedSlots > 0;
              const isUsed = idx < usedSlots;
              return (
                <div
                  key={idx}
                  className="h-11 rounded-xl flex flex-col items-center justify-center font-mono-num text-[10px]"
                  style={{
                    background: isRoot
                      ? 'var(--td-accent-soft)'
                      : isUsed
                        ? 'var(--td-blue-soft)'
                        : 'var(--td-surface-2)',
                    border: `1px solid ${isRoot ? 'var(--td-accent)' : 'var(--td-border)'}`,
                    color: isRoot ? 'var(--td-accent)' : isUsed ? 'var(--td-blue)' : 'var(--td-ink-3)',
                  }}
                >
                  <span>S{idx}</span>
                  <span className="text-[8px] uppercase">{isRoot ? 'ROOT' : isUsed ? 'HOT' : 'FREE'}</span>
                </div>
              );
            })}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono-num">
            {[
              { k: 'Root Anchor', v: prefix?.fork_counts.root_anchor ?? 0 },
              { k: 'Turn Fork', v: prefix?.fork_counts.turn_boundary ?? 0 },
              { k: 'Exact Hit', v: prefix?.fork_counts.exact ?? 0 },
              { k: 'Cold Prefill', v: prefix?.fork_counts.cold ?? 0 },
            ].map((item) => (
              <div key={item.k} className="td-card-subtle px-3 py-2 flex justify-between items-center">
                <span className="text-[11px]" style={{ color: 'var(--td-ink-3)' }}>{item.k}</span>
                <span className="font-semibold" style={{ color: 'var(--td-ink)' }}>{item.v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Coding Agent Config Snippets */}
      {configs && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[14px] font-semibold font-display flex items-center gap-2" style={{ color: 'var(--td-ink)' }}>
              <Layers className="w-4 h-4" style={{ color: 'var(--td-accent)' }} /> Client Integration ({configs.base_url})
            </h2>
            <div
              className="inline-flex rounded-full p-0.5"
              style={{ background: 'var(--td-surface-2)', border: '1px solid var(--td-border)' }}
            >
              {(['linux', 'windows'] as const).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setClaudeOs(k)}
                  className="px-3 py-1 rounded-full text-[11px] font-medium capitalize cursor-pointer transition-colors"
                  style={{
                    background: claudeOs === k ? 'var(--td-surface)' : 'transparent',
                    color: claudeOs === k ? 'var(--td-ink)' : 'var(--td-ink-3)',
                  }}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <CopySnippetCard
              title="Claude Code (Anthropic /v1/messages)"
              badge={claudeOs.toUpperCase()}
              code={claudeOs === 'linux' ? configs.claude_code.linux : configs.claude_code.windows}
            />
            <CopySnippetCard
              title="OpenCode (opencode.json)"
              badge="OPENAI /v1"
              code={configs.opencode.content}
            />
            <CopySnippetCard
              title="Cline / Roo Code (VS Code Provider)"
              badge="JSON"
              code={JSON.stringify(configs.cline, null, 2)}
            />
            <CopySnippetCard
              title="Python OpenAI SDK & cURL"
              badge="SDK"
              code={`${configs.python}\n# cURL:\n${configs.curl}`}
            />
          </div>
        </div>
      )}
    </div>
  );
};
