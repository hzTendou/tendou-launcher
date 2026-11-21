import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  LineChart,
  PanelLeft,
  PanelRight,
  Pencil,
  Pin,
  Plus,
  RotateCcw,
  Search,
  Square,
  Trash2,
  X,
} from 'lucide-react';
import { MarkdownRenderer } from '../components/MarkdownRenderer';
import { TendouMark } from '../components/TendouMark';
import { useChats } from '../context/ChatContext';
import type { ChatContextPoint, ChatMessageRecord, ChatSummary, StudioState } from '../types/api';

interface PlaygroundPageProps {
  studio: StudioState | null;
}

function formatTokensCompact(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.round(ms)}ms`;
}

function groupSummariesByDate(items: ChatSummary[]): Array<{ label: string; chats: ChatSummary[] }> {
  const pinned: ChatSummary[] = [];
  const today: ChatSummary[] = [];
  const yesterday: ChatSummary[] = [];
  const week: ChatSummary[] = [];
  const older: ChatSummary[] = [];

  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const startYesterday = startToday - 86400;
  const startWeek = startToday - 7 * 86400;

  for (const item of items) {
    if (item.pinned) {
      pinned.push(item);
    } else if (item.updated_at >= startToday) {
      today.push(item);
    } else if (item.updated_at >= startYesterday) {
      yesterday.push(item);
    } else if (item.updated_at >= startWeek) {
      week.push(item);
    } else {
      older.push(item);
    }
  }

  const groups: Array<{ label: string; chats: ChatSummary[] }> = [];
  if (pinned.length) groups.push({ label: 'Pinned', chats: pinned });
  if (today.length) groups.push({ label: 'Today', chats: today });
  if (yesterday.length) groups.push({ label: 'Yesterday', chats: yesterday });
  if (week.length) groups.push({ label: 'Previous 7 Days', chats: week });
  if (older.length) groups.push({ label: 'Earlier', chats: older });
  return groups;
}

function ContextGrowthChart({ points }: { points: ChatContextPoint[] }) {
  const width = 264;
  const height = 116;
  const pad = { l: 28, r: 28, t: 10, b: 18 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;

  if (points.length === 0) {
    return (
      <div
        className="h-28 rounded-xl flex items-center justify-center text-[11.5px] text-center px-4"
        style={{ background: 'var(--td-surface-2)', color: 'var(--td-ink-3)', border: '1px solid var(--td-border)' }}
      >
        Send turns in this chat to plot how Decode TPS and TTFT evolve as context grows.
      </div>
    );
  }

  const maxTps = Math.max(20, ...points.map((p) => p.decode_tps));
  const maxTtft = Math.max(250, ...points.map((p) => p.ttft_ms));

  const xFor = (i: number) =>
    points.length === 1 ? pad.l + plotW / 2 : pad.l + (i / (points.length - 1)) * plotW;

  const tpsPts = points.map((p, i) => `${xFor(i).toFixed(1)},${(pad.t + plotH - (p.decode_tps / maxTps) * plotH).toFixed(1)}`);
  const ttftPts = points.map((p, i) => `${xFor(i).toFixed(1)},${(pad.t + plotH - (p.ttft_ms / maxTtft) * plotH).toFixed(1)}`);

  return (
    <div
      className="rounded-xl p-2.5"
      style={{ background: 'var(--td-surface-2)', border: '1px solid var(--td-border)' }}
    >
      <div className="flex items-center justify-between text-[10px] font-mono-num mb-1.5 px-1">
        <span className="flex items-center gap-1.5" style={{ color: 'var(--td-accent)' }}>
          <span className="w-2 h-[2px] inline-block rounded" style={{ background: 'var(--td-accent)' }} />
          Decode TPS
        </span>
        <span className="flex items-center gap-1.5" style={{ color: 'var(--td-blue)' }}>
          <span className="w-2 h-[2px] inline-block rounded" style={{ background: 'var(--td-blue)' }} />
          TTFT ms
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-[112px]">
        {[0, 0.5, 1].map((g) => {
          const y = pad.t + plotH * (1 - g);
          return (
            <g key={g}>
              <line x1={pad.l} y1={y} x2={pad.l + plotW} y2={y} stroke="var(--td-border)" strokeDasharray="2 3" />
              <text x={pad.l - 4} y={y + 3} textAnchor="end" fill="var(--td-ink-3)" className="font-mono-num text-[8.5px]">
                {Math.round(maxTps * g)}
              </text>
              <text x={pad.l + plotW + 4} y={y + 3} textAnchor="start" fill="var(--td-ink-3)" className="font-mono-num text-[8.5px]">
                {maxTtft >= 1000 ? `${((maxTtft * g) / 1000).toFixed(1)}s` : `${Math.round(maxTtft * g)}`}
              </text>
            </g>
          );
        })}

        {points.length >= 2 && (
          <>
            <polyline
              fill="none"
              stroke="var(--td-blue)"
              strokeWidth="1.4"
              strokeDasharray="3 2"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={ttftPts.join(' ')}
            />
            <polyline
              fill="none"
              stroke="var(--td-accent)"
              strokeWidth="1.9"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={tpsPts.join(' ')}
            />
          </>
        )}

        {points.map((p, i) => {
          const cx = xFor(i);
          const cyTps = pad.t + plotH - (p.decode_tps / maxTps) * plotH;
          const cyTtft = pad.t + plotH - (p.ttft_ms / maxTtft) * plotH;
          return (
            <g key={p.message_id}>
              <circle cx={cx} cy={cyTtft} r="2.2" fill="var(--td-blue)" />
              <circle cx={cx} cy={cyTps} r="2.8" fill="var(--td-accent)" />
              {(i === 0 || i === points.length - 1 || points.length <= 5) && (
                <text x={cx} y={height - 3} textAnchor="middle" fill="var(--td-ink-3)" className="font-mono-num text-[8px]">
                  {formatTokensCompact(p.total_tokens)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ReasoningAccordion({ thinking, streaming }: { thinking: string; streaming?: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mb-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 text-[12px] font-medium transition-opacity hover:opacity-80 cursor-pointer"
        style={{ color: 'var(--td-ink-3)' }}
      >
        {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        <span>{streaming ? 'Thinking…' : 'Thought process'}</span>
        <span className="font-mono-num text-[11px] opacity-75">
          ({Math.max(1, Math.round(thinking.length / 4))} tok)
        </span>
      </button>
      {open && (
        <div
          className="mt-2 pl-3.5 py-2 text-[12.5px] leading-relaxed whitespace-pre-wrap break-words"
          style={{
            borderLeft: '2px solid var(--td-border-strong)',
            color: 'var(--td-ink-2)',
          }}
        >
          {thinking}
        </div>
      )}
    </div>
  );
}

export const PlaygroundPage: React.FC<PlaygroundPageProps> = ({ studio }) => {
  const {
    summaries,
    activeChatId,
    activeChat,
    activeContextCurve,
    streamingChatIds,
    isActiveStreaming,
    loadingChat,
    draftInput,
    setDraftInput,
    draftParams,
    draftSystemPrompt,
    updateActiveParams,
    updateActiveSystemPrompt,
    refreshChatList,
    selectChat,
    startNewChat,
    renameChat,
    togglePinChat,
    deleteChat,
    clearActiveChat,
    sendMessage,
    regenerateLast,
    editAndResend,
    stopGeneration,
    exportActiveChat,
  } = useChats();

  const [historyOpen, setHistoryOpen] = useState<boolean>(() => {
    try {
      const saved = window.localStorage.getItem('tendou_pg_history_open');
      return saved === null ? true : saved === '1';
    } catch {
      return true;
    }
  });
  const [inspectorOpen, setInspectorOpen] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem('tendou_pg_inspector_open') === '1';
    } catch {
      return false;
    }
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editMsgDraft, setEditMsgDraft] = useState('');
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const searchMountedRef = useRef<boolean>(false);

  useEffect(() => {
    try {
      window.localStorage.setItem('tendou_pg_history_open', historyOpen ? '1' : '0');
    } catch {
      // ignore storage errors
    }
  }, [historyOpen]);

  useEffect(() => {
    try {
      window.localStorage.setItem('tendou_pg_inspector_open', inspectorOpen ? '1' : '0');
    } catch {
      // ignore storage errors
    }
  }, [inspectorOpen]);

  const messages: ChatMessageRecord[] = activeChat?.messages ?? [];
  const ctxLimit = Number(studio?.running_settings?.ctx_size ?? studio?.settings?.ctx_size ?? 65536);

  // Current cumulative context footprint of active conversation
  const currentContextTokens = useMemo(() => {
    if (activeContextCurve.length > 0) {
      return activeContextCurve[activeContextCurve.length - 1].total_tokens;
    }
    return messages.reduce((acc, m) => acc + Math.ceil((m.content.length + (m.thinking?.length ?? 0)) / 3.8), 0);
  }, [activeContextCurve, messages]);

  const contextPct = Math.min(100, (currentContextTokens / Math.max(1, ctxLimit)) * 100);

  // Filter chat list smoothly (skip redundant refetch on initial mount with empty search)
  useEffect(() => {
    if (!searchMountedRef.current) {
      searchMountedRef.current = true;
      if (!searchQuery.trim()) return;
    }
    const timer = window.setTimeout(() => {
      void refreshChatList(searchQuery);
    }, 180);
    return () => window.clearTimeout(timer);
  }, [searchQuery, refreshChatList]);

  // Smooth auto-scroll when new tokens arrive in the active chat
  const lastMsg = messages.length > 0 ? messages[messages.length - 1] : null;
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: isActiveStreaming ? 'auto' : 'smooth' });
  }, [activeChatId, messages.length, lastMsg?.content.length, lastMsg?.thinking?.length, isActiveStreaming]);

  // Auto-resize composer textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(180, Math.max(44, el.scrollHeight))}px`;
  }, [draftInput]);

  const groupedChats = useMemo(() => groupSummariesByDate(summaries), [summaries]);

  const handleCopyMessage = (msg: ChatMessageRecord) => {
    void navigator.clipboard?.writeText(msg.content).catch(() => {});
    setCopiedMsgId(msg.id);
    window.setTimeout(() => setCopiedMsgId(null), 1500);
  };

  const handleExport = async () => {
    const md = await exportActiveChat();
    if (!md) return;
    void navigator.clipboard?.writeText(md).catch(() => {});
    try {
      const slug = (activeChat?.title || 'tendou-chat')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '')
        .slice(0, 48) || 'tendou-chat';
      const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${slug}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      // clipboard copy still succeeded
    }
    setExportNotice(true);
    window.setTimeout(() => setExportNotice(false), 1800);
  };

  const commitRename = async (chatId: string) => {
    const next = renameDraft.trim();
    setRenamingId(null);
    if (next) {
      await renameChat(chatId, next);
    }
  };

  const starters = [
    {
      title: 'Lock-Free C++20 Ring Buffer',
      prompt: 'Write a cache-line aligned SPSC lock-free ring buffer in C++20 with acquire-release memory ordering and false sharing analysis.',
    },
    {
      title: 'Long-Context Stress Test',
      prompt: 'Design a comprehensive 3-tier MoE inference architecture (VRAM + Hot RAM + NVMe) with mathematical latency bounds and Python pseudocode.',
    },
    {
      title: 'Debug Async Race Condition',
      prompt: 'Explain subtle memory-ordering bugs when publishing a lookup table across a CPU worker thread and a CUDA kernel stream, and show the exact fence pattern.',
    },
    {
      title: 'System Design Review',
      prompt: 'Compare continuous batching vs prefix-tree KV forking for multi-agent coding workloads with shared 12K-token system prompts.',
    },
  ];

  return (
    <div className="h-full flex overflow-hidden" style={{ background: 'var(--td-canvas)' }}>
      {/* Left Rail: ChatGPT-style Persistent Chat History */}
      {historyOpen && (
        <aside
          className="w-[268px] shrink-0 flex flex-col overflow-hidden"
          style={{
            background: 'var(--td-surface)',
            borderRight: '1px solid var(--td-border)',
          }}
        >
          <div className="p-3 space-y-2.5" style={{ borderBottom: '1px solid var(--td-border)' }}>
            <button
              type="button"
              onClick={startNewChat}
              className="td-btn-primary w-full py-2 px-3 rounded-xl text-[12.5px] font-medium flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              New Chat
            </button>

            <div className="relative">
              <Search
                className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                style={{ color: 'var(--td-ink-3)' }}
              />
              <input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search conversations…"
                className="w-full pl-8 pr-3 py-1.5 rounded-xl text-[12px] focus:outline-none transition-colors"
                style={{
                  background: 'var(--td-surface-2)',
                  border: '1px solid var(--td-border)',
                  color: 'var(--td-ink)',
                }}
              />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-2 py-2.5 space-y-4">
            {groupedChats.length === 0 ? (
              <div className="px-3 py-8 text-center text-[12px]" style={{ color: 'var(--td-ink-3)' }}>
                {searchQuery ? 'No matching conversations.' : 'Your conversation history will appear here.'}
              </div>
            ) : (
              groupedChats.map((group) => (
                <div key={group.label}>
                  <div
                    className="px-2.5 mb-1 text-[10.5px] font-medium tracking-wide uppercase"
                    style={{ color: 'var(--td-ink-3)' }}
                  >
                    {group.label}
                  </div>
                  <div className="space-y-0.5">
                    {group.chats.map((chat) => {
                      const isSelected = chat.id === activeChatId;
                      const isStreamingThis = streamingChatIds.includes(chat.id);
                      const isRenaming = renamingId === chat.id;

                      return (
                        <div
                          key={chat.id}
                          onClick={() => {
                            if (!isRenaming) void selectChat(chat.id);
                          }}
                          className="group relative rounded-xl px-2.5 py-2 cursor-pointer transition-colors"
                          style={{
                            background: isSelected ? 'var(--td-surface-2)' : 'transparent',
                            border: isSelected ? '1px solid var(--td-border)' : '1px solid transparent',
                          }}
                        >
                          {isRenaming ? (
                            <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                              <input
                                autoFocus
                                value={renameDraft}
                                onChange={(e) => setRenameDraft(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') void commitRename(chat.id);
                                  if (e.key === 'Escape') setRenamingId(null);
                                }}
                                onBlur={() => void commitRename(chat.id)}
                                className="flex-1 min-w-0 px-1.5 py-0.5 rounded-md text-xs focus:outline-none"
                                style={{
                                  background: 'var(--td-canvas)',
                                  border: '1px solid var(--td-border-strong)',
                                  color: 'var(--td-ink)',
                                }}
                              />
                              <button
                                type="button"
                                onClick={() => void commitRename(chat.id)}
                                className="p-1 cursor-pointer"
                                style={{ color: 'var(--td-green)' }}
                              >
                                <Check className="w-3 h-3" />
                              </button>
                            </div>
                          ) : (
                            <>
                              <div className="flex items-center gap-1.5 pr-12">
                                {isStreamingThis && (
                                  <span
                                    className="w-1.5 h-1.5 rounded-full shrink-0 td-live-dot"
                                    style={{ background: 'var(--td-accent)' }}
                                    title="Generating in background"
                                  />
                                )}
                                <span
                                  className="text-[12.5px] font-medium truncate"
                                  style={{ color: isSelected ? 'var(--td-ink)' : 'var(--td-ink-2)' }}
                                >
                                  {chat.title}
                                </span>
                              </div>

                              <div
                                className="mt-0.5 flex items-center gap-2 text-[10.5px] font-mono-num"
                                style={{ color: 'var(--td-ink-3)' }}
                              >
                                <span>{chat.turn_count}t</span>
                                {chat.total_tokens > 0 && (
                                  <>
                                    <span>·</span>
                                    <span>{formatTokensCompact(chat.total_tokens)} ctx</span>
                                  </>
                                )}
                                {chat.last_decode_tps > 0 && (
                                  <>
                                    <span>·</span>
                                    <span style={{ color: 'var(--td-accent)' }}>{chat.last_decode_tps} t/s</span>
                                  </>
                                )}
                              </div>

                              {/* Hover Actions */}
                              <div
                                className="absolute right-1.5 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center gap-0.5 px-1 py-0.5 rounded-lg"
                                style={{ background: 'var(--td-surface-2)' }}
                                onClick={(e) => e.stopPropagation()}
                              >
                                <button
                                  type="button"
                                  onClick={() => void togglePinChat(chat.id)}
                                  title={chat.pinned ? 'Unpin' : 'Pin'}
                                  className="p-1 rounded-md cursor-pointer hover:opacity-80"
                                  style={{ color: chat.pinned ? 'var(--td-accent)' : 'var(--td-ink-3)' }}
                                >
                                  <Pin className="w-3 h-3" />
                                </button>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setRenamingId(chat.id);
                                    setRenameDraft(chat.title);
                                  }}
                                  title="Rename"
                                  className="p-1 rounded-md cursor-pointer hover:opacity-80"
                                  style={{ color: 'var(--td-ink-3)' }}
                                >
                                  <Pencil className="w-3 h-3" />
                                </button>
                                <button
                                  type="button"
                                  onClick={() => void deleteChat(chat.id)}
                                  title="Delete"
                                  className="p-1 rounded-md cursor-pointer hover:opacity-80"
                                  style={{ color: 'var(--td-red)' }}
                                >
                                  <Trash2 className="w-3 h-3" />
                                </button>
                              </div>
                            </>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>
      )}

      {/* Center Column: Serene Chat Canvas + Floating Composer */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Conversation Toolbar */}
        <div
          className="h-12 px-4 shrink-0 flex items-center justify-between gap-3"
          style={{ borderBottom: '1px solid var(--td-border)' }}
        >
          <div className="flex items-center gap-2.5 min-w-0">
            <button
              type="button"
              onClick={() => setHistoryOpen((v) => !v)}
              className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors cursor-pointer"
              style={{ color: 'var(--td-ink-2)', background: 'var(--td-surface-2)' }}
              title={historyOpen ? 'Hide chat history' : 'Show chat history'}
            >
              <PanelLeft className="w-3.5 h-3.5" />
            </button>

            <div className="min-w-0">
              <div className="text-[13.5px] font-medium truncate" style={{ color: 'var(--td-ink)' }}>
                {activeChat ? activeChat.title : 'New Conversation'}
              </div>
            </div>
          </div>

          {/* Right Controls: Live Context Pill + Inspector + Export */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setInspectorOpen((v) => !v)}
              className="flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-mono-num cursor-pointer transition-colors"
              style={{
                background: inspectorOpen ? 'var(--td-accent-soft)' : 'var(--td-surface-2)',
                border: '1px solid var(--td-border)',
                color: inspectorOpen ? 'var(--td-accent)' : 'var(--td-ink-2)',
              }}
              title="Toggle Context Growth Curve & Sampling Inspector"
            >
              <LineChart className="w-3 h-3" />
              <span>
                {currentContextTokens.toLocaleString()} / {formatTokensCompact(ctxLimit)}
              </span>
              <span className="w-10 h-1 rounded-full overflow-hidden hidden sm:inline-block" style={{ background: 'var(--td-surface-3)' }}>
                <span
                  className="block h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.max(contextPct, currentContextTokens > 0 ? 4 : 0)}%`,
                    background: contextPct > 85 ? 'var(--td-red)' : 'var(--td-accent)',
                  }}
                />
              </span>
            </button>

            {activeChat && messages.length > 0 && (
              <>
                <button
                  type="button"
                  onClick={() => void handleExport()}
                  className="w-7 h-7 rounded-lg flex items-center justify-center cursor-pointer transition-colors"
                  style={{
                    color: exportNotice ? 'var(--td-green)' : 'var(--td-ink-2)',
                    background: 'var(--td-surface-2)',
                  }}
                  title="Copy conversation as Markdown"
                >
                  {exportNotice ? <Check className="w-3.5 h-3.5" /> : <Download className="w-3.5 h-3.5" />}
                </button>
                <button
                  type="button"
                  onClick={() => void clearActiveChat()}
                  className="w-7 h-7 rounded-lg flex items-center justify-center cursor-pointer transition-colors"
                  style={{ color: 'var(--td-ink-2)', background: 'var(--td-surface-2)' }}
                  title="Clear messages in this chat"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </>
            )}

            <button
              type="button"
              onClick={() => setInspectorOpen((v) => !v)}
              className="w-7 h-7 rounded-lg flex items-center justify-center cursor-pointer transition-colors"
              style={{
                color: inspectorOpen ? 'var(--td-ink)' : 'var(--td-ink-2)',
                background: 'var(--td-surface-2)',
              }}
              title="Toggle Context & Parameters Inspector"
            >
              <PanelRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Scrollable Editorial Message Stream */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-6">
          <div className="max-w-[780px] mx-auto py-8">
            {loadingChat ? (
              <div className="py-24 text-center text-xs" style={{ color: 'var(--td-ink-3)' }}>
                Restoring conversation…
              </div>
            ) : messages.length === 0 ? (
              <div className="py-12 sm:py-20 flex flex-col items-center text-center">
                <div
                  className="w-12 h-12 rounded-2xl flex items-center justify-center mb-5"
                  style={{
                    background: 'var(--td-surface)',
                    border: '1px solid var(--td-border-strong)',
                    boxShadow: 'var(--td-shadow-card)',
                    color: 'var(--td-ink)',
                  }}
                >
                  <TendouMark className="w-10 h-10" />
                </div>

                <h1 className="text-[26px] font-semibold font-display tracking-tight" style={{ color: 'var(--td-ink)' }}>
                  What shall we build?
                </h1>
                <p className="text-[13.5px] mt-2 max-w-md leading-relaxed" style={{ color: 'var(--td-ink-2)' }}>
                  Running <span className="font-mono-num" style={{ color: 'var(--td-ink)' }}>{studio?.active_model.id ?? 'local model'}</span>{' '}
                  with persistent history, background-resilient streaming, and live context-growth telemetry.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-9 w-full">
                  {starters.map((s) => (
                    <button
                      key={s.title}
                      type="button"
                      onClick={() => void sendMessage(s.prompt)}
                      className="text-left p-4 rounded-2xl transition-all cursor-pointer"
                      style={{
                        background: 'var(--td-surface)',
                        border: '1px solid var(--td-border)',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--td-border-strong)')}
                      onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--td-border)')}
                    >
                      <div className="text-[13px] font-medium" style={{ color: 'var(--td-ink)' }}>
                        {s.title}
                      </div>
                      <div className="text-[12px] mt-1 line-clamp-2 leading-relaxed" style={{ color: 'var(--td-ink-3)' }}>
                        {s.prompt}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-7">
                {messages.map((msg, msgIdx) => {
                  const isUser = msg.role === 'user';
                  const isLastAssistant = !isUser && msgIdx === messages.length - 1;

                  if (isUser) {
                    const isEditing = editingMsgId === msg.id;
                    return (
                      <div key={msg.id} className="group flex flex-col items-end">
                        {isEditing ? (
                          <div
                            className="w-full max-w-[640px] rounded-2xl p-3 space-y-2.5"
                            style={{
                              background: 'var(--td-surface)',
                              border: '1px solid var(--td-border-strong)',
                            }}
                          >
                            <textarea
                              autoFocus
                              rows={3}
                              value={editMsgDraft}
                              onChange={(e) => setEditMsgDraft(e.target.value)}
                              className="w-full bg-transparent text-[14px] leading-relaxed focus:outline-none resize-none"
                              style={{ color: 'var(--td-ink)' }}
                            />
                            <div className="flex justify-end gap-2">
                              <button
                                type="button"
                                onClick={() => setEditingMsgId(null)}
                                className="td-btn-secondary px-3 py-1 rounded-full text-xs cursor-pointer"
                              >
                                Cancel
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  const next = editMsgDraft;
                                  setEditingMsgId(null);
                                  void editAndResend(msg.id, next);
                                }}
                                className="td-btn-primary px-3.5 py-1 rounded-full text-xs font-medium cursor-pointer"
                              >
                                Save &amp; Send
                              </button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div
                              className="max-w-[82%] px-4 py-2.5 rounded-[20px] rounded-br-[6px] text-[14.5px] leading-[1.65] whitespace-pre-wrap break-words"
                              style={{
                                background: 'var(--td-user-bubble)',
                                color: 'var(--td-ink)',
                              }}
                            >
                              {msg.content}
                            </div>
                            <div className="mt-1 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                type="button"
                                onClick={() => handleCopyMessage(msg)}
                                className="p-1 rounded-md cursor-pointer hover:opacity-80"
                                style={{ color: 'var(--td-ink-3)' }}
                                title="Copy prompt"
                              >
                                {copiedMsgId === msg.id ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                              </button>
                              {!isActiveStreaming && (
                                <button
                                  type="button"
                                  onClick={() => {
                                    setEditingMsgId(msg.id);
                                    setEditMsgDraft(msg.content);
                                  }}
                                  className="p-1 rounded-md cursor-pointer hover:opacity-80"
                                  style={{ color: 'var(--td-ink-3)' }}
                                  title="Edit & re-branch from here"
                                >
                                  <Pencil className="w-3 h-3" />
                                </button>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    );
                  }

                  // Assistant message — Apple / ChatGPT editorial layout (no clunky card box)
                  return (
                    <div key={msg.id} className="group flex items-start gap-3.5">
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                        style={{
                          background: 'var(--td-surface)',
                          border: '1px solid var(--td-border)',
                          color: 'var(--td-ink)',
                        }}
                      >
                        <TendouMark className="w-6 h-6" />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <span className="text-[12px] font-medium" style={{ color: 'var(--td-ink-2)' }}>
                            {msg.model_id || studio?.active_model.id || 'Tendou'}
                          </span>
                          {msg.streaming && msg.metrics && (
                            <span
                              className="inline-flex items-center gap-1.5 text-[11px] font-mono-num"
                              style={{ color: 'var(--td-accent)' }}
                            >
                              <span className="w-1.5 h-1.5 rounded-full td-live-dot" style={{ background: 'var(--td-accent)' }} />
                              {(msg.metrics.live_tps ?? msg.metrics.decode_tps).toFixed(1)} tok/s ·{' '}
                              {msg.metrics.completion_tokens} tok
                            </span>
                          )}
                        </div>

                        {msg.thinking && (
                          <ReasoningAccordion thinking={msg.thinking} streaming={Boolean(msg.streaming && !msg.content)} />
                        )}

                        <MarkdownRenderer content={msg.content} streaming={Boolean(msg.streaming)} />

                        {!msg.streaming && (
                          <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2">
                            {msg.metrics ? (
                              <div
                                className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[11px] font-mono-num"
                                style={{ color: 'var(--td-ink-3)' }}
                              >
                                <span>
                                  TTFT <strong style={{ color: 'var(--td-ink-2)' }}>{formatDuration(msg.metrics.ttft_ms)}</strong>
                                </span>
                                <span>·</span>
                                <span>
                                  <strong style={{ color: 'var(--td-accent)' }}>{msg.metrics.decode_tps.toFixed(1)}</strong> tok/s
                                </span>
                                <span>·</span>
                                <span>
                                  {msg.metrics.total_tokens.toLocaleString()} ctx ({msg.metrics.prompt_tokens} in /{' '}
                                  {msg.metrics.completion_tokens} out)
                                </span>
                                {msg.metrics.prefix_fork_kind && msg.metrics.prefix_fork_kind !== 'cold' && (
                                  <span style={{ color: 'var(--td-green)' }}>
                                    · KV {msg.metrics.prefix_fork_kind.replace('_', ' ')}
                                  </span>
                                )}
                                {msg.metrics.finish_reason === 'length' && !msg.stopped && (
                                  <span className="inline-flex items-center gap-1.5" style={{ color: 'var(--td-amber)' }}>
                                    <span>· output limit reached</span>
                                    {isLastAssistant && !isActiveStreaming && (
                                      <button
                                        type="button"
                                        onClick={() => void sendMessage('Continue exactly where you left off without repeating.')}
                                        className="underline font-medium cursor-pointer hover:opacity-80"
                                        style={{ color: 'var(--td-accent)' }}
                                      >
                                        Continue
                                      </button>
                                    )}
                                  </span>
                                )}
                                {msg.stopped && <span style={{ color: 'var(--td-amber)' }}>· stopped</span>}
                              </div>
                            ) : (
                              <div />
                            )}

                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                type="button"
                                onClick={() => handleCopyMessage(msg)}
                                className="p-1.5 rounded-lg cursor-pointer transition-colors"
                                style={{ color: copiedMsgId === msg.id ? 'var(--td-green)' : 'var(--td-ink-3)' }}
                                title="Copy response"
                              >
                                {copiedMsgId === msg.id ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                              </button>
                              {isLastAssistant && !isActiveStreaming && (
                                <button
                                  type="button"
                                  onClick={() => void regenerateLast()}
                                  className="p-1.5 rounded-lg cursor-pointer transition-colors"
                                  style={{ color: 'var(--td-ink-3)' }}
                                  title="Regenerate response"
                                >
                                  <RotateCcw className="w-3.5 h-3.5" />
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Floating Docked Composer */}
        <div className="px-4 sm:px-6 pb-5 pt-2 shrink-0">
          <div className="max-w-[780px] mx-auto">
            <div className="td-composer rounded-[24px] p-2.5 pl-4">
              <textarea
                ref={textareaRef}
                rows={1}
                value={draftInput}
                onChange={(e) => setDraftInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    void sendMessage();
                  }
                }}
                placeholder="Message Tendou…"
                className="w-full bg-transparent text-[14.5px] leading-relaxed focus:outline-none resize-none max-h-[180px]"
                style={{ color: 'var(--td-ink)' }}
              />

              <div className="flex items-center justify-between pt-1.5 gap-2">
                <div className="flex items-center gap-2 text-[11px] font-mono-num" style={{ color: 'var(--td-ink-3)' }}>
                  <span className="truncate max-w-[200px]">{studio?.active_model.id ?? 'tendou-local'}</span>
                  <span>·</span>
                  <span>T={draftParams.temperature.toFixed(2)}</span>
                  <span>·</span>
                  <span>
                    {draftParams.max_tokens === 0
                      ? '∞ Max'
                      : draftParams.max_tokens >= 1024
                        ? `${(draftParams.max_tokens / 1024).toFixed(draftParams.max_tokens % 1024 === 0 ? 0 : 1)}K Max`
                        : `${draftParams.max_tokens} Max`}
                  </span>
                  <span>·</span>
                  <span>{draftParams.stream ? 'Stream' : 'Batch'}</span>
                </div>

                {isActiveStreaming ? (
                  <button
                    type="button"
                    onClick={() => stopGeneration()}
                    className="w-8 h-8 rounded-full flex items-center justify-center cursor-pointer transition-transform active:scale-95"
                    style={{ background: 'var(--td-red)', color: '#ffffff' }}
                    title="Stop generation"
                  >
                    <Square className="w-3 h-3 fill-current" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => void sendMessage()}
                    disabled={!draftInput.trim()}
                    className="td-btn-primary w-8 h-8 rounded-full flex items-center justify-center disabled:opacity-25 cursor-pointer"
                    title="Send (Enter)"
                  >
                    <ArrowUp className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Collapsible Drawer: Context Growth Curve & Sampling Parameters */}
      {inspectorOpen && (
        <aside
          className="w-[310px] shrink-0 overflow-y-auto p-4 space-y-5"
          style={{
            background: 'var(--td-surface)',
            borderLeft: '1px solid var(--td-border)',
          }}
        >
          <div className="flex items-center justify-between">
            <h2 className="text-[13px] font-semibold font-display" style={{ color: 'var(--td-ink)' }}>
              Context &amp; Parameters
            </h2>
            <button
              type="button"
              onClick={() => setInspectorOpen(false)}
              className="p-1 rounded-lg cursor-pointer"
              style={{ color: 'var(--td-ink-3)' }}
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Context Growth Performance */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-medium uppercase tracking-wider" style={{ color: 'var(--td-ink-3)' }}>
                Context Scaling
              </span>
              <span className="font-mono-num" style={{ color: 'var(--td-ink-2)' }}>
                {contextPct.toFixed(1)}% of {formatTokensCompact(ctxLimit)}
              </span>
            </div>

            <ContextGrowthChart points={activeContextCurve} />

            {activeContextCurve.length > 0 && (
              <div
                className="rounded-xl overflow-hidden text-[10.5px] font-mono-num"
                style={{ border: '1px solid var(--td-border)' }}
              >
                <div
                  className="grid grid-cols-4 px-2.5 py-1.5 font-medium"
                  style={{ background: 'var(--td-surface-2)', color: 'var(--td-ink-3)' }}
                >
                  <span>Turn</span>
                  <span className="text-right">Ctx</span>
                  <span className="text-right">TTFT</span>
                  <span className="text-right">TPS</span>
                </div>
                <div className="max-h-36 overflow-y-auto divide-y" style={{ borderColor: 'var(--td-border)' }}>
                  {activeContextCurve.map((pt) => (
                    <div
                      key={pt.message_id}
                      className="grid grid-cols-4 px-2.5 py-1.5"
                      style={{ color: 'var(--td-ink-2)', borderColor: 'var(--td-border)' }}
                    >
                      <span>#{pt.turn}</span>
                      <span className="text-right">{formatTokensCompact(pt.total_tokens)}</span>
                      <span className="text-right" style={{ color: 'var(--td-blue)' }}>
                        {formatDuration(pt.ttft_ms)}
                      </span>
                      <span className="text-right" style={{ color: 'var(--td-accent)' }}>
                        {pt.decode_tps.toFixed(1)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="h-px" style={{ background: 'var(--td-border)' }} />

          {/* Sampling Controls */}
          <div className="space-y-4 text-xs">
            <div>
              <label className="block text-[11px] font-medium uppercase tracking-wider mb-1.5" style={{ color: 'var(--td-ink-3)' }}>
                System Prompt
              </label>
              <textarea
                rows={4}
                value={draftSystemPrompt}
                onChange={(e) => updateActiveSystemPrompt(e.target.value)}
                className="w-full rounded-xl p-2.5 text-[12px] leading-relaxed focus:outline-none resize-none"
                style={{
                  background: 'var(--td-surface-2)',
                  border: '1px solid var(--td-border)',
                  color: 'var(--td-ink)',
                }}
              />
            </div>

            <div>
              <div className="flex justify-between mb-1.5">
                <span style={{ color: 'var(--td-ink-2)' }}>Temperature</span>
                <span className="font-mono-num" style={{ color: 'var(--td-ink)' }}>
                  {draftParams.temperature.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={draftParams.temperature}
                onChange={(e) => updateActiveParams({ temperature: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1.5">
                <span style={{ color: 'var(--td-ink-2)' }}>Top-P</span>
                <span className="font-mono-num" style={{ color: 'var(--td-ink)' }}>
                  {draftParams.top_p.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={draftParams.top_p}
                onChange={(e) => updateActiveParams({ top_p: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1.5">
                <span style={{ color: 'var(--td-ink-2)' }}>Max Output Tokens</span>
                <span
                  className="font-mono-num"
                  style={{ color: draftParams.max_tokens === 0 ? 'var(--td-accent)' : 'var(--td-ink)' }}
                >
                  {draftParams.max_tokens === 0 ? '∞ Unlimited' : draftParams.max_tokens.toLocaleString()}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {[
                  { val: 2048, label: '2K' },
                  { val: 4096, label: '4K' },
                  { val: 8192, label: '8K' },
                  { val: 16384, label: '16K' },
                  { val: 32768, label: '32K' },
                  { val: 0, label: '∞ No Limit' },
                ].map((opt) => (
                  <button
                    key={opt.val}
                    type="button"
                    onClick={() => updateActiveParams({ max_tokens: opt.val })}
                    className="py-1.5 rounded-lg text-[11px] font-mono-num cursor-pointer transition-colors"
                    style={{
                      background: draftParams.max_tokens === opt.val ? 'var(--td-cta-bg)' : 'var(--td-surface-2)',
                      color: draftParams.max_tokens === opt.val ? 'var(--td-cta-ink)' : 'var(--td-ink-2)',
                      border: '1px solid var(--td-border)',
                    }}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-[10.5px] leading-snug" style={{ color: 'var(--td-ink-3)' }}>
                {draftParams.max_tokens === 0
                  ? 'Generates until natural stop or context window capacity.'
                  : `Caps each response at ${draftParams.max_tokens.toLocaleString()} tokens.`}
              </p>
            </div>

            <div className="flex items-center justify-between pt-1">
              <span style={{ color: 'var(--td-ink-2)' }}>SSE Streaming</span>
              <button
                type="button"
                onClick={() => updateActiveParams({ stream: !draftParams.stream })}
                className="w-10 h-6 rounded-full p-0.5 transition-colors cursor-pointer"
                style={{
                  background: draftParams.stream ? 'var(--td-green)' : 'var(--td-surface-3)',
                }}
              >
                <span
                  className={`block w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    draftParams.stream ? 'translate-x-4' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>
        </aside>
      )}
    </div>
  );
};
