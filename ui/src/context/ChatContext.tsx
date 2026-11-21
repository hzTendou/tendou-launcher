import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE, studioApi } from '../api/client';
import type {
  ChatContextPoint,
  ChatConversation,
  ChatMessageRecord,
  ChatParams,
  ChatSummary,
  StudioState,
} from '../types/api';

const DEFAULT_SYSTEM_PROMPT =
  'You are Tendou Assistant, a concise, accurate software and systems engineering collaborator.';

const DEFAULT_PARAMS: ChatParams = {
  temperature: 0.6,
  top_p: 0.95,
  max_tokens: 4096,
  stream: true,
};

const ACTIVE_CHAT_STORAGE_KEY = 'tendou_studio_active_chat_id';
const DEFAULT_PARAMS_STORAGE_KEY = 'tendou_studio_default_params';

function loadSavedDefaultParams(): ChatParams {
  try {
    const raw = window.localStorage.getItem(DEFAULT_PARAMS_STORAGE_KEY);
    if (!raw) return DEFAULT_PARAMS;
    const parsed = JSON.parse(raw) as Partial<ChatParams>;
    return {
      temperature: typeof parsed.temperature === 'number' ? parsed.temperature : DEFAULT_PARAMS.temperature,
      top_p: typeof parsed.top_p === 'number' ? parsed.top_p : DEFAULT_PARAMS.top_p,
      max_tokens:
        typeof parsed.max_tokens === 'number' && parsed.max_tokens >= 0
          ? parsed.max_tokens
          : DEFAULT_PARAMS.max_tokens,
      stream: typeof parsed.stream === 'boolean' ? parsed.stream : DEFAULT_PARAMS.stream,
    };
  } catch {
    return DEFAULT_PARAMS;
  }
}

function estimateTokens(text: string): number {
  const cleaned = text.trim();
  if (!cleaned) return 0;
  return Math.max(1, Math.ceil(cleaned.length / 3.8));
}

function deriveLocalTitle(text: string): string {
  const firstLine = (text || '')
    .trim()
    .split(/\r?\n/)
    .map((l) => l.trim())
    .find((l) => l.length > 0);
  if (!firstLine) return 'New Chat';
  const cleaned = firstLine.replace(/\s+/g, ' ').replace(/^[#*`"' ]+/, '').trim();
  if (!cleaned) return 'New Chat';
  if (cleaned.length <= 54) return cleaned;
  const cut = cleaned.slice(0, 54);
  const lastSpace = cut.lastIndexOf(' ');
  return `${(lastSpace > 20 ? cut.slice(0, lastSpace) : cut).replace(/[.,;:]+$/, '')}…`;
}

function buildContextCurveLocal(conv: ChatConversation | null): ChatContextPoint[] {
  if (!conv) return [];
  const pts: ChatContextPoint[] = [];
  let turn = 0;
  for (const m of conv.messages) {
    if (m.role !== 'assistant' || !m.metrics) continue;
    turn += 1;
    pts.push({
      turn,
      message_id: m.id,
      prompt_tokens: m.metrics.prompt_tokens,
      completion_tokens: m.metrics.completion_tokens,
      total_tokens: m.metrics.total_tokens || m.metrics.prompt_tokens + m.metrics.completion_tokens,
      ttft_ms: m.metrics.ttft_ms,
      decode_tps: m.metrics.decode_tps,
      prefill_tps: m.metrics.prefill_tps ?? null,
      duration_ms: m.metrics.duration_ms,
      context_limit: m.metrics.context_limit,
      prefix_fork_kind: m.metrics.prefix_fork_kind ?? null,
    });
  }
  return pts;
}

function toSummary(conv: ChatConversation): ChatSummary {
  let totalTokens = 0;
  let lastDecodeTps = 0;
  let lastTtftMs = 0;
  let turnCount = 0;
  let lastPreview = '';
  for (const m of conv.messages) {
    if (m.role === 'assistant') {
      turnCount += 1;
      if (m.metrics) {
        totalTokens = Math.max(
          totalTokens,
          m.metrics.total_tokens || m.metrics.prompt_tokens + m.metrics.completion_tokens,
        );
        if (m.metrics.decode_tps > 0) lastDecodeTps = Number(m.metrics.decode_tps.toFixed(1));
        if (m.metrics.ttft_ms > 0) lastTtftMs = Number(m.metrics.ttft_ms.toFixed(1));
      }
    }
    if (m.content.trim()) {
      lastPreview = m.content.replace(/\s+/g, ' ').trim().slice(0, 110);
    }
  }
  return {
    id: conv.id,
    title: conv.title,
    model_id: conv.model_id,
    pinned: conv.pinned,
    created_at: conv.created_at,
    updated_at: conv.updated_at,
    message_count: conv.messages.length,
    turn_count: turnCount,
    total_tokens: totalTokens,
    last_decode_tps: lastDecodeTps,
    last_ttft_ms: lastTtftMs,
    last_preview: lastPreview,
  };
}

function sortSummaries(items: ChatSummary[]): ChatSummary[] {
  return [...items].sort((a, b) => {
    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
    return b.updated_at - a.updated_at;
  });
}

interface ChatContextValue {
  summaries: ChatSummary[];
  activeChatId: string | null;
  activeChat: ChatConversation | null;
  activeContextCurve: ChatContextPoint[];
  streamingChatIds: string[];
  isActiveStreaming: boolean;
  loadingList: boolean;
  loadingChat: boolean;
  draftInput: string;
  setDraftInput: (val: string) => void;
  draftParams: ChatParams;
  draftSystemPrompt: string;
  setDraftSystemPrompt: (val: string) => void;
  updateActiveParams: (patch: Partial<ChatParams>) => void;
  updateActiveSystemPrompt: (val: string) => void;
  refreshChatList: (query?: string) => Promise<void>;
  selectChat: (chatId: string) => Promise<void>;
  startNewChat: () => void;
  renameChat: (chatId: string, title: string) => Promise<void>;
  togglePinChat: (chatId: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  clearActiveChat: () => Promise<void>;
  sendMessage: (textOverride?: string) => Promise<void>;
  regenerateLast: () => Promise<void>;
  editAndResend: (messageId: string, newContent: string) => Promise<void>;
  stopGeneration: (chatId?: string) => void;
  exportActiveChat: () => Promise<string | null>;
}

const ChatContext = createContext<ChatContextValue | null>(null);

export const ChatProvider: React.FC<{ studio: StudioState | null; children: React.ReactNode }> = ({
  studio,
  children,
}) => {
  const [summaries, setSummaries] = useState<ChatSummary[]>([]);
  const [conversations, setConversations] = useState<Record<string, ChatConversation>>({});
  const [activeChatId, setActiveChatId] = useState<string | null>(() => {
    try {
      return window.localStorage.getItem(ACTIVE_CHAT_STORAGE_KEY);
    } catch {
      return null;
    }
  });
  const [streamingIds, setStreamingIds] = useState<string[]>([]);
  const [loadingList, setLoadingList] = useState<boolean>(true);
  const [loadingChat, setLoadingChat] = useState<boolean>(false);
  const [draftInput, setDraftInput] = useState<string>('');
  const [draftParams, setDraftParams] = useState<ChatParams>(() => loadSavedDefaultParams());
  const [draftSystemPrompt, setDraftSystemPrompt] = useState<string>(DEFAULT_SYSTEM_PROMPT);

  const conversationsRef = useRef<Record<string, ChatConversation>>({});
  const controllersRef = useRef<Map<string, AbortController>>(new Map());
  const studioRef = useRef<StudioState | null>(studio);
  const creatingChatRef = useRef<boolean>(false);
  const deletedChatIdsRef = useRef<Set<string>>(new Set());
  const chatEpochRef = useRef<Map<string, number>>(new Map());
  const pendingSendChatIdsRef = useRef<Set<string>>(new Set());
  const loadingChatIdsRef = useRef<Set<string>>(new Set());
  const patchTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const pendingPatchRef = useRef<Map<string, { params?: ChatParams; system_prompt?: string }>>(new Map());

  const bumpChatEpoch = useCallback((chatId: string): number => {
    const next = (chatEpochRef.current.get(chatId) ?? 0) + 1;
    chatEpochRef.current.set(chatId, next);
    return next;
  }, []);

  const scheduleMetadataPatch = useCallback(
    (chatId: string, patch: { params?: ChatParams; system_prompt?: string }) => {
      const merged = { ...(pendingPatchRef.current.get(chatId) ?? {}), ...patch };
      pendingPatchRef.current.set(chatId, merged);
      const prevTimer = patchTimersRef.current.get(chatId);
      if (prevTimer) clearTimeout(prevTimer);
      const timer = setTimeout(() => {
        patchTimersRef.current.delete(chatId);
        const payload = pendingPatchRef.current.get(chatId);
        pendingPatchRef.current.delete(chatId);
        if (!payload || deletedChatIdsRef.current.has(chatId)) return;
        void studioApi.updateChat(chatId, payload).catch(() => {});
      }, 250);
      patchTimersRef.current.set(chatId, timer);
    },
    [],
  );

  useEffect(() => {
    const timers = patchTimersRef.current;
    return () => {
      for (const t of timers.values()) clearTimeout(t);
      timers.clear();
    };
  }, []);

  useEffect(() => {
    studioRef.current = studio;
  }, [studio]);

  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  useEffect(() => {
    try {
      if (activeChatId) {
        window.localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, activeChatId);
      } else {
        window.localStorage.removeItem(ACTIVE_CHAT_STORAGE_KEY);
      }
    } catch {
      // ignore storage failure
    }
  }, [activeChatId]);

  const upsertLocalConversation = useCallback((conv: ChatConversation) => {
    setConversations((prev) => {
      const next = { ...prev, [conv.id]: conv };
      conversationsRef.current = next;
      return next;
    });
    const sum = toSummary(conv);
    setSummaries((prev) => {
      const filtered = prev.filter((item) => item.id !== conv.id);
      return sortSummaries([sum, ...filtered]);
    });
  }, []);

  const selectChat = useCallback(async (chatId: string) => {
    setActiveChatId(chatId);
    // If actively streaming or already loaded in memory, keep live in-memory state
    if (conversationsRef.current[chatId]) {
      return;
    }
    loadingChatIdsRef.current.add(chatId);
    setLoadingChat(true);
    try {
      const conv = await studioApi.getChat(chatId);
      if (deletedChatIdsRef.current.has(chatId)) return;
      setConversations((prev) => {
        const next = { ...prev, [conv.id]: conv };
        conversationsRef.current = next;
        return next;
      });
    } catch {
      setActiveChatId((cur) => (cur === chatId ? null : cur));
    } finally {
      loadingChatIdsRef.current.delete(chatId);
      setLoadingChat(loadingChatIdsRef.current.size > 0);
    }
  }, []);

  const refreshChatList = useCallback(
    async (query?: string) => {
      setLoadingList(true);
      try {
        const res = await studioApi.listChats(query);
        setSummaries((prev) => {
          // Preserve live in-memory summaries for any currently streaming chats
          const streamingSet = new Set(controllersRef.current.keys());
          const merged = res.chats.map((s) =>
            streamingSet.has(s.id) && conversationsRef.current[s.id]
              ? toSummary(conversationsRef.current[s.id])
              : s,
          );
          for (const sid of streamingSet) {
            if (!merged.some((m) => m.id === sid) && conversationsRef.current[sid]) {
              merged.push(toSummary(conversationsRef.current[sid]));
            }
          }
          return sortSummaries(merged.length > 0 ? merged : query ? [] : prev.filter((p) => streamingSet.has(p.id)));
        });
      } catch {
        // tolerate transient offline state
      } finally {
        setLoadingList(false);
      }
    },
    [],
  );

  // Initial load on app boot — restore persisted chats from disk
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingList(true);
      try {
        const res = await studioApi.listChats();
        if (cancelled) return;
        setSummaries(sortSummaries(res.chats));
        const storedId = (() => {
          try {
            return window.localStorage.getItem(ACTIVE_CHAT_STORAGE_KEY);
          } catch {
            return null;
          }
        })();
        const targetId =
          storedId && res.chats.some((c) => c.id === storedId)
            ? storedId
            : res.chats.length > 0
              ? res.chats[0].id
              : null;
        if (targetId) {
          setActiveChatId(targetId);
          const conv = await studioApi.getChat(targetId);
          if (!cancelled) {
            setConversations((prev) => {
              const next = { ...prev, [conv.id]: conv };
              conversationsRef.current = next;
              return next;
            });
          }
        }
      } catch {
        // ignore if server still booting
      } finally {
        if (!cancelled) setLoadingList(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeChat = activeChatId ? conversations[activeChatId] ?? null : null;

  const activeContextCurve = useMemo(() => buildContextCurveLocal(activeChat), [activeChat]);

  const startNewChat = useCallback(() => {
    setActiveChatId(null);
    setDraftInput('');
  }, []);

  const updateActiveParams = useCallback(
    (patch: Partial<ChatParams>) => {
      setDraftParams((prev) => {
        const next = { ...prev, ...patch };
        try {
          window.localStorage.setItem(DEFAULT_PARAMS_STORAGE_KEY, JSON.stringify(next));
        } catch {
          // ignore storage failure
        }
        return next;
      });
      if (!activeChatId || !conversationsRef.current[activeChatId]) {
        return;
      }
      const cur = conversationsRef.current[activeChatId];
      const nextParams: ChatParams = { ...cur.params, ...patch };
      const updated: ChatConversation = { ...cur, params: nextParams, updated_at: Date.now() / 1000 };
      upsertLocalConversation(updated);
      scheduleMetadataPatch(activeChatId, { params: nextParams });
    },
    [activeChatId, scheduleMetadataPatch, upsertLocalConversation],
  );

  const updateActiveSystemPrompt = useCallback(
    (val: string) => {
      if (!activeChatId || !conversationsRef.current[activeChatId]) {
        setDraftSystemPrompt(val);
        return;
      }
      const cur = conversationsRef.current[activeChatId];
      const updated: ChatConversation = { ...cur, system_prompt: val, updated_at: Date.now() / 1000 };
      upsertLocalConversation(updated);
      scheduleMetadataPatch(activeChatId, { system_prompt: val });
    },
    [activeChatId, scheduleMetadataPatch, upsertLocalConversation],
  );

  const renameChat = useCallback(
    async (chatId: string, title: string) => {
      const trimmed = title.trim();
      if (!trimmed) return;
      const existing = conversationsRef.current[chatId];
      if (existing) {
        upsertLocalConversation({ ...existing, title: trimmed, updated_at: Date.now() / 1000 });
      } else {
        setSummaries((prev) => prev.map((s) => (s.id === chatId ? { ...s, title: trimmed } : s)));
      }
      try {
        const saved = await studioApi.updateChat(chatId, { title: trimmed });
        if (deletedChatIdsRef.current.has(chatId)) return;
        const latest = conversationsRef.current[chatId];
        if (latest) {
          // Preserve live in-memory messages (especially while streaming) — merge title metadata only
          upsertLocalConversation({ ...latest, title: saved.title, updated_at: Math.max(latest.updated_at, saved.updated_at) });
        } else {
          setSummaries((prev) =>
            sortSummaries(prev.map((s) => (s.id === chatId ? { ...s, title: saved.title, updated_at: saved.updated_at } : s))),
          );
        }
      } catch {
        // keep optimistic rename
      }
    },
    [upsertLocalConversation],
  );

  const togglePinChat = useCallback(
    async (chatId: string) => {
      const existing = conversationsRef.current[chatId];
      const curPinned = existing ? existing.pinned : summaries.find((s) => s.id === chatId)?.pinned ?? false;
      const nextPinned = !curPinned;
      if (existing) {
        upsertLocalConversation({ ...existing, pinned: nextPinned });
      } else {
        setSummaries((prev) =>
          sortSummaries(prev.map((s) => (s.id === chatId ? { ...s, pinned: nextPinned } : s))),
        );
      }
      try {
        const saved = await studioApi.updateChat(chatId, { pinned: nextPinned });
        if (deletedChatIdsRef.current.has(chatId)) return;
        const latest = conversationsRef.current[chatId];
        if (latest) {
          // Preserve live in-memory messages (especially while streaming) — merge pin state only
          upsertLocalConversation({ ...latest, pinned: saved.pinned });
        } else {
          setSummaries((prev) =>
            sortSummaries(prev.map((s) => (s.id === chatId ? { ...s, pinned: saved.pinned } : s))),
          );
        }
      } catch {
        // ignore
      }
    },
    [summaries, upsertLocalConversation],
  );

  const stopGeneration = useCallback((chatId?: string) => {
    const target = chatId ?? activeChatId;
    if (!target) return;
    const ctrl = controllersRef.current.get(target);
    if (ctrl) {
      ctrl.abort();
      controllersRef.current.delete(target);
      setStreamingIds(Array.from(controllersRef.current.keys()));
    }
  }, [activeChatId]);

  const deleteChat = useCallback(
    async (chatId: string) => {
      deletedChatIdsRef.current.add(chatId);
      const pendingTimer = patchTimersRef.current.get(chatId);
      if (pendingTimer) {
        clearTimeout(pendingTimer);
        patchTimersRef.current.delete(chatId);
      }
      pendingPatchRef.current.delete(chatId);
      pendingSendChatIdsRef.current.delete(chatId);
      bumpChatEpoch(chatId);
      stopGeneration(chatId);
      setSummaries((prev) => prev.filter((s) => s.id !== chatId));
      setConversations((prev) => {
        const next = { ...prev };
        delete next[chatId];
        conversationsRef.current = next;
        return next;
      });
      setActiveChatId((cur) => (cur === chatId ? null : cur));
      try {
        await studioApi.deleteChat(chatId);
      } catch {
        // ignore
      }
    },
    [bumpChatEpoch, stopGeneration],
  );

  const clearActiveChat = useCallback(async () => {
    if (!activeChatId) {
      setDraftInput('');
      return;
    }
    bumpChatEpoch(activeChatId);
    stopGeneration(activeChatId);
    const cur = conversationsRef.current[activeChatId];
    if (!cur) return;
    const cleared: ChatConversation = {
      ...cur,
      messages: [],
      title: 'New Chat',
      updated_at: Date.now() / 1000,
    };
    upsertLocalConversation(cleared);
    try {
      await studioApi.updateChat(activeChatId, { messages: [], title: 'New Chat' });
    } catch {
      // ignore
    }
  }, [activeChatId, bumpChatEpoch, stopGeneration, upsertLocalConversation]);

  const runCompletionForChat = useCallback(
    async (conv: ChatConversation, historyUpToUser: ChatMessageRecord[]) => {
      const chatId = conv.id;
      if (controllersRef.current.has(chatId) || deletedChatIdsRef.current.has(chatId)) return;

      const myEpoch = bumpChatEpoch(chatId);
      const isStale = () =>
        deletedChatIdsRef.current.has(chatId) || chatEpochRef.current.get(chatId) !== myEpoch;

      const curStudio = studioRef.current;
      const activeModelId = curStudio?.active_model.id || conv.model_id || 'tendou-local';
      const ctxLimit = Number(
        curStudio?.running_settings?.ctx_size ?? curStudio?.settings?.ctx_size ?? 65536,
      );

      const asstId = `a-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const asstPlaceholder: ChatMessageRecord = {
        id: asstId,
        role: 'assistant',
        content: '',
        thinking: null,
        created_at: Date.now() / 1000,
        model_id: activeModelId,
        streaming: true,
        stopped: false,
        metrics: {
          ttft_ms: 0,
          decode_tps: 0,
          prefill_tps: null,
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
          duration_ms: 0,
          context_limit: ctxLimit,
          live_tps: 0,
        },
      };

      const baseMessages = [...historyUpToUser, asstPlaceholder];
      const workingConv: ChatConversation = {
        ...conv,
        model_id: activeModelId,
        messages: baseMessages,
        updated_at: Date.now() / 1000,
      };
      upsertLocalConversation(workingConv);

      const apiMessages = [
        ...(conv.system_prompt.trim() ? [{ role: 'system', content: conv.system_prompt.trim() }] : []),
        ...historyUpToUser.map((m) => ({ role: m.role, content: m.content })),
      ];
      const estPromptTokens = apiMessages.reduce((acc, m) => acc + estimateTokens(m.content) + 4, 4);

      const controller = new AbortController();
      controllersRef.current.set(chatId, controller);
      setStreamingIds(Array.from(controllersRef.current.keys()));

      const t0 = performance.now();
      let firstTokenAt: number | null = null;
      let tokCount = 0;
      let usagePromptTokens = 0;
      let usageCompletionTokens = 0;
      let usageCachedTokens = 0;
      let serverTtftMs: number | null = null;
      let serverDecodeTps: number | null = null;
      let serverPrefillTps: number | null = null;
      let serverForkKind: string | null = null;
      let finishReason: string | null = null;
      let rawContent = '';
      let rawReasoning = '';
      let stoppedByUser = false;
      let lastPersistAt = t0;
      const windowTicks: Array<{ t: number; n: number }> = [];

      const buildUpdatedAssistant = (done: boolean, isNonStream: boolean): ChatMessageRecord => {
        const now = performance.now();
        const durationMs = Math.max(1, now - t0);
        const clientTtftMs = firstTokenAt !== null ? firstTokenAt - t0 : durationMs;
        const ttftMs = serverTtftMs !== null && serverTtftMs > 0 ? serverTtftMs : clientTtftMs;
        const elapsedSec = Math.max(0.01, durationMs / 1000);
        const decSec = isNonStream
          ? elapsedSec
          : firstTokenAt !== null
            ? Math.max(0.01, (now - firstTokenAt) / 1000)
            : elapsedSec;
        const finalCompToks = usageCompletionTokens > 0 ? usageCompletionTokens : tokCount;
        const finalPromptToks = usagePromptTokens > 0 ? usagePromptTokens : estPromptTokens;
        const clientAvgTps = isNonStream
          ? finalCompToks / elapsedSec
          : finalCompToks > 1
            ? (finalCompToks - 1) / decSec
            : finalCompToks / decSec;
        const avgTps =
          done && serverDecodeTps !== null && serverDecodeTps > 0 ? serverDecodeTps : clientAvgTps;
        const prefillTps =
          serverPrefillTps !== null && serverPrefillTps > 0
            ? serverPrefillTps
            : ttftMs > 1
              ? finalPromptToks / (ttftMs / 1000)
              : null;

        const cutoff = now - 500;
        while (windowTicks.length > 1 && windowTicks[0].t < cutoff) windowTicks.shift();
        let liveTps = avgTps;
        if (!isNonStream && windowTicks.length >= 2) {
          const spanS = (now - windowTicks[0].t) / 1000;
          if (spanS >= 0.03) {
            const cnt = windowTicks.reduce((acc, x) => acc + x.n, 0) - windowTicks[0].n;
            liveTps = cnt / spanS;
          }
        }

        let thinking: string | null = rawReasoning.trim() || null;
        let visible = rawContent;
        const thinkMatch = rawContent.match(/^\s*<think>([\s\S]*?)(?:<\/think>|$)([\s\S]*)$/);
        if (thinkMatch) {
          const inlineThink = thinkMatch[1].trim();
          thinking = thinking ? `${thinking}\n${inlineThink}` : inlineThink || null;
          visible = thinkMatch[2].trimStart();
        }

        const resolvedForkKind =
          serverForkKind ?? (usageCachedTokens > 0 ? 'turn_boundary' : done ? 'cold' : null);

        return {
          ...asstPlaceholder,
          content: visible,
          thinking,
          streaming: !done,
          stopped: stoppedByUser,
          metrics: {
            ttft_ms: Number(ttftMs.toFixed(1)),
            decode_tps: Number(avgTps.toFixed(2)),
            prefill_tps: prefillTps !== null ? Number(prefillTps.toFixed(1)) : null,
            prompt_tokens: finalPromptToks,
            completion_tokens: finalCompToks,
            total_tokens: finalPromptToks + finalCompToks,
            duration_ms: Number(durationMs.toFixed(1)),
            context_limit: ctxLimit,
            prefix_fork_kind: resolvedForkKind,
            finish_reason: finishReason,
            live_tps: Number((done ? avgTps : liveTps).toFixed(2)),
          },
        };
      };

      const applyAssistantToState = (done: boolean, isNonStream: boolean, forcePersist = false) => {
        // Never mutate state or persist back to disk if this chat was deleted, cleared, or superseded
        if (isStale()) return;
        const updatedMsg = buildUpdatedAssistant(done, isNonStream);
        const curConv = conversationsRef.current[chatId];
        if (!curConv) return;
        const nextMsgs = curConv.messages.map((m) => (m.id === asstId ? updatedMsg : m));
        upsertLocalConversation({
          ...curConv,
          messages: nextMsgs,
          updated_at: Date.now() / 1000,
        });
        const now = performance.now();
        if (forcePersist || done || now - lastPersistAt >= 2500) {
          lastPersistAt = now;
          const persistable: ChatMessageRecord = { ...updatedMsg, streaming: false };
          void studioApi.upsertChatMessage(chatId, persistable).catch(() => {});
        }
      };

      const ingestTimingsAndUsage = (obj: Record<string, unknown>) => {
        const usage = obj.usage as Record<string, unknown> | undefined;
        if (usage) {
          if (typeof usage.prompt_tokens === 'number' && usage.prompt_tokens > 0) {
            usagePromptTokens = usage.prompt_tokens;
          }
          if (typeof usage.completion_tokens === 'number' && usage.completion_tokens > 0) {
            usageCompletionTokens = usage.completion_tokens;
          }
          const details = usage.prompt_tokens_details as Record<string, unknown> | undefined;
          const cached =
            typeof details?.cached_tokens === 'number'
              ? details.cached_tokens
              : typeof usage.cache_read_input_tokens === 'number'
                ? usage.cache_read_input_tokens
                : 0;
          if (cached > 0) usageCachedTokens = cached;
        }
        const timings = obj.timings as Record<string, unknown> | undefined;
        if (timings) {
          if (typeof timings.ttft_ms === 'number' && timings.ttft_ms > 0) serverTtftMs = timings.ttft_ms;
          if (typeof timings.decode_tps === 'number' && timings.decode_tps > 0) serverDecodeTps = timings.decode_tps;
          if (typeof timings.prefill_tps === 'number' && timings.prefill_tps > 0) serverPrefillTps = timings.prefill_tps;
          if (typeof timings.prefix_fork_kind === 'string' && timings.prefix_fork_kind) {
            serverForkKind = timings.prefix_fork_kind;
          }
          if (typeof timings.cached_tokens === 'number' && timings.cached_tokens > 0) {
            usageCachedTokens = timings.cached_tokens;
          }
        }
      };

      try {
        const res = await fetch(`${API_BASE}/v1/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Tendou-Session-Id': `studio-${chatId}`,
          },
          signal: controller.signal,
          body: JSON.stringify({
            model: activeModelId,
            user: `studio-${chatId}`,
            messages: apiMessages,
            temperature: conv.params.temperature,
            top_p: conv.params.top_p,
            max_tokens: conv.params.max_tokens,
            stream: conv.params.stream,
            ...(conv.params.stream ? { stream_options: { include_usage: true } } : {}),
          }),
        });

        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`HTTP ${res.status}: ${errText}`);
        }

        if (!conv.params.stream) {
          const json = await res.json();
          firstTokenAt = performance.now();
          ingestTimingsAndUsage(json);
          if (typeof json.choices?.[0]?.finish_reason === 'string') {
            finishReason = json.choices[0].finish_reason;
          }
          rawContent = json.choices?.[0]?.message?.content ?? '';
          rawReasoning = json.choices?.[0]?.message?.reasoning_content ?? '';
          tokCount =
            usageCompletionTokens ||
            Math.max(1, estimateTokens(`${rawContent} ${rawReasoning}`));
          applyAssistantToState(true, true, true);
        } else {
          const reader = res.body?.getReader();
          const decoder = new TextDecoder();
          let buf = '';
          const processSseLine = (line: string) => {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith('data:')) return;
            const payload = trimmedLine.slice(5).trim();
            if (!payload || payload === '[DONE]') return;
            let parsed: Record<string, unknown>;
            try {
              parsed = JSON.parse(payload);
            } catch {
              return;
            }
            if (parsed.error) {
              const errMsg =
                typeof parsed.error === 'string'
                  ? parsed.error
                  : typeof parsed.error === 'object'
                    ? String((parsed.error as Record<string, unknown>).message || 'Engine inference error')
                    : 'Engine inference error';
              throw new Error(errMsg);
            }
            ingestTimingsAndUsage(parsed);
            const choices = parsed.choices as Array<Record<string, unknown>> | undefined;
            if (typeof choices?.[0]?.finish_reason === 'string' && choices[0].finish_reason) {
              finishReason = choices[0].finish_reason;
            }
            const delta = choices?.[0]?.delta as Record<string, unknown> | undefined;
            const rPiece = typeof delta?.reasoning_content === 'string' ? delta.reasoning_content : '';
            const cPiece = typeof delta?.content === 'string' ? delta.content : '';
            if (rPiece || cPiece) {
              const now = performance.now();
              if (firstTokenAt === null) firstTokenAt = now;
              const chunkToks = Math.max(1, Math.round((rPiece.length + cPiece.length) / 4));
              tokCount += chunkToks;
              windowTicks.push({ t: now, n: chunkToks });
              if (rPiece) rawReasoning += rPiece;
              if (cPiece) rawContent += cPiece;
              applyAssistantToState(false, false, false);
            }
          };
          if (reader) {
            while (true) {
              const { value, done } = await reader.read();
              if (done) break;
              buf += decoder.decode(value, { stream: true });
              const lines = buf.split('\n');
              buf = lines.pop() ?? '';
              for (const line of lines) {
                processSseLine(line);
              }
            }
            buf += decoder.decode();
            if (buf.trim()) {
              for (const line of buf.split('\n')) {
                processSseLine(line);
              }
            }
          }
          applyAssistantToState(true, false, true);
        }
      } catch (err) {
        if ((err as Error).name === 'AbortError') {
          stoppedByUser = true;
        } else {
          rawContent += `\n\n[Error: ${err instanceof Error ? err.message : String(err)}]`;
        }
        applyAssistantToState(true, !conv.params.stream, true);
      } finally {
        if (controllersRef.current.get(chatId) === controller) {
          controllersRef.current.delete(chatId);
        }
        setStreamingIds(Array.from(controllersRef.current.keys()));
      }
    },
    [bumpChatEpoch, upsertLocalConversation],
  );

  const sendMessage = useCallback(
    async (textOverride?: string) => {
      const trimmed = (textOverride !== undefined ? textOverride : draftInput).trim();
      if (!trimmed) return;

      const curId = activeChatId;
      if (
        curId &&
        (controllersRef.current.has(curId) ||
          pendingSendChatIdsRef.current.has(curId) ||
          loadingChatIdsRef.current.has(curId))
      ) {
        return;
      }
      if (!curId && creatingChatRef.current) return;

      let conv: ChatConversation | null = curId ? conversationsRef.current[curId] ?? null : null;
      if (curId && !conv) return;
      const activeModelId = studioRef.current?.active_model.id || 'tendou-local';

      if (!conv) {
        creatingChatRef.current = true;
        if (textOverride === undefined) setDraftInput('');
        const title = deriveLocalTitle(trimmed);
        try {
          conv = await studioApi.createChat({
            title,
            model_id: activeModelId,
            system_prompt: draftSystemPrompt,
            params: draftParams,
          });
        } catch {
          const now = Date.now() / 1000;
          conv = {
            id: `chat_${Date.now()}_local`,
            title,
            model_id: activeModelId,
            system_prompt: draftSystemPrompt,
            pinned: false,
            created_at: now,
            updated_at: now,
            params: { ...draftParams },
            messages: [],
          };
        } finally {
          creatingChatRef.current = false;
        }
        deletedChatIdsRef.current.delete(conv.id);
        setActiveChatId(conv.id);
      }

      if (pendingSendChatIdsRef.current.has(conv.id) || controllersRef.current.has(conv.id)) return;
      pendingSendChatIdsRef.current.add(conv.id);

      const userMsg: ChatMessageRecord = {
        id: `u-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        role: 'user',
        content: trimmed,
        created_at: Date.now() / 1000,
      };

      const nextTitle = conv.title === 'New Chat' ? deriveLocalTitle(trimmed) : conv.title;
      const nextHistory = [...conv.messages, userMsg];
      const updatedConv: ChatConversation = {
        ...conv,
        title: nextTitle,
        messages: nextHistory,
        updated_at: Date.now() / 1000,
      };
      upsertLocalConversation(updatedConv);
      if (textOverride === undefined) setDraftInput('');

      try {
        try {
          await studioApi.upsertChatMessage(conv.id, userMsg);
        } catch {
          // continue inference even if disk persist transiently fails
        }
        if (deletedChatIdsRef.current.has(conv.id)) return;
        void runCompletionForChat(updatedConv, nextHistory);
      } finally {
        pendingSendChatIdsRef.current.delete(conv.id);
      }
    },
    [activeChatId, draftInput, draftParams, draftSystemPrompt, runCompletionForChat, upsertLocalConversation],
  );

  const regenerateLast = useCallback(async () => {
    if (
      !activeChatId ||
      controllersRef.current.has(activeChatId) ||
      pendingSendChatIdsRef.current.has(activeChatId)
    ) {
      return;
    }
    const conv = conversationsRef.current[activeChatId];
    if (!conv || conv.messages.length === 0) return;

    let cutIdx = conv.messages.length;
    while (cutIdx > 0 && conv.messages[cutIdx - 1].role === 'assistant') {
      cutIdx -= 1;
    }
    if (cutIdx === 0) return;
    pendingSendChatIdsRef.current.add(conv.id);
    try {
      bumpChatEpoch(conv.id);
      const trimmedHistory = conv.messages.slice(0, cutIdx);
      const updatedConv: ChatConversation = { ...conv, messages: trimmedHistory, updated_at: Date.now() / 1000 };
      upsertLocalConversation(updatedConv);
      try {
        await studioApi.updateChat(conv.id, { messages: trimmedHistory });
      } catch {
        // ignore
      }
      if (deletedChatIdsRef.current.has(conv.id)) return;
      void runCompletionForChat(updatedConv, trimmedHistory);
    } finally {
      pendingSendChatIdsRef.current.delete(conv.id);
    }
  }, [activeChatId, bumpChatEpoch, runCompletionForChat, upsertLocalConversation]);

  const editAndResend = useCallback(
    async (messageId: string, newContent: string) => {
      const trimmed = newContent.trim();
      if (
        !activeChatId ||
        !trimmed ||
        controllersRef.current.has(activeChatId) ||
        pendingSendChatIdsRef.current.has(activeChatId)
      ) {
        return;
      }
      const conv = conversationsRef.current[activeChatId];
      if (!conv) return;
      const idx = conv.messages.findIndex((m) => m.id === messageId);
      if (idx < 0) return;
      pendingSendChatIdsRef.current.add(conv.id);
      try {
        bumpChatEpoch(conv.id);
        const updatedUser: ChatMessageRecord = {
          ...conv.messages[idx],
          content: trimmed,
          created_at: Date.now() / 1000,
        };
        const truncated = [...conv.messages.slice(0, idx), updatedUser];
        const updatedConv: ChatConversation = {
          ...conv,
          title: idx === 0 ? deriveLocalTitle(trimmed) : conv.title,
          messages: truncated,
          updated_at: Date.now() / 1000,
        };
        upsertLocalConversation(updatedConv);
        try {
          await studioApi.updateChat(conv.id, { messages: truncated, title: updatedConv.title });
        } catch {
          // ignore
        }
        if (deletedChatIdsRef.current.has(conv.id)) return;
        void runCompletionForChat(updatedConv, truncated);
      } finally {
        pendingSendChatIdsRef.current.delete(conv.id);
      }
    },
    [activeChatId, bumpChatEpoch, runCompletionForChat, upsertLocalConversation],
  );

  const exportActiveChat = useCallback(async (): Promise<string | null> => {
    if (!activeChatId) return null;
    try {
      const res = await studioApi.exportChat(activeChatId);
      return res.markdown;
    } catch {
      return null;
    }
  }, [activeChatId]);

  const value: ChatContextValue = {
    summaries,
    activeChatId,
    activeChat,
    activeContextCurve,
    streamingChatIds: streamingIds,
    isActiveStreaming: Boolean(activeChatId && streamingIds.includes(activeChatId)),
    loadingList,
    loadingChat,
    draftInput,
    setDraftInput,
    draftParams: activeChat ? activeChat.params : draftParams,
    draftSystemPrompt: activeChat ? activeChat.system_prompt : draftSystemPrompt,
    setDraftSystemPrompt,
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
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};

export function useChats(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error('useChats must be used inside ChatProvider');
  }
  return ctx;
}
