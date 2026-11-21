import type {
  AgentConfigsResponse,
  BenchmarkHistoryVersion,
  ChatConversation,
  ChatMessageRecord,
  ChatParams,
  ChatSummary,
  ConfigPreset,
  ConfigResponse,
  LocalModelEntry,
  QuickBenchmarkResult,
  SettingsSchemaResponse,
  StudioState,
} from '../types/api';

function resolveApiBase(): string {
  if (typeof window === 'undefined') return '';
  const proto = window.location.protocol;
  const host = window.location.hostname;
  if (proto === 'tauri:' || proto === 'file:' || host === 'tauri.localhost') {
    return 'http://127.0.0.1:8000';
  }
  return '';
}

export const API_BASE = resolveApiBase();

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch {
      // ignore parse failure
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const studioApi = {
  getState: () => requestJson<StudioState>('/api/ui/state'),
  getModels: (ctxSize = 8192) =>
    requestJson<{ models: LocalModelEntry[]; count: number; active_model_path: string | null; extra_dirs: string[] }>(
      `/api/ui/models?ctx_size=${ctxSize}`,
    ),
  scanModels: (directory: string) =>
    requestJson<{ scanned: string; models: LocalModelEntry[]; count: number; extra_dirs: string[] }>(
      '/api/ui/models/scan',
      { method: 'POST', body: JSON.stringify({ directory }) },
    ),
  getConfig: () => requestJson<ConfigResponse>('/api/ui/config'),
  saveConfig: (settings: Record<string, unknown>) =>
    requestJson<ConfigResponse>('/api/ui/config', { method: 'PUT', body: JSON.stringify(settings) }),
  getConfigSchema: () => requestJson<SettingsSchemaResponse>('/api/ui/config/schema'),
  getPresets: () => requestJson<{ presets: ConfigPreset[] }>('/api/ui/config/presets'),
  startEngine: (body?: { settings?: Record<string, unknown>; model_path?: string; save_profile?: boolean }) =>
    requestJson<{ status: string; model?: string; message?: string }>('/api/ui/engine/start', {
      method: 'POST',
      body: JSON.stringify(body ?? {}),
    }),
  stopEngine: () =>
    requestJson<{ status: string }>('/api/ui/engine/stop', { method: 'POST' }),
  restartEngine: (body?: { settings?: Record<string, unknown>; model_path?: string; save_profile?: boolean }) =>
    requestJson<{ status: string; model?: string; applied_diff?: Array<{ key: string; old: unknown; new: unknown }> }>(
      '/api/ui/engine/restart',
      { method: 'POST', body: JSON.stringify(body ?? {}) },
    ),
  getAgentConfigs: () => requestJson<AgentConfigsResponse>('/api/ui/agents/configs'),
  runBenchmark: (body: { prompt?: string; max_tokens?: number; temperature?: number }) =>
    requestJson<QuickBenchmarkResult>('/api/ui/benchmark/run', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getBenchmarkHistory: () =>
    requestJson<{ versions: BenchmarkHistoryVersion[]; count: number; last_live_run: QuickBenchmarkResult | null }>(
      '/api/ui/benchmark/history',
    ),
  // Persistent ChatGPT-style Chat History
  listChats: (query?: string) =>
    requestJson<{ chats: ChatSummary[]; count: number }>(
      query && query.trim() ? `/api/ui/chats?q=${encodeURIComponent(query.trim())}` : '/api/ui/chats',
    ),
  createChat: (body?: { title?: string; model_id?: string; system_prompt?: string; params?: Partial<ChatParams> }) =>
    requestJson<ChatConversation>('/api/ui/chats', {
      method: 'POST',
      body: JSON.stringify(body ?? {}),
    }),
  getChat: (chatId: string) => requestJson<ChatConversation>(`/api/ui/chats/${encodeURIComponent(chatId)}`),
  updateChat: (
    chatId: string,
    body: {
      title?: string;
      model_id?: string;
      system_prompt?: string;
      pinned?: boolean;
      params?: ChatParams;
      messages?: ChatMessageRecord[];
    },
  ) =>
    requestJson<ChatConversation>(`/api/ui/chats/${encodeURIComponent(chatId)}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  upsertChatMessage: (chatId: string, msg: ChatMessageRecord) =>
    requestJson<ChatConversation>(`/api/ui/chats/${encodeURIComponent(chatId)}/messages`, {
      method: 'POST',
      body: JSON.stringify(msg),
    }),
  deleteChat: (chatId: string) =>
    requestJson<{ deleted: boolean; id: string }>(`/api/ui/chats/${encodeURIComponent(chatId)}`, {
      method: 'DELETE',
    }),
  exportChat: (chatId: string) =>
    requestJson<{ id: string; title: string; markdown: string }>(`/api/ui/chats/${encodeURIComponent(chatId)}/export`),
};
