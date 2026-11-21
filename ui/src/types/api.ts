export type EngineStatus = 'stopped' | 'starting' | 'ready' | 'restarting' | 'error';

export type NavTab =
  | 'dashboard'
  | 'playground'
  | 'models'
  | 'settings'
  | 'agents'
  | 'benchmarks';

export interface TpsHistoryPoint {
  ts: number;
  tps: number;
  vram_mb: number;
  generating: boolean;
}

export interface TelemetrySnapshot {
  ts: number;
  engine_status: EngineStatus;
  tps: {
    live_decode_tps: number;
    ema_decode_tps: number;
    peak_decode_tps: number;
    last_prefill_tps: number;
    last_ttft_ms: number;
    last_turn_decode_tps: number;
    last_turn_tokens: number;
    is_generating: boolean;
    tokens_this_turn: number;
    tokens_session_total: number;
    requests_total: number;
    active_request_id: string | null;
    last_prefix_fork_kind: string;
    tps_history: TpsHistoryPoint[];
  };
  gpu: {
    name: string;
    used_mb: number;
    total_mb: number;
    peak_mb: number;
    util_pct: number;
    temp_c: number;
  };
  system: {
    ram_used_gb: number;
    ram_available_gb: number;
    ram_total_gb: number;
    ram_cached_gb?: number;
    cpu_util_pct: number;
    engine_rss_mb?: number;
    engine_locked_mb?: number;
    engine_threads?: number;
    engine_cpu_pct?: number;
    hot_ram_mb?: number;
    hot_ram_mode?: string;
  };
  moe_cache: {
    tier0_vram_hits: number;
    tier1_ram_hits: number;
    tier2_nvme_misses: number;
    tier0_vram_pct: number;
    tier1_ram_pct: number;
    tier2_nvme_pct: number;
    avg_dynamic_k: number;
  };
  prefix_cache: {
    max_slots: number;
    slots_used: number;
    hit_rate_pct: number;
    fork_counts: {
      root_anchor: number;
      turn_boundary: number;
      exact: number;
      cold: number;
    };
  };
  swarm: {
    active_http_requests: number;
    queue_depth: number;
    active_session_id: string | null;
    last_session_id: string | null;
    waiters: Array<Record<string, unknown>>;
  };
}

export interface StudioState {
  status: EngineStatus;
  last_error: string | null;
  version: string;
  os: 'linux' | 'windows';
  platform: string;
  host: string;
  port: number;
  exe_path: string | null;
  active_model: {
    id: string;
    path: string | null;
    backend: string;
    uptime_sec: number;
  };
  hardware: {
    has_gpu: boolean;
    gpu_name: string;
    vram_total_mb: number;
    vram_free_mb: number;
    cpu_cores_physical: number;
    cpu_cores_logical: number;
    has_avx512_vnni: boolean;
    has_avx2: boolean;
    ram_total_gb: number;
    ram_avail_gb: number;
    is_windows: boolean;
  };
  autotune_plan: Record<string, unknown> | null;
  settings: Record<string, unknown>;
  running_settings?: Record<string, unknown>;
  pending_restart: boolean;
  telemetry: TelemetrySnapshot;
}

export interface LocalModelEntry {
  path: string;
  name: string;
  filename: string;
  architecture: string;
  block_count: number;
  expert_count: number;
  expert_used_count: number;
  file_size_gb: number;
  quant_type: string;
  is_moe: boolean;
  is_mtp_head: boolean;
  shards: number;
  active: boolean;
  autotune: {
    threads: number;
    gpu_layers: number;
    vram_cache_mb: number;
    moe_cache_mib: number;
    cache_type_k: string;
    cache_type_v: string;
    vram_layer_cap: string;
    ram_lock_policy: string;
    boost_level?: number;
    mtp_mode: string;
    estimated_vram_peak_mb: number;
    estimated_decode_tps: number;
    expert_layer_vram_mb: number;
    gpu_layer_policy: string;
    kv_policy: string;
  };
}

export interface SettingFieldSchema {
  key: string;
  label: string;
  cli_flag: string;
  group: string;
  type: 'string' | 'number' | 'slider' | 'boolean' | 'tribool' | 'select';
  min?: number;
  max?: number;
  step?: number;
  choices?: Array<string | number>;
  quick_values?: Array<string | number>;
  nullable?: boolean;
  placeholder?: string;
  description: string;
  safety_warning?: string | null;
}

export interface SettingsSchemaResponse {
  groups: Array<{ id: string; label: string; description: string }>;
  fields: SettingFieldSchema[];
  total_flags: number;
  ram_total_gb: number;
  vram_total_mb: number;
}

export interface ConfigPreset {
  id: string;
  name: string;
  badge: string;
  description: string;
  overrides: Record<string, unknown>;
}

export interface ConfigResponse {
  settings: Record<string, unknown>;
  running_settings?: Record<string, unknown>;
  diff_from_running: Array<{ key: string; old: unknown; new: unknown }>;
  cli: { linux: string; windows: string };
  profile_path?: string;
}

export interface AgentConfigsResponse {
  base_url: string;
  openai_url: string;
  anthropic_url: string;
  model_id: string;
  claude_code: { linux: string; windows: string };
  opencode: { filename: string; content: string };
  cline: Record<string, unknown>;
  curl: string;
  python: string;
}

export interface QuickBenchmarkResult {
  timestamp: string;
  version: string;
  model_id: string;
  cold: { pass: string; ttft_ms: number; decode_tps: number; tokens: number; elapsed_s: number; preview: string };
  warm: { pass: string; ttft_ms: number; decode_tps: number; tokens: number; elapsed_s: number; preview: string };
  ttft_speedup_x: number;
  vram_used_mb: number;
  vram_peak_mb: number;
}

export interface BenchmarkHistoryVersion {
  label: string;
  version: string;
  timestamp?: string;
  file: string;
  models: Record<
    string,
    {
      vram_peak_mib: number;
      gpu_layers?: number;
      cache_type_k?: string;
      workloads: Array<{
        workload_id: string;
        category: string;
        name: string;
        opencode?: { cold_ttft_ms: number; warm_ttft_ms: number; best_decode_tps: number; warm_decode_tps: number };
        claude_code?: { cold_ttft_ms: number; warm_ttft_ms: number; best_decode_tps: number; warm_decode_tps: number };
        cline?: { cold_ttft_ms: number; warm_ttft_ms: number; best_decode_tps: number; warm_decode_tps: number };
      }>;
    }
  >;
}

export interface LogEntry {
  seq: number;
  ts: number;
  source: 'engine' | 'server';
  level: string;
  line: string;
}

export interface ChatTurnMetrics {
  ttft_ms: number;
  decode_tps: number;
  prefill_tps?: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  duration_ms: number;
  context_limit: number;
  prefix_fork_kind?: string | null;
  finish_reason?: string | null;
  live_tps?: number;
}

export interface ChatMessageRecord {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking?: string | null;
  created_at: number;
  model_id?: string | null;
  metrics?: ChatTurnMetrics | null;
  stopped?: boolean;
  streaming?: boolean;
}

export interface ChatParams {
  temperature: number;
  top_p: number;
  max_tokens: number;
  stream: boolean;
}

export interface ChatContextPoint {
  turn: number;
  message_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  ttft_ms: number;
  decode_tps: number;
  prefill_tps?: number | null;
  duration_ms: number;
  context_limit: number;
  prefix_fork_kind?: string | null;
}

export interface ChatSummary {
  id: string;
  title: string;
  model_id?: string | null;
  pinned: boolean;
  created_at: number;
  updated_at: number;
  message_count: number;
  turn_count: number;
  total_tokens: number;
  last_decode_tps: number;
  last_ttft_ms: number;
  last_preview: string;
}

export interface ChatConversation {
  id: string;
  title: string;
  model_id?: string | null;
  system_prompt: string;
  pinned: boolean;
  created_at: number;
  updated_at: number;
  params: ChatParams;
  messages: ChatMessageRecord[];
  context_curve?: ChatContextPoint[];
}
