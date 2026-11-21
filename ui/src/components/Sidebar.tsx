import React from 'react';
import {
  Activity,
  BarChart3,
  Bot,
  Box,
  MessageSquare,
  Moon,
  SlidersHorizontal,
  Sun,
} from 'lucide-react';
import type { NavTab, StudioState } from '../types/api';
import { TendouMark } from './TendouMark';

interface SidebarProps {
  activeTab: NavTab;
  onSelect: (tab: NavTab) => void;
  studio: StudioState | null;
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  streamingChatsCount: number;
}

const NAV_ITEMS: Array<{
  id: NavTab;
  label: string;
  icon: React.FC<{ className?: string; style?: React.CSSProperties }>;
}> = [
  { id: 'playground', label: 'Chats', icon: MessageSquare },
  { id: 'dashboard', label: 'Overview', icon: Activity },
  { id: 'models', label: 'Models', icon: Box },
  { id: 'settings', label: 'Engine', icon: SlidersHorizontal },
  { id: 'agents', label: 'Agents', icon: Bot },
  { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelect,
  studio,
  theme,
  onToggleTheme,
  streamingChatsCount,
}) => {
  return (
    <aside
      className="w-[232px] shrink-0 flex flex-col justify-between select-none"
      style={{
        background: 'var(--td-sidebar)',
        borderRight: '1px solid var(--td-border)',
      }}
    >
      <div>
        {/* Brand Identity */}
        <div
          className="h-14 px-4 flex items-center justify-between"
          style={{ borderBottom: '1px solid var(--td-border)' }}
        >
          <div className="flex items-center gap-2.5 min-w-0">
            <div
              className="w-8 h-8 rounded-[10px] flex items-center justify-center shrink-0"
              style={{
                background: 'var(--td-surface-2)',
                border: '1px solid var(--td-border-strong)',
                color: 'var(--td-ink)',
              }}
            >
              <TendouMark className="w-7 h-7" />
            </div>
            <div className="min-w-0">
              <div className="text-[13.5px] font-semibold font-display tracking-tight truncate" style={{ color: 'var(--td-ink)' }}>
                Tendou Studio
              </div>
              <div className="text-[10.5px] font-mono-num truncate" style={{ color: 'var(--td-ink-3)' }}>
                v{studio?.version ?? '0.16.3-stable'}
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={onToggleTheme}
            title={theme === 'dark' ? 'Switch to Alabaster Light' : 'Switch to Obsidian Dark'}
            className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors cursor-pointer"
            style={{
              color: 'var(--td-ink-2)',
              background: 'transparent',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--td-surface-2)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            {theme === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="p-2.5 space-y-0.5">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelect(item.id)}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-[11px] text-left transition-all cursor-pointer"
                style={{
                  background: active ? 'var(--td-surface)' : 'transparent',
                  border: active ? '1px solid var(--td-border)' : '1px solid transparent',
                  boxShadow: active ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                  color: active ? 'var(--td-ink)' : 'var(--td-ink-2)',
                }}
              >
                <Icon
                  className="w-[15px] h-[15px] shrink-0"
                  style={{ color: active ? 'var(--td-accent)' : 'var(--td-ink-3)' }}
                />
                <span className="text-[13px] font-medium flex-1 truncate">{item.label}</span>

                {item.id === 'playground' && streamingChatsCount > 0 && (
                  <span
                    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-mono-num"
                    style={{ background: 'var(--td-accent-soft)', color: 'var(--td-accent)' }}
                    title={`${streamingChatsCount} conversation(s) streaming in background`}
                  >
                    <span className="w-1.5 h-1.5 rounded-full td-live-dot" style={{ background: 'var(--td-accent)' }} />
                    {streamingChatsCount}
                  </span>
                )}

                {item.id === 'settings' && studio?.pending_restart && (
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: 'var(--td-accent)' }}
                    title="Pending Hot-Swap changes"
                  />
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Hardware & Active Model Footnote */}
      <div className="p-3.5" style={{ borderTop: '1px solid var(--td-border)' }}>
        <div
          className="p-3 rounded-[13px] space-y-1.5"
          style={{
            background: 'var(--td-surface)',
            border: '1px solid var(--td-border)',
          }}
        >
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider font-medium" style={{ color: 'var(--td-ink-3)' }}>
            <span>Active Model</span>
            <span className="font-mono-num" style={{ color: 'var(--td-accent)' }}>
              {studio?.hardware.has_avx512_vnni ? 'AVX-512' : 'AVX2'}
            </span>
          </div>
          <div
            className="text-[12px] font-medium truncate"
            style={{ color: 'var(--td-ink)' }}
            title={studio?.active_model.path ?? ''}
          >
            {studio?.active_model.id ?? 'No model loaded'}
          </div>
          <div className="text-[10.5px] font-mono-num truncate" style={{ color: 'var(--td-ink-3)' }}>
            {studio?.hardware.gpu_name?.replace('NVIDIA GeForce ', '') ?? 'CUDA GPU'} ·{' '}
            {studio?.hardware.cpu_cores_physical ?? 8}C/{studio?.hardware.cpu_cores_logical ?? 16}T
          </div>
        </div>
      </div>
    </aside>
  );
};
