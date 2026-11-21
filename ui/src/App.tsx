import React, { useEffect, useState } from 'react';
import { studioApi } from './api/client';
import { Sidebar } from './components/Sidebar';
import { TopTelemetryBar } from './components/TopTelemetryBar';
import { ChatProvider, useChats } from './context/ChatContext';
import { useTelemetry } from './hooks/useTelemetry';
import { AgentHubPage } from './pages/AgentHubPage';
import { BenchmarksLogsPage } from './pages/BenchmarksLogsPage';
import { DashboardPage } from './pages/DashboardPage';
import { ModelsPage } from './pages/ModelsPage';
import { PlaygroundPage } from './pages/PlaygroundPage';
import { SettingsPage } from './pages/SettingsPage';
import type { NavTab } from './types/api';

const THEME_STORAGE_KEY = 'tendou_studio_theme';

const StudioShell: React.FC<{
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  studio: ReturnType<typeof useTelemetry>['studio'];
  telemetry: ReturnType<typeof useTelemetry>['telemetry'];
  connected: boolean;
  error: string | null;
  refreshState: () => Promise<void>;
}> = ({ theme, onToggleTheme, studio, telemetry, connected, error, refreshState }) => {
  const [activeTab, setActiveTab] = useState<NavTab>('playground');
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const { streamingChatIds } = useChats();

  const runLifecycle = async (action: 'start' | 'restart' | 'stop') => {
    setBusyAction(action);
    setActionError(null);
    try {
      if (action === 'start') await studioApi.startEngine();
      else if (action === 'restart') await studioApi.restartEngine();
      else await studioApi.stopEngine();
      await refreshState();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : `Engine ${action} failed`);
    } finally {
      setBusyAction(null);
    }
  };

  return (
    <div
      className="h-screen w-screen overflow-hidden flex"
      style={{ background: 'var(--td-canvas)', color: 'var(--td-ink)' }}
    >
      <Sidebar
        activeTab={activeTab}
        onSelect={setActiveTab}
        studio={studio}
        theme={theme}
        onToggleTheme={onToggleTheme}
        streamingChatsCount={streamingChatIds.length}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopTelemetryBar
          studio={studio}
          telemetry={telemetry}
          connected={connected}
          busyAction={busyAction}
          onStart={() => void runLifecycle('start')}
          onRestart={() => void runLifecycle('restart')}
          onStop={() => void runLifecycle('stop')}
        />

        {(error || actionError) && (
          <div
            className="px-5 py-2 text-xs flex items-center justify-between"
            style={{
              background: 'var(--td-red-soft)',
              borderBottom: '1px solid var(--td-border)',
              color: 'var(--td-red)',
            }}
          >
            <span>{actionError || error}</span>
            {actionError && (
              <button
                type="button"
                onClick={() => setActionError(null)}
                className="underline font-medium cursor-pointer"
              >
                Dismiss
              </button>
            )}
          </div>
        )}

        <main className="flex-1 min-h-0 overflow-y-auto">
          {activeTab === 'playground' && <PlaygroundPage studio={studio} />}
          {activeTab === 'dashboard' && <DashboardPage studio={studio} telemetry={telemetry} />}
          {activeTab === 'models' && (
            <ModelsPage
              ctxSize={Number(studio?.settings?.ctx_size ?? 65536)}
              onModelLoaded={() => void refreshState()}
            />
          )}
          {activeTab === 'settings' && <SettingsPage onApplied={() => void refreshState()} />}
          {activeTab === 'agents' && <AgentHubPage telemetry={telemetry} />}
          {activeTab === 'benchmarks' && <BenchmarksLogsPage />}
        </main>
      </div>
    </div>
  );
};

export const App: React.FC = () => {
  const { studio, telemetry, connected, error, refreshState } = useTelemetry();
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
      return saved === 'light' ? 'light' : 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.classList.toggle('dark', theme === 'dark');
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  return (
    <ChatProvider studio={studio}>
      <StudioShell
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
        studio={studio}
        telemetry={telemetry}
        connected={connected}
        error={error}
        refreshState={refreshState}
      />
    </ChatProvider>
  );
};

export default App;
