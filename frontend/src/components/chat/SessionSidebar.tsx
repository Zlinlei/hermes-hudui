import { useTranslation } from '../../i18n'

interface SessionSidebarProps {
  sessions: Array<{ id: string; title: string; backend_type: string; is_active: boolean }>
  activeSessionId: string | null
  onSelect: (sessionId: string) => void
  onCreate: () => void
  loading?: boolean
  streamingSessionIds?: string[]
}

export default function SessionSidebar({
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  loading,
  streamingSessionIds,
}: SessionSidebarProps) {
  const { t } = useTranslation()

  return (
    <div
      className="h-full flex flex-col"
      style={{ borderRight: '1px solid var(--hud-border)', background: 'var(--hud-bg-panel)' }}
    >
      {/* Header */}
      <div
        className="px-3 py-2 border-b flex items-center justify-between"
        style={{ borderColor: 'var(--hud-border)' }}
      >
        <span className="text-[11px] uppercase tracking-widest" style={{ color: 'var(--hud-text-dim)' }}>
          {t('sessions.title')}
        </span>
        <button
          onClick={onCreate}
          disabled={loading}
          className="text-[11px] px-2 py-0.5 cursor-pointer"
          style={{
            background: 'var(--hud-primary)',
            color: 'var(--hud-bg-deep)',
            opacity: loading ? 0.5 : 1,
          }}
        >
          + {t('chat.newSession')}
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="p-3 text-[12px]" style={{ color: 'var(--hud-text-dim)' }}>
            {t('sessions.noSessions')}
          </div>
        ) : (
          sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => onSelect(session.id)}
              className="w-full px-3 py-2 text-left cursor-pointer"
              style={{
                background: activeSessionId === session.id ? 'var(--hud-bg-hover)' : 'transparent',
                borderLeft: activeSessionId === session.id ? '2px solid var(--hud-primary)' : '2px solid transparent',
              }}
            >
              <div className="text-[13px] font-bold flex items-center justify-between gap-1" style={{ color: 'var(--hud-text)' }}>
                <span className="truncate">{session.title}</span>
                {streamingSessionIds?.includes(session.id) && (
                  <span className="text-[9px] animate-pulse shrink-0" style={{ color: 'var(--hud-primary)' }}>
                    ●
                  </span>
                )}
              </div>
              <div className="text-[11px] flex items-center gap-1" style={{ color: 'var(--hud-text-dim)' }}>
                <span
                  style={{
                    color: session.backend_type === 'direct' ? 'var(--hud-success)' : 'var(--hud-warning)',
                  }}
                >
                  ●
                </span>
                {session.backend_type}
                {!session.is_active && ` (${t('sessions.ended')})`}
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  )
}
