import { useApi } from '../hooks/useApi'
import Panel, { Stat, CapacityBar, Sparkline } from './Panel'

function GrowthDelta({ snapshots }: { snapshots: any[] }) {
  if (!snapshots || snapshots.length < 2) {
    return (
      <div className="text-[12px]" style={{ color: 'var(--hud-text-dim)' }}>
        {snapshots?.length === 1 ? 'First snapshot recorded — delta available after next snapshot.' : 'No snapshots yet. Run hermes-hud snapshot to start tracking.'}
      </div>
    )
  }

  const current = snapshots[snapshots.length - 1]
  const previous = snapshots[snapshots.length - 2]

  const fields = [
    { key: 'sessions', label: 'Sessions' },
    { key: 'messages', label: 'Messages' },
    { key: 'tool_calls', label: 'Tool Calls' },
    { key: 'skills', label: 'Skills' },
    { key: 'custom_skills', label: 'Custom Skills' },
    { key: 'memory_entries', label: 'Memory Entries' },
    { key: 'user_entries', label: 'User Entries' },
    { key: 'tokens', label: 'Tokens' },
  ]

  return (
    <div className="space-y-0.5 text-[12px]">
      <div className="flex justify-between mb-2" style={{ color: 'var(--hud-text-dim)' }}>
        <span>{snapshots.length} snapshots</span>
        <span>{previous.timestamp?.slice(0, 10)} → {current.timestamp?.slice(0, 10)}</span>
      </div>
      {fields.map(({ key, label }) => {
        const cur = current[key] || 0
        const prev = previous[key] || 0
        const delta = cur - prev
        if (delta === 0) return (
          <div key={key} className="flex justify-between py-0.5">
            <span style={{ color: 'var(--hud-text-dim)' }}>= {label}</span>
            <span>{cur.toLocaleString()}</span>
          </div>
        )
        return (
          <div key={key} className="flex justify-between py-0.5">
            <span style={{ color: delta > 0 ? 'var(--hud-success)' : 'var(--hud-error)' }}>
              {delta > 0 ? '↑' : '↓'} {label}
            </span>
            <span>
              <span style={{ color: 'var(--hud-text-dim)' }}>{prev.toLocaleString()} → </span>
              <span>{cur.toLocaleString()}</span>
              <span style={{ color: delta > 0 ? 'var(--hud-success)' : 'var(--hud-error)' }}>
                {' '}({delta > 0 ? '+' : ''}{delta.toLocaleString()})
              </span>
            </span>
          </div>
        )
      })}
    </div>
  )
}

export default function DashboardPanel() {
  const { data, isLoading } = useApi('/state', 30000)
  const { data: snapData } = useApi('/snapshots', 60000)

  if (isLoading || !data) {
    return (
      <Panel title="Overview" className="col-span-full">
        <div className="glow text-[11px] animate-pulse">Collecting state...</div>
      </Panel>
    )
  }

  const { config, memory, user, skills, sessions } = data
  const dailyMessages = sessions?.daily_stats?.map((d: any) => d.messages) || []
  const dailyTokens = sessions?.daily_stats?.map((d: any) => d.tokens) || []

  return (
    <>
      {/* Row 1: overview + activity sparkline */}
      <Panel title="Overview">
        <div className="text-[12px] mb-3" style={{ color: 'var(--hud-text-dim)' }}>
          {config?.provider}<span style={{ color: 'var(--hud-primary)' }}>/</span>{config?.model}
        </div>
        <div className="grid grid-cols-2 gap-3 mb-4">
          <Stat value={sessions?.total_sessions || 0} label="Sessions" />
          <Stat value={(sessions?.total_messages || 0).toLocaleString()} label="Messages" />
          <Stat value={(sessions?.total_tool_calls || 0).toLocaleString()} label="Tool Calls" />
          <Stat value={skills?.total || 0} label="Skills" />
        </div>

        <CapacityBar value={memory?.total_chars || 0} max={memory?.max_chars || 2200} label="MEMORY" />
        <CapacityBar value={user?.total_chars || 0} max={user?.max_chars || 1375} label="USER" />

        {sessions?.date_range?.[0] && (
          <div className="text-[12px] mt-3" style={{ color: 'var(--hud-text-dim)' }}>
            {new Date(sessions.date_range[0]).toLocaleDateString()} → {new Date(sessions.date_range[1]).toLocaleDateString()}
          </div>
        )}
      </Panel>

      <Panel title="Activity" className="col-span-2">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-wider mb-1" style={{ color: 'var(--hud-text-dim)' }}>
              Messages/day · last {dailyMessages.length}d
            </div>
            <Sparkline values={dailyMessages} width={360} height={45} />
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-wider mb-1" style={{ color: 'var(--hud-text-dim)' }}>
              Tokens/day
            </div>
            <Sparkline values={dailyTokens} width={360} height={45} />
          </div>
        </div>
        <div className="mt-3 text-[12px] grid grid-cols-5 gap-1">
          {sessions?.daily_stats?.slice(-10).map((d: any) => (
            <div key={d.date} className="text-center py-1" style={{ background: 'var(--hud-bg-panel)' }}>
              <div style={{ color: 'var(--hud-text-dim)' }}>{d.date.slice(5)}</div>
              <div style={{ color: 'var(--hud-primary)' }}>{d.sessions}s</div>
              <div>{d.messages}m</div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Row 2: tools + growth delta + platforms */}
      <Panel title="Top Tools">
        <div className="text-[12px] space-y-0.5">
          {sessions?.tool_usage && Object.entries(sessions.tool_usage)
            .sort((a: any, b: any) => b[1] - a[1])
            .slice(0, 12)
            .map(([tool, count]: any) => {
              const maxCount = Math.max(...Object.values(sessions.tool_usage as Record<string, number>))
              const pct = (count / maxCount) * 100
              return (
                <div key={tool} className="flex items-center gap-2">
                  <span className="w-[110px] truncate" style={{ color: 'var(--hud-text)' }}>
                    {tool.replace('mcp_', '').replace('browser_', 'b_')}
                  </span>
                  <div className="flex-1 h-[3px]" style={{ background: 'var(--hud-bg-hover)' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: 'var(--hud-primary)' }} />
                  </div>
                  <span className="tabular-nums w-10 text-right" style={{ color: 'var(--hud-text-dim)' }}>
                    {count}
                  </span>
                </div>
              )
            })}
        </div>
      </Panel>

      <Panel title="Growth Delta">
        <GrowthDelta snapshots={snapData?.snapshots || []} />
      </Panel>

      <Panel title="Platforms">
        <div className="space-y-2">
          {sessions?.by_source && Object.entries(sessions.by_source).map(([src, count]: any) => (
            <div key={src} className="flex justify-between text-[11px] py-1 px-2" style={{ borderLeft: '2px solid var(--hud-border)' }}>
              <span style={{ color: 'var(--hud-primary)' }}>{src}</span>
              <span>{count} sessions</span>
            </div>
          ))}
        </div>
        {sessions?.daily_stats?.length > 0 && (
          <div className="mt-3 text-[12px]" style={{ color: 'var(--hud-text-dim)' }}>
            Total tokens: {(sessions?.total_tokens || 0).toLocaleString()}
          </div>
        )}
      </Panel>
    </>
  )
}
