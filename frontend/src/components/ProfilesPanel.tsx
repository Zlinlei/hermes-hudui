import { useApi } from '../hooks/useApi'
import Panel, { CapacityBar } from './Panel'

export default function ProfilesPanel() {
  const { data, isLoading } = useApi('/profiles', 30000)

  if (isLoading || !data) {
    return <Panel title="Profiles" className="col-span-full"><div className="glow text-[11px] animate-pulse">Loading...</div></Panel>
  }

  const profiles = data.profiles || []

  return (
    <Panel title="Agent Profiles" className="col-span-full">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {profiles.map((p: any) => (
          <div key={p.name} className="p-3" style={{ background: 'var(--hud-bg-panel)', border: '1px solid var(--hud-border)' }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full"
                style={{ background: p.gateway_status === 'active' ? 'var(--hud-success)' : 'var(--hud-text-dim)' }} />
              <span className="font-bold text-[11px]" style={{ color: 'var(--hud-primary)' }}>
                {p.name}{p.is_default ? ' ★' : ''}
              </span>
            </div>
            <div className="text-[11px] space-y-0.5" style={{ color: 'var(--hud-text-dim)' }}>
              <div>{p.provider}/{p.model || 'no model'}</div>
              {p.soul_summary && <div className="italic truncate" style={{ color: 'var(--hud-text)' }}>{p.soul_summary}</div>}
              <div className="grid grid-cols-3 gap-1 mt-2 text-center">
                <div><span style={{ color: 'var(--hud-primary)' }}>{p.session_count}</span> sess</div>
                <div><span style={{ color: 'var(--hud-primary)' }}>{p.message_count}</span> msgs</div>
                <div><span style={{ color: 'var(--hud-primary)' }}>{p.skill_count}</span> skills</div>
              </div>
            </div>
            <div className="mt-2">
              <CapacityBar value={p.memory_chars || 0} max={p.memory_max_chars || 2200} label="MEM" />
              <CapacityBar value={p.user_chars || 0} max={p.user_max_chars || 1375} label="USR" />
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}
