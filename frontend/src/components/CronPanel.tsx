import { useApi } from '../hooks/useApi'
import Panel from './Panel'

function timeAgo(iso: string): string {
  if (!iso) return 'never'
  const d = new Date(iso)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const mins = Math.floor(diffMs / 60000)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}

export default function CronPanel() {
  const { data, isLoading } = useApi('/cron', 30000)

  if (isLoading || !data) {
    return <Panel title="Cron Jobs" className="col-span-full"><div className="glow text-[11px] animate-pulse">Loading...</div></Panel>
  }

  const jobs = data.jobs || data || []
  if (!Array.isArray(jobs) || jobs.length === 0) {
    return <Panel title="Cron Jobs" className="col-span-full"><div className="text-[12px]" style={{ color: 'var(--hud-text-dim)' }}>No cron jobs configured</div></Panel>
  }

  return (
    <Panel title="Cron Jobs" className="col-span-full">
      <div className="space-y-3">
        {jobs.map((job: any) => {
          const isActive = job.enabled && !job.paused_reason
          return (
            <div key={job.id} className="p-3" style={{ background: 'var(--hud-bg-panel)', border: '1px solid var(--hud-border)' }}>
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: isActive ? 'var(--hud-success)' : 'var(--hud-text-dim)' }} />
                <span className="font-bold text-[11px]" style={{ color: 'var(--hud-primary)' }}>
                  {job.name || job.id}
                </span>
                <span className="text-[11px] px-1.5 py-0.5 ml-auto"
                  style={{
                    background: 'var(--hud-bg-hover)',
                    color: job.state === 'scheduled' ? 'var(--hud-success)' : 'var(--hud-text-dim)'
                  }}>
                  {job.state || 'unknown'}
                </span>
              </div>

              <div className="grid grid-cols-4 gap-4 text-[12px]">
                <div>
                  <div className="uppercase tracking-wider" style={{ color: 'var(--hud-text-dim)', fontSize: '9px' }}>Schedule</div>
                  <div style={{ color: 'var(--hud-primary)' }}>{job.schedule_display || job.schedule || '-'}</div>
                </div>
                <div>
                  <div className="uppercase tracking-wider" style={{ color: 'var(--hud-text-dim)', fontSize: '9px' }}>Last Run</div>
                  <div>
                    {job.last_run_at ? timeAgo(job.last_run_at) : 'never'}
                    {job.last_status && (
                      <span className="ml-1" style={{ color: job.last_status === 'ok' ? 'var(--hud-success)' : 'var(--hud-error)' }}>
                        {job.last_status === 'ok' ? '✔' : '✗'}
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  <div className="uppercase tracking-wider" style={{ color: 'var(--hud-text-dim)', fontSize: '9px' }}>Next Run</div>
                  <div>{job.next_run_at ? new Date(job.next_run_at).toLocaleString() : '-'}</div>
                </div>
                <div>
                  <div className="uppercase tracking-wider" style={{ color: 'var(--hud-text-dim)', fontSize: '9px' }}>Deliver</div>
                  <div style={{ color: 'var(--hud-accent)' }}>{job.deliver || '-'}</div>
                </div>
              </div>

              {job.repeat_completed != null && (
                <div className="mt-2 text-[11px]" style={{ color: 'var(--hud-text-dim)' }}>
                  Runs completed: {job.repeat_completed}{job.repeat_total ? ` / ${job.repeat_total}` : ''}
                  {job.skills?.length > 0 && <span className="ml-2">Skills: {job.skills.join(', ')}</span>}
                </div>
              )}

              {job.prompt && (
                <div className="mt-2 text-[12px] truncate" style={{ color: 'var(--hud-text-dim)' }}>
                  {job.prompt.slice(0, 120)}...
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Panel>
  )
}
