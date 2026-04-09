import { useApi } from '../hooks/useApi'
import Panel from './Panel'

export default function ProjectsPanel() {
  const { data, isLoading } = useApi('/projects', 60000)

  if (isLoading || !data) {
    return <Panel title="Projects" className="col-span-full"><div className="glow text-[11px] animate-pulse">Loading...</div></Panel>
  }

  const projects = data.projects || data || []
  if (!Array.isArray(projects) || projects.length === 0) {
    return <Panel title="Projects" className="col-span-full"><div className="text-[12px]" style={{ color: 'var(--hud-text-dim)' }}>No projects found in ~/projects</div></Panel>
  }

  // Sort: dirty first, then by last modified
  const sorted = [...projects].sort((a: any, b: any) => {
    if (a.dirty_files && !b.dirty_files) return -1
    if (!a.dirty_files && b.dirty_files) return 1
    return (b.last_commit_ts || 0) - (a.last_commit_ts || 0)
  })

  return (
    <Panel title="Projects" className="col-span-full">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
        {sorted.map((p: any) => (
          <div key={p.name} className="p-2.5 text-[12px]"
            style={{
              background: 'var(--hud-bg-panel)',
              borderLeft: `3px solid ${p.dirty_files > 0 ? 'var(--hud-warning)' : p.is_git ? 'var(--hud-primary)' : 'var(--hud-text-dim)'}`,
            }}>
            <div className="flex items-center justify-between mb-1">
              <span className="font-bold text-[11px]" style={{ color: 'var(--hud-primary)' }}>{p.name}</span>
              <span style={{ color: p.dirty_files > 0 ? 'var(--hud-warning)' : 'var(--hud-success)', fontSize: '9px' }}>
                {p.dirty_files > 0 ? `${p.dirty_files} modified` : p.is_git ? 'clean' : 'no git'}
              </span>
            </div>

            {p.is_git && (
              <>
                <div className="flex gap-3 mb-1" style={{ color: 'var(--hud-text-dim)' }}>
                  {p.branch && <span>⎇ {p.branch}</span>}
                  {p.total_commits != null && <span>{p.total_commits} commits</span>}
                  {p.last_commit_ago && <span>{p.last_commit_ago}</span>}
                </div>
                {p.last_commit_msg && (
                  <div className="truncate" style={{ color: 'var(--hud-text)' }}>
                    {p.last_commit_msg}
                  </div>
                )}
              </>
            )}

            <div className="flex gap-2 mt-1">
              {p.languages?.map((lang: string) => (
                <span key={lang} className="px-1.5 py-0.5" style={{ background: 'var(--hud-bg-hover)', fontSize: '9px' }}>
                  {lang}
                </span>
              ))}
              {[p.has_readme && 'README', p.has_package_json && 'npm', p.has_pyproject && 'py']
                .filter(Boolean).map((badge: any) => (
                  <span key={badge} className="px-1.5 py-0.5" style={{ background: 'var(--hud-bg-hover)', fontSize: '9px', color: 'var(--hud-text-dim)' }}>
                    {badge}
                  </span>
                ))}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}
