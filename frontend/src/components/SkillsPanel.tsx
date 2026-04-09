import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import Panel from './Panel'

export default function SkillsPanel() {
  const { data, isLoading } = useApi('/skills', 60000)
  const [expandedCat, setExpandedCat] = useState<string | null>(null)

  if (isLoading || !data) {
    return <Panel title="Skills" className="col-span-full"><div className="glow text-[11px] animate-pulse">Loading...</div></Panel>
  }

  const byCategory = data.by_category || {}
  const catCounts = data.category_counts || {}
  const recentlyMod = data.recently_modified || []

  return (
    <>
      <Panel title="Skill Library" className="col-span-2">
        <div className="flex gap-1 flex-wrap mb-3">
          <span className="text-[12px] px-2 py-0.5" style={{ background: 'var(--hud-bg-panel)', color: 'var(--hud-primary)' }}>
            {data.total || 0} total
          </span>
          <span className="text-[12px] px-2 py-0.5" style={{ background: 'var(--hud-bg-panel)', color: 'var(--hud-accent)' }}>
            {data.custom_count || 0} custom
          </span>
        </div>
        <div className="space-y-0.5 text-[12px]">
          {Object.entries(catCounts)
            .sort((a: any, b: any) => b[1] - a[1])
            .map(([cat, count]: any) => (
            <div key={cat}>
              <button
                onClick={() => setExpandedCat(expandedCat === cat ? null : cat)}
                className="flex justify-between w-full py-1 px-2 transition-colors"
                style={{ background: expandedCat === cat ? 'var(--hud-bg-hover)' : 'transparent' }}
              >
                <span>
                  <span style={{ color: 'var(--hud-primary)' }}>{expandedCat === cat ? '▾' : '▸'}</span>
                  {' '}{cat}
                </span>
                <span style={{ color: 'var(--hud-text-dim)' }}>{count}</span>
              </button>
              {expandedCat === cat && byCategory[cat] && (
                <div className="pl-4 pb-1">
                  {byCategory[cat].map((s: any) => (
                    <div key={s.name} className="py-0.5 flex justify-between">
                      <span>{s.name}</span>
                      <span style={{ color: 'var(--hud-text-dim)' }}>{s.description?.slice(0, 50)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Recently Modified">
        <div className="space-y-1.5 text-[12px]">
          {recentlyMod.map((s: any) => (
            <div key={s.name} className="py-1 px-2" style={{ borderLeft: '2px solid var(--hud-border)' }}>
              <div className="flex justify-between">
                <span style={{ color: 'var(--hud-primary)' }}>{s.name}</span>
                <span style={{ color: 'var(--hud-text-dim)' }}>
                  {s.modified_at ? new Date(s.modified_at).toLocaleDateString() : ''}
                </span>
              </div>
              <div style={{ color: 'var(--hud-text-dim)' }}>{s.category}</div>
            </div>
          ))}
        </div>
      </Panel>
    </>
  )
}
