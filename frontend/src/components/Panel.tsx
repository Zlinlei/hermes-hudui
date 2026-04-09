import type { ReactNode } from 'react'

interface PanelProps {
  title: string
  children: ReactNode
  className?: string
}

export default function Panel({ title, children, className = '' }: PanelProps) {
  return (
    <div className={`hud-panel ${className}`}>
      <div className="hud-panel-title">{title}</div>
      <div className="hud-panel-content">
        {children}
      </div>
    </div>
  )
}

export function Stat({ value, label, delta }: { value: string | number; label: string; delta?: string }) {
  const isNeg = delta?.startsWith('-')
  return (
    <div className="text-center">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {delta && <div className={`stat-delta ${isNeg ? 'negative' : ''}`}>{delta}</div>}
    </div>
  )
}

export function CapacityBar({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0
  const level = pct > 90 ? 'critical' : pct > 70 ? 'warn' : 'ok'
  return (
    <div className="mb-2">
      <div className="flex justify-between text-[11px] mb-1">
        <span style={{ color: 'var(--hud-text-dim)' }}>{label}</span>
        <span>
          <span style={{ color: 'var(--hud-primary)' }}>{value.toLocaleString()}</span>
          <span style={{ color: 'var(--hud-text-dim)' }}>/{max.toLocaleString()} ({pct.toFixed(0)}%)</span>
        </span>
      </div>
      <div className="capacity-bar">
        <div className={`capacity-bar-fill ${level}`} style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
    </div>
  )
}

export function Sparkline({ values, width = 100, height = 20 }: { values: number[]; width?: number; height?: number }) {
  if (!values.length) return null
  const max = Math.max(...values, 1)
  const points = values.map((v, i) => {
    const x = (i / Math.max(values.length - 1, 1)) * width
    const y = height - (v / max) * height
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={width} height={height} className="inline-block">
      <polyline
        points={points}
        fill="none"
        stroke="var(--hud-primary)"
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}
