import { useState, useEffect } from 'react'
import { useTheme, THEMES } from '../hooks/useTheme'

const TABS = [
  { id: 'dashboard', label: 'Dashboard', key: '1' },
  { id: 'memory', label: 'Memory', key: '2' },
  { id: 'skills', label: 'Skills', key: '3' },
  { id: 'sessions', label: 'Sessions', key: '4' },
  { id: 'cron', label: 'Cron', key: '5' },
  { id: 'projects', label: 'Projects', key: '6' },
  { id: 'health', label: 'Health', key: '7' },
  { id: 'profiles', label: 'Profiles', key: '8' },
  { id: 'patterns', label: 'Patterns', key: '9' },
] as const

export type TabId = typeof TABS[number]['id']

interface TopBarProps {
  activeTab: TabId
  onTabChange: (tab: TabId) => void
}

export default function TopBar({ activeTab, onTabChange }: TopBarProps) {
  const { theme, setTheme, scanlines, setScanlines } = useTheme()
  const [showThemePicker, setShowThemePicker] = useState(false)
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Number keys for tabs
      const num = parseInt(e.key)
      if (num >= 1 && num <= TABS.length && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const target = e.target as HTMLElement
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
        onTabChange(TABS[num - 1].id)
      }
      // R to refresh
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) {
        const target = e.target as HTMLElement
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
        window.location.reload()
      }
      // T to toggle theme picker
      if (e.key === 't' && !e.metaKey && !e.ctrlKey) {
        const target = e.target as HTMLElement
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
        setShowThemePicker(p => !p)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onTabChange])

  return (
    <div className="flex items-center gap-1 px-3 py-1.5 border-b"
         style={{ borderColor: 'var(--hud-border)', background: 'var(--hud-bg-surface)' }}>
      {/* Logo */}
      <span className="gradient-text font-bold text-sm mr-3 tracking-wider">☤ HERMES</span>

      {/* Tabs */}
      <div className="flex gap-0.5 flex-1">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className="px-2.5 py-1 text-[12px] tracking-widest uppercase transition-all duration-150"
            style={{
              color: activeTab === tab.id ? 'var(--hud-primary)' : 'var(--hud-text-dim)',
              background: activeTab === tab.id ? 'var(--hud-bg-panel)' : 'transparent',
              borderBottom: activeTab === tab.id ? '1px solid var(--hud-primary)' : '1px solid transparent',
              textShadow: activeTab === tab.id ? '0 0 8px var(--hud-primary-glow)' : 'none',
            }}
          >
            <span className="opacity-40 mr-1">{tab.key}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Theme picker */}
      <div className="relative">
        <button
          onClick={() => setShowThemePicker(p => !p)}
          className="px-2 py-1 text-[12px] tracking-wider uppercase"
          style={{ color: 'var(--hud-text-dim)' }}
          title="Theme (T)"
        >
          ◆ Theme
        </button>
        {showThemePicker && (
          <div className="absolute right-0 top-full mt-1 z-50 py-1 min-w-[180px]"
               style={{ background: 'var(--hud-bg-panel)', border: '1px solid var(--hud-border)' }}>
            {THEMES.map(t => (
              <button
                key={t.id}
                onClick={() => { setTheme(t.id); setShowThemePicker(false) }}
                className="block w-full text-left px-3 py-1.5 text-[11px] transition-colors"
                style={{
                  color: theme === t.id ? 'var(--hud-primary)' : 'var(--hud-text)',
                  background: theme === t.id ? 'var(--hud-bg-hover)' : 'transparent',
                }}
              >
                {t.icon} {t.label}
              </button>
            ))}
            <div className="border-t my-1" style={{ borderColor: 'var(--hud-border)' }} />
            <button
              onClick={() => setScanlines(!scanlines)}
              className="block w-full text-left px-3 py-1.5 text-[11px]"
              style={{ color: 'var(--hud-text-dim)' }}
            >
              {scanlines ? '▣' : '□'} Scanlines
            </button>
          </div>
        )}
      </div>

      {/* Clock */}
      <span className="text-[11px] ml-2 tabular-nums" style={{ color: 'var(--hud-text-dim)' }}>
        {time.toLocaleTimeString('en-US', { hour12: false })}
      </span>
    </div>
  )
}
