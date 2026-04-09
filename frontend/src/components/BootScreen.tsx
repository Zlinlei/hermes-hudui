import { useState, useEffect } from 'react'

const HERMES_ASCII = `
 ██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗
 ██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝
 ███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗
 ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║
 ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝
`.trim()

const BOOT_LINES = [
  '☤ HERMES HUD v0.1.0',
  '',
  'Initializing consciousness monitor...',
  'Reading ~/.hermes/state.db',
  'Scanning memory banks',
  'Indexing skill library',
  'Checking service health',
  'Profiling agent processes',
  '',
  '"I think, therefore I process."',
  '',
  'Systems ready.',
]

interface BootScreenProps {
  onComplete: () => void
}

export default function BootScreen({ onComplete }: BootScreenProps) {
  const [visibleLines, setVisibleLines] = useState(0)
  const [asciiVisible, setAsciiVisible] = useState(false)
  const [fadeOut, setFadeOut] = useState(false)

  useEffect(() => {
    // Show ASCII art first
    const asciiTimer = setTimeout(() => setAsciiVisible(true), 200)

    // Then type lines one by one
    const lineTimers = BOOT_LINES.map((_, i) =>
      setTimeout(() => setVisibleLines(i + 1), 600 + i * 120)
    )

    // Fade out and complete
    const fadeTimer = setTimeout(() => setFadeOut(true), 600 + BOOT_LINES.length * 120 + 400)
    const completeTimer = setTimeout(onComplete, 600 + BOOT_LINES.length * 120 + 900)

    return () => {
      clearTimeout(asciiTimer)
      lineTimers.forEach(clearTimeout)
      clearTimeout(fadeTimer)
      clearTimeout(completeTimer)
    }
  }, [onComplete])

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center z-50 transition-opacity duration-500"
      style={{
        background: 'var(--hud-bg-deep)',
        opacity: fadeOut ? 0 : 1,
      }}
      onClick={onComplete}
    >
      {/* ASCII logo */}
      <pre
        className="gradient-text text-[11px] leading-tight mb-6 transition-opacity duration-300"
        style={{ opacity: asciiVisible ? 1 : 0 }}
      >
        {HERMES_ASCII}
      </pre>

      {/* Boot text */}
      <div className="text-[11px] w-[400px]">
        {BOOT_LINES.slice(0, visibleLines).map((line, i) => (
          <div key={i} className="py-0.5" style={{
            color: line.startsWith('"') ? 'var(--hud-accent)' :
                   line.startsWith('☤') ? 'var(--hud-primary)' :
                   line === 'Systems ready.' ? 'var(--hud-success)' :
                   'var(--hud-text-dim)',
            fontStyle: line.startsWith('"') ? 'italic' : 'normal',
          }}>
            {line}
            {i === visibleLines - 1 && (
              <span className="animate-pulse" style={{ color: 'var(--hud-primary)' }}>█</span>
            )}
          </div>
        ))}
      </div>

      <div className="absolute bottom-6 text-[11px]" style={{ color: 'var(--hud-text-dim)' }}>
        click to skip
      </div>
    </div>
  )
}
