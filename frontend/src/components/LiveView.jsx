import { useState, useEffect, useRef, useMemo } from 'react'
import { ptzCommand, gotoBound, getCameraBounds } from '../api.js'
import { useIsMobile } from '../hooks/useIsMobile.js'

const C = {
  bg:      '#0f172a',
  surface: '#1e293b',
  border:  '#334155',
  text:    '#f1f5f9',
  muted:   '#94a3b8',
  accent:  '#3b82f6',
  green:   '#22c55e',
  red:     '#ef4444',
  yellow:  '#eab308',
  purple:  '#a855f7',
  orange:  '#f97316',
}

export default function LiveView({ scanning, ptzPos }) {
  const isMobile = useIsMobile()
  const [frameSrc, setFrameSrc] = useState(null)
  const [error, setError]       = useState(null)
  const [fps, setFps]           = useState(null)
  const [bounds, setBounds]     = useState(null)
  const lastFrameTime = useRef(null)

  useEffect(() => {
    getCameraBounds().then(setBounds).catch(() => {})
  }, [])

  // Poll /api/live-frame continuously while this tab is mounted
  useEffect(() => {
    let running = true

    const fetchFrame = async () => {
      if (!running) return
      try {
        const r = await fetch('/api/live-frame')
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const data = await r.json()
        if (!running) return
        setFrameSrc(`data:image/jpeg;base64,${data.image_b64}`)
        setError(null)
        // Compute instantaneous FPS
        const now = Date.now()
        if (lastFrameTime.current) {
          const delta = now - lastFrameTime.current
          setFps(Math.round(1000 / delta))
        }
        lastFrameTime.current = Date.now()
      } catch (e) {
        if (running) setError(e.message)
      }
      if (running) setTimeout(fetchFrame, 100)
    }

    fetchFrame()
    return () => { running = false }
  }, [])

  // Compute all 4 corners from the stored left/right/top/bottom bounds
  const corners = useMemo(() => {
    if (!bounds) return null
    const { left, right, top, bottom } = bounds
    if (left == null || right == null || top == null || bottom == null) return null
    return {
      top_left:     { pan: left,  tilt: top    },
      top_right:    { pan: right, tilt: top    },
      bottom_left:  { pan: left,  tilt: bottom },
      bottom_right: { pan: right, tilt: bottom },
    }
  }, [bounds])

  const sendPtz = (action, speed = 3) => ptzCommand(action, speed).catch(() => {})
  const stopPtz = () => {
    ptzCommand('stop').catch(() => {})
    ptzCommand('zoomstop').catch(() => {})
  }

  // Arrow key PTZ control (slow speed for precision)
  useEffect(() => {
    const pressed = new Set()
    const KEY_ACTION = {
      ArrowUp:    'up',
      ArrowDown:  'down',
      ArrowLeft:  'left',
      ArrowRight: 'right',
    }
    const onKeyDown = (e) => {
      const action = KEY_ACTION[e.key]
      if (!action || pressed.has(e.key)) return
      e.preventDefault()
      pressed.add(e.key)
      sendPtz(action, 3)
    }
    const onKeyUp = (e) => {
      if (!KEY_ACTION[e.key]) return
      pressed.delete(e.key)
      if (pressed.size === 0) ptzCommand('stop').catch(() => {})
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [])

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: isMobile ? 'column' : 'row', overflow: 'hidden', background: C.bg }}>

      {/* ── Camera feed ── */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', background: '#000', overflow: 'hidden' }}>
        {frameSrc ? (
          <img
            src={frameSrc}
            alt="Live camera feed"
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', display: 'block' }}
          />
        ) : (
          <div style={{ color: C.muted, fontSize: 14, textAlign: 'center', padding: 24 }}>
            {error
              ? <><div style={{ color: C.red, marginBottom: 6 }}>Camera unavailable</div><div style={{ fontSize: 12 }}>{error}</div></>
              : 'Connecting to camera…'
            }
          </div>
        )}

        {/* Top-left: LIVE badge + FPS */}
        <div style={{ position: 'absolute', top: 10, left: 10, display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ background: 'rgba(0,0,0,0.65)', color: C.red, padding: '3px 8px', borderRadius: 4, fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%', background: C.red, display: 'inline-block',
              animation: 'livePulse 1.2s ease-in-out infinite',
            }} />
            LIVE
          </span>
          {fps !== null && (
            <span style={{ background: 'rgba(0,0,0,0.55)', color: C.muted, padding: '3px 7px', borderRadius: 4, fontSize: 11 }}>
              {fps} fps
            </span>
          )}
        </div>

        {/* Top-right: scanning warning */}
        {scanning && (
          <div style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(234,179,8,0.88)', color: '#000', padding: '4px 10px', borderRadius: 4, fontSize: 11, fontWeight: 600 }}>
            ⏳ Scan in progress — camera is moving
          </div>
        )}

        {/* Bottom-left: PTZ position */}
        {ptzPos && (ptzPos.pan !== null || ptzPos.tilt !== null) && (
          <div style={{ position: 'absolute', bottom: 10, left: 10, background: 'rgba(0,0,0,0.6)', color: C.muted, padding: '3px 9px', borderRadius: 4, fontSize: 11, fontFamily: 'monospace' }}>
            Pan {ptzPos.pan ?? '--'}  ·  Tilt {ptzPos.tilt ?? '--'}  ·  Zoom {ptzPos.zoom ?? '--'}
          </div>
        )}

        {/* Pulse keyframe (injected once) */}
        <style>{`
          @keyframes livePulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.3; }
          }
        `}</style>
      </div>

      {/* ── PTZ sidebar / bottom bar ── */}
      <div style={{
        width: isMobile ? '100%' : 164,
        maxHeight: isMobile ? 180 : 'none',
        background: C.surface,
        borderTop:  isMobile ? `1px solid ${C.border}` : 'none',
        borderLeft: isMobile ? 'none' : `1px solid ${C.border}`,
        padding: 16,
        display: 'flex',
        flexDirection: isMobile ? 'row' : 'column',
        gap: 16,
        alignItems: isMobile ? 'center' : 'stretch',
        flexShrink: 0,
        overflowX: isMobile ? 'auto' : 'visible',
      }}>
        <div style={{ color: C.muted, fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', whiteSpace: 'nowrap', alignSelf: 'center' }}>
          PTZ
        </div>

        {/* D-pad */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 40px)', gridTemplateRows: 'repeat(3, 40px)', gap: 4 }}>
          <div />
          <PtzBtn label="▲" onStart={() => sendPtz('up')}    onEnd={stopPtz} />
          <div />
          <PtzBtn label="◀" onStart={() => sendPtz('left')}  onEnd={stopPtz} />
          <PtzBtn label="⌂" onStart={() => sendPtz('home')}  onEnd={() => {}} />
          <PtzBtn label="▶" onStart={() => sendPtz('right')} onEnd={stopPtz} />
          <div />
          <PtzBtn label="▼" onStart={() => sendPtz('down')}  onEnd={stopPtz} />
          <div />
        </div>

        {/* Zoom */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <PtzBtn label="＋ Zoom" wide onStart={() => sendPtz('zoomin')}  onEnd={() => ptzCommand('zoomstop').catch(() => {})} />
          <PtzBtn label="－ Zoom" wide onStart={() => sendPtz('zoomout')} onEnd={() => ptzCommand('zoomstop').catch(() => {})} />
        </div>

        {/* Bounds shortcuts — 2×2 grid */}
        {corners && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ color: C.muted, fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', textAlign: 'center' }}>
              Bounds
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              <GoToBoundBtn label="↖ TL" color={C.green}  onClick={() => gotoBound('top_left').catch(() => {})} />
              <GoToBoundBtn label="↗ TR" color={C.purple} onClick={() => gotoBound('top_right').catch(() => {})} />
              <GoToBoundBtn label="↙ BL" color={C.orange} onClick={() => gotoBound('bottom_left').catch(() => {})} />
              <GoToBoundBtn label="↘ BR" color={C.accent} onClick={() => gotoBound('bottom_right').catch(() => {})} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── One-shot button for going to a saved bound position ──────────────────────
function GoToBoundBtn({ label, onClick, color }) {
  const borderColor = color || C.border
  return (
    <button
      onClick={onClick}
      style={{
        background: '#0f172a',
        border: `1px solid ${borderColor}`,
        color: color || C.text,
        borderRadius: 6,
        cursor: 'pointer',
        fontWeight: 700,
        fontSize: 11,
        height: 34,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      {label}
    </button>
  )
}

// ── PTZ button that fires on press and sends stop on release ──────────────────
function PtzBtn({ label, onStart, onEnd, wide }) {
  const held = useRef(false)

  const start = () => {
    held.current = true
    onStart()
  }
  const end = () => {
    if (!held.current) return
    held.current = false
    onEnd()
  }

  return (
    <button
      onMouseDown={start}
      onMouseUp={end}
      onMouseLeave={end}
      onTouchStart={e => { e.preventDefault(); start() }}
      onTouchEnd={end}
      style={{
        background: '#0f172a',
        border: `1px solid ${C.border}`,
        color: C.text,
        borderRadius: 6,
        cursor: 'pointer',
        fontWeight: 600,
        fontSize: wide ? 12 : 14,
        width:  wide ? 132 : 40,
        height: 40,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      {label}
    </button>
  )
}
