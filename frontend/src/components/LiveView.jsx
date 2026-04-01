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

export default function LiveView({ scanning, ptzPos, scanProgress = 0, totalPositions = 0 }) {
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
        width: isMobile ? '100%' : 180,
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
        overflowY: isMobile ? 'visible' : 'auto',
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

        {/* Scan Path Map */}
        <ScanPathMap bounds={bounds} ptzPos={ptzPos} scanning={scanning} scanProgress={scanProgress} totalPositions={totalPositions} isMobile={isMobile} />
      </div>
    </div>
  )
}

// ── Scan Path Map — X/Y grid showing expected course + completed path ────────
function ScanPathMap({ bounds, ptzPos, scanning, scanProgress, totalPositions, isMobile }) {
  const grid = useMemo(() => {
    if (!bounds) return null
    const { left, right, top, bottom, zoom: z } = bounds
    if (left == null || right == null || top == null || bottom == null) return null

    const zoom = Math.max(1, z || 10000)
    const panStep = Math.max(25, Math.floor(1000000 / zoom))
    const tiltStep = Math.max(25, Math.floor(panStep * 0.65))
    const panRange = Math.abs(right - left)
    const tiltRange = Math.abs(bottom - top)
    const cols = Math.max(1, Math.ceil(panRange / panStep) + 1)
    const rows = Math.max(1, Math.ceil(tiltRange / tiltStep) + 1)

    // Boustrophedon path (matches backend _calibrated_scan)
    const positions = []
    for (let col = 0; col < cols; col++) {
      const panFrac = col / Math.max(cols - 1, 1)
      const pan = Math.round(left + (right - left) * panFrac)
      const rowOrder = col % 2 === 0
        ? Array.from({ length: rows }, (_, i) => i)
        : Array.from({ length: rows }, (_, i) => rows - 1 - i)
      for (const row of rowOrder) {
        const tiltFrac = row / Math.max(rows - 1, 1)
        const tilt = Math.round(top + (bottom - top) * tiltFrac)
        positions.push({ pan, tilt })
      }
    }

    return { positions, rows, cols, zoom, left, right, top, bottom }
  }, [bounds])

  if (!grid) return null

  const { positions, rows, cols, zoom, left, right, top, bottom } = grid

  // How many positions are completed during a scan (progress 0-88% = capture phase)
  const completedCount = scanning && totalPositions > 0
    ? Math.min(Math.round((Math.min(scanProgress, 88) / 88) * positions.length), positions.length)
    : 0

  // SVG dimensions — preserve pan/tilt aspect ratio
  const svgW = isMobile ? 160 : 148
  const padL = 30, padR = 8, padT = 8, padB = 20
  const mapW = svgW - padL - padR
  const aspect = Math.abs(bottom - top) / Math.max(Math.abs(right - left), 1)
  const mapH = Math.max(40, Math.min(100, Math.round(mapW * aspect)))
  const svgH = mapH + padT + padB

  const toX = pan => padL + ((pan - left) / Math.max(right - left, 1)) * mapW
  const toY = tilt => padT + ((tilt - top) / Math.max(bottom - top, 1)) * mapH

  // Path strings
  const fullPath = positions.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${toX(p.pan).toFixed(1)},${toY(p.tilt).toFixed(1)}`
  ).join(' ')

  const donePath = completedCount > 1
    ? positions.slice(0, completedCount).map((p, i) =>
        `${i === 0 ? 'M' : 'L'}${toX(p.pan).toFixed(1)},${toY(p.tilt).toFixed(1)}`
      ).join(' ')
    : null

  // Current PTZ position mapped to SVG
  const hasCur = ptzPos && ptzPos.pan != null && ptzPos.tilt != null
  const curX = hasCur ? toX(ptzPos.pan) : 0
  const curY = hasCur ? toY(ptzPos.tilt) : 0
  // Clamp to map area for display
  const curInBounds = hasCur
    && ptzPos.pan >= Math.min(left, right) - 50 && ptzPos.pan <= Math.max(left, right) + 50
    && ptzPos.tilt >= Math.min(top, bottom) - 50 && ptzPos.tilt <= Math.max(top, bottom) + 50

  // Tick values for axes (show ~3-4 values along each axis)
  const xTicks = []
  const xCount = Math.min(cols, 4)
  for (let i = 0; i < xCount; i++) {
    const frac = i / Math.max(xCount - 1, 1)
    const val = Math.round(left + (right - left) * frac)
    xTicks.push(val)
  }
  const yTicks = []
  const yCount = Math.min(rows, 4)
  for (let i = 0; i < yCount; i++) {
    const frac = i / Math.max(yCount - 1, 1)
    const val = Math.round(top + (bottom - top) * frac)
    yTicks.push(val)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <div style={{ color: C.muted, fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Scan Path · {cols}×{rows}
      </div>

      <svg width={svgW} height={svgH} style={{ background: '#0a0f1a', borderRadius: 6, border: `1px solid ${C.border}` }}>
        {/* Grid background lines */}
        {xTicks.map(v => (
          <line key={`gx${v}`} x1={toX(v)} y1={padT} x2={toX(v)} y2={padT + mapH}
            stroke={C.border} strokeWidth={0.5} strokeDasharray="2,3" opacity={0.4} />
        ))}
        {yTicks.map(v => (
          <line key={`gy${v}`} x1={padL} y1={toY(v)} x2={padL + mapW} y2={toY(v)}
            stroke={C.border} strokeWidth={0.5} strokeDasharray="2,3" opacity={0.4} />
        ))}

        {/* Expected path (full route) */}
        <path d={fullPath} fill="none" stroke={C.border} strokeWidth={1.2} strokeDasharray="4,3" opacity={0.6} />

        {/* Completed path */}
        {donePath && (
          <path d={donePath} fill="none" stroke={C.green} strokeWidth={2} strokeLinecap="round" />
        )}

        {/* Grid point dots */}
        {positions.map((p, i) => (
          <circle key={i} cx={toX(p.pan)} cy={toY(p.tilt)} r={2.2}
            fill={i < completedCount ? C.green : '#1e293b'}
            stroke={i < completedCount ? C.green : C.border}
            strokeWidth={0.8}
          />
        ))}

        {/* Start marker */}
        {positions.length > 0 && (
          <circle cx={toX(positions[0].pan)} cy={toY(positions[0].tilt)} r={3.5}
            fill="none" stroke={C.green} strokeWidth={1.5} />
        )}

        {/* Current camera position */}
        {curInBounds && (
          <>
            <circle cx={curX} cy={curY} r={6} fill={C.accent} opacity={0.2}>
              <animate attributeName="r" values="6;10;6" dur="1.5s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.25;0.08;0.25" dur="1.5s" repeatCount="indefinite" />
            </circle>
            <circle cx={curX} cy={curY} r={3.5} fill={C.accent} />
          </>
        )}

        {/* X axis labels (pan) */}
        {xTicks.map(v => (
          <text key={`xl${v}`} x={toX(v)} y={svgH - 2} fontSize={7} fill={C.muted}
            textAnchor="middle" fontFamily="monospace">{v}</text>
        ))}

        {/* Y axis labels (tilt) */}
        {yTicks.map(v => (
          <text key={`yl${v}`} x={padL - 3} y={toY(v) + 3} fontSize={7} fill={C.muted}
            textAnchor="end" fontFamily="monospace">{v}</text>
        ))}
      </svg>

      {/* Zoom display centered underneath */}
      <div style={{
        fontSize: 11, fontFamily: 'monospace', fontWeight: 600,
        color: C.accent, background: C.bg, borderRadius: 4,
        padding: '2px 8px', border: `1px solid ${C.border}`,
      }}>
        Zoom: {zoom}
      </div>

      {/* Current position readout */}
      {hasCur && (
        <div style={{ fontSize: 9, fontFamily: 'monospace', color: C.muted, textAlign: 'center' }}>
          X:{ptzPos.pan} Y:{ptzPos.tilt}
        </div>
      )}
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
