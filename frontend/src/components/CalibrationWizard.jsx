/**
 * Calibration Wizard — Camera Bounds Setup
 * Move camera to top-left and bottom-right corners, set zoom, save.
 * Live feed streams continuously with no buffer for real-time feedback.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { ptzCommand, getPtzPosition, getCameraBounds, saveCameraBounds } from '../api.js'

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
}

export default function CalibrationWizard({ onClose }) {
  const [frameSrc, setFrameSrc]     = useState(null)
  const [frameError, setFrameError] = useState(null)
  const [fps, setFps]               = useState(null)

  const [pos, setPos]               = useState({ pan: null, tilt: null, zoom: null })
  const [posError, setPosError]     = useState(false)

  const [topLeft, setTopLeft]       = useState(null)   // {pan, tilt, zoom}
  const [botRight, setBotRight]     = useState(null)   // {pan, tilt, zoom}
  const [scanZoom, setScanZoom]     = useState(null)   // standalone zoom override

  const [saving, setSaving]         = useState(false)
  const [saved, setSaved]           = useState(false)
  const [status, setStatus]         = useState('')

  const lastFrameTime = useRef(null)
  const liveRunning   = useRef(false)
  const posInterval   = useRef(null)

  // ── Live feed — 10 fps polling against persistent backend stream ─────────
  useEffect(() => {
    liveRunning.current = true

    // Tell the backend to open the persistent RTSP capture thread
    fetch('/api/live-frame/start', { method: 'POST' }).catch(() => {})

    const fetchFrame = async () => {
      if (!liveRunning.current) return
      try {
        const r = await fetch('/api/live-frame')
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const data = await r.json()
        if (!liveRunning.current) return
        setFrameSrc(`data:image/jpeg;base64,${data.image_b64}`)
        setFrameError(null)
        const now = Date.now()
        if (lastFrameTime.current) setFps(Math.round(1000 / (now - lastFrameTime.current)))
        lastFrameTime.current = now
      } catch (e) {
        if (liveRunning.current) setFrameError(e.message)
      }
      // Target 10 fps: schedule next fetch 100 ms from now
      if (liveRunning.current) setTimeout(fetchFrame, 100)
    }

    fetchFrame()
    return () => {
      liveRunning.current = false
      // Release the persistent stream when the wizard closes
      fetch('/api/live-frame/stop', { method: 'POST' }).catch(() => {})
    }
  }, [])

  // ── PTZ position poll ─────────────────────────────────────────────────────
  useEffect(() => {
    const pollPos = async () => {
      try {
        const p = await getPtzPosition()
        setPos(p)
        setPosError(false)
      } catch {
        setPosError(true)
      }
    }
    pollPos()
    posInterval.current = setInterval(pollPos, 500)
    return () => clearInterval(posInterval.current)
  }, [])

  // ── Load existing bounds on mount ─────────────────────────────────────────
  useEffect(() => {
    getCameraBounds().then(b => {
      if (b.top_left)     setTopLeft(b.top_left)
      if (b.bottom_right) setBotRight(b.bottom_right)
      if (b.zoom != null) setScanZoom(b.zoom)
    }).catch(() => {})
  }, [])

  // ── PTZ control helpers ───────────────────────────────────────────────────
  const sendPtz  = useCallback((action, speed = 8) => ptzCommand(action, speed).catch(() => {}), [])
  const stopPan  = useCallback(() => ptzCommand('stop').catch(() => {}), [])
  const stopZoom = useCallback(() => ptzCommand('zoomstop').catch(() => {}), [])

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
      ptzCommand(action, 3).catch(() => {})
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

  // ── Bounds capture ────────────────────────────────────────────────────────
  const captureTopLeft = () => {
    if (pos.pan == null) return
    setTopLeft({ pan: pos.pan, tilt: pos.tilt, zoom: pos.zoom })
    setStatus('Top-left corner set.')
  }

  const captureBotRight = () => {
    if (pos.pan == null) return
    setBotRight({ pan: pos.pan, tilt: pos.tilt, zoom: pos.zoom })
    setStatus('Bottom-right corner set.')
  }

  const captureZoom = () => {
    if (pos.zoom == null) return
    setScanZoom(pos.zoom)
    setStatus(`Scan zoom set to ${pos.zoom}.`)
  }

  // ── Save ──────────────────────────────────────────────────────────────────
  const save = async () => {
    if (!topLeft && !botRight && scanZoom == null) {
      setStatus('Nothing to save — set at least one bound.')
      return
    }
    setSaving(true)
    try {
      await saveCameraBounds({ top_left: topLeft, bottom_right: botRight, zoom: scanZoom })
      setSaved(true)
      setStatus('Bounds saved!')
      setTimeout(onClose, 900)
    } catch (e) {
      setStatus(`Error saving: ${e.message}`)
    } finally {
      setSaving(false)
    }
  }

  const posStr = p =>
    p ? `Pan ${p.pan}  Tilt ${p.tilt}  Zoom ${p.zoom}` : 'Not set'

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
      <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 16, width: '98vw', maxWidth: 1300, height: '92vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Header */}
        <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: '14px 20px', display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0 }}>
          <span style={{ fontSize: 17, fontWeight: 700 }}>⚙ Camera Calibration</span>
          <span style={{ fontSize: 12, color: C.muted }}>Set scan bounds — move camera, then click Set buttons</span>
          {status && <span style={{ marginLeft: 'auto', fontSize: 12, color: C.yellow }}>{status}</span>}
          <button onClick={onClose} style={{ marginLeft: status ? 0 : 'auto', background: 'transparent', border: `1px solid ${C.border}`, color: C.muted, borderRadius: 8, padding: '5px 12px', cursor: 'pointer', fontSize: 13 }}>✕ Close</button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

          {/* ── Live feed ── */}
          <div style={{ flex: 1, position: 'relative', background: '#000', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {frameSrc ? (
              <img
                src={frameSrc}
                alt="Live camera feed"
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', display: 'block' }}
              />
            ) : (
              <div style={{ color: C.muted, fontSize: 14, textAlign: 'center', padding: 24 }}>
                {frameError
                  ? <><div style={{ color: C.red, marginBottom: 6 }}>Camera unavailable</div><div style={{ fontSize: 12 }}>{frameError}</div></>
                  : 'Connecting to camera…'
                }
              </div>
            )}

            {/* LIVE badge + FPS */}
            <div style={{ position: 'absolute', top: 10, left: 10, display: 'flex', gap: 6, alignItems: 'center' }}>
              <span style={{ background: 'rgba(0,0,0,0.65)', color: C.red, padding: '3px 8px', borderRadius: 4, fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: C.red, display: 'inline-block', animation: 'livePulse 1.2s ease-in-out infinite' }} />
                LIVE
              </span>
              {fps !== null && (
                <span style={{ background: 'rgba(0,0,0,0.55)', color: C.muted, padding: '3px 7px', borderRadius: 4, fontSize: 11 }}>
                  {fps} fps
                </span>
              )}
            </div>

            {/* Current position overlay */}
            <div style={{ position: 'absolute', bottom: 10, left: 10, background: 'rgba(0,0,0,0.65)', color: posError ? C.red : C.muted, padding: '4px 10px', borderRadius: 4, fontSize: 11, fontFamily: 'monospace' }}>
              {posError
                ? 'Position unavailable'
                : `Pan ${pos.pan ?? '--'}  ·  Tilt ${pos.tilt ?? '--'}  ·  Zoom ${pos.zoom ?? '--'}`
              }
            </div>

            {/* Bounds corners overlay */}
            {topLeft && (
              <div style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(34,197,94,0.85)', color: '#000', padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', fontWeight: 700 }}>
                TL SET
              </div>
            )}
            {botRight && (
              <div style={{ position: 'absolute', bottom: 10, right: 10, background: 'rgba(59,130,246,0.85)', color: '#fff', padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', fontWeight: 700 }}>
                BR SET
              </div>
            )}

            <style>{`@keyframes livePulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
          </div>

          {/* ── Right sidebar ── */}
          <div style={{ width: 220, background: C.surface, borderLeft: `1px solid ${C.border}`, display: 'flex', flexDirection: 'column', gap: 0, flexShrink: 0, overflowY: 'auto' }}>

            {/* PTZ section */}
            <Section label="PTZ Controls">
              {/* D-pad */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 48px)', gridTemplateRows: 'repeat(3, 48px)', gap: 5, margin: '0 auto' }}>
                <div />
                <PtzBtn label="▲" onStart={() => sendPtz('up')}    onEnd={stopPan} />
                <div />
                <PtzBtn label="◀" onStart={() => sendPtz('left')}  onEnd={stopPan} />
                <PtzBtn label="⌂" onStart={() => sendPtz('home')}  onEnd={() => {}} />
                <PtzBtn label="▶" onStart={() => sendPtz('right')} onEnd={stopPan} />
                <div />
                <PtzBtn label="▼" onStart={() => sendPtz('down')}  onEnd={stopPan} />
                <div />
              </div>

              {/* Zoom */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 4 }}>
                <PtzBtn label="＋ Zoom" wide onStart={() => sendPtz('zoomin', 4)}  onEnd={stopZoom} />
                <PtzBtn label="－ Zoom" wide onStart={() => sendPtz('zoomout', 4)} onEnd={stopZoom} />
              </div>
            </Section>

            <Divider />

            {/* Top Left */}
            <Section label="Top Left Corner">
              <button onClick={captureTopLeft} disabled={pos.pan == null} style={actionBtn(C.green, pos.pan == null)}>
                Set Top Left
              </button>
              <div style={{ fontSize: 10, color: topLeft ? C.green : C.muted, fontFamily: 'monospace', marginTop: 4, lineHeight: 1.5 }}>
                {topLeft
                  ? <>Pan {topLeft.pan}<br />Tilt {topLeft.tilt}<br />Zoom {topLeft.zoom}</>
                  : 'Not set'
                }
              </div>
            </Section>

            <Divider />

            {/* Bottom Right */}
            <Section label="Bottom Right Corner">
              <button onClick={captureBotRight} disabled={pos.pan == null} style={actionBtn(C.accent, pos.pan == null)}>
                Set Bottom Right
              </button>
              <div style={{ fontSize: 10, color: botRight ? C.accent : C.muted, fontFamily: 'monospace', marginTop: 4, lineHeight: 1.5 }}>
                {botRight
                  ? <>Pan {botRight.pan}<br />Tilt {botRight.tilt}<br />Zoom {botRight.zoom}</>
                  : 'Not set'
                }
              </div>
            </Section>

            <Divider />

            {/* Zoom */}
            <Section label="Scan Zoom">
              <button onClick={captureZoom} disabled={pos.zoom == null} style={actionBtn(C.yellow, pos.zoom == null)}>
                Set Zoom
              </button>
              <div style={{ fontSize: 10, color: scanZoom != null ? C.yellow : C.muted, fontFamily: 'monospace', marginTop: 4 }}>
                {scanZoom != null ? `Zoom ${scanZoom}` : 'Not set (uses corner values)'}
              </div>
            </Section>

            <Divider />

            {/* Save */}
            <div style={{ padding: '14px 16px' }}>
              <button
                onClick={save}
                disabled={saving || saved}
                style={{
                  width: '100%',
                  padding: '10px 0',
                  background: saved ? C.green : saving ? C.border : C.accent,
                  color: '#fff',
                  border: 'none',
                  borderRadius: 8,
                  fontWeight: 700,
                  fontSize: 14,
                  cursor: saving || saved ? 'default' : 'pointer',
                }}
              >
                {saved ? '✓ Saved!' : saving ? 'Saving…' : 'Save Bounds'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Section({ label, children }) {
  return (
    <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: C.muted }}>{label}</div>
      {children}
    </div>
  )
}

function Divider() {
  return <div style={{ height: 1, background: C.border, flexShrink: 0 }} />
}

function PtzBtn({ label, onStart, onEnd, wide }) {
  const held = useRef(false)
  const start = () => { held.current = true; onStart() }
  const end   = () => { if (!held.current) return; held.current = false; onEnd() }

  return (
    <button
      onMouseDown={start}
      onMouseUp={end}
      onMouseLeave={end}
      onTouchStart={e => { e.preventDefault(); start() }}
      onTouchEnd={end}
      style={{
        background: C.bg,
        border: `1px solid ${C.border}`,
        color: C.text,
        borderRadius: 6,
        cursor: 'pointer',
        fontWeight: 600,
        fontSize: wide ? 12 : 14,
        width: wide ? '100%' : 48,
        height: 48,
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

const actionBtn = (color, disabled) => ({
  width: '100%',
  padding: '8px 0',
  background: disabled ? C.border : color,
  color: disabled ? C.muted : color === C.yellow ? '#000' : '#fff',
  border: 'none',
  borderRadius: 7,
  fontWeight: 700,
  fontSize: 13,
  cursor: disabled ? 'default' : 'pointer',
  opacity: disabled ? 0.6 : 1,
})
