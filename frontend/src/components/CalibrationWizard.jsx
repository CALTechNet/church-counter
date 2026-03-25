/**
 * Calibration Wizard — Camera Bounds Setup
 * Set left/right pan edges, top/bottom tilt edges, and scan zoom independently.
 * Live feed streams continuously with no buffer for real-time feedback.
 */
import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { ptzCommand, getPtzPosition, getCameraBounds, saveCameraBounds, gotoBound } from '../api.js'

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

export default function CalibrationWizard({ onClose }) {
  const [frameSrc, setFrameSrc]     = useState(null)
  const [frameError, setFrameError] = useState(null)
  const [fps, setFps]               = useState(null)

  const [pos, setPos]               = useState({ pan: null, tilt: null, zoom: null })
  const [posError, setPosError]     = useState(false)

  // Individual edge bounds
  const [leftPan,    setLeftPan]    = useState(null)
  const [rightPan,   setRightPan]   = useState(null)
  const [topTilt,    setTopTilt]    = useState(null)
  const [bottomTilt, setBottomTilt] = useState(null)
  const [scanZoom,   setScanZoom]   = useState(null)

  // Lens distortion correction
  const [lensK1, setLensK1] = useState(-0.32)
  const [lensK2, setLensK2] = useState(0.12)

  const [saving, setSaving]         = useState(false)
  const [saved, setSaved]           = useState(false)
  const [status, setStatus]         = useState('')

  const lastFrameTime = useRef(null)
  const liveRunning   = useRef(false)
  const posInterval   = useRef(null)

  // Computed 4 corners from stored edges
  const corners = useMemo(() => {
    if (leftPan == null || rightPan == null || topTilt == null || bottomTilt == null) return null
    return {
      top_left:     { pan: leftPan,  tilt: topTilt },
      top_right:    { pan: rightPan, tilt: topTilt },
      bottom_left:  { pan: leftPan,  tilt: bottomTilt },
      bottom_right: { pan: rightPan, tilt: bottomTilt },
    }
  }, [leftPan, rightPan, topTilt, bottomTilt])

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
      if (liveRunning.current) setTimeout(fetchFrame, 100)
    }

    fetchFrame()
    return () => {
      liveRunning.current = false
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
      if (b.left   != null) setLeftPan(b.left)
      if (b.right  != null) setRightPan(b.right)
      if (b.top    != null) setTopTilt(b.top)
      if (b.bottom != null) setBottomTilt(b.bottom)
      if (b.zoom   != null) setScanZoom(b.zoom)
      if (b.lens_k1 != null) setLensK1(b.lens_k1)
      if (b.lens_k2 != null) setLensK2(b.lens_k2)
    }).catch(() => {})
  }, [])

  // ── PTZ control helpers ───────────────────────────────────────────────────
  const sendPtz  = useCallback((action, speed = 3) => ptzCommand(action, speed).catch(() => {}), [])
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
      if (e.target.tagName === 'INPUT') return
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
  const captureLeft   = () => { if (pos.pan  == null) return; setLeftPan(pos.pan);     setStatus(`Left edge set (pan ${pos.pan}).`) }
  const captureRight  = () => { if (pos.pan  == null) return; setRightPan(pos.pan);    setStatus(`Right edge set (pan ${pos.pan}).`) }
  const captureTop    = () => { if (pos.tilt == null) return; setTopTilt(pos.tilt);    setStatus(`Top edge set (tilt ${pos.tilt}).`) }
  const captureBottom = () => { if (pos.tilt == null) return; setBottomTilt(pos.tilt); setStatus(`Bottom edge set (tilt ${pos.tilt}).`) }
  const captureZoom   = () => { if (pos.zoom == null) return; setScanZoom(pos.zoom);   setStatus(`Scan zoom set to ${pos.zoom}.`) }

  // ── Save ──────────────────────────────────────────────────────────────────
  const save = async () => {
    if (leftPan == null && rightPan == null && topTilt == null && bottomTilt == null && scanZoom == null) {
      setStatus('Nothing to save — set at least one bound.')
      return
    }
    setSaving(true)
    try {
      await saveCameraBounds({
        left:   leftPan,
        right:  rightPan,
        top:    topTilt,
        bottom: bottomTilt,
        zoom:   scanZoom,
        lens_k1: lensK1,
        lens_k2: lensK2,
      })
      setSaved(true)
      setStatus('Bounds saved!')
      setTimeout(onClose, 900)
    } catch (e) {
      setStatus(`Error saving: ${e.message}`)
    } finally {
      setSaving(false)
    }
  }

  const allCornersSet = corners != null

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
      <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 16, width: '98vw', maxWidth: 1300, height: '92vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Header */}
        <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: '14px 20px', display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0 }}>
          <span style={{ fontSize: 17, fontWeight: 700 }}>⚙ Camera Calibration</span>
          <span style={{ fontSize: 12, color: C.muted }}>Set left/right pan edges, top/bottom tilt edges, then set zoom</span>
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

            {/* Corner overlay badges */}
            {corners && (
              <>
                <div style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(34,197,94,0.85)', color: '#000', padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', fontWeight: 700 }}>
                  TL
                </div>
                <div style={{ position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)', background: 'rgba(168,85,247,0.85)', color: '#fff', padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', fontWeight: 700 }}>
                  TR
                </div>
                <div style={{ position: 'absolute', bottom: 10, right: 10, background: 'rgba(59,130,246,0.85)', color: '#fff', padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', fontWeight: 700 }}>
                  BR
                </div>
                <div style={{ position: 'absolute', bottom: 10, left: '50%', transform: 'translateX(-50%)', background: 'rgba(249,115,22,0.85)', color: '#fff', padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', fontWeight: 700 }}>
                  BL
                </div>
              </>
            )}

            <style>{`@keyframes livePulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
          </div>

          {/* ── Right sidebar ── */}
          <div style={{ width: 240, background: C.surface, borderLeft: `1px solid ${C.border}`, display: 'flex', flexDirection: 'column', gap: 0, flexShrink: 0, overflowY: 'auto' }}>

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

            {/* Bounding edges — capture or type */}
            <Section label="Set Bounds">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                {/* Pan edges */}
                <EdgeBtn label="◀ Set Left"  color={C.green}  set={leftPan   != null} value={leftPan}   unit="pan"  onClick={captureLeft}   disabled={pos.pan == null} />
                <EdgeBtn label="Set Right ▶" color={C.accent} set={rightPan  != null} value={rightPan}  unit="pan"  onClick={captureRight}  disabled={pos.pan == null} />
                {/* Tilt edges */}
                <EdgeBtn label="▲ Set Top"   color={C.yellow} set={topTilt   != null} value={topTilt}   unit="tilt" onClick={captureTop}    disabled={pos.tilt == null} />
                <EdgeBtn label="Set Bot ▼"   color={C.orange} set={bottomTilt!= null} value={bottomTilt}unit="tilt" onClick={captureBottom} disabled={pos.tilt == null} />
              </div>
              <div style={{ fontSize: 9, color: C.muted, marginTop: 2 }}>Or type VISCA values below:</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                <NumericInput label="Left Pan"   color={C.green}  value={leftPan}    onChange={v => { setLeftPan(v);    setStatus(`Left edge set (pan ${v}).`) }} />
                <NumericInput label="Right Pan"  color={C.accent} value={rightPan}   onChange={v => { setRightPan(v);   setStatus(`Right edge set (pan ${v}).`) }} />
                <NumericInput label="Top Tilt"   color={C.yellow} value={topTilt}    onChange={v => { setTopTilt(v);    setStatus(`Top edge set (tilt ${v}).`) }} />
                <NumericInput label="Bot Tilt"   color={C.orange} value={bottomTilt} onChange={v => { setBottomTilt(v); setStatus(`Bottom edge set (tilt ${v}).`) }} />
              </div>
            </Section>

            <Divider />

            {/* Scan Zoom */}
            <Section label="Scan Zoom">
              <button onClick={captureZoom} disabled={pos.zoom == null} style={actionBtn(C.yellow, pos.zoom == null)}>
                Set Zoom
              </button>
              <NumericInput label="Zoom" color={C.yellow} value={scanZoom} min={0} max={65535} onChange={v => { setScanZoom(v); setStatus(`Scan zoom set to ${v}.`) }} />
            </Section>

            <Divider />

            {/* Lens Distortion Correction */}
            <Section label="Lens Correction">
              <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>
                Corrects barrel distortion (camera curve). Negative k1 fixes barrel, positive fixes pincushion. Set both to 0 to disable.
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <label style={{ fontSize: 10, color: C.muted, width: 24, flexShrink: 0 }}>k1</label>
                  <input
                    type="range" min="-1" max="1" step="0.01" value={lensK1}
                    onChange={e => setLensK1(parseFloat(e.target.value))}
                    style={{ flex: 1, accentColor: C.accent }}
                  />
                  <input
                    type="number" step="0.01" value={lensK1}
                    onChange={e => setLensK1(parseFloat(e.target.value) || 0)}
                    style={{ width: 54, background: C.bg, border: `1px solid ${C.border}`, color: C.text, borderRadius: 4, padding: '2px 4px', fontSize: 11, textAlign: 'center' }}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <label style={{ fontSize: 10, color: C.muted, width: 24, flexShrink: 0 }}>k2</label>
                  <input
                    type="range" min="-0.5" max="0.5" step="0.01" value={lensK2}
                    onChange={e => setLensK2(parseFloat(e.target.value))}
                    style={{ flex: 1, accentColor: C.accent }}
                  />
                  <input
                    type="number" step="0.01" value={lensK2}
                    onChange={e => setLensK2(parseFloat(e.target.value) || 0)}
                    style={{ width: 54, background: C.bg, border: `1px solid ${C.border}`, color: C.text, borderRadius: 4, padding: '2px 4px', fontSize: 11, textAlign: 'center' }}
                  />
                </div>
                <button
                  onClick={() => { setLensK1(-0.32); setLensK2(0.12); setStatus('Lens correction reset to defaults.') }}
                  style={{ background: 'transparent', border: `1px solid ${C.border}`, color: C.muted, borderRadius: 4, padding: '3px 8px', fontSize: 10, cursor: 'pointer' }}
                >
                  Reset to defaults
                </button>
              </div>
            </Section>

            <Divider />

            {/* Go-to corners 2×2 matrix */}
            <Section label="Go To Corner">
              {allCornersSet ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                  <GotoBtn label="↖ TL" color={C.green}  onClick={() => gotoBound('top_left').catch(() => {})} />
                  <GotoBtn label="↗ TR" color={C.purple} onClick={() => gotoBound('top_right').catch(() => {})} />
                  <GotoBtn label="↙ BL" color={C.orange} onClick={() => gotoBound('bottom_left').catch(() => {})} />
                  <GotoBtn label="↘ BR" color={C.accent} onClick={() => gotoBound('bottom_right').catch(() => {})} />
                </div>
              ) : (
                <div style={{ fontSize: 10, color: C.muted }}>Set all 4 edges to enable.</div>
              )}
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

/** A capture button showing current value if set */
function EdgeBtn({ label, color, set, value, unit, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: disabled ? C.border : set ? color : C.bg,
        border: `1px solid ${set ? color : C.border}`,
        color: disabled ? C.muted : set ? (color === C.yellow ? '#000' : '#fff') : C.text,
        borderRadius: 7,
        fontWeight: 700,
        fontSize: 10,
        padding: '7px 4px',
        cursor: disabled ? 'default' : 'pointer',
        opacity: disabled ? 0.6 : 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
        userSelect: 'none',
      }}
    >
      <span>{label}</span>
      {set && <span style={{ fontFamily: 'monospace', fontSize: 9, opacity: 0.85 }}>{unit} {value}</span>}
    </button>
  )
}

/** One-shot goto corner button */
function GotoBtn({ label, color, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: C.bg,
        border: `1px solid ${color}`,
        color: color,
        borderRadius: 6,
        fontWeight: 700,
        fontSize: 11,
        padding: '8px 0',
        cursor: 'pointer',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      {label}
    </button>
  )
}

/** Inline numeric input for typing VISCA values directly */
function NumericInput({ label, color, value, onChange, min = -32768, max = 32767 }) {
  const [draft, setDraft] = useState('')
  const [editing, setEditing] = useState(false)

  const commit = () => {
    setEditing(false)
    const trimmed = draft.trim()
    if (trimmed === '' || isNaN(trimmed)) return
    const n = Math.max(min, Math.min(max, parseInt(trimmed, 10)))
    onChange(n)
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ fontSize: 9, color: color, fontWeight: 700, width: 52, flexShrink: 0 }}>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        placeholder={value != null ? String(value) : '--'}
        value={editing ? draft : (value != null ? String(value) : '')}
        onFocus={() => { setEditing(true); setDraft(value != null ? String(value) : '') }}
        onBlur={commit}
        onKeyDown={e => { if (e.key === 'Enter') { e.target.blur() } }}
        onChange={e => setDraft(e.target.value)}
        style={{
          flex: 1,
          minWidth: 0,
          background: C.bg,
          border: `1px solid ${value != null ? color : C.border}`,
          color: C.text,
          borderRadius: 4,
          padding: '4px 6px',
          fontSize: 11,
          fontFamily: 'monospace',
          outline: 'none',
          width: '100%',
        }}
      />
    </div>
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
