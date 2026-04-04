import { useState, useEffect, useCallback } from 'react'
import PhotoView from './components/PhotoView.jsx'
import AttendanceGraph from './components/AttendanceGraph.jsx'
import DataTable from './components/DataTable.jsx'
import LiveView from './components/LiveView.jsx'
import CalibrationWizard from './components/CalibrationWizard.jsx'
import SettingsModal from './components/SettingsModal.jsx'
import { getStatus, triggerScan, cancelScan, createWebSocket, getPtzPosition, getSettings } from './api.js'
import { VERSION } from './version.js'
import { useIsMobile } from './hooks/useIsMobile.js'

const TABS = ['Photo', 'Attendance', 'Data', 'Live View']
const SERVICE_TYPES = ['Manual', 'Sunday Morning', 'Sunday Midday', 'Wednesday Evening']

const C = {
  bg:       '#0f172a',
  surface:  '#1e293b',
  border:   '#334155',
  text:     '#f1f5f9',
  muted:    '#94a3b8',
  accent:   '#3b82f6',
  green:    '#22c55e',
  red:      '#ef4444',
  yellow:   '#eab308',
}

export default function App() {
  const isMobile = useIsMobile()
  const [tab, setTab]             = useState(0)
  const [scanState, setScanState] = useState({ running: false, progress: 0, message: 'Idle' })
  const [seatStates, setSeatStates] = useState({})
  const [latestCount, setLatestCount] = useState(null)
  const [latestTs, setLatestTs]   = useState(null)
  const [latestService, setLatestService] = useState(null)
  const [imageB64, setImageB64]       = useState(null)
  const [rawImageB64, setRawImageB64] = useState(null)
  const [scanStartedAt, setScanStartedAt] = useState(null)
  const [totalPositions, setTotalPositions] = useState(null)
  const [processingStartedAt, setProcessingStartedAt] = useState(null)
  const [now, setNow] = useState(Date.now())
  const [zoomImageB64, setZoomImageB64] = useState(null)
  const [serviceType, setServiceType] = useState('Manual')
  const [showCal, setShowCal]         = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [hasCalibration, setHasCalibration] = useState(false)
  const [toast, setToast]         = useState(null)
  const [ptzPos, setPtzPos]       = useState(null)
  const [churchName, setChurchName] = useState('Lakeshore Church')
  const [rooms, setRooms]         = useState([])
  const [selectedRoom, setSelectedRoom] = useState(null)

  const showToast = (msg, color = C.green) => {
    setToast({ msg, color })
    setTimeout(() => setToast(null), 3500)
  }

  // Bootstrap status + settings
  useEffect(() => {
    getStatus().then(s => {
      setScanState({ running: s.running, progress: s.progress, message: s.message })
      setSeatStates(s.seat_states || {})
      setLatestCount(s.latest_count)
      setLatestTs(s.latest_timestamp)
      setLatestService(s.latest_service)
      setHasCalibration(s.has_calibration)
    }).catch(() => {})
    getSettings().then(s => {
      if (s.church_name) setChurchName(s.church_name)
      if (s.rooms && s.rooms.length > 0) {
        setRooms(s.rooms)
        setSelectedRoom(s.rooms[0].id)
      }
    }).catch(() => {})
  }, [])

  // WebSocket
  useEffect(() => {
    let ws
    const connect = () => {
      ws = createWebSocket(msg => {
        if (msg.type === 'progress') {
          setScanState({ running: true, progress: msg.progress, message: msg.message })
          // Detect transition to processing phase (camera going home / stitching)
          if (msg.progress >= 90) {
            setProcessingStartedAt(prev => prev || Date.now())
          }
        } else if (msg.type === 'scan_started') {
          setScanState({ running: true, progress: 0, message: 'Starting…' })
          setScanStartedAt(Date.now())
          setProcessingStartedAt(null)
          setTotalPositions(msg.total_positions || null)
        } else if (msg.type === 'scan_complete') {
          setScanState({ running: false, progress: 100, message: 'Complete!' })
          setScanStartedAt(null)
          setProcessingStartedAt(null)
          setTotalPositions(null)
          setSeatStates(msg.seat_states || {})
          setLatestCount(msg.count)
          setLatestTs(msg.timestamp)
          setLatestService(msg.service_type)
          setImageB64(msg.image_b64)
          setRawImageB64(msg.raw_image_b64 ?? null)
          setZoomImageB64(msg.zoom_image_b64)
          showToast(`Scan complete${msg.room ? ` (${msg.room})` : ''} — ${msg.count} people detected`)
        } else if (msg.type === 'scan_cancelled') {
          setScanState({ running: false, progress: 0, message: 'Cancelled' })
          setScanStartedAt(null)
          setProcessingStartedAt(null)
          setTotalPositions(null)
          showToast('Scan cancelled', C.yellow)
        } else if (msg.type === 'scan_error') {
          setScanState(s => ({ ...s, running: false, message: `Error: ${msg.error}` }))
          setScanStartedAt(null)
          setProcessingStartedAt(null)
          setTotalPositions(null)
          showToast(`Scan error: ${msg.error}`, C.red)
        } else if (msg.type === 'state') {
          setScanState({ running: msg.running, progress: msg.progress, message: msg.message })
          setSeatStates(msg.latest_seat_states || {})
          setLatestCount(msg.latest_count)
          setLatestTs(msg.latest_timestamp)
          setLatestService(msg.latest_service)
          setImageB64(msg.latest_image_b64)
          setRawImageB64(msg.latest_raw_image_b64 ?? null)
          setZoomImageB64(msg.latest_zoom_image_b64)
        }
      })
      ws.onclose = () => setTimeout(connect, 3000)
    }
    connect()
    return () => ws && ws.close()
  }, [])

  // Tick every second during scan for time estimates
  useEffect(() => {
    if (!scanState.running) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [scanState.running])

  // PTZ position polling — every second
  useEffect(() => {
    const poll = () => getPtzPosition().then(setPtzPos).catch(() => {})
    poll()
    const id = setInterval(poll, 1000)
    return () => clearInterval(id)
  }, [])

  const handleScan = async () => {
    try {
      await triggerScan(serviceType, rooms.length > 1 ? selectedRoom : null)
    } catch (e) {
      showToast(e.message, C.red)
    }
  }

  const handleCancel = async () => {
    try {
      await cancelScan()
    } catch (e) {
      showToast(e.message, C.red)
    }
  }

  const occupiedCount = Object.values(seatStates).filter(v => v === 'occupied').length
  const totalSeats    = Object.keys(seatStates).length

  return (
    <div style={{ minHeight: '100vh', background: C.bg, color: C.text, fontFamily: 'system-ui, sans-serif', display: 'flex', flexDirection: 'column' }}>

      {/* Header */}
      <header style={{
        background: C.surface, borderBottom: `1px solid ${C.border}`,
        padding: isMobile ? '8px 16px' : '0 24px',
        display: 'flex', alignItems: 'center', gap: isMobile ? 8 : 16,
        minHeight: 60, flexWrap: isMobile ? 'wrap' : 'nowrap',
      }}>
        {/* Branding */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: isMobile ? '1 1 auto' : '0 0 auto' }}>
          <span style={{ fontSize: isMobile ? 18 : 22, fontWeight: 700, letterSpacing: '-0.5px', whiteSpace: 'nowrap' }}>⛪ {churchName}</span>
          {!isMobile && <span style={{ color: C.muted, fontSize: 14 }}>Attendance Counter</span>}
        </div>

        {/* PTZ position status */}
        {!isMobile && ptzPos && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11, color: C.muted, fontFamily: 'monospace', background: C.bg, borderRadius: 6, padding: '4px 10px', border: `1px solid ${C.border}` }}>
            <span style={{ color: C.border, fontSize: 10, fontFamily: 'system-ui', letterSpacing: '0.05em', textTransform: 'uppercase', marginRight: 2 }}>PTZ</span>
            <span>X: {ptzPos.pan  ?? '--'}</span>
            <span>Y: {ptzPos.tilt ?? '--'}</span>
            <span>Z: {ptzPos.zoom ?? '--'}</span>
          </div>
        )}

        {/* Latest count — inline on mobile, right-aligned on desktop */}
        {latestCount !== null && (
          <div style={{ textAlign: isMobile ? 'left' : 'right', marginRight: isMobile ? 0 : 8, flex: isMobile ? '0 0 auto' : '0 0 auto', marginLeft: isMobile ? 'auto' : 0 }}>
            <div style={{ fontSize: isMobile ? 20 : 24, fontWeight: 700, color: C.green, lineHeight: 1 }}>{latestCount}</div>
            <div style={{ fontSize: 10, color: C.muted, whiteSpace: 'nowrap' }}>
              {isMobile ? '' : `${latestService} · `}{latestTs ? new Date(latestTs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
            </div>
          </div>
        )}

        {/* Controls row — wraps to second line on mobile */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          flex: isMobile ? '1 1 100%' : '0 0 auto',
          marginLeft: isMobile ? 0 : 'auto',
          flexWrap: 'wrap',
        }}>
          <select
            value={serviceType}
            onChange={e => setServiceType(e.target.value)}
            disabled={scanState.running}
            style={{ background: C.surface, border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '6px 8px', fontSize: 12, cursor: 'pointer', flex: isMobile ? '1 1 auto' : '0 0 auto' }}
          >
            {SERVICE_TYPES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          {rooms.length > 1 && (
            <select
              value={selectedRoom || ''}
              onChange={e => setSelectedRoom(e.target.value)}
              disabled={scanState.running}
              style={{ background: C.surface, border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '6px 8px', fontSize: 12, cursor: 'pointer', flex: isMobile ? '1 1 auto' : '0 0 auto' }}
            >
              {rooms.map(r => <option key={r.id} value={r.id}>{r.name || r.id}</option>)}
            </select>
          )}

          <button
            onClick={handleScan}
            disabled={scanState.running}
            style={{
              background: scanState.running ? C.border : C.accent,
              color: '#fff', border: 'none', borderRadius: 8,
              padding: '8px 16px', fontWeight: 600, fontSize: 13,
              cursor: scanState.running ? 'not-allowed' : 'pointer',
              transition: 'background 0.2s', whiteSpace: 'nowrap',
            }}
          >
            {scanState.running ? '⏳ Scanning…' : '▶ Scan Now'}
          </button>

          {/* Cancel button — shown while a scan is running */}
          {scanState.running && (
            <button
              onClick={handleCancel}
              style={{ background: C.red, color: '#fff', border: 'none', borderRadius: 8, padding: '8px 12px', fontWeight: 600, fontSize: 12, cursor: 'pointer', whiteSpace: 'nowrap' }}
              title="Cancel the current scan"
            >
              ✕ Cancel
            </button>
          )}

          <button
            onClick={() => setShowCal(true)}
            style={{ background: 'transparent', color: C.muted, border: `1px solid ${C.border}`, borderRadius: 8, padding: '7px 12px', fontSize: 12, cursor: 'pointer', whiteSpace: 'nowrap' }}
          >
            ⚙{!isMobile && ' Calibrate'}
          </button>

          <button
            onClick={() => setShowSettings(true)}
            title="Settings"
            style={{ background: 'transparent', color: C.muted, border: `1px solid ${C.border}`, borderRadius: 8, padding: '7px 12px', fontSize: 12, cursor: 'pointer', whiteSpace: 'nowrap' }}
          >
            {isMobile ? '⚙' : '⚙ Settings'}
          </button>
        </div>
      </header>

      {/* Progress bar */}
      {scanState.running && (() => {
        const isProcessing = scanState.progress >= 90
        const isInitiating = scanState.progress === 0
        const SECS_PER_FRAME = 5 // ~5 seconds per scanner frame
        const PROCESSING_SECS = 600 // ~10 minutes for stitching + counting
        const formatTime = (secs) => {
          const m = Math.floor(secs / 60)
          const s = Math.floor(secs % 60)
          return m > 0 ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`
        }
        let timeStr = ''
        if (scanStartedAt) {
          if (!isProcessing) {
            // Capture phase: estimate based on 5 seconds per frame
            const elapsed = (now - scanStartedAt) / 1000
            if (totalPositions && totalPositions > 0) {
              const captureTotal = totalPositions * SECS_PER_FRAME
              const captureRemaining = Math.max(0, captureTotal - elapsed)
              timeStr = `~${formatTime(Math.round(captureRemaining + PROCESSING_SECS))} remaining`
            } else {
              // Fallback: estimate from elapsed time + progress
              const pct = Math.max(scanState.progress, 1)
              const captureTotal = (elapsed / pct) * 88
              const captureRemaining = Math.max(0, captureTotal - elapsed)
              timeStr = `~${formatTime(Math.round(captureRemaining + PROCESSING_SECS))} remaining`
            }
          } else if (processingStartedAt) {
            // Processing phase: 10-min countdown for stitching + counting
            const processingElapsed = (now - processingStartedAt) / 1000
            const remaining = Math.max(0, PROCESSING_SECS - processingElapsed)
            timeStr = `~${formatTime(Math.round(remaining))} remaining`
          }
        }
        const displayMessage = isInitiating
          ? '✓ Scan initiated — starting camera sweep…'
          : isProcessing && scanState.progress < 100
            ? 'Stitching & counting — please wait…'
            : scanState.message
        return (
          <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: '8px 24px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ flex: 1, background: C.border, borderRadius: 4, height: 8, overflow: 'hidden', position: 'relative' }}>
              {isInitiating ? (
                /* Indeterminate animated bar while scan is initiating */
                <div style={{
                  position: 'absolute', top: 0, left: 0,
                  width: '30%', height: '100%',
                  background: `linear-gradient(90deg, transparent, ${C.accent}, transparent)`,
                  borderRadius: 4,
                  animation: 'scanSlide 1.5s ease-in-out infinite',
                }} />
              ) : (
                <div style={{
                  width: isProcessing ? '100%' : `${scanState.progress}%`,
                  height: '100%',
                  background: isProcessing ? C.yellow : C.accent,
                  transition: 'width 0.5s',
                  borderRadius: 4,
                  ...(isProcessing ? { animation: 'pulse 2s ease-in-out infinite', opacity: 0.8 } : {}),
                }} />
              )}
            </div>
            <span style={{ fontSize: 12, color: isInitiating ? C.green : C.muted, fontWeight: isInitiating ? 600 : 400, whiteSpace: 'nowrap', minWidth: 220 }}>
              {displayMessage} {!isInitiating && scanState.progress < 90 ? `(${scanState.progress}%)` : ''}
            </span>
            {timeStr && (
              <span style={{
                fontSize: 13, fontWeight: 700, fontFamily: 'monospace',
                color: isProcessing ? C.yellow : C.accent,
                whiteSpace: 'nowrap',
                background: C.bg, borderRadius: 6, padding: '3px 10px',
                border: `1px solid ${isProcessing ? C.yellow : C.accent}`,
              }}>
                {timeStr}
              </span>
            )}
          </div>
        )
      })()}

      {/* Tabs */}
      <nav style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: isMobile ? '0 8px' : '0 24px', display: 'flex', gap: 0 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            background: 'transparent', border: 'none', color: tab === i ? C.text : C.muted,
            borderBottom: tab === i ? `2px solid ${C.accent}` : '2px solid transparent',
            padding: isMobile ? '12px 16px' : '12px 20px', cursor: 'pointer', fontWeight: tab === i ? 600 : 400,
            fontSize: isMobile ? 13 : 14, transition: 'color 0.15s', flex: isMobile ? 1 : '0 0 auto', textAlign: 'center',
          }}>{t}</button>
        ))}
      </nav>

      {/* Content */}
      <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {tab === 0 && <PhotoView imageB64={imageB64} rawImageB64={rawImageB64} zoomImageB64={zoomImageB64} scanning={scanState.running} />}
        {tab === 1 && <AttendanceGraph />}
        {tab === 2 && <DataTable />}
        {tab === 3 && <LiveView scanning={scanState.running} ptzPos={ptzPos} scanProgress={scanState.progress} totalPositions={totalPositions} />}
      </main>

      {/* Calibration modal */}
      {showCal && (
        <CalibrationWizard
          onClose={() => { setShowCal(false); setHasCalibration(true) }}
        />
      )}

      {/* Settings modal */}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onSave={s => {
            if (s.church_name) setChurchName(s.church_name)
            if (s.rooms && s.rooms.length > 0) {
              setRooms(s.rooms)
              if (!s.rooms.find(r => r.id === selectedRoom)) {
                setSelectedRoom(s.rooms[0].id)
              }
            }
          }}
        />
      )}

      {/* Footer */}
      <footer style={{
        background: C.surface, borderTop: `1px solid ${C.border}`,
        padding: '6px 24px', display: 'flex', alignItems: 'center', justifyContent: 'center',
        gap: 8, flexShrink: 0,
      }}>
        <span style={{ color: '#475569', fontSize: 11 }}>
          © 2026 · Clayton Little · For Lakeshore Church
        </span>
        <span style={{ color: C.border, fontSize: 11 }}>|</span>
        <span style={{ color: '#334155', fontSize: 11, fontFamily: 'monospace' }}>{VERSION}</span>
      </footer>

      {/* Toast */}
      {toast && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24, background: toast.color,
          color: '#fff', padding: '12px 20px', borderRadius: 10, fontWeight: 600,
          fontSize: 14, boxShadow: '0 4px 24px rgba(0,0,0,0.4)', zIndex: 9999,
          animation: 'fadeIn 0.2s ease',
        }}>{toast.msg}</div>
      )}
    </div>
  )
}
