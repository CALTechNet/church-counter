import { useState, useEffect, useCallback } from 'react'
import PhotoView from './components/PhotoView.jsx'
import AttendanceGraph from './components/AttendanceGraph.jsx'
import DataTable from './components/DataTable.jsx'
import CalibrationWizard from './components/CalibrationWizard.jsx'
import { getStatus, triggerScan, createWebSocket } from './api.js'
import { VERSION } from './version.js'

const TABS = ['Photo', 'Attendance', 'Data']
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
  const [tab, setTab]             = useState(0)
  const [scanState, setScanState] = useState({ running: false, progress: 0, message: 'Idle' })
  const [seatStates, setSeatStates] = useState({})
  const [latestCount, setLatestCount] = useState(null)
  const [latestTs, setLatestTs]   = useState(null)
  const [latestService, setLatestService] = useState(null)
  const [imageB64, setImageB64]       = useState(null)
  const [scanStartedAt, setScanStartedAt] = useState(null)
  const [zoomImageB64, setZoomImageB64] = useState(null)
  const [serviceType, setServiceType] = useState('Manual')
  const [showCal, setShowCal]     = useState(false)
  const [hasCalibration, setHasCalibration] = useState(false)
  const [toast, setToast]         = useState(null)

  const showToast = (msg, color = C.green) => {
    setToast({ msg, color })
    setTimeout(() => setToast(null), 3500)
  }

  // Bootstrap status
  useEffect(() => {
    getStatus().then(s => {
      setScanState({ running: s.running, progress: s.progress, message: s.message })
      setSeatStates(s.seat_states || {})
      setLatestCount(s.latest_count)
      setLatestTs(s.latest_timestamp)
      setLatestService(s.latest_service)
      setHasCalibration(s.has_calibration)
    }).catch(() => {})
  }, [])

  // WebSocket
  useEffect(() => {
    let ws
    const connect = () => {
      ws = createWebSocket(msg => {
        if (msg.type === 'progress') {
          setScanState({ running: true, progress: msg.progress, message: msg.message })
        } else if (msg.type === 'scan_started') {
          setScanState({ running: true, progress: 0, message: 'Starting…' })
          setScanStartedAt(Date.now())
        } else if (msg.type === 'scan_complete') {
          setScanState({ running: false, progress: 100, message: 'Complete!' })
          setScanStartedAt(null)
          setSeatStates(msg.seat_states || {})
          setLatestCount(msg.count)
          setLatestTs(msg.timestamp)
          setLatestService(msg.service_type)
          setImageB64(msg.image_b64)
          setZoomImageB64(msg.zoom_image_b64)
          showToast(`Scan complete — ${msg.count} people detected`)
        } else if (msg.type === 'scan_error') {
          setScanState(s => ({ ...s, running: false, message: `Error: ${msg.error}` }))
          setScanStartedAt(null)
          showToast(`Scan error: ${msg.error}`, C.red)
        } else if (msg.type === 'state') {
          setScanState({ running: msg.running, progress: msg.progress, message: msg.message })
          setSeatStates(msg.latest_seat_states || {})
          setLatestCount(msg.latest_count)
          setLatestTs(msg.latest_timestamp)
          setLatestService(msg.latest_service)
          setImageB64(msg.latest_image_b64)
          setZoomImageB64(msg.latest_zoom_image_b64)
        }
      })
      ws.onclose = () => setTimeout(connect, 3000)
    }
    connect()
    return () => ws && ws.close()
  }, [])

  const handleScan = async () => {
    try {
      await triggerScan(serviceType)
    } catch (e) {
      showToast(e.message, C.red)
    }
  }

  const occupiedCount = Object.values(seatStates).filter(v => v === 'occupied').length
  const totalSeats    = Object.keys(seatStates).length

  return (
    <div style={{ minHeight: '100vh', background: C.bg, color: C.text, fontFamily: 'system-ui, sans-serif', display: 'flex', flexDirection: 'column' }}>

      {/* Header */}
      <header style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: '0 24px', display: 'flex', alignItems: 'center', gap: 16, height: 60 }}>
        <span style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.5px' }}>⛪ Lakeshore Church</span>
        <span style={{ color: C.muted, fontSize: 14 }}>Attendance Counter</span>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
          {latestCount !== null && (
            <div style={{ textAlign: 'right', marginRight: 8 }}>
              <div style={{ fontSize: 24, fontWeight: 700, color: C.green, lineHeight: 1 }}>{latestCount}</div>
              <div style={{ fontSize: 11, color: C.muted }}>
                {latestService} · {latestTs ? new Date(latestTs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
              </div>
            </div>
          )}

          <select
            value={serviceType}
            onChange={e => setServiceType(e.target.value)}
            disabled={scanState.running}
            style={{ background: C.surface, border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '6px 10px', fontSize: 13, cursor: 'pointer' }}
          >
            {SERVICE_TYPES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          <button
            onClick={handleScan}
            disabled={scanState.running}
            style={{
              background: scanState.running ? C.border : C.accent,
              color: '#fff', border: 'none', borderRadius: 8,
              padding: '8px 20px', fontWeight: 600, fontSize: 14,
              cursor: scanState.running ? 'not-allowed' : 'pointer',
              transition: 'background 0.2s',
            }}
          >
            {scanState.running ? '⏳ Scanning…' : '▶ Scan Now'}
          </button>

          {/* Force reset — shown when scan has been running >5 min */}
          {scanState.running && scanStartedAt && (Date.now() - scanStartedAt) > 300000 && (
            <button
              onClick={() => { setScanState({ running: false, progress: 0, message: 'Idle' }); setScanStartedAt(null) }}
              style={{ background: C.red, color: '#fff', border: 'none', borderRadius: 8, padding: '8px 14px', fontWeight: 600, fontSize: 13, cursor: 'pointer' }}
              title="Force reset stuck scan state"
            >
              ✕ Reset
            </button>
          )}

          <button
            onClick={() => setShowCal(true)}
            style={{ background: 'transparent', color: C.muted, border: `1px solid ${C.border}`, borderRadius: 8, padding: '7px 14px', fontSize: 13, cursor: 'pointer' }}
          >
            ⚙ Calibrate
          </button>
        </div>
      </header>

      {/* Progress bar */}
      {scanState.running && (
        <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: '6px 24px', display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ flex: 1, background: C.border, borderRadius: 4, height: 6, overflow: 'hidden' }}>
            <div style={{ width: `${scanState.progress}%`, height: '100%', background: C.accent, transition: 'width 0.5s', borderRadius: 4 }} />
          </div>
          <span style={{ fontSize: 12, color: C.muted, whiteSpace: 'nowrap', minWidth: 180 }}>
            {scanState.message} ({scanState.progress}%)
          </span>
        </div>
      )}

      {/* Tabs */}
      <nav style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: '0 24px', display: 'flex', gap: 0 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            background: 'transparent', border: 'none', color: tab === i ? C.text : C.muted,
            borderBottom: tab === i ? `2px solid ${C.accent}` : '2px solid transparent',
            padding: '12px 20px', cursor: 'pointer', fontWeight: tab === i ? 600 : 400,
            fontSize: 14, transition: 'color 0.15s',
          }}>{t}</button>
        ))}


      </nav>

      {/* Content */}
      <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {tab === 0 && <PhotoView imageB64={imageB64} zoomImageB64={zoomImageB64} scanning={scanState.running} />}
        {tab === 1 && <AttendanceGraph />}
        {tab === 2 && <DataTable />}
      </main>

      {/* Calibration modal */}
      {showCal && (
        <CalibrationWizard
          onClose={() => { setShowCal(false); setHasCalibration(true) }}
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
