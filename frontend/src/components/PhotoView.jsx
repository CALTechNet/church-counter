import { useState, useEffect } from 'react'
import { useIsMobile } from '../hooks/useIsMobile.js'

const C = { bg: '#0f172a', surface: '#1e293b', muted: '#94a3b8', border: '#334155', text: '#f1f5f9' }

const VIEW_MODES = [
  { key: 'clean',   label: 'Before' },
  { key: 'overlay', label: 'AI Overlay' },
  { key: 'count',   label: 'Count' },
]

export default function PhotoView({ imageB64: propB64, rawImageB64: propRawB64, scanning }) {
  const isMobile = useIsMobile()
  const [scans, setScans]             = useState([])   // newest-first
  const [selectedId, setSelectedId]   = useState(null)
  const [imageB64, setImageB64]       = useState(propB64)
  const [rawImageB64, setRawImageB64] = useState(propRawB64 ?? null)
  const [ts, setTs]                   = useState(null)
  const [count, setCount]             = useState(null)
  const [loadingImg, setLoadingImg]   = useState(false)
  const [zoom, setZoom]               = useState(1)
  const [viewMode, setViewMode]       = useState('overlay')
  const [manualCount, setManualCount] = useState('')

  // Fetch the full scan list (newest-first)
  const loadScans = () =>
    fetch('/api/attendance')
      .then(r => r.json())
      .then(data => {
        const list = (Array.isArray(data) ? data : (data.records || []))
          .slice()
          .reverse()   // newest first for the dropdown
        setScans(list)
        return list
      })
      .catch(() => [])

  // Fetch and display a specific scan's image
  const fetchScanImage = async (id, scan) => {
    setLoadingImg(true)
    setSelectedId(id)
    try {
      const res  = await fetch(`/api/attendance/${id}/image`)
      if (!res.ok) throw new Error('No image')
      const data = await res.json()
      setImageB64(data.image_b64)
      setRawImageB64(data.raw_image_b64 ?? null)
      setTs(scan?.timestamp ?? data.timestamp)
      setCount(scan ? (scan.count || 0) + (scan.manual_add || 0) : data.count)
    } catch {
      setImageB64(null)
      setRawImageB64(null)
    } finally {
      setLoadingImg(false)
    }
  }

  // On mount: load list + show latest if no live image
  useEffect(() => {
    loadScans().then(list => {
      if (!propB64 && list.length > 0) {
        fetchScanImage(list[0].id, list[0])
      }
    })
  }, [])

  // When a new scan completes, reload the list and show the fresh image
  useEffect(() => {
    if (!propB64) return
    setImageB64(propB64)
    setRawImageB64(propRawB64 ?? null)
    setTs(null)
    setCount(null)
    setSelectedId(null)
    setManualCount('')
    loadScans().then(list => {
      // Auto-select the newest entry so the dropdown stays in sync
      if (list.length > 0) setSelectedId(list[0].id)
    })
  }, [propB64])

  const handleSelect = (e) => {
    const id   = Number(e.target.value)
    const scan = scans.find(s => s.id === id)
    setManualCount('')
    fetchScanImage(id, scan)
  }

  // Scan numbers: scans[] is newest-first, total count tells us the 1-based number
  const scanNumber = (idx) => scans.length - idx

  const displayTs = ts ?? scans.find(s => s.id === selectedId)?.timestamp

  // Which image to show based on view mode
  const activeImage = viewMode === 'clean' ? (rawImageB64 ?? imageB64) : imageB64

  return (
    <div style={{ flex:1, display:'flex', flexDirection:'column', background: C.bg, position:'relative' }}>

      {/* Toolbar */}
      <div style={{ background: C.surface, borderBottom:`1px solid ${C.border}`, padding: isMobile ? '8px 12px' : '8px 20px', display:'flex', alignItems:'center', gap:8, flexWrap:'wrap' }}>

        {/* Scan picker */}
        <div style={{ display:'flex', alignItems:'center', gap:6, flex: isMobile ? '1 1 auto' : '0 0 auto' }}>
          <span style={{ fontSize:12, color:C.muted, whiteSpace:'nowrap' }}>Scan:</span>
          <select
            value={selectedId ?? ''}
            onChange={handleSelect}
            disabled={scanning || scans.length === 0}
            style={{
              background:'#0f172a', border:`1px solid ${C.border}`, color:C.text,
              borderRadius:6, padding:'4px 8px', fontSize:12, cursor:'pointer',
              flex: isMobile ? '1 1 0' : '0 0 auto', maxWidth: isMobile ? '100%' : 360, minWidth:0,
            }}
          >
            {scans.length === 0 && <option value="">No scans yet</option>}
            {scans.map((s, idx) => {
              const total = (s.count || 0) + (s.manual_add || 0)
              const dt    = new Date(s.timestamp).toLocaleString([], {
                month:'short', day:'numeric', year:'2-digit',
                hour:'2-digit', minute:'2-digit',
              })
              return (
                <option key={s.id} value={s.id}>
                  #{scanNumber(idx)} · {dt} · {s.service_type || 'Manual'} · {total} people
                </option>
              )
            })}
          </select>
        </div>

        {/* View mode toggle */}
        <div style={{ display:'flex', gap:2, background:'#0f172a', borderRadius:8, padding:2, border:`1px solid ${C.border}` }}>
          {VIEW_MODES.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setViewMode(key)}
              style={{
                background: viewMode === key ? '#3b82f6' : 'transparent',
                border: 'none',
                color: viewMode === key ? '#fff' : C.muted,
                borderRadius:6, padding:'3px 10px', cursor:'pointer', fontSize:12,
                fontWeight: viewMode === key ? 600 : 400,
                transition: 'all 0.15s',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Info label — hidden on mobile to save space */}
        {!isMobile && (
          <span style={{ fontSize:12, color:C.muted }}>
            {displayTs
              ? `Captured ${new Date(displayTs).toLocaleString()}`
              : 'Latest scan'}
            {count !== null && ` · AI count: ${count}`}
          </span>
        )}

        {/* Zoom + download */}
        <div style={{ marginLeft:'auto', display:'flex', gap:6 }}>
          {[['＋', 1.3], ['1:1', 0], ['－', 0.75]].map(([lbl, f]) => (
            <button key={lbl}
              onClick={() => f === 0 ? setZoom(1) : setZoom(z => Math.max(0.3, Math.min(8, z * f)))}
              style={{ background:'transparent', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'4px 10px', cursor:'pointer', fontSize:13 }}>
              {lbl}
            </button>
          ))}
          {activeImage && !isMobile && (
            <a href={`data:image/jpeg;base64,${activeImage}`} download={viewMode === 'clean' ? 'scan-raw.jpg' : 'scan-annotated.jpg'}
              style={{ background:'transparent', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'4px 10px', textDecoration:'none', fontSize:13 }}>
              ↓
            </a>
          )}
        </div>
      </div>

      {/* Count mode panel */}
      {viewMode === 'count' && (
        <div style={{ background:'#1e3a5f', borderBottom:`1px solid #2563eb`, padding: '12px 20px', display:'flex', alignItems:'center', gap:20, flexWrap:'wrap' }}>
          <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
            <span style={{ fontSize:11, color:'#93c5fd', textTransform:'uppercase', letterSpacing:'0.05em' }}>AI Count</span>
            <span style={{ fontSize:28, fontWeight:700, color:'#fff', lineHeight:1 }}>{count ?? '—'}</span>
          </div>
          <div style={{ width:1, height:40, background:'#2563eb' }} />
          <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
            <span style={{ fontSize:11, color:'#93c5fd', textTransform:'uppercase', letterSpacing:'0.05em' }}>Your Count</span>
            <input
              type="number"
              min="0"
              placeholder="Enter count…"
              value={manualCount}
              onChange={e => setManualCount(e.target.value)}
              style={{ background:'#0f172a', border:`1px solid #2563eb`, color:'#fff', borderRadius:6, padding:'4px 10px', fontSize:18, fontWeight:600, width:130 }}
            />
          </div>
          {manualCount !== '' && count !== null && (
            <>
              <div style={{ width:1, height:40, background:'#2563eb' }} />
              <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
                <span style={{ fontSize:11, color:'#93c5fd', textTransform:'uppercase', letterSpacing:'0.05em' }}>Difference</span>
                <span style={{
                  fontSize:22, fontWeight:700, lineHeight:1,
                  color: Math.abs(Number(manualCount) - count) <= 3 ? '#4ade80' : '#f87171'
                }}>
                  {Number(manualCount) > count ? '+' : ''}{Number(manualCount) - count}
                </span>
              </div>
            </>
          )}
          <span style={{ fontSize:12, color:'#60a5fa', marginLeft: isMobile ? 0 : 'auto' }}>
            Switch to "Before" for a clean view to hand-count
          </span>
        </div>
      )}

      {/* Image area */}
      <div style={{ flex:1, overflow:'auto', display:'flex', alignItems:'flex-start', justifyContent:'center', padding:20 }}>
        {loadingImg ? (
          <div style={{ alignSelf:'center', color:C.muted, fontSize:14 }}>Loading image…</div>
        ) : activeImage ? (
          <div style={{ position:'relative', display:'inline-block' }}>
            <img
              src={`data:image/jpeg;base64,${activeImage}`}
              alt={viewMode === 'clean' ? 'Raw scan (no annotations)' : 'Stitched scan with AI detections'}
              style={{
                maxWidth: zoom === 1 ? '100%' : 'none',
                width: zoom !== 1 ? `${zoom * 100}%` : undefined,
                borderRadius:8, boxShadow:'0 4px 32px rgba(0,0,0,0.5)',
                display:'block',
              }}
            />
            {viewMode === 'clean' && !rawImageB64 && (
              <div style={{ position:'absolute', top:8, left:8, background:'rgba(0,0,0,0.7)', color:'#fbbf24', fontSize:11, padding:'4px 8px', borderRadius:4 }}>
                Raw image unavailable for older scans — showing annotated
              </div>
            )}
          </div>
        ) : (
          <div style={{ alignSelf:'center', display:'flex', flexDirection:'column', alignItems:'center', gap:12, color:C.muted }}>
            <div style={{ fontSize:48 }}>📷</div>
            <div style={{ fontSize:16 }}>No scan photo yet</div>
            <div style={{ fontSize:13 }}>Run a scan to see the stitched room photo with AI detections</div>
          </div>
        )}
      </div>

      {/* Scanning overlay */}
      {scanning && (
        <div style={{ position:'absolute', inset:0, background:'rgba(15,23,42,0.6)', display:'flex', alignItems:'center', justifyContent:'center', pointerEvents:'none' }}>
          <div style={{ fontSize:18, color:'#fff', fontWeight:700 }}>⏳ Scanning…</div>
        </div>
      )}
    </div>
  )
}
