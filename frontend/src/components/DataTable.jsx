import { useState, useEffect, useRef } from 'react'
import { getAttendance } from '../api.js'

const C = { bg:'#0f172a', surface:'#1e293b', border:'#334155', text:'#f1f5f9', muted:'#94a3b8', accent:'#3b82f6', green:'#22c55e', red:'#ef4444' }
const SVC_COLORS = { 'Sunday Morning':'#3b82f6','Sunday Midday':'#a855f7','Wednesday Evening':'#f59e0b','Manual':'#64748b','Test':'#f43f5e' }

export default function DataTable() {
  const [scans, setScans]       = useState([])
  const [loading, setLoading]   = useState(true)
  const [filter, setFilter]     = useState('All')
  const [sortDesc, setSortDesc] = useState(true)
  const [modal, setModal]       = useState(null)
  const [editNotes, setEditNotes]             = useState('')
  const [editManualAdd, setEditManualAdd]     = useState(0)
  const [editServiceType, setEditServiceType] = useState('Manual')
  const [saving, setSaving]                   = useState(false)
  const [showArchived, setShowArchived]       = useState(false)

  // Manual entry modal
  const [showManualEntry, setShowManualEntry]   = useState(false)
  const [manualDateTime, setManualDateTime]     = useState('')
  const [manualServiceType, setManualServiceType] = useState('Sunday Morning')
  const [manualCount, setManualCount]           = useState('')
  const [manualNotes, setManualNotes]           = useState('')
  const [savingManual, setSavingManual]         = useState(false)

  const SERVICE_TYPES = ['Manual', 'Sunday Morning', 'Sunday Midday', 'Wednesday Evening']

  const openManualEntry = () => {
    const now = new Date()
    now.setSeconds(0, 0)
    setManualDateTime(now.toISOString().slice(0, 16))
    setManualServiceType('Sunday Morning')
    setManualCount('')
    setManualNotes('')
    setShowManualEntry(true)
  }

  const saveManualEntry = async () => {
    if (!manualCount || isNaN(Number(manualCount))) return
    setSavingManual(true)
    try {
      const res = await fetch('/api/attendance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp:    new Date(manualDateTime).toISOString(),
          service_type: manualServiceType,
          count:        Number(manualCount),
          notes:        manualNotes || null,
        }),
      })
      if (!res.ok) throw new Error('Save failed')
      setShowManualEntry(false)
      loadScans()
    } catch (e) {
      alert('Failed to save: ' + e.message)
    } finally {
      setSavingManual(false)
    }
  }

  // Zoom / pan
  const [zoom, setZoom]       = useState(1)
  const [pan, setPan]         = useState({ x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const lastMouse             = useRef({ x: 0, y: 0 })

  const loadScans = (inclArchived = showArchived) => {
    const url = inclArchived ? '/api/attendance?include_archived=true' : '/api/attendance'
    fetch(url).then(r => r.json()).then(s => { setScans(s); setLoading(false) }).catch(() => setLoading(false))
  }

  useEffect(() => { loadScans(showArchived) }, [showArchived])

  // Reset zoom/pan when a different scan is opened
  useEffect(() => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [modal?.scan?.id])

  const services = ['All', ...Array.from(new Set(scans.map(s => s.service_type || 'Manual')))]

  const filtered = scans
    .filter(s => filter === 'All' || (s.service_type || 'Manual') === filter)
    .sort((a, b) => sortDesc
      ? new Date(b.timestamp) - new Date(a.timestamp)
      : new Date(a.timestamp) - new Date(b.timestamp))

  // Navigation: always oldest→newest regardless of table sort
  const sortedFiltered = [...filtered].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
  const modalIdx = modal ? sortedFiltered.findIndex(s => s.id === modal.scan.id) : -1
  const hasPrev  = modalIdx > 0
  const hasNext  = modalIdx < sortedFiltered.length - 1

  // Ref so the keyboard handler always has fresh values without re-registering
  const navRef = useRef(null)
  navRef.current = { sortedFiltered, modalIdx, hasPrev, hasNext }

  const openRow = async (scan) => {
    setEditNotes(scan.notes || '')
    setEditManualAdd(scan.manual_add || 0)
    setEditServiceType(scan.service_type || 'Manual')
    setModal({ scan, imageB64: null, imageLoading: true })
    try {
      const res  = await fetch(`/api/attendance/${scan.id}/image`)
      if (!res.ok) throw new Error('No image')
      const data = await res.json()
      setModal(m => ({ ...m, imageB64: data.image_b64, imageLoading: false }))
    } catch {
      setModal(m => ({ ...m, imageLoading: false }))
    }
  }

  const navigateModal = (dir) => {
    const { sortedFiltered, modalIdx } = navRef.current
    const next = sortedFiltered[modalIdx + dir]
    if (next) openRow(next)
  }

  // Keyboard: ←/→ to navigate, Escape to close
  useEffect(() => {
    if (!modal) return
    const handler = (e) => {
      if (e.key === 'ArrowLeft'  && navRef.current.hasPrev) navigateModal(-1)
      if (e.key === 'ArrowRight' && navRef.current.hasNext) navigateModal(+1)
      if (e.key === 'Escape') setModal(null)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [!!modal])  // only re-register on open/close

  const toggleArchive = async (scan) => {
    const newVal = !scan.archived
    try {
      const res = await fetch(`/api/attendance/${scan.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ archived: newVal }),
      })
      if (!res.ok) throw new Error('Failed')
      setModal(null)
      loadScans(showArchived)
    } catch (e) {
      alert('Failed to archive: ' + e.message)
    }
  }

  const saveEdit = async () => {
    if (!modal) return
    setSaving(true)
    try {
      const res = await fetch(`/api/attendance/${modal.scan.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: editNotes, manual_add: Number(editManualAdd), service_type: editServiceType }),
      })
      if (!res.ok) throw new Error('Save failed')
      const updated = await res.json()
      setScans(prev => prev.map(s => s.id === updated.id ? updated : s))
      setModal(m => ({ ...m, scan: updated }))
    } catch (e) {
      alert('Failed to save: ' + e.message)
    } finally {
      setSaving(false)
    }
  }

  const exportCSV = () => {
    const header = 'ID,Timestamp,Service,AI Count,Adjustment,Total,Notes'
    const rows = filtered.map(s =>
      [s.id, s.timestamp, s.service_type || 'Manual', s.count, s.manual_add || 0, (s.count||0)+(s.manual_add||0), s.notes || ''].join(',')
    )
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url
    a.download = `attendance_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ── Zoom / pan handlers ────────────────────────────────────────────────────
  const handleWheel = (e) => {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.15 : 0.87
    setZoom(z => Math.min(Math.max(z * factor, 1), 10))
  }

  const handleMouseDown = (e) => {
    if (zoom <= 1) return
    setPanning(true)
    lastMouse.current = { x: e.clientX, y: e.clientY }
    e.preventDefault()
  }

  const handleMouseMove = (e) => {
    if (!panning) return
    const dx = e.clientX - lastMouse.current.x
    const dy = e.clientY - lastMouse.current.y
    setPan(p => ({ x: p.x + dx / zoom, y: p.y + dy / zoom }))
    lastMouse.current = { x: e.clientX, y: e.clientY }
  }

  const handleMouseUp = () => setPanning(false)
  const resetZoom     = () => { setZoom(1); setPan({ x: 0, y: 0 }) }

  // ── Adjustment label helper ────────────────────────────────────────────────
  const fmtAdjust = (v) => {
    if (!v) return { label: '—', color: C.muted, weight: 400 }
    if (v > 0) return { label: `+${v}`, color: C.green, weight: 600 }
    return { label: String(v), color: C.red, weight: 600 }
  }

  // ── Arrow button component ────────────────────────────────────────────────
  const NavArrow = ({ dir, disabled, onClick }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      title={dir === -1 ? 'Previous (older)  ←' : 'Next (newer)  →'}
      style={{
        flexShrink: 0,
        width: 44, height: 44,
        borderRadius: '50%',
        border: `1px solid ${disabled ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.25)'}`,
        background: disabled ? 'transparent' : 'rgba(255,255,255,0.07)',
        color: disabled ? 'rgba(255,255,255,0.15)' : '#fff',
        fontSize: 22, display: 'flex', alignItems: 'center', justifyContent: 'center',
        cursor: disabled ? 'default' : 'pointer',
        transition: 'background 0.15s, border-color 0.15s',
        userSelect: 'none',
      }}
      onMouseEnter={e => { if (!disabled) e.currentTarget.style.background = 'rgba(255,255,255,0.15)' }}
      onMouseLeave={e => { e.currentTarget.style.background = disabled ? 'transparent' : 'rgba(255,255,255,0.07)' }}
    >
      {dir === -1 ? '‹' : '›'}
    </button>
  )

  if (loading) return (
    <div style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center', color:C.muted }}>Loading…</div>
  )

  return (
    <div style={{ flex:1, display:'flex', flexDirection:'column', background:C.bg }}>

      {/* Toolbar */}
      <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:'10px 20px', display:'flex', alignItems:'center', gap:12, flexWrap:'wrap' }}>
        <span style={{ fontSize:14, color:C.muted }}>{filtered.length} records</span>
        <div style={{ display:'flex', gap:6, marginLeft:8 }}>
          {services.map(s => (
            <button key={s} onClick={() => setFilter(s)} style={{
              background: filter===s ? C.accent : 'transparent',
              border: `1px solid ${filter===s ? C.accent : C.border}`,
              color: filter===s ? '#fff' : C.muted,
              borderRadius:20, padding:'3px 12px', cursor:'pointer', fontSize:12,
            }}>{s}</button>
          ))}
        </div>
        <button onClick={() => setSortDesc(d => !d)} style={{ background:'transparent', border:`1px solid ${C.border}`, color:C.muted, borderRadius:6, padding:'4px 12px', cursor:'pointer', fontSize:12, marginLeft:'auto' }}>
          {sortDesc ? '↓ Newest first' : '↑ Oldest first'}
        </button>
        <button onClick={() => setShowArchived(v => !v)} style={{
          background: showArchived ? '#78350f' : 'transparent',
          border: `1px solid ${showArchived ? '#b45309' : C.border}`,
          color: showArchived ? '#fde68a' : C.muted,
          borderRadius:6, padding:'4px 12px', cursor:'pointer', fontSize:12,
        }}>
          {showArchived ? '📦 Hide Archived' : '📦 Show Archived'}
        </button>
        <button onClick={openManualEntry} style={{ background:'#16a34a', color:'#fff', border:'none', borderRadius:6, padding:'6px 16px', cursor:'pointer', fontWeight:600, fontSize:13 }}>
          + Manual Entry
        </button>
        <button onClick={exportCSV} style={{ background:C.accent, color:'#fff', border:'none', borderRadius:6, padding:'6px 16px', cursor:'pointer', fontWeight:600, fontSize:13 }}>
          ↓ Export CSV
        </button>
      </div>

      {/* Table */}
      <div style={{ flex:1, overflow:'auto' }}>
        {!filtered.length ? (
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:200, color:C.muted, gap:8 }}>
            <div style={{ fontSize:36 }}>📋</div>
            <div>No data yet — run a scan to record attendance</div>
          </div>
        ) : (
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:14 }}>
            <thead>
              <tr style={{ background:C.surface, position:'sticky', top:0 }}>
                {['Date & Time', 'Service', 'Total', 'AI Count', 'Adjust', 'Notes'].map(h => (
                  <th key={h} style={{
                    padding:'10px 20px',
                    textAlign: ['Total','AI Count','Adjust'].includes(h) ? 'right' : 'left',
                    color:C.muted, fontWeight:600, fontSize:12, textTransform:'uppercase',
                    letterSpacing:'0.05em', borderBottom:`1px solid ${C.border}`,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => {
                const svc    = s.service_type || 'Manual'
                const total  = (s.count || 0) + (s.manual_add || 0)
                const adjust = fmtAdjust(s.manual_add)
                return (
                  <tr key={s.id} onClick={() => openRow(s)}
                    style={{ borderBottom:`1px solid ${C.border}`, background: s.archived ? 'rgba(120,53,15,0.15)' : i%2===0 ? 'transparent' : 'rgba(30,41,59,0.3)', cursor:'pointer', opacity: s.archived ? 0.6 : 1 }}
                    onMouseEnter={e => e.currentTarget.style.background='rgba(59,130,246,0.08)'}
                    onMouseLeave={e => e.currentTarget.style.background= s.archived ? 'rgba(120,53,15,0.15)' : i%2===0 ? 'transparent' : 'rgba(30,41,59,0.3)'}>
                    <td style={{ padding:'10px 20px', color:C.text }}>
                      {new Date(s.timestamp).toLocaleString()}
                      {s.archived && <span style={{ marginLeft:8, fontSize:10, background:'#92400e', color:'#fde68a', borderRadius:4, padding:'1px 6px', fontWeight:700 }}>ARCHIVED</span>}
                    </td>
                    <td style={{ padding:'10px 20px' }}>
                      <span style={{ background:(SVC_COLORS[svc]||'#64748b')+'22', color:SVC_COLORS[svc]||'#64748b', borderRadius:12, padding:'2px 10px', fontSize:12, fontWeight:600 }}>{svc}</span>
                    </td>
                    <td style={{ padding:'10px 20px', color:C.text, fontWeight:700, fontSize:16, textAlign:'right' }}>{total}</td>
                    <td style={{ padding:'10px 20px', color:C.muted, textAlign:'right' }}>{s.count || 0}</td>
                    <td style={{ padding:'10px 20px', color:adjust.color, textAlign:'right', fontWeight:adjust.weight }}>{adjust.label}</td>
                    <td style={{ padding:'10px 20px', color:C.muted, fontSize:13, maxWidth:200, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{s.notes || '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Manual entry modal */}
      {showManualEntry && (
        <div onClick={() => setShowManualEntry(false)} style={{
          position:'fixed', inset:0, background:'rgba(0,0,0,0.85)',
          display:'flex', alignItems:'center', justifyContent:'center',
          zIndex:9999, padding:24,
        }}>
          <div onClick={e => e.stopPropagation()} style={{
            background:C.surface, borderRadius:12, border:`1px solid ${C.border}`,
            width:'min(95vw, 440px)', display:'flex', flexDirection:'column',
          }}>
            <div style={{ padding:'14px 20px', borderBottom:`1px solid ${C.border}`, display:'flex', justifyContent:'space-between', alignItems:'center' }}>
              <span style={{ color:C.text, fontWeight:600, fontSize:15 }}>Manual Entry</span>
              <button onClick={() => setShowManualEntry(false)} style={{ background:'transparent', border:'none', color:C.muted, fontSize:20, cursor:'pointer' }}>✕</button>
            </div>
            <div style={{ padding:20, display:'flex', flexDirection:'column', gap:16 }}>

              <div>
                <label style={{ color:C.muted, fontSize:12, display:'block', marginBottom:6 }}>DATE & TIME</label>
                <input
                  type="datetime-local"
                  value={manualDateTime}
                  onChange={e => setManualDateTime(e.target.value)}
                  style={{ width:'100%', background:'#0f172a', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'8px 10px', fontSize:13, boxSizing:'border-box' }}
                />
              </div>

              <div>
                <label style={{ color:C.muted, fontSize:12, display:'block', marginBottom:6 }}>SERVICE TYPE</label>
                <select
                  value={manualServiceType}
                  onChange={e => setManualServiceType(e.target.value)}
                  style={{ width:'100%', background:'#0f172a', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'8px 10px', fontSize:13, cursor:'pointer' }}
                >
                  {SERVICE_TYPES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>

              <div>
                <label style={{ color:C.muted, fontSize:12, display:'block', marginBottom:6 }}>ATTENDANCE COUNT</label>
                <input
                  type="number"
                  min="0"
                  placeholder="e.g. 342"
                  value={manualCount}
                  onChange={e => setManualCount(e.target.value)}
                  style={{ width:'100%', background:'#0f172a', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'8px 10px', fontSize:16, textAlign:'center', boxSizing:'border-box' }}
                />
              </div>

              <div>
                <label style={{ color:C.muted, fontSize:12, display:'block', marginBottom:6 }}>NOTES <span style={{ color:'#475569' }}>(optional)</span></label>
                <textarea
                  value={manualNotes}
                  onChange={e => setManualNotes(e.target.value)}
                  placeholder="e.g. Easter Sunday, special event…"
                  rows={3}
                  style={{ width:'100%', background:'#0f172a', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'8px 10px', fontSize:13, resize:'vertical', boxSizing:'border-box' }}
                />
              </div>

              <button
                onClick={saveManualEntry}
                disabled={savingManual || !manualCount}
                style={{
                  background: (savingManual || !manualCount) ? C.border : '#16a34a',
                  color:'#fff', border:'none', borderRadius:8, padding:'10px 0',
                  fontWeight:600, fontSize:14, cursor: (savingManual || !manualCount) ? 'not-allowed' : 'pointer',
                }}>
                {savingManual ? 'Saving…' : '+ Add Entry'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Row modal */}
      {modal && (
        <div onClick={() => setModal(null)} style={{
          position:'fixed', inset:0, background:'rgba(0,0,0,0.85)',
          display:'flex', alignItems:'center', justifyContent:'center',
          zIndex:9999, padding:'24px 16px', gap:12,
        }}>

          {/* Left arrow (older) */}
          <NavArrow dir={-1} disabled={!hasPrev} onClick={e => { e.stopPropagation(); navigateModal(-1) }} />

          {/* Modal card */}
          <div onClick={e => e.stopPropagation()} style={{
            background:C.surface, borderRadius:12, border:`1px solid ${C.border}`,
            width:'min(95vw, 1100px)', maxHeight:'90vh', overflow:'hidden',
            display:'flex', flexDirection:'column', flex:'0 1 auto',
          }}>

            {/* Modal header */}
            <div style={{ padding:'12px 20px', borderBottom:`1px solid ${C.border}`, display:'flex', justifyContent:'space-between', alignItems:'center', gap:8 }}>
              <div style={{ display:'flex', alignItems:'center', gap:12, minWidth:0 }}>
                <span style={{ color:C.text, fontWeight:600, fontSize:14, whiteSpace:'nowrap' }}>
                  {new Date(modal.scan.timestamp).toLocaleString()}
                </span>
                <span style={{ color:C.muted, fontSize:13, whiteSpace:'nowrap' }}>
                  {modal.scan.service_type || 'Manual'}
                </span>
              </div>
              <div style={{ display:'flex', alignItems:'center', gap:12, flexShrink:0 }}>
                {/* Position counter */}
                <span style={{ color:C.muted, fontSize:12, whiteSpace:'nowrap' }}>
                  {modalIdx + 1} / {sortedFiltered.length}
                </span>
                <button onClick={() => setModal(null)} style={{ background:'transparent', border:'none', color:C.muted, fontSize:20, cursor:'pointer', lineHeight:1 }}>✕</button>
              </div>
            </div>

            <div style={{ display:'flex', flex:1, overflow:'hidden', minHeight:0 }}>

              {/* Left — zoomable image */}
              <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
                <div
                  onWheel={handleWheel}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                  style={{
                    flex:1, overflow:'hidden', padding:16,
                    display:'flex', alignItems:'flex-start', justifyContent:'center',
                    cursor: zoom > 1 ? (panning ? 'grabbing' : 'grab') : 'zoom-in',
                    userSelect:'none',
                  }}>
                  {modal.imageLoading ? (
                    <div style={{ color:C.muted, alignSelf:'center' }}>Loading image…</div>
                  ) : modal.imageB64 ? (
                    <img
                      src={`data:image/jpeg;base64,${modal.imageB64}`}
                      alt="Scan"
                      draggable={false}
                      style={{
                        maxWidth:'100%', borderRadius:8, display:'block',
                        transform:`scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                        transformOrigin:'top center',
                        transition: panning ? 'none' : 'transform 0.1s ease',
                      }}
                    />
                  ) : (
                    <div style={{ color:C.muted, alignSelf:'center' }}>No image saved for this scan</div>
                  )}
                </div>

                {/* Zoom controls */}
                {modal.imageB64 && (
                  <div style={{ padding:'8px 16px', borderTop:`1px solid ${C.border}`, display:'flex', alignItems:'center', gap:10, background:C.bg, flexShrink:0 }}>
                    <button onClick={() => setZoom(z => Math.max(z * 0.8, 1))}
                      style={{ background:C.border, border:'none', color:C.text, borderRadius:6, width:28, height:28, cursor:'pointer', fontSize:16, lineHeight:'28px' }}>−</button>
                    <span style={{ color:C.muted, fontSize:12, minWidth:44, textAlign:'center' }}>{Math.round(zoom * 100)}%</span>
                    <button onClick={() => setZoom(z => Math.min(z * 1.25, 10))}
                      style={{ background:C.border, border:'none', color:C.text, borderRadius:6, width:28, height:28, cursor:'pointer', fontSize:16, lineHeight:'28px' }}>+</button>
                    <button onClick={resetZoom}
                      style={{ background:'transparent', border:`1px solid ${C.border}`, color:C.muted, borderRadius:6, padding:'3px 10px', cursor:'pointer', fontSize:11 }}>Reset</button>
                    <span style={{ color:C.muted, fontSize:11, marginLeft:4 }}>Scroll to zoom · Drag to pan</span>
                  </div>
                )}
              </div>

              {/* Right — edit panel */}
              <div style={{ width:260, borderLeft:`1px solid ${C.border}`, padding:20, display:'flex', flexDirection:'column', gap:20, overflowY:'auto' }}>

                {/* Count summary */}
                <div style={{ background:'rgba(59,130,246,0.08)', borderRadius:8, padding:12, border:`1px solid ${C.border}` }}>
                  <div style={{ color:C.muted, fontSize:11, marginBottom:8 }}>COUNT SUMMARY</div>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
                    <span style={{ color:C.muted, fontSize:13 }}>AI detected</span>
                    <span style={{ color:C.text, fontWeight:600 }}>{modal.scan.count || 0}</span>
                  </div>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:8 }}>
                    <span style={{ color:C.muted, fontSize:13 }}>Adjustment</span>
                    <span style={{ color: Number(editManualAdd) < 0 ? C.red : C.green, fontWeight:600 }}>
                      {Number(editManualAdd) > 0 ? `+${editManualAdd}` : (editManualAdd || 0)}
                    </span>
                  </div>
                  <div style={{ borderTop:`1px solid ${C.border}`, paddingTop:8, display:'flex', justifyContent:'space-between' }}>
                    <span style={{ color:C.text, fontSize:14, fontWeight:600 }}>Total</span>
                    <span style={{ color:C.green, fontSize:20, fontWeight:700 }}>{(modal.scan.count || 0) + (Number(editManualAdd) || 0)}</span>
                  </div>
                </div>

                {/* Service type */}
                <div>
                  <label style={{ color:C.muted, fontSize:12, display:'block', marginBottom:6 }}>SERVICE TYPE</label>
                  <select
                    value={editServiceType}
                    onChange={e => setEditServiceType(e.target.value)}
                    style={{ width:'100%', background:'#0f172a', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'8px 10px', fontSize:13, cursor:'pointer' }}
                  >
                    {SERVICE_TYPES.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                {/* Adjustment input — allows negative */}
                <div>
                  <label style={{ color:C.muted, fontSize:12, display:'block', marginBottom:4 }}>ADJUSTMENT</label>
                  <div style={{ color:C.muted, fontSize:11, marginBottom:8 }}>Positive: missed people · Negative: fix duplicates</div>
                  <div style={{ display:'flex', gap:8, alignItems:'center' }}>
                    <button onClick={() => setEditManualAdd(v => Number(v) - 1)}
                      style={{ background:C.border, border:'none', color:C.text, borderRadius:6, width:32, height:32, cursor:'pointer', fontSize:16 }}>−</button>
                    <input
                      type="number"
                      value={editManualAdd}
                      onChange={e => setEditManualAdd(Number(e.target.value))}
                      style={{ flex:1, background:'#0f172a', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'6px 10px', fontSize:16, textAlign:'center' }}
                    />
                    <button onClick={() => setEditManualAdd(v => Number(v) + 1)}
                      style={{ background:C.border, border:'none', color:C.text, borderRadius:6, width:32, height:32, cursor:'pointer', fontSize:16 }}>+</button>
                  </div>
                </div>

                {/* Notes */}
                <div>
                  <label style={{ color:C.muted, fontSize:12, display:'block', marginBottom:6 }}>NOTES</label>
                  <textarea
                    value={editNotes}
                    onChange={e => setEditNotes(e.target.value)}
                    placeholder="Add notes about this service…"
                    rows={4}
                    style={{ width:'100%', background:'#0f172a', border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'8px 10px', fontSize:13, resize:'vertical', boxSizing:'border-box' }}
                  />
                </div>

                {/* Save */}
                <button onClick={saveEdit} disabled={saving}
                  style={{ background: saving ? C.border : C.accent, color:'#fff', border:'none', borderRadius:8, padding:'10px 0', fontWeight:600, fontSize:14, cursor: saving ? 'not-allowed' : 'pointer' }}>
                  {saving ? 'Saving…' : '💾 Save Changes'}
                </button>

                {/* Archive / Unarchive */}
                <button onClick={() => toggleArchive(modal.scan)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${modal.scan.archived ? '#16a34a' : '#b45309'}`,
                    color: modal.scan.archived ? C.green : '#f59e0b',
                    borderRadius:8, padding:'8px 0', fontWeight:600, fontSize:13, cursor:'pointer',
                  }}>
                  {modal.scan.archived ? '📦 Unarchive' : '📦 Archive'}
                </button>
              </div>
            </div>
          </div>

          {/* Right arrow (newer) */}
          <NavArrow dir={+1} disabled={!hasNext} onClick={e => { e.stopPropagation(); navigateModal(+1) }} />

        </div>
      )}
    </div>
  )
}
