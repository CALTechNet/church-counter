import { useState, useEffect, useRef } from 'react'

const C = { bg: '#0f172a', muted: '#94a3b8', occupied: '#22c55e', empty: '#ef4444', pending: '#4b5563' }

export default function SeatMap({ seatStates, scanning }) {
  const [svgContent, setSvgContent] = useState(null)
  const [zoom, setZoom]             = useState(1)
  const [pan, setPan]               = useState({ x: 0, y: 0 })
  const [dragging, setDragging]     = useState(false)
  const dragStart = useRef(null)

  useEffect(() => {
    fetch('/api/svg')
      .then(r => r.text())
      .then(setSvgContent)
      .catch(() => {})
  }, [])

  const coloredSvg = applyColors(svgContent, seatStates)

  const onWheel = e => {
    e.preventDefault()
    setZoom(z => Math.max(0.25, Math.min(6, z * (e.deltaY < 0 ? 1.12 : 0.9))))
  }
  const onMouseDown = e => {
    setDragging(true)
    dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }
  }
  const onMouseMove = e => {
    if (!dragging) return
    setPan({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y })
  }
  const onMouseUp = () => setDragging(false)

  if (!svgContent) return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted, fontSize: 16 }}>
      Loading seat map…
    </div>
  )

  return (
    <div
      style={{ flex: 1, position: 'relative', overflow: 'hidden', background: C.bg, cursor: dragging ? 'grabbing' : 'grab', userSelect: 'none' }}
      onWheel={onWheel} onMouseDown={onMouseDown} onMouseMove={onMouseMove}
      onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
    >
      <div
        style={{ transform: `translate(calc(-50% + ${pan.x}px), calc(-50% + ${pan.y}px)) scale(${zoom})`, position: 'absolute', top: '50%', left: '50%', transition: dragging ? 'none' : 'none' }}
        dangerouslySetInnerHTML={{ __html: coloredSvg }}
      />

      {/* Zoom controls */}
      <div style={{ position: 'absolute', bottom: 20, right: 20, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {[['＋', () => setZoom(z => Math.min(6, z * 1.25))],
          ['⟳', () => { setZoom(1); setPan({ x: 0, y: 0 }) }],
          ['－', () => setZoom(z => Math.max(0.25, z * 0.8))]].map(([lbl, fn]) => (
          <button key={lbl} onClick={fn} style={{ width:36, height:36, background:'rgba(30,41,59,0.95)', border:'1px solid #334155', color:'#f1f5f9', borderRadius:8, cursor:'pointer', fontSize:16, fontWeight:600 }}>{lbl}</button>
        ))}
      </div>

      {/* Legend */}
      <div style={{ position: 'absolute', bottom: 20, left: 20, background: 'rgba(15,23,42,0.92)', borderRadius: 10, padding: '10px 16px', border: '1px solid #334155' }}>
        {[[C.occupied,'Occupied'],[C.empty,'Empty'],[C.pending,'Not scanned']].map(([color, label]) => (
          <div key={label} style={{ display:'flex', alignItems:'center', gap:8, marginBottom:4, fontSize:13 }}>
            <div style={{ width:12, height:12, borderRadius:'50%', background:color, flexShrink:0 }} />
            <span style={{ color:'#94a3b8' }}>{label}</span>
          </div>
        ))}
      </div>

      {scanning && (
        <div style={{ position:'absolute', inset:0, background:'rgba(15,23,42,0.55)', display:'flex', alignItems:'center', justifyContent:'center', pointerEvents:'none' }}>
          <div style={{ fontSize:18, color:'#fff', fontWeight:700, background:'rgba(30,41,59,0.9)', padding:'12px 24px', borderRadius:12 }}>⏳ Scanning in progress…</div>
        </div>
      )}
    </div>
  )
}

function applyColors(svgContent, seatStates) {
  if (!svgContent) return null
  if (!seatStates || !Object.keys(seatStates).length) return svgContent
  let result = svgContent
  for (const [id, state] of Object.entries(seatStates)) {
    // Try to swap existing class
    const withClass = result.replace(
      new RegExp(`(id="${escRe(id)}"[^/]*?class=")[^"]*(")`,'g'),
      `$1seat ${state}$2`
    )
    if (withClass !== result) { result = withClass; continue }
    // Add class attribute
    result = result.replace(
      new RegExp(`(id="${escRe(id)}"[^/]*?)(/?>)`,'g'),
      `$1 class="seat ${state}" $2`
    )
  }
  return result
}

function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') }
