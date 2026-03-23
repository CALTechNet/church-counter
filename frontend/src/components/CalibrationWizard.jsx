/**
 * Calibration Wizard
 * Step 1: Capture a live photo from the camera (or use latest scan image)
 * Step 2: User clicks a seat in the PHOTO, then clicks the same seat in the SVG
 * Step 3: Repeat for at least 4 seats (more = better homography accuracy)
 * Step 4: Save all points
 */
import { useState, useEffect, useRef } from 'react'
import { getCapture, saveCalPoint, getCalibration, clearCalibration, ptzCommand } from '../api.js'

const C = { bg:'#0f172a', surface:'#1e293b', border:'#334155', text:'#f1f5f9', muted:'#94a3b8', accent:'#3b82f6', green:'#22c55e', red:'#ef4444', yellow:'#eab308' }

const STEPS = ['Capture Photo', 'Mark Calibration Points', 'Review & Save']

export default function CalibrationWizard({ onClose }) {
  const [step, setStep]           = useState(0)
  const [photoB64, setPhotoB64]   = useState(null)
  const [svgContent, setSvgContent] = useState(null)
  const [points, setPoints]       = useState([])   // [{seat_id, photo_x, photo_y, svg_x, svg_y}]
  const [phase, setPhase]         = useState('photo')  // 'photo' | 'svg'
  const [pending, setPending]     = useState(null) // partial point while picking
  const [existingCal, setExisting] = useState({})
  const [loading, setLoading]     = useState(false)
  const [status, setStatus]       = useState('')
  const [seatIdInput, setSeatIdInput] = useState('')
  const photoRef = useRef()
  const svgRef   = useRef()

  useEffect(() => {
    fetch('/api/svg').then(r=>r.text()).then(setSvgContent)
    getCalibration().then(setExisting)
  }, [])

  const capture = async () => {
    setLoading(true); setStatus('Capturing from camera…')
    try {
      const d = await getCapture()
      setPhotoB64(d.image_b64)
      setStatus('Capture successful!')
      setTimeout(() => setStep(1), 800)
    } catch(e) {
      setStatus(`Error: ${e.message}`)
    } finally { setLoading(false) }
  }

  const onPhotoClick = e => {
    if (phase !== 'photo') return
    const rect = e.currentTarget.getBoundingClientRect()
    const img = e.currentTarget
    const scaleX = img.naturalWidth / img.width
    const scaleY = img.naturalHeight / img.height
    const px = (e.clientX - rect.left) * scaleX
    const py = (e.clientY - rect.top)  * scaleY
    setPending({ photo_x: Math.round(px), photo_y: Math.round(py) })
    setPhase('svg')
    setStatus(`Photo point set at (${Math.round(px)}, ${Math.round(py)}). Now click the same seat in the SVG map.`)
  }

  const onSvgClick = e => {
    if (phase !== 'svg' || !pending) return
    // Get click in SVG coordinate space
    const svgEl = svgRef.current?.querySelector('svg')
    if (!svgEl) return
    const pt = svgEl.createSVGPoint()
    pt.x = e.clientX; pt.y = e.clientY
    const svgP = pt.matrixTransform(svgEl.getScreenCTM().inverse())

    // Check if a seat element was clicked
    const target = e.target
    const id = target.id || target.closest('[id]')?.id || ''

    const seatId = id.startsWith('seat_') ? id : (seatIdInput.trim() || `cal_${points.length + 1}`)
    const newPt = { seat_id: seatId, ...pending, svg_x: Math.round(svgP.x), svg_y: Math.round(svgP.y) }
    setPoints(p => [...p.filter(x=>x.seat_id!==seatId), newPt])
    setPending(null)
    setPhase('photo')
    setSeatIdInput('')
    setStatus(`✓ Point "${seatId}" saved. Click another seat in the photo, or proceed to Review.`)
  }

  const save = async () => {
    if (points.length < 4) { setStatus('Need at least 4 points for accurate mapping.'); return }
    setLoading(true); setStatus('Saving calibration…')
    try {
      for (const pt of points) await saveCalPoint(pt)
      setStatus(`Saved ${points.length} calibration points!`)
      setTimeout(onClose, 1000)
    } catch(e) { setStatus(`Error: ${e.message}`) }
    finally { setLoading(false) }
  }

  const clearAll = async () => {
    await clearCalibration()
    setPoints([])
    setExisting({})
    setStatus('Calibration cleared.')
  }

  return (
    <div style={{ position:'fixed', inset:0, background:'rgba(0,0,0,0.75)', zIndex:1000, display:'flex', alignItems:'center', justifyContent:'center', padding:20 }}>
      <div style={{ background:C.bg, border:`1px solid ${C.border}`, borderRadius:16, width:'95vw', maxWidth:1200, maxHeight:'95vh', display:'flex', flexDirection:'column', overflow:'hidden' }}>

        {/* Header */}
        <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:'16px 24px', display:'flex', alignItems:'center', gap:16 }}>
          <span style={{ fontSize:18, fontWeight:700 }}>⚙ Calibration Wizard</span>
          <span style={{ fontSize:13, color:C.muted }}>Link photo coordinates → SVG seat positions</span>
          {/* Step indicators */}
          <div style={{ marginLeft:'auto', display:'flex', gap:8 }}>
            {STEPS.map((s,i)=>(
              <div key={s} onClick={()=>i<=step&&setStep(i)} style={{
                padding:'4px 14px', borderRadius:20, fontSize:12, fontWeight:600, cursor: i<=step?'pointer':'default',
                background: step===i?C.accent:step>i?C.green+'33':'transparent',
                border:`1px solid ${step===i?C.accent:step>i?C.green:C.border}`,
                color: step===i?'#fff':step>i?C.green:C.muted,
              }}>{i+1}. {s}</div>
            ))}
          </div>
          <button onClick={onClose} style={{ background:'transparent', border:`1px solid ${C.border}`, color:C.muted, borderRadius:8, padding:'6px 14px', cursor:'pointer' }}>✕ Close</button>
        </div>

        {/* Status bar */}
        {status && (
          <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:'8px 24px', fontSize:13, color:C.yellow }}>{status}</div>
        )}

        {/* Step 0: Capture */}
        {step === 0 && (
          <div style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:20, padding:40 }}>
            <div style={{ fontSize:48 }}>📷</div>
            <div style={{ fontSize:18, fontWeight:600 }}>Capture a Live Photo</div>
            <div style={{ fontSize:14, color:C.muted, textAlign:'center', maxWidth:480 }}>
              This will grab a frame from the camera. Position the camera so the full seating area is visible before capturing.
            </div>
            {/* PTZ nudge controls */}
            <div style={{ display:'grid', gridTemplateColumns:'repeat(3,44px)', gap:4 }}>
              {[['▲','up'],['',''],['▼','down']].map(([lbl,act])=>
                act ? <button key={lbl+act} onMouseDown={()=>ptzCommand(act,8)} onMouseUp={()=>ptzCommand('stop')} style={ptzBtn}>{lbl}</button> : <div key="empty"/>
              )}
              {[['◀','left'],['■','stop'],['▶','right']].map(([lbl,act])=>
                <button key={lbl+act} onMouseDown={()=>ptzCommand(act,8)} onMouseUp={()=>act!=='stop'&&ptzCommand('stop')} style={ptzBtn}>{lbl}</button>
              )}
            </div>
            <div style={{ display:'flex', gap:10 }}>
              <button onClick={()=>ptzCommand('zoomin',4)} onMouseUp={()=>ptzCommand('zoomstop')} style={{...ptzBtn,width:'auto',padding:'8px 16px'}}>🔍＋</button>
              <button onClick={()=>ptzCommand('zoomout',4)} onMouseUp={()=>ptzCommand('zoomstop')} style={{...ptzBtn,width:'auto',padding:'8px 16px'}}>🔍－</button>
            </div>
            <button onClick={capture} disabled={loading} style={{ background:C.accent, color:'#fff', border:'none', borderRadius:10, padding:'12px 32px', fontSize:16, fontWeight:700, cursor:'pointer' }}>
              {loading ? 'Capturing…' : '📷 Capture Photo'}
            </button>
            {Object.keys(existingCal).length > 0 && (
              <div style={{ fontSize:13, color:C.muted }}>
                Existing calibration: {Object.keys(existingCal).length} points ·{' '}
                <span style={{ color:C.red, cursor:'pointer', textDecoration:'underline' }} onClick={clearAll}>Clear all</span>
              </div>
            )}
          </div>
        )}

        {/* Step 1: Mark points */}
        {step === 1 && (
          <div style={{ flex:1, display:'flex', overflow:'hidden' }}>
            {/* Photo panel */}
            <div style={{ flex:1, display:'flex', flexDirection:'column', borderRight:`1px solid ${C.border}`, overflow:'hidden' }}>
              <div style={{ padding:'8px 16px', background:C.surface, borderBottom:`1px solid ${C.border}`, fontSize:13, display:'flex', alignItems:'center', gap:12 }}>
                <span style={{ color: phase==='photo'?C.yellow:C.muted, fontWeight: phase==='photo'?700:400 }}>
                  {phase==='photo' ? '👆 Click a seat in this photo' : '✓ Photo point set'}
                </span>
                <span style={{ marginLeft:'auto', color:C.muted }}>{points.length} points collected</span>
              </div>
              <div style={{ flex:1, overflow:'auto', position:'relative' }}>
                <img
                  src={`data:image/jpeg;base64,${photoB64}`}
                  style={{ display:'block', maxWidth:'100%', cursor: phase==='photo'?'crosshair':'default' }}
                  onClick={onPhotoClick}
                  alt="Camera capture"
                />
                {/* Show marked points */}
                {points.map((pt, i) => (
                  <div key={pt.seat_id} style={{
                    position:'absolute',
                    left: `${pt.photo_x / (photoRef.current?.naturalWidth || 1920) * 100}%`,
                    top:  `${pt.photo_y / (photoRef.current?.naturalHeight || 1080) * 100}%`,
                    transform:'translate(-50%,-50%)',
                    width:16, height:16, borderRadius:'50%', background:C.green, border:'2px solid #fff',
                    display:'flex', alignItems:'center', justifyContent:'center', fontSize:10, color:'#fff', fontWeight:700,
                    pointerEvents:'none',
                  }}>{i+1}</div>
                ))}
                {pending && (
                  <div style={{ position:'absolute', bottom:12, left:12, background:'rgba(234,179,8,0.9)', color:'#000', borderRadius:8, padding:'6px 12px', fontSize:13, fontWeight:600 }}>
                    Click matching seat in SVG →
                  </div>
                )}
              </div>
            </div>

            {/* SVG panel */}
            <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
              <div style={{ padding:'8px 16px', background:C.surface, borderBottom:`1px solid ${C.border}`, fontSize:13, display:'flex', alignItems:'center', gap:12 }}>
                <span style={{ color: phase==='svg'?C.yellow:C.muted, fontWeight: phase==='svg'?700:400 }}>
                  {phase==='svg' ? '👆 Click the matching seat in this map' : 'SVG seat map'}
                </span>
                {phase==='svg' && (
                  <input
                    placeholder="Seat ID (optional, auto from click)"
                    value={seatIdInput}
                    onChange={e=>setSeatIdInput(e.target.value)}
                    style={{ marginLeft:'auto', background:C.bg, border:`1px solid ${C.border}`, color:C.text, borderRadius:6, padding:'3px 10px', fontSize:12, width:200 }}
                  />
                )}
              </div>
              <div ref={svgRef} style={{ flex:1, overflow:'auto', cursor: phase==='svg'?'crosshair':'default' }}
                onClick={onSvgClick}
                dangerouslySetInnerHTML={{ __html: svgContent }}
              />
            </div>
          </div>
        )}

        {/* Step 2: Review */}
        {step === 2 && (
          <div style={{ flex:1, display:'flex', flexDirection:'column', padding:24, gap:16, overflow:'auto' }}>
            <div style={{ fontSize:16, fontWeight:600 }}>Review Calibration Points ({points.length})</div>
            {points.length < 4 && (
              <div style={{ background:'rgba(239,68,68,0.15)', border:`1px solid ${C.red}`, borderRadius:8, padding:'10px 16px', fontSize:13, color:C.red }}>
                ⚠ Need at least 4 points for homography. Go back and add more.
              </div>
            )}
            <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(200px,1fr))', gap:10 }}>
              {points.map((pt,i) => (
                <div key={pt.seat_id} style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, padding:'10px 14px', fontSize:12 }}>
                  <div style={{ color:C.green, fontWeight:700, marginBottom:4 }}>#{i+1} {pt.seat_id}</div>
                  <div style={{ color:C.muted }}>Photo: ({pt.photo_x}, {pt.photo_y})</div>
                  <div style={{ color:C.muted }}>SVG:   ({pt.svg_x}, {pt.svg_y})</div>
                  <button onClick={()=>setPoints(p=>p.filter(x=>x.seat_id!==pt.seat_id))}
                    style={{ marginTop:6, background:'transparent', border:`1px solid ${C.red}`, color:C.red, borderRadius:4, padding:'2px 8px', cursor:'pointer', fontSize:11 }}>Remove</button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ background:C.surface, borderTop:`1px solid ${C.border}`, padding:'12px 24px', display:'flex', gap:12, justifyContent:'flex-end' }}>
          {step > 0 && <button onClick={()=>setStep(s=>s-1)} style={secBtn}>← Back</button>}
          {step === 1 && points.length >= 4 && <button onClick={()=>setStep(2)} style={primBtn}>Review {points.length} Points →</button>}
          {step === 1 && points.length < 4 && <span style={{ color:C.muted, fontSize:13, alignSelf:'center' }}>Add {4-points.length} more point{4-points.length!==1?'s':''} to continue</span>}
          {step === 2 && <button onClick={save} disabled={loading||points.length<4} style={{ ...primBtn, background: loading?C.border:C.green }}>{loading?'Saving…':'✓ Save Calibration'}</button>}
        </div>
      </div>
    </div>
  )
}

const ptzBtn = { width:44, height:44, background:'#1e293b', border:'1px solid #334155', color:'#f1f5f9', borderRadius:8, cursor:'pointer', fontSize:18, fontWeight:700 }
const primBtn = { background:C.accent, color:'#fff', border:'none', borderRadius:8, padding:'8px 20px', cursor:'pointer', fontWeight:600, fontSize:14 }
const secBtn  = { background:'transparent', border:'1px solid #334155', color:'#94a3b8', borderRadius:8, padding:'8px 20px', cursor:'pointer', fontSize:14 }
