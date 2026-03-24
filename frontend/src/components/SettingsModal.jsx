import { useState, useEffect } from 'react'
import { getSettings, saveSettings, getCameraBounds } from '../api.js'

const C = {
  bg:      '#0f172a',
  surface: '#1e293b',
  border:  '#334155',
  text:    '#f1f5f9',
  muted:   '#94a3b8',
  accent:  '#3b82f6',
  green:   '#22c55e',
  red:     '#ef4444',
}

const label  = { fontSize: 12, color: C.muted, marginBottom: 4, display: 'block' }
const inputStyle  = { width: '100%', background: C.bg, border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '8px 10px', fontSize: 13, boxSizing: 'border-box', outline: 'none' }
const section      = { marginBottom: 22 }
const sectionTitle = { fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted, marginBottom: 12, paddingBottom: 6, borderBottom: `1px solid ${C.border}` }
const field        = { marginBottom: 12 }

let _nextId = 1
function genId() { return `room_${Date.now()}_${_nextId++}` }

function defaultRoom() {
  return {
    id: genId(),
    name: '',
    camera_type: 'ptz_optics',
    camera_ip: '',
    camera_user: 'admin',
    camera_pass: 'admin',
    scan_mode: 'preset',
    preset_start: 100,
    preset_end: 131,
    rtsp_url: '',
  }
}

export default function SettingsModal({ onClose, onSave }) {
  const [cfg, setCfg] = useState({
    church_name: 'Lakeshore Church',
    rooms: [],
  })
  const [saving, setSaving]   = useState(false)
  const [msg, setMsg]         = useState(null)
  const [bounds, setBounds]   = useState(null)
  const [expandedRoom, setExpandedRoom] = useState(null)

  useEffect(() => {
    getSettings()
      .then(s => {
        const rooms = s.rooms && s.rooms.length > 0
          ? s.rooms
          : [{
              id: 'default',
              name: 'Sanctuary',
              camera_type: 'ptz_optics',
              camera_ip: s.camera_ip || '10.10.140.140',
              camera_user: s.camera_user || 'admin',
              camera_pass: s.camera_pass || 'admin',
              scan_mode: s.scan_mode || 'preset',
              preset_start: s.preset_start ?? 100,
              preset_end: s.preset_end ?? 131,
              rtsp_url: '',
            }]
        setCfg({ church_name: s.church_name || 'Lakeshore Church', rooms })
        if (rooms.length > 0) setExpandedRoom(rooms[0].id)
      })
      .catch(() => {})
    getCameraBounds()
      .then(b => setBounds(b))
      .catch(() => {})
  }, [])

  const set = (key, val) => setCfg(prev => ({ ...prev, [key]: val }))

  const updateRoom = (roomId, key, val) => {
    setCfg(prev => ({
      ...prev,
      rooms: prev.rooms.map(r => r.id === roomId ? { ...r, [key]: val } : r),
    }))
  }

  const addRoom = () => {
    const newRoom = defaultRoom()
    setCfg(prev => ({ ...prev, rooms: [...prev.rooms, newRoom] }))
    setExpandedRoom(newRoom.id)
  }

  const removeRoom = (roomId) => {
    setCfg(prev => {
      const rooms = prev.rooms.filter(r => r.id !== roomId)
      return { ...prev, rooms }
    })
    if (expandedRoom === roomId) {
      setCfg(prev => {
        if (prev.rooms.length > 0) setExpandedRoom(prev.rooms[0].id)
        return prev
      })
    }
  }

  const handleSave = async () => {
    setSaving(true)
    setMsg(null)
    try {
      // Also keep flat camera fields synced from first room for backward compat
      const firstRoom = cfg.rooms[0]
      const payload = {
        church_name: cfg.church_name,
        rooms: cfg.rooms,
      }
      if (firstRoom && firstRoom.camera_type === 'ptz_optics') {
        payload.camera_ip = firstRoom.camera_ip
        payload.camera_user = firstRoom.camera_user
        payload.camera_pass = firstRoom.camera_pass
        payload.scan_mode = firstRoom.scan_mode
        payload.preset_start = firstRoom.preset_start
        payload.preset_end = firstRoom.preset_end
      }
      await saveSettings(payload)
      setMsg({ text: 'Settings saved!', ok: true })
      setTimeout(() => { onSave?.(payload); onClose() }, 1000)
    } catch (e) {
      setMsg({ text: `Error: ${e.message}`, ok: false })
    } finally {
      setSaving(false)
    }
  }

  const calGrid = (() => {
    const tl = bounds?.top_left
    const br = bounds?.bottom_right
    if (!tl || !br) return null
    const zoom = Math.max(1, bounds?.zoom || tl?.zoom || 10000)
    const step = Math.max(25, Math.floor(1250000 / zoom) - 50)
    const cols = Math.max(1, Math.ceil(Math.abs(br.pan - tl.pan) / step) + 1)
    const rows = Math.max(1, Math.ceil(Math.abs(br.tilt - tl.tilt) / step) + 1)
    return { cols, rows, total: cols * rows, zoom, step }
  })()

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
    }}>
      <div style={{
        background: C.surface, borderRadius: 12, border: `1px solid ${C.border}`,
        width: '100%', maxWidth: 520, maxHeight: '90vh', overflowY: 'auto',
        padding: 24, boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
      }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>Settings</h2>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: C.muted, fontSize: 20, cursor: 'pointer', padding: '0 4px', lineHeight: 1 }}>✕</button>
        </div>

        {/* Church */}
        <div style={section}>
          <div style={sectionTitle}>Church</div>
          <div style={field}>
            <label style={label}>Church Name</label>
            <input style={inputStyle} value={cfg.church_name} onChange={e => set('church_name', e.target.value)} />
          </div>
        </div>

        {/* Rooms / Cameras */}
        <div style={section}>
          <div style={{ ...sectionTitle, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Rooms & Cameras</span>
            <button
              onClick={addRoom}
              title="Add room"
              style={{
                background: C.accent, color: '#fff', border: 'none', borderRadius: 6,
                width: 26, height: 26, fontSize: 16, fontWeight: 700, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', lineHeight: 1,
              }}
            >+</button>
          </div>

          {cfg.rooms.map((room, idx) => {
            const isExpanded = expandedRoom === room.id
            const presetCount = Math.max(0, (room.preset_end ?? 131) - (room.preset_start ?? 100) + 1)

            return (
              <div key={room.id} style={{
                background: C.bg, borderRadius: 8, border: `1px solid ${C.border}`,
                marginBottom: 10, overflow: 'hidden',
              }}>
                {/* Room header — click to expand/collapse */}
                <div
                  onClick={() => setExpandedRoom(isExpanded ? null : room.id)}
                  style={{
                    padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 10,
                    cursor: 'pointer', userSelect: 'none',
                  }}
                >
                  <span style={{ fontSize: 12, color: C.muted, transform: isExpanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>▶</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: C.text, flex: 1 }}>
                    {room.name || `Room ${idx + 1}`}
                  </span>
                  <span style={{ fontSize: 11, color: C.muted, background: C.surface, borderRadius: 4, padding: '2px 8px' }}>
                    {room.camera_type === 'rtsp' ? 'RTSP' : 'PTZ Optics'}
                  </span>
                  {cfg.rooms.length > 1 && (
                    <button
                      onClick={e => { e.stopPropagation(); removeRoom(room.id) }}
                      title="Remove room"
                      style={{
                        background: 'transparent', border: 'none', color: C.red,
                        fontSize: 14, cursor: 'pointer', padding: '0 4px', lineHeight: 1,
                      }}
                    >✕</button>
                  )}
                </div>

                {/* Room details — expanded */}
                {isExpanded && (
                  <div style={{ padding: '0 14px 14px' }}>
                    {/* Room name */}
                    <div style={field}>
                      <label style={label}>Room Name</label>
                      <input
                        style={inputStyle}
                        value={room.name}
                        onChange={e => updateRoom(room.id, 'name', e.target.value)}
                        placeholder="e.g. Sanctuary, Fellowship Hall"
                      />
                    </div>

                    {/* Camera type */}
                    <div style={field}>
                      <label style={label}>Camera Type</label>
                      <select
                        style={{ ...inputStyle, cursor: 'pointer' }}
                        value={room.camera_type}
                        onChange={e => updateRoom(room.id, 'camera_type', e.target.value)}
                      >
                        <option value="ptz_optics">PTZ Optics</option>
                        <option value="rtsp">Generic RTSP</option>
                      </select>
                    </div>

                    {/* PTZ Optics fields */}
                    {room.camera_type === 'ptz_optics' && (
                      <>
                        <div style={field}>
                          <label style={label}>IP Address</label>
                          <input
                            style={inputStyle}
                            value={room.camera_ip}
                            onChange={e => updateRoom(room.id, 'camera_ip', e.target.value)}
                            placeholder="10.10.140.140"
                          />
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                          <div style={field}>
                            <label style={label}>Username</label>
                            <input style={inputStyle} value={room.camera_user} onChange={e => updateRoom(room.id, 'camera_user', e.target.value)} />
                          </div>
                          <div style={field}>
                            <label style={label}>Password</label>
                            <input type="password" style={inputStyle} value={room.camera_pass} onChange={e => updateRoom(room.id, 'camera_pass', e.target.value)} />
                          </div>
                        </div>

                        {/* Scan mode */}
                        <div style={field}>
                          <label style={label}>Scan Mode</label>
                          <select
                            style={{ ...inputStyle, cursor: 'pointer' }}
                            value={room.scan_mode}
                            onChange={e => updateRoom(room.id, 'scan_mode', e.target.value)}
                          >
                            <option value="preset">Preset Scanning</option>
                            <option value="calibrated">Calibrated Scanning</option>
                          </select>
                        </div>

                        {room.scan_mode === 'preset' && (
                          <div style={{ background: C.surface, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
                            <p style={{ margin: '0 0 10px', fontSize: 11, color: C.muted }}>
                              Visits saved camera presets in sequence.
                            </p>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                              <div>
                                <label style={label}>First Preset #</label>
                                <input type="number" style={inputStyle} value={room.preset_start} min={0} max={255}
                                  onChange={e => updateRoom(room.id, 'preset_start', parseInt(e.target.value) || 0)} />
                              </div>
                              <div>
                                <label style={label}>Last Preset #</label>
                                <input type="number" style={inputStyle} value={room.preset_end} min={0} max={255}
                                  onChange={e => updateRoom(room.id, 'preset_end', parseInt(e.target.value) || 0)} />
                              </div>
                            </div>
                            <p style={{ margin: '8px 0 0', fontSize: 11, color: C.muted }}>
                              Total: {presetCount} preset{presetCount !== 1 ? 's' : ''}
                            </p>
                          </div>
                        )}

                        {room.scan_mode === 'calibrated' && (
                          <div style={{ background: C.surface, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
                            <p style={{ margin: '0 0 10px', fontSize: 11, color: C.muted }}>
                              Moves camera through a grid of absolute positions using saved bounds.
                            </p>
                            {calGrid ? (
                              <div style={{ fontSize: 11, color: C.muted }}>
                                <div style={{ marginBottom: 2 }}>Estimated frames: <strong style={{ color: C.text }}>{calGrid.total}</strong></div>
                                <div>{calGrid.cols} col{calGrid.cols !== 1 ? 's' : ''} × {calGrid.rows} row{calGrid.rows !== 1 ? 's' : ''} · zoom {calGrid.zoom} · step {calGrid.step}</div>
                                <div>~{Math.max(1, Math.round((calGrid.total * 1.5 + 3) / 60))} min</div>
                              </div>
                            ) : (
                              <p style={{ margin: 0, fontSize: 11, color: C.muted, fontStyle: 'italic' }}>
                                No bounds saved — set in Calibration tab first.
                              </p>
                            )}
                          </div>
                        )}
                      </>
                    )}

                    {/* Generic RTSP fields */}
                    {room.camera_type === 'rtsp' && (
                      <div style={field}>
                        <label style={label}>RTSP URL</label>
                        <input
                          style={inputStyle}
                          value={room.rtsp_url}
                          onChange={e => updateRoom(room.id, 'rtsp_url', e.target.value)}
                          placeholder="rtsp://192.168.1.100:554/stream"
                        />
                        <p style={{ margin: '6px 0 0', fontSize: 11, color: C.muted }}>
                          A single frame will be captured from this feed for each scan.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}

          {cfg.rooms.length === 0 && (
            <p style={{ fontSize: 12, color: C.muted, fontStyle: 'italic', textAlign: 'center', padding: 16 }}>
              No rooms configured. Click + to add one.
            </p>
          )}
        </div>

        {/* Status message */}
        {msg && (
          <div style={{
            background: msg.ok ? C.green + '22' : C.red + '22',
            border: `1px solid ${msg.ok ? C.green : C.red}`,
            borderRadius: 6, padding: '8px 12px', marginBottom: 12,
            fontSize: 13, color: msg.ok ? C.green : C.red,
          }}>
            {msg.text}
          </div>
        )}

        {/* Footer */}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: `1px solid ${C.border}`, color: C.muted, borderRadius: 8, padding: '8px 16px', cursor: 'pointer', fontSize: 13 }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            style={{ background: C.accent, color: '#fff', border: 'none', borderRadius: 8, padding: '8px 18px', cursor: saving ? 'not-allowed' : 'pointer', fontSize: 13, fontWeight: 600, opacity: saving ? 0.7 : 1 }}
          >
            {saving ? 'Saving…' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  )
}
