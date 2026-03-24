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
const input  = { width: '100%', background: C.bg, border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '8px 10px', fontSize: 13, boxSizing: 'border-box', outline: 'none' }
const section      = { marginBottom: 22 }
const sectionTitle = { fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted, marginBottom: 12, paddingBottom: 6, borderBottom: `1px solid ${C.border}` }
const field        = { marginBottom: 12 }

export default function SettingsModal({ onClose, onSave }) {
  const [cfg, setCfg] = useState({
    church_name:  'Lakeshore Church',
    camera_ip:    '10.10.140.140',
    camera_user:  'admin',
    camera_pass:  'admin',
    scan_mode:    'preset',
    preset_start: 100,
    preset_end:   131,
  })
  const [saving, setSaving]       = useState(false)
  const [msg,    setMsg]          = useState(null)
  const [bounds, setBounds]       = useState(null)

  useEffect(() => {
    getSettings()
      .then(s => setCfg(prev => ({ ...prev, ...s })))
      .catch(() => {})
    getCameraBounds()
      .then(b => setBounds(b))
      .catch(() => {})
  }, [])

  const set = (key, val) => setCfg(prev => ({ ...prev, [key]: val }))

  const handleSave = async () => {
    setSaving(true)
    setMsg(null)
    try {
      await saveSettings(cfg)
      setMsg({ text: 'Settings saved!', ok: true })
      setTimeout(() => { onSave?.(cfg); onClose() }, 1000)
    } catch (e) {
      setMsg({ text: `Error: ${e.message}`, ok: false })
    } finally {
      setSaving(false)
    }
  }

  const presetCount = Math.max(0, (cfg.preset_end ?? 131) - (cfg.preset_start ?? 100) + 1)

  const calGrid = (() => {
    const tl = bounds?.top_left
    const br = bounds?.bottom_right
    if (!tl || !br) return null
    const zoom     = Math.max(1, bounds?.zoom || tl?.zoom || 10000)
    const step     = Math.max(1, Math.floor(25 * 10000 / zoom))
    const cols     = Math.max(1, Math.ceil(Math.abs(br.pan  - tl.pan)  / step) + 1)
    const rows     = Math.max(1, Math.ceil(Math.abs(br.tilt - tl.tilt) / step) + 1)
    return { cols, rows, total: cols * rows, zoom, step }
  })()

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
    }}>
      <div style={{
        background: C.surface, borderRadius: 12, border: `1px solid ${C.border}`,
        width: '100%', maxWidth: 480, maxHeight: '90vh', overflowY: 'auto',
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
            <input style={input} value={cfg.church_name} onChange={e => set('church_name', e.target.value)} />
          </div>
        </div>

        {/* Camera */}
        <div style={section}>
          <div style={sectionTitle}>Camera</div>
          <div style={field}>
            <label style={label}>IP Address</label>
            <input style={input} value={cfg.camera_ip} onChange={e => set('camera_ip', e.target.value)} placeholder="10.10.140.140" />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div style={field}>
              <label style={label}>Username</label>
              <input style={input} value={cfg.camera_user} onChange={e => set('camera_user', e.target.value)} />
            </div>
            <div style={field}>
              <label style={label}>Password</label>
              <input type="password" style={input} value={cfg.camera_pass} onChange={e => set('camera_pass', e.target.value)} />
            </div>
          </div>
        </div>

        {/* Scan Mode */}
        <div style={section}>
          <div style={sectionTitle}>Scan Mode</div>
          <div style={field}>
            <label style={label}>Scanning Method</label>
            <select
              style={{ ...input, cursor: 'pointer' }}
              value={cfg.scan_mode}
              onChange={e => set('scan_mode', e.target.value)}
            >
              <option value="preset">Preset Scanning</option>
              <option value="calibrated">Calibrated Scanning</option>
            </select>
          </div>

          {cfg.scan_mode === 'preset' && (
            <div style={{ background: C.bg, borderRadius: 8, padding: 16, border: `1px solid ${C.border}` }}>
              <p style={{ margin: '0 0 12px', fontSize: 12, color: C.muted }}>
                Visits saved camera presets in sequence. Presets must be programmed into the camera in advance.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={label}>First Preset #</label>
                  <input type="number" style={input} value={cfg.preset_start} min={0} max={255}
                    onChange={e => set('preset_start', parseInt(e.target.value) || 0)} />
                </div>
                <div>
                  <label style={label}>Last Preset #</label>
                  <input type="number" style={input} value={cfg.preset_end} min={0} max={255}
                    onChange={e => set('preset_end', parseInt(e.target.value) || 0)} />
                </div>
              </div>
              <p style={{ margin: '10px 0 0', fontSize: 11, color: C.muted }}>
                Total: {presetCount} preset{presetCount !== 1 ? 's' : ''}
              </p>
            </div>
          )}

          {cfg.scan_mode === 'calibrated' && (
            <div style={{ background: C.bg, borderRadius: 8, padding: 16, border: `1px solid ${C.border}` }}>
              <p style={{ margin: '0 0 12px', fontSize: 12, color: C.muted }}>
                Moves the camera left-to-right through a grid of absolute positions using the bounds saved in Calibration. Grid density is calculated from the saved zoom level.
              </p>
              {calGrid ? (
                <div style={{ background: C.surface, borderRadius: 6, padding: '10px 14px', border: `1px solid ${C.border}` }}>
                  <div style={{ fontSize: 13, color: C.text, marginBottom: 4 }}>
                    Estimated frames: <strong>{calGrid.total}</strong>
                  </div>
                  <div style={{ fontSize: 11, color: C.muted }}>
                    {calGrid.cols} col{calGrid.cols !== 1 ? 's' : ''} × {calGrid.rows} row{calGrid.rows !== 1 ? 's' : ''} &nbsp;·&nbsp; zoom {calGrid.zoom} &nbsp;·&nbsp; step {calGrid.step} units
                  </div>
                </div>
              ) : (
                <p style={{ margin: 0, fontSize: 11, color: C.muted, fontStyle: 'italic' }}>
                  No calibration bounds saved — set bounds in the Calibration tab first.
                </p>
              )}
            </div>
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
