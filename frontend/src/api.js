const BASE = ''  // same origin in production; vite proxy in dev

export async function getStatus()      { return _get('/api/status') }
export async function getAttendance()  { return _get('/api/attendance') }
export async function getImage()       { return _get('/api/scan/image') }
export async function getCalibration() { return _get('/api/calibration') }
export async function getCapture()     { return _get('/api/capture') }

export async function triggerScan(serviceType = 'Manual') {
  return _post(`/api/scan/trigger?service_type=${encodeURIComponent(serviceType)}`)
}
export async function saveCalPoint(pt) { return _post('/api/calibration', pt) }
export async function deleteCalPoint(seatId) {
  const r = await fetch(`${BASE}/api/calibration/${seatId}`, { method: 'DELETE' })
  return r.json()
}
export async function clearCalibration() {
  const r = await fetch(`${BASE}/api/calibration`, { method: 'DELETE' })
  return r.json()
}
export async function ptzCommand(action, speed = 10) {
  return _post(`/api/ptz/${action}?speed=${speed}`)
}
export async function getPtzPosition() { return _get('/api/ptz/position') }

async function _get(path) {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}
async function _post(path, body) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}

export function createWebSocket(onMessage) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws`)
  ws.onmessage = e => onMessage(JSON.parse(e.data))
  ws.onerror   = e => console.error('WS error', e)
  return ws
}
