import React, { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { Download, RefreshCw } from 'lucide-react'

const SERVICE_COLORS = {
  'Sunday AM':  '#3b82f6',  // blue
  'Sunday PM':  '#8b5cf6',  // purple
  'Wednesday':  '#10b981',  // green
  'Manual':     '#f59e0b',  // amber
}

export default function Graphs() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const [viewRange, setViewRange] = useState({ start: 0, end: 100 }) // percent of data

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/attendance')
      const data = await res.json()
      setRecords(Array.isArray(data) ? data : (data.records || []))
    } catch (e) {
      console.error('Failed to fetch attendance', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])

  const exportCsv = async () => {
    const res = await fetch('/api/attendance/export')
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `attendance_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Build chart data — one point per scan
  const TOTAL_SEATS = 1075

  const chartData = records.map(r => ({
    date: new Date(r.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' }),
    timestamp: r.timestamp,
    service: r.service_type,
    occupied: r.count,
    total: TOTAL_SEATS,
    percentage: Math.round(r.count / TOTAL_SEATS * 100),
  }))

  // Separate series by service type for the 3-line chart
  const services = [...new Set(records.map(r => r.service_type))].filter(Boolean)

  // For line chart with gaps: build date-keyed object
  const allDates = [...new Set(chartData.map(d => d.date))]
  const lineData = allDates.map(date => {
    const point = { date }
    for (const svc of services) {
      const rec = chartData.find(d => d.date === date && d.service === svc)
      point[svc] = rec ? rec.occupied : null
    }
    return point
  })

  // Slice for zoom
  const totalPoints = lineData.length
  const startIdx = Math.floor(viewRange.start / 100 * totalPoints)
  const endIdx   = Math.ceil(viewRange.end   / 100 * totalPoints)
  const visibleData = lineData.slice(startIdx, endIdx)

  // Stats
  const totalScans = records.length
  const avgOccupied = records.length > 0
    ? Math.round(records.reduce((s, r) => s + r.count, 0) / records.length)
    : 0
  const maxRecord = records.reduce((best, r) => r.count > (best?.count || 0) ? r : best, null)

  return (
    <div className="p-4 flex flex-col gap-5">
      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Total Scans" value={totalScans} />
        <StatCard label="Avg Attendance" value={avgOccupied} sub={`${Math.round(avgOccupied / 1075 * 100)}% capacity`} />
        <StatCard label="All-Time High" value={maxRecord?.count ?? '—'}
          sub={maxRecord ? new Date(maxRecord.timestamp).toLocaleDateString() : ''} />
        <StatCard label="Total Seats" value="1,075" />
      </div>

      {/* Main chart */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-white font-medium text-sm">Attendance Over Time</h2>
          <div className="flex gap-2">
            <button onClick={fetchData}
              className="bg-slate-700 hover:bg-slate-600 p-1.5 rounded text-slate-300">
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
            <button onClick={exportCsv}
              className="flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded text-slate-300 text-xs">
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
          </div>
        </div>

        {loading ? (
          <div className="h-64 flex items-center justify-center text-slate-500">Loading...</div>
        ) : records.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-slate-500">
            <div className="text-center">
              <div className="text-3xl mb-2">📊</div>
              <div className="text-sm">No attendance data yet</div>
              <div className="text-xs text-slate-600 mt-1">Run your first scan to start tracking</div>
            </div>
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={visibleData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: '#334155' }}
                />
                <YAxis
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  labelStyle={{ color: '#94a3b8', fontSize: 11 }}
                  itemStyle={{ fontSize: 12 }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 12, color: '#94a3b8' }}
                />
                {services.map(svc => (
                  <Line
                    key={svc}
                    type="monotone"
                    dataKey={svc}
                    stroke={SERVICE_COLORS[svc] || '#94a3b8'}
                    strokeWidth={2}
                    dot={{ r: 3, fill: SERVICE_COLORS[svc] || '#94a3b8' }}
                    activeDot={{ r: 5 }}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>

            {/* Zoom/scroll slider */}
            {totalPoints > 10 && (
              <div className="mt-4">
                <div className="text-xs text-slate-500 mb-1">Zoom range (drag to scroll)</div>
                <div className="flex gap-2 items-center">
                  <span className="text-xs text-slate-600">All time</span>
                  <input type="range" min="0" max="90" value={viewRange.start}
                    onChange={e => {
                      const v = Number(e.target.value)
                      setViewRange(r => ({ start: Math.min(v, r.end - 10), end: r.end }))
                    }}
                    className="flex-1 accent-blue-500" />
                  <input type="range" min="10" max="100" value={viewRange.end}
                    onChange={e => {
                      const v = Number(e.target.value)
                      setViewRange(r => ({ start: r.start, end: Math.max(v, r.start + 10) }))
                    }}
                    className="flex-1 accent-blue-500" />
                  <button onClick={() => setViewRange({ start: 0, end: 100 })}
                    className="text-xs text-blue-400 hover:text-blue-300 whitespace-nowrap">
                    Reset zoom
                  </button>
                </div>
                <div className="text-xs text-slate-600 text-center mt-1">
                  Showing {visibleData.length} of {totalPoints} data points
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Data table */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
          <h2 className="text-white font-medium text-sm">Scan History</h2>
          <span className="text-slate-500 text-xs">{records.length} records</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 text-xs border-b border-slate-700">
                <th className="text-left px-4 py-2">Date & Time</th>
                <th className="text-left px-4 py-2">Service</th>
                <th className="text-right px-4 py-2">Occupied</th>
                <th className="text-right px-4 py-2">Total</th>
                <th className="text-right px-4 py-2">%</th>
              </tr>
            </thead>
            <tbody>
              {[...records].reverse().slice(0, 50).map(r => (
                <tr key={r.id} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                  <td className="px-4 py-2 text-slate-300 text-xs">
                    {new Date(r.timestamp).toLocaleString()}
                  </td>
                  <td className="px-4 py-2">
                    <span className="px-2 py-0.5 rounded text-xs font-medium"
                      style={{
                        background: (SERVICE_COLORS[r.service_type] || '#475569') + '33',
                        color: SERVICE_COLORS[r.service_type] || '#94a3b8'
                      }}>
                      {r.service_type}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-right text-white font-medium">{r.count}</td>
                  <td className="px-4 py-2 text-right text-slate-400">1,075</td>
                  <td className="px-4 py-2 text-right">
                    <span className={`text-xs font-medium ${
                      r.count / 1075 > 0.8 ? 'text-green-400' :
                      r.count / 1075 > 0.5 ? 'text-yellow-400' : 'text-slate-400'
                    }`}>
                      {Math.round(r.count / 1075 * 100)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {records.length === 0 && (
            <div className="text-center py-8 text-slate-500 text-sm">No records yet</div>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, sub }) {
  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="text-slate-400 text-xs mb-1">{label}</div>
      <div className="text-white text-2xl font-bold">{value}</div>
      {sub && <div className="text-slate-500 text-xs mt-0.5">{sub}</div>}
    </div>
  )
}
