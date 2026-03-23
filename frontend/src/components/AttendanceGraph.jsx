import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'
import { useIsMobile } from '../hooks/useIsMobile.js'

const TOTAL_SEATS = 1075

const SERVICE_COLORS = {
  'Sunday Morning':   '#3b82f6',
  'Sunday Midday':    '#8b5cf6',
  'Wednesday Evening':'#10b981',
  'Manual':           '#f59e0b',
  'Test':             '#f43f5e',
}

const C = {
  bg:      '#0f172a',
  surface: '#1e293b',
  border:  '#334155',
  text:    '#f1f5f9',
  muted:   '#94a3b8',
  accent:  '#3b82f6',
  green:   '#22c55e',
  yellow:  '#eab308',
}

const RANGE_OPTIONS = [
  { value: 'all',     label: 'All Time'    },
  { value: 'week',    label: 'Last Week'   },
  { value: 'month',   label: 'Last Month'  },
  { value: 'quarter', label: 'Last Quarter' },
  { value: 'year',    label: 'Last Year'   },
  { value: 'custom',  label: 'Custom…'     },
]

export default function AttendanceGraph() {
  const isMobile = useIsMobile()
  const [records, setRecords]           = useState([])
  const [tableRecords, setTableRecords] = useState([])
  const [loading, setLoading]           = useState(true)
  const [viewRange, setViewRange]       = useState({ start: 0, end: 100 })
  const [showArchived, setShowArchived] = useState(false)

  // Date range filter
  const [dateRange, setDateRange]     = useState('all')
  const [customStart, setCustomStart] = useState('')
  const [customEnd, setCustomEnd]     = useState('')

  // Fetch non-archived records for chart + stats
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res  = await fetch('/api/attendance')
      const data = await res.json()
      setRecords(Array.isArray(data) ? data : (data.records || []))
    } catch (e) {
      console.error('Failed to fetch attendance', e)
    } finally {
      setLoading(false)
    }
  }, [])

  // Fetch records for the history table (optionally includes archived)
  const fetchTableRecords = useCallback(async (inclArchived) => {
    try {
      const url  = inclArchived ? '/api/attendance?include_archived=true' : '/api/attendance'
      const res  = await fetch(url)
      const data = await res.json()
      setTableRecords(Array.isArray(data) ? data : (data.records || []))
    } catch (e) {
      console.error('Failed to fetch table records', e)
    }
  }, [])

  useEffect(() => {
    fetchData()
    fetchTableRecords(false)
  }, [fetchData, fetchTableRecords])

  const handleRefresh = () => {
    fetchData()
    fetchTableRecords(showArchived)
  }

  const toggleShowArchived = () => {
    const next = !showArchived
    setShowArchived(next)
    fetchTableRecords(next)
  }

  const toggleArchiveRow = async (r) => {
    await fetch(`/api/attendance/${r.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ archived: !r.archived }),
    })
    fetchData()
    fetchTableRecords(showArchived)
  }

  // ── Date range filtering ───────────────────────────────────────────────────
  const now = new Date()
  const cutoffs = {
    week:    new Date(now - 7   * 86400000),
    month:   new Date(now - 30  * 86400000),
    quarter: new Date(now - 90  * 86400000),
    year:    new Date(now - 365 * 86400000),
  }

  const applyDateFilter = (list) => list.filter(r => {
    const ts = new Date(r.timestamp)
    if (dateRange === 'custom') {
      if (customStart && ts < new Date(customStart))              return false
      if (customEnd   && ts > new Date(customEnd + 'T23:59:59')) return false
      return true
    }
    if (dateRange in cutoffs) return ts >= cutoffs[dateRange]
    return true
  })

  const filteredRecords      = applyDateFilter(records)
  const filteredTableRecords = applyDateFilter(tableRecords)

  const exportCsv = () => {
    const header = 'ID,Timestamp,Service,AI Count,Adjustment,Total,Notes,Archived'
    const rows = filteredTableRecords.map(r =>
      [r.id, r.timestamp, r.service_type || 'Manual', r.count, r.manual_add || 0,
       (r.count||0)+(r.manual_add||0), r.notes || '', r.archived ? 'Yes' : 'No'].join(',')
    )
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url
    a.download = `attendance_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ── Chart data ─────────────────────────────────────────────────────────────
  // Total = AI count + any manual adjustment (can be negative for duplicate fixes)
  const total = r => (r.count || 0) + (r.manual_add || 0)

  const chartData = filteredRecords.map(r => ({
    date:      new Date(r.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' }),
    timestamp: r.timestamp,
    service:   r.service_type,
    count:     total(r),
  }))

  const services = [...new Set(filteredRecords.map(r => r.service_type))].filter(Boolean)
  const allDates = [...new Set(chartData.map(d => d.date))]
  const lineData = allDates.map(date => {
    const point = { date }
    for (const svc of services) {
      const rec  = chartData.find(d => d.date === date && d.service === svc)
      point[svc] = rec ? rec.count : null
    }
    return point
  })

  const totalPoints = lineData.length
  const startIdx    = Math.floor(viewRange.start / 100 * totalPoints)
  const endIdx      = Math.ceil(viewRange.end   / 100 * totalPoints)
  const visibleData = lineData.slice(startIdx, endIdx)

  // ── Stats (reflect current filter, non-archived only) ─────────────────────
  const totalScans  = filteredRecords.length
  const avgOccupied = filteredRecords.length > 0
    ? Math.round(filteredRecords.reduce((s, r) => s + total(r), 0) / filteredRecords.length) : 0
  const maxRecord   = filteredRecords.reduce((best, r) => total(r) > total(best || {}) ? r : best, null)

  return (
    <div style={{ padding: isMobile ? 12 : 24, display: 'flex', flexDirection: 'column', gap: isMobile ? 12 : 20, overflowY: 'auto', flex: 1 }}>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: isMobile ? 'repeat(2, 1fr)' : 'repeat(4, 1fr)', gap: isMobile ? 8 : 12 }}>
        <StatCard label="Scans in Range"  value={totalScans} />
        <StatCard label="Avg Attendance"  value={avgOccupied} sub={`${Math.round(avgOccupied / TOTAL_SEATS * 100)}% capacity`} />
        <StatCard label="High in Range"   value={maxRecord ? total(maxRecord) : '—'} sub={maxRecord ? new Date(maxRecord.timestamp).toLocaleDateString() : ''} />
        <StatCard label="Total Seats"     value="1,075" />
      </div>

      {/* Chart */}
      <div style={{ background: C.surface, borderRadius: 12, padding: isMobile ? 12 : 20, border: `1px solid ${C.border}` }}>
        {/* Chart header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: dateRange === 'custom' ? 12 : 16, flexWrap: 'wrap', gap: 8 }}>
          <span style={{ color: C.text, fontWeight: 600, fontSize: 14 }}>Attendance Over Time</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            {/* Date range picker */}
            <select
              value={dateRange}
              onChange={e => { setDateRange(e.target.value); setViewRange({ start: 0, end: 100 }) }}
              style={{ background: '#0f172a', border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '5px 10px', fontSize: 12, cursor: 'pointer' }}
            >
              {RANGE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            <button onClick={handleRefresh}
              style={{ background: '#334155', border: 'none', borderRadius: 6, padding: '6px 10px', color: C.muted, cursor: 'pointer', fontSize: 12 }}>
              ↻ Refresh
            </button>
            <button onClick={exportCsv}
              style={{ background: '#334155', border: 'none', borderRadius: 6, padding: '6px 12px', color: C.muted, cursor: 'pointer', fontSize: 12 }}>
              ↓ Export CSV
            </button>
          </div>
        </div>

        {/* Custom date inputs */}
        {dateRange === 'custom' && (
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <label style={{ color: C.muted, fontSize: 12 }}>From</label>
              <input
                type="date"
                value={customStart}
                onChange={e => setCustomStart(e.target.value)}
                style={{ background: '#0f172a', border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '4px 8px', fontSize: 12 }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <label style={{ color: C.muted, fontSize: 12 }}>To</label>
              <input
                type="date"
                value={customEnd}
                onChange={e => setCustomEnd(e.target.value)}
                style={{ background: '#0f172a', border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '4px 8px', fontSize: 12 }}
              />
            </div>
            {(customStart || customEnd) && (
              <button onClick={() => { setCustomStart(''); setCustomEnd('') }}
                style={{ background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 11 }}>
                Clear
              </button>
            )}
          </div>
        )}

        {loading ? (
          <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}>Loading…</div>
        ) : filteredRecords.length === 0 ? (
          <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted, flexDirection: 'column', gap: 8 }}>
            <div style={{ fontSize: 32 }}>📊</div>
            <div style={{ fontSize: 14 }}>{records.length === 0 ? 'No attendance data yet' : 'No records in selected range'}</div>
            {records.length === 0 && <div style={{ fontSize: 12, color: '#475569' }}>Run your first scan to start tracking</div>}
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={isMobile ? 220 : 300}>
              <LineChart data={visibleData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: C.muted, fontSize: 11 }} tickLine={false} axisLine={{ stroke: C.border }} />
                <YAxis tick={{ fill: C.muted, fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                <Tooltip
                  contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8 }}
                  labelStyle={{ color: C.muted, fontSize: 11 }}
                  itemStyle={{ fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: C.muted }} />
                {services.map(svc => (
                  <Line key={svc} type="monotone" dataKey={svc}
                    stroke={SERVICE_COLORS[svc] || C.muted}
                    strokeWidth={2}
                    dot={{ r: 3, fill: SERVICE_COLORS[svc] || C.muted }}
                    activeDot={{ r: 5 }}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>

            {totalPoints > 10 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 11, color: '#475569', marginBottom: 4 }}>Zoom range</div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input type="range" min="0" max="90" value={viewRange.start}
                    onChange={e => { const v = Number(e.target.value); setViewRange(r => ({ start: Math.min(v, r.end - 10), end: r.end })) }}
                    style={{ flex: 1, accentColor: C.accent }} />
                  <input type="range" min="10" max="100" value={viewRange.end}
                    onChange={e => { const v = Number(e.target.value); setViewRange(r => ({ start: r.start, end: Math.max(v, r.start + 10) })) }}
                    style={{ flex: 1, accentColor: C.accent }} />
                  <button onClick={() => setViewRange({ start: 0, end: 100 })}
                    style={{ background: 'none', border: 'none', color: C.accent, cursor: 'pointer', fontSize: 12, whiteSpace: 'nowrap' }}>
                    Reset
                  </button>
                </div>
                <div style={{ fontSize: 11, color: '#475569', textAlign: 'center', marginTop: 4 }}>
                  Showing {visibleData.length} of {totalPoints} points
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Table */}
      <div style={{ background: C.surface, borderRadius: 12, border: `1px solid ${C.border}`, overflow: 'hidden' }}>
        <div style={{ padding: '12px 20px', borderBottom: `1px solid ${C.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: C.text, fontWeight: 600, fontSize: 14 }}>Scan History</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ color: C.muted, fontSize: 12 }}>{filteredTableRecords.length} records</span>
            <button
              onClick={toggleShowArchived}
              style={{
                background: showArchived ? '#78350f33' : '#334155',
                border: `1px solid ${showArchived ? '#d97706' : C.border}`,
                borderRadius: 6, padding: '4px 10px',
                color: showArchived ? '#d97706' : C.muted,
                cursor: 'pointer', fontSize: 11, fontWeight: 600,
              }}>
              {showArchived ? '🗃 Hide Archived' : '🗃 Show Archived'}
            </button>
          </div>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ color: C.muted, fontSize: 11, borderBottom: `1px solid ${C.border}` }}>
                <th style={{ textAlign: 'left',  padding: '8px 16px' }}>Date & Time</th>
                <th style={{ textAlign: 'left',  padding: '8px 16px' }}>Service</th>
                <th style={{ textAlign: 'right', padding: '8px 16px' }}>Count</th>
                <th style={{ textAlign: 'right', padding: '8px 16px' }}>Capacity</th>
                <th style={{ textAlign: 'right', padding: '8px 16px' }}>%</th>
                <th style={{ textAlign: 'center', padding: '8px 16px' }}></th>
              </tr>
            </thead>
            <tbody>
              {[...filteredTableRecords].reverse().slice(0, 50).map(r => (
                <tr key={r.id} style={{
                  borderBottom: `1px solid ${C.border}33`,
                  background: r.archived ? '#78350f18' : 'transparent',
                  opacity: r.archived ? 0.7 : 1,
                }}>
                  <td style={{ padding: '8px 16px', color: C.muted, fontSize: 12 }}>
                    {new Date(r.timestamp).toLocaleString()}
                    {r.archived && (
                      <span style={{
                        marginLeft: 8, padding: '1px 6px', borderRadius: 3,
                        background: '#78350f44', color: '#d97706',
                        fontSize: 10, fontWeight: 700, verticalAlign: 'middle',
                      }}>ARCHIVED</span>
                    )}
                  </td>
                  <td style={{ padding: '8px 16px' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                      background: (SERVICE_COLORS[r.service_type] || '#475569') + '33',
                      color: SERVICE_COLORS[r.service_type] || C.muted,
                    }}>{r.service_type}</span>
                  </td>
                  <td style={{ padding: '8px 16px', textAlign: 'right', color: C.text, fontWeight: 600 }}>{total(r)}</td>
                  <td style={{ padding: '8px 16px', textAlign: 'right', color: C.muted }}>1,075</td>
                  <td style={{ padding: '8px 16px', textAlign: 'right' }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: total(r) / TOTAL_SEATS > 0.8 ? C.green : total(r) / TOTAL_SEATS > 0.5 ? C.yellow : C.muted }}>
                      {Math.round(total(r) / TOTAL_SEATS * 100)}%
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                    <button
                      onClick={() => toggleArchiveRow(r)}
                      title={r.archived ? 'Unarchive' : 'Archive'}
                      style={{
                        background: 'none',
                        border: `1px solid ${r.archived ? '#d97706' : C.border}`,
                        borderRadius: 4, padding: '2px 8px',
                        color: r.archived ? '#d97706' : C.muted,
                        cursor: 'pointer', fontSize: 11,
                      }}>
                      {r.archived ? 'Unarchive' : 'Archive'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filteredTableRecords.length === 0 && (
            <div style={{ textAlign: 'center', padding: 32, color: C.muted, fontSize: 14 }}>No records in selected range</div>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 12, padding: 16, border: '1px solid #334155' }}>
      <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 24, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#475569', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  )
}
