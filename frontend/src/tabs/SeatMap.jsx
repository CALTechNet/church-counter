import React, { useState, useEffect, useRef } from 'react'

const COLORS = {
  occupied:  '#22c55e',  // green
  empty:     '#ef4444',  // red
  unscanned: '#64748b',  // grey
}

export default function SeatMap({ seatData, scanStatus }) {
  const [svgContent, setSvgContent] = useState(null)
  const [svgSize, setSvgSize] = useState({ width: 800, height: 600 })
  const [tooltip, setTooltip] = useState(null)
  const containerRef = useRef(null)

  useEffect(() => {
    // Load the seat map SVG
    fetch('/api/seatmap')
      .then(r => r.text())
      .then(svg => {
        // Extract viewBox dimensions
        const vbMatch = svg.match(/viewBox="([^"]+)"/)
        if (vbMatch) {
          const parts = vbMatch[1].split(/[\s,]+/).map(Number)
          if (parts.length === 4) {
            setSvgSize({ width: parts[2], height: parts[3] })
          }
        }
        setSvgContent(svg)
      })
      .catch(() => {
        // Use placeholder if no SVG loaded yet
        setSvgContent(null)
      })
  }, [])

  // Build seat lookup map
  const seatMap = {}
  if (seatData?.seats) {
    for (const seat of seatData.seats) {
      seatMap[seat.id] = seat
    }
  }

  const getSeatColor = (seatId) => {
    if (!seatData || !seatData.seats || seatData.seats.length === 0) return COLORS.unscanned
    const seat = seatMap[seatId]
    if (!seat) return COLORS.unscanned
    return seat.occupied ? COLORS.occupied : COLORS.empty
  }

  const totalSeats = seatData?.total || 0
  const occupied   = seatData?.occupied || 0
  const empty      = totalSeats - occupied
  const hasData    = seatData && seatData.seats && seatData.seats.length > 0

  return (
    <div className="p-4 h-full flex flex-col gap-4">
      {/* Summary badges */}
      <div className="flex gap-4 flex-wrap">
        <StatBadge color="#22c55e" label="Occupied" value={occupied} />
        <StatBadge color="#ef4444" label="Empty" value={empty} />
        <StatBadge color="#64748b" label="Total" value={totalSeats || '—'} />
        {hasData && (
          <StatBadge color="#3b82f6" label="Fullness"
            value={`${seatData.percentage}%`} />
        )}
        {seatData?.timestamp && (
          <div className="text-xs text-slate-500 self-center ml-auto">
            Last scan: {new Date(seatData.timestamp).toLocaleString()}
            {seatData.service && ` · ${seatData.service}`}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-slate-400">
        <LegendDot color={COLORS.occupied}  label="Occupied" />
        <LegendDot color={COLORS.empty}     label="Empty" />
        <LegendDot color={COLORS.unscanned} label="Not yet scanned" />
      </div>

      {/* SVG Map */}
      <div ref={containerRef} className="flex-1 bg-slate-800 rounded-xl overflow-hidden relative border border-slate-700">
        {svgContent ? (
          <SeatMapSVG
            svgContent={svgContent}
            svgSize={svgSize}
            seatData={seatData}
            getSeatColor={getSeatColor}
            onSeatHover={setTooltip}
          />
        ) : (
          <PlaceholderMap
            seatData={seatData}
            getSeatColor={getSeatColor}
            onSeatHover={setTooltip}
          />
        )}

        {/* Scanning overlay */}
        {scanStatus?.status === 'scanning' && (
          <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center rounded-xl">
            <div className="text-center">
              <div className="text-blue-400 text-lg mb-2 animate-pulse">🔍 Scanning room...</div>
              <div className="text-slate-300 text-sm">{scanStatus.message}</div>
            </div>
          </div>
        )}

        {/* Tooltip */}
        {tooltip && (
          <div
            className="absolute bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-white pointer-events-none z-10"
            style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}
          >
            <div className="font-medium">{tooltip.id}</div>
            {tooltip.section && <div className="text-slate-400">{tooltip.section} · Row {tooltip.row}</div>}
            <div className={tooltip.occupied ? 'text-green-400' : 'text-red-400'}>
              {tooltip.occupied ? 'Occupied' : 'Empty'}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Render the real SVG with seat dots overlaid ─────────────────────────────

function SeatMapSVG({ svgContent, svgSize, seatData, getSeatColor, onSeatHover }) {
  const seats = seatData?.seats || []

  return (
    <div className="w-full h-full overflow-auto">
      <svg
        viewBox={`0 0 ${svgSize.width} ${svgSize.height}`}
        className="w-full h-full"
        style={{ minHeight: 400 }}
      >
        {/* Base SVG floor plan rendered as foreignObject or image */}
        <image
          href="/api/seatmap"
          x="0" y="0"
          width={svgSize.width}
          height={svgSize.height}
          preserveAspectRatio="xMidYMid meet"
          opacity="0.4"
        />
        {/* Seat dots */}
        {seats.map(seat => (
          <circle
            key={seat.id}
            cx={seat.svg_x}
            cy={seat.svg_y}
            r={4}
            fill={getSeatColor(seat.id)}
            className="seat-dot"
            onMouseEnter={(e) => onSeatHover({
              x: e.clientX, y: e.clientY,
              id: seat.id, occupied: seat.occupied,
              section: seat.section, row: seat.row
            })}
            onMouseLeave={() => onSeatHover(null)}
          />
        ))}
      </svg>
    </div>
  )
}

// ── Placeholder map (fan-shaped grid) when no SVG loaded ────────────────────

function PlaceholderMap({ seatData, getSeatColor, onSeatHover }) {
  // Generate a fan-shaped placeholder seat layout
  // Main floor: 28 rows, curved
  // Balcony: 12 rows, outer arc
  const seats = generatePlaceholderSeats()
  const seatMap = {}
  if (seatData?.seats) {
    for (const s of seatData.seats) seatMap[s.id] = s
  }

  return (
    <div className="w-full h-full overflow-auto p-4">
      <svg viewBox="0 0 900 700" className="w-full" style={{ minHeight: 400 }}>
        {/* Stage */}
        <rect x="300" y="20" width="300" height="60" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="2" />
        <text x="450" y="56" textAnchor="middle" fill="#64748b" fontSize="14">STAGE</text>

        {/* Camera indicator */}
        <circle cx="450" cy="50" r="6" fill="#3b82f6" opacity="0.8" />
        <text x="450" y="42" textAnchor="middle" fill="#3b82f6" fontSize="9">📷</text>

        {/* Seats */}
        {seats.map(seat => {
          const mapped = seatMap[seat.id]
          const color = mapped
            ? (mapped.occupied ? COLORS.occupied : COLORS.empty)
            : COLORS.unscanned
          return (
            <circle
              key={seat.id}
              cx={seat.x}
              cy={seat.y}
              r={seat.balcony ? 3.5 : 4.5}
              fill={color}
              className="seat-dot"
              opacity={seat.balcony ? 0.8 : 1}
              onMouseEnter={(e) => onSeatHover({
                x: e.clientX, y: e.clientY,
                id: seat.id,
                occupied: mapped?.occupied || false,
                section: seat.balcony ? 'Balcony' : 'Main Floor',
                row: seat.row
              })}
              onMouseLeave={() => onSeatHover(null)}
            />
          )
        })}

        {/* Section labels */}
        <text x="150" y="400" textAnchor="middle" fill="#475569" fontSize="11">LEFT</text>
        <text x="450" y="550" textAnchor="middle" fill="#475569" fontSize="11">CENTER</text>
        <text x="750" y="400" textAnchor="middle" fill="#475569" fontSize="11">RIGHT</text>
        <text x="450" y="680" textAnchor="middle" fill="#475569" fontSize="10">BALCONY (rear arc)</text>
      </svg>
      <div className="text-center text-xs text-slate-600 mt-2">
        Placeholder layout — drop your seatmap.svg into /static/ to use real layout
      </div>
    </div>
  )
}

function generatePlaceholderSeats() {
  const seats = []
  const cx = 450, stageBottom = 90

  // Main floor — fan shape, 28 rows
  for (let row = 0; row < 28; row++) {
    const radius = 100 + row * 16
    const angleSpread = 0.55 + row * 0.008
    const seatsInRow = Math.max(8, Math.floor(row * 2.5 + 12))
    for (let s = 0; s < seatsInRow; s++) {
      const angle = -angleSpread / 2 + (s / (seatsInRow - 1)) * angleSpread
      const x = cx + Math.sin(angle) * radius
      const y = stageBottom + Math.cos(angle) * radius * 0.75 + 30
      if (y > 60 && y < 580) {
        seats.push({ id: `MF-${row + 1}-${s + 1}`, x, y, row: row + 1, balcony: false })
      }
    }
  }

  // Balcony — outer arc, 12 rows
  for (let row = 0; row < 12; row++) {
    const radius = 560 + row * 12
    const angleSpread = 0.7
    const seatsInRow = Math.floor(row * 1.5 + 20)
    for (let s = 0; s < seatsInRow; s++) {
      const angle = -angleSpread / 2 + (s / (seatsInRow - 1)) * angleSpread
      const x = cx + Math.sin(angle) * radius
      const y = stageBottom + Math.cos(angle) * radius * 0.5 + 20
      if (x > 30 && x < 870 && y > 80 && y < 690) {
        seats.push({ id: `BAL-${row + 1}-${s + 1}`, x, y, row: row + 1, balcony: true })
      }
    }
  }

  return seats
}

// ── Small components ─────────────────────────────────────────────────────────

function StatBadge({ color, label, value }) {
  return (
    <div className="flex items-center gap-2 bg-slate-800 rounded-lg px-3 py-2 border border-slate-700">
      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-slate-400 text-xs">{label}</span>
      <span className="text-white text-sm font-semibold">{value}</span>
    </div>
  )
}

function LegendDot({ color, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      <span>{label}</span>
    </div>
  )
}
