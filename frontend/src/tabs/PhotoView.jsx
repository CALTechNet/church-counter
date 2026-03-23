import React, { useState, useEffect, useRef } from 'react'
import { RefreshCw, ZoomIn, ZoomOut } from 'lucide-react'

export default function PhotoView({ scanStatus }) {
  const [photoUrl, setPhotoUrl] = useState(null)
  const [loading, setLoading] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [dragging, setDragging] = useState(false)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [dragStart, setDragStart] = useState(null)
  const imgRef = useRef(null)

  const loadPhoto = () => {
    setLoading(true)
    const url = `/api/photo/latest?t=${Date.now()}`
    setPhotoUrl(url)
    setLoading(false)
  }

  useEffect(() => { loadPhoto() }, [])

  // Reload photo when scan completes
  useEffect(() => {
    if (scanStatus?.status === 'complete') {
      setTimeout(loadPhoto, 500)
    }
  }, [scanStatus?.status])

  const handleWheel = (e) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? -0.1 : 0.1
    setZoom(z => Math.min(4, Math.max(0.5, z + delta)))
  }

  const handleMouseDown = (e) => {
    setDragging(true)
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y })
  }

  const handleMouseMove = (e) => {
    if (!dragging || !dragStart) return
    setOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
  }

  const handleMouseUp = () => setDragging(false)

  const resetView = () => { setZoom(1); setOffset({ x: 0, y: 0 }) }

  return (
    <div className="p-4 h-full flex flex-col gap-3">
      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-slate-400 text-xs">Stitched panorama with AI detections</span>
        <div className="ml-auto flex items-center gap-2">
          <button onClick={() => setZoom(z => Math.min(4, z + 0.25))}
            className="bg-slate-700 hover:bg-slate-600 p-1.5 rounded text-slate-300">
            <ZoomIn className="w-4 h-4" />
          </button>
          <span className="text-slate-400 text-xs w-10 text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
            className="bg-slate-700 hover:bg-slate-600 p-1.5 rounded text-slate-300">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button onClick={resetView}
            className="bg-slate-700 hover:bg-slate-600 px-2 py-1.5 rounded text-slate-300 text-xs">
            Reset
          </button>
          <button onClick={loadPhoto}
            className="bg-slate-700 hover:bg-slate-600 p-1.5 rounded text-slate-300">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Photo viewport */}
      <div
        className="flex-1 bg-slate-900 rounded-xl border border-slate-700 overflow-hidden relative"
        style={{ minHeight: 400, cursor: dragging ? 'grabbing' : 'grab' }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {photoUrl ? (
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{ transform: `translate(${offset.x}px, ${offset.y}px)` }}
          >
            <img
              ref={imgRef}
              src={photoUrl}
              alt="Room scan"
              style={{ transform: `scale(${zoom})`, transformOrigin: 'center', transition: dragging ? 'none' : 'transform 0.1s' }}
              className="max-w-none"
              onError={() => setPhotoUrl(null)}
            />
          </div>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500">
            <div className="text-center">
              <div className="text-4xl mb-3">📷</div>
              <div className="text-sm">No scan photo yet</div>
              <div className="text-xs text-slate-600 mt-1">Run a scan to capture the room</div>
            </div>
          </div>
        )}

        {/* Scanning overlay */}
        {scanStatus?.status === 'scanning' && (
          <div className="absolute inset-0 bg-slate-900/70 flex items-center justify-center">
            <div className="text-center">
              <div className="text-blue-400 text-lg animate-pulse mb-2">📷 Capturing room...</div>
              <div className="text-slate-300 text-sm">{scanStatus.message}</div>
              <div className="mt-3 h-1.5 w-48 bg-slate-700 rounded-full overflow-hidden mx-auto">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-500"
                  style={{ width: `${scanStatus.progress}%` }}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="text-xs text-slate-500">
        Scroll to zoom · Drag to pan · Green boxes = detected people
      </div>
    </div>
  )
}
