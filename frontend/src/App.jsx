import { useState, useEffect, useRef } from 'react'
import NeuralCanvas from './components/NeuralCanvas'

const API_BASE = import.meta.env.VITE_API_URL || ''

function App() {
  // Clock state
  const [timeStr, setTimeStr] = useState('--:--:--')
  
  // App UI states
  const [dragOver, setDragOver] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  
  // Scan results state
  const [results, setResults] = useState(null)

  // DOM / stream references
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  // Clock runner
  useEffect(() => {
    function tick() {
      const n = new Date()
      setTimeStr(n.toLocaleTimeString('en-IN', { hour12: false }))
    }
    tick()
    const timer = setInterval(tick, 1000)
    return () => clearInterval(timer)
  }, [])

  // Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  // Camera controls
  const startCamera = async (e) => {
    e.stopPropagation()
    setErrorMsg(null)
    setResults(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setCameraActive(true)
    } catch (err) {
      const m = {
        NotFoundError: 'No camera device found.',
        NotAllowedError: 'Camera permission denied.',
        NotReadableError: 'Camera in use by another app.'
      }
      setErrorMsg('📷 ' + (m[err.name] || 'Camera error: ' + err.message))
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
  }

  const capturePhoto = (e) => {
    e.stopPropagation()
    if (!streamRef.current || !videoRef.current) return
    
    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    
    canvas.toBlob((blob) => {
      if (blob) {
        stopCamera()
        sendFile(new File([blob], 'capture.jpg', { type: 'image/jpeg' }))
      }
    }, 'image/jpeg', 0.95)
  }

  // Upload handlers
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (cameraActive) return
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      sendFile(e.dataTransfer.files[0])
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    if (cameraActive) return
    setDragOver(true)
  }

  const handleDragLeave = () => {
    setDragOver(false)
  }

  const handleZoneClick = () => {
    if (!cameraActive && fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      sendFile(e.target.files[0])
      e.target.value = '' // Clear input so same file can be scanned again
    }
  }

  // Main OCR network request
  const sendFile = async (file) => {
    stopCamera()
    setErrorMsg(null)
    setResults(null)
    
    if (!file.type.match('image.*')) {
      setErrorMsg('⛔ Invalid file. Upload PNG, JPG, or JPEG.')
      return
    }

    setProcessing(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE}/api/v1/scan-pan`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        let detail = 'This image does not match a formal PAN card. Please upload a valid PAN card.'
        try {
          const resJson = await response.json()
          if (resJson.detail) detail = resJson.detail
        } catch (_) {}
        setErrorMsg(detail)
        return
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setErrorMsg('🌐 Network error: ' + err.message)
    } finally {
      setProcessing(false)
    }
  }

  return (
    <>
      {/* Dynamic Background */}
      <NeuralCanvas />
      <video 
        className="bg-video" 
        autoPlay 
        muted 
        loop 
        playsInline
        src={`${API_BASE}/luma.mp4`}
      />
      <div className="grid-overlay" />

      <div className="page-wrap">
        {/* HUD Top Bar */}
        <header className="hud-bar">
          <div className="hud-logo">PAN<span>·</span>AI</div>
          <div className="hud-meta">
            <div className="hud-dot" />
            <span>NEURAL ENGINE v2.0</span>
            <span className="hud-clock">{timeStr}</span>
            <span>OCR·ACTIVE</span>
          </div>
        </header>

        {/* Main Application Card */}
        <main>
          <div className="ai-card">
            <div className="card-strip" />

            <div className="title-section">
              <h1 className="ai-heading" data-text="PAN Card Verification">PAN Card Verification</h1>
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: 8 }}>
                <div className="ai-pill"><span className="ai-pill-dot" />AI ENGINE ACTIVE</div>
              </div>
              <p className="ai-subtitle">
                Upload or drop a clear PAN card image. The AI neural engine will automatically extract and validate all KYC parameters.
              </p>
            </div>

            {/* Drop / Scanner Zone */}
            <div 
              className={`scanner-zone ${dragOver ? 'drag-over' : ''}`} 
              onClick={handleZoneClick}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input 
                type="file" 
                ref={fileInputRef}
                accept="image/png,image/jpeg,image/jpg" 
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
              
              <div className="brk brk-tl" /><div className="brk brk-tr" />
              <div className="brk brk-bl" /><div className="brk brk-br" />
              <div className="zone-beam" />

              {/* Upload Prompt */}
              {!cameraActive && (
                <div style={{ textAlign: 'center', pointerEvents: 'none', padding: '18px 0' }}>
                  <div className="scan-icon">📄</div>
                  <p className="scan-text-main"><span>Click to select</span> or drag and drop</p>
                  <p className="scan-text-sub">PNG · JPG · JPEG &nbsp;|&nbsp; FULL RESOLUTION SUPPORTED</p>
                </div>
              )}

              {/* Camera Container */}
              {cameraActive && (
                <div style={{ width: '100%', padding: '0 4px', zIndex: 10 }}>
                  <video 
                    ref={videoRef}
                    className="cam-feed" 
                    autoPlay 
                    playsInline 
                  />
                  <div className="cam-btns">
                    <button onClick={capturePhoto} className="btn-capture">⚡ Capture &amp; Scan</button>
                    <button onClick={(e) => { e.stopPropagation(); stopCamera(); }} className="btn-cancel">✕ Cancel</button>
                  </div>
                </div>
              )}

              {/* Processing Overlay */}
              {processing && (
                <div className="processing-overlay">
                  <div style={{ position: 'relative', width: 50, height: 50, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div className="proc-ring" />
                    <div className="proc-ring-2" />
                    <span style={{ fontSize: 16, position: 'absolute' }}>🔍</span>
                  </div>
                  <div className="proc-label">NEURAL SCAN IN PROGRESS</div>
                  <div className="proc-steps">
                    <div className="proc-dot" /><div className="proc-dot" /><div className="proc-dot" />
                  </div>
                </div>
              )}
            </div>

            {/* Toggle Camera Button */}
            {!cameraActive && (
              <button className="btn-camera" onClick={startCamera} type="button">
                <span>📹</span> ACTIVATE LIVE CAMERA SCAN
              </button>
            )}

            {/* Error Banner */}
            {errorMsg && (
              <div className="error-banner">
                <div className="err-body">
                  <div className="err-icon">⚠</div>
                  <div style={{ flexGrow: 1, minWidth: 0 }}>
                    <div className="err-label">⬡ PAN Card Not Detected</div>
                    <div className="err-msg">{errorMsg}</div>
                  </div>
                  <button className="err-close" onClick={() => setErrorMsg(null)}>✕</button>
                </div>
                <div className="err-footer">💡 TIP: Good lighting · Full card visible · Not blurry · Try camera scan</div>
              </div>
            )}

            {/* Results Card */}
            {results && (
              <div>
                <div className="res-wrap">
                  <div className="res-header">
                    <span className="res-title">⬡ Extracted KYC Parameters</span>
                    <span className={`res-badge ${results.is_valid_individual ? 'badge-valid' : 'badge-amber'}`}>
                      {results.is_valid_individual ? 'VALID INDIVIDUAL' : 'NOT RECOGNIZED AS INDIVIDUAL'}
                    </span>
                  </div>
                  <div className="field-grid">
                    <div className="field-row">
                      <div className="field-label">Scan Visibility Accuracy</div>
                      <div className="acc-wrap">
                        <span className="acc-val">{(results.visibility_accuracy || 0).toFixed(1)}%</span>
                        <div className="acc-track">
                          <div 
                            className="acc-fill" 
                            style={{ 
                              width: `${results.visibility_accuracy || 0}%`,
                              background: (results.visibility_accuracy || 0) > 75 
                                ? 'linear-gradient(90deg,#10b981,#00f5ff)' 
                                : (results.visibility_accuracy || 0) > 40 
                                ? 'linear-gradient(90deg,#f59e0b,#fbbf24)' 
                                : 'linear-gradient(90deg,#ef4444,#f87171)'
                            }} 
                          />
                        </div>
                      </div>
                    </div>
                    <div className="field-row">
                      <div className="field-label">Permanent Account Number (PAN)</div>
                      <div className="field-value mono">{results.pan_number || 'NOT FOUND'}</div>
                    </div>
                    <div className="field-row">
                      <div className="field-label">Full Legal Name</div>
                      <div className="field-value green">{results.name || 'NOT FOUND'}</div>
                    </div>
                    <div className="field-row">
                      <div className="field-label">Father's / Guardian Name</div>
                      <div className="field-value">{results.father_name || 'NOT FOUND'}</div>
                    </div>
                    <div className="field-row">
                      <div className="field-label">Date of Birth</div>
                      <div className="field-value mono" style={{ color: '#f1f5f9', letterSpacing: '.06em' }}>
                        {results.date_of_birth || 'NOT FOUND'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>

        <footer className="hud-footer">
          POWERED BY OPENCV · FASTAPI · TESSERACT OCR &nbsp;|&nbsp; NO DATA STORED ON SERVER
        </footer>
      </div>
    </>
  )
}

export default App
