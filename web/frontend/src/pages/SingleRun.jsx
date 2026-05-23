import { useState, useRef, useEffect } from 'react'

const API_BASE = '/api'

export default function SingleRun() {
  const [password, setPassword] = useState('')
  const [method, setMethod] = useState('serial')
  const [attackMode, setAttackMode] = useState(4)
  const [threads, setThreads] = useState(4)
  const [processes, setProcesses] = useState(4)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const wsRef = useRef(null)

  const handleStart = async () => {
    if (!password.trim()) return
    setRunning(true)
    setProgress(null)
    setResult(null)

    try {
      // Create job via REST
      const res = await fetch(`${API_BASE}/crack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          password, method, attack_mode: attackMode, threads, processes
        }),
      })
      const { job_id } = await res.json()

      // Connect WebSocket for live updates
      const wsUrl = `ws://localhost:8000/ws/crack/${job_id}`
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.type === 'progress') {
          setProgress(data)
        } else if (data.type === 'complete') {
          setResult(data.result)
          setRunning(false)
        } else if (data.type === 'error') {
          setRunning(false)
        }
      }
      ws.onerror = () => setRunning(false)
      ws.onclose = () => setRunning(false)
    } catch (err) {
      setRunning(false)
    }
  }

  const handleStop = () => {
    if (wsRef.current) wsRef.current.close()
    setRunning(false)
  }

  useEffect(() => {
    return () => { if (wsRef.current) wsRef.current.close() }
  }, [])

  const formatNumber = (n) => {
    if (!n) return '0'
    return n.toLocaleString()
  }

  return (
    <div className="animate-fade-in">
      <h1 className="section-title">Single Run</h1>
      <p className="section-subtitle">Choose a method and crack a password with live progress</p>

      <div className="two-col">
        {/* Left: Configuration */}
        <div className="card">
          <h3 style={{ marginBottom: '1.5rem', color: 'var(--accent-cyan)' }}>⚙️ Configuration</h3>

          <div className="form-group">
            <label className="form-label">Password to crack</label>
            <input
              id="password-input"
              type="text"
              className="form-input"
              placeholder="Enter a password..."
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={running}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Method</label>
            <select id="method-select" className="form-select" value={method}
              onChange={(e) => setMethod(e.target.value)} disabled={running}>
              <option value="serial">🔢 Serial (baseline)</option>
              <option value="openmp">🧵 OpenMP (threads)</option>
              <option value="mpi">🌐 MPI (distributed)</option>
              <option value="hybrid">⚡ MPI + OpenMP (hybrid)</option>
              <option value="cuda">🚀 MPI + CUDA (GPU)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Attack Mode</label>
            <select id="attack-select" className="form-select" value={attackMode}
              onChange={(e) => setAttackMode(Number(e.target.value))} disabled={running}>
              <option value={1}>🔓 Brute Force</option>
              <option value={2}>📖 Dictionary</option>
              <option value={3}>🔧 Rule-Based</option>
              <option value={4}>🎯 All Three</option>
            </select>
          </div>

          {(method === 'openmp' || method === 'hybrid') && (
            <div className="form-group">
              <label className="form-label">
                OpenMP Threads: <span className="form-range-value">{threads}</span>
              </label>
              <input type="range" className="form-range" min="1" max="16"
                value={threads} onChange={(e) => setThreads(Number(e.target.value))}
                disabled={running} />
            </div>
          )}

          {(method === 'mpi' || method === 'hybrid' || method === 'cuda') && (
            <div className="form-group">
              <label className="form-label">
                MPI Processes: <span className="form-range-value">{processes}</span>
              </label>
              <input type="range" className="form-range" min="2" max="8"
                value={processes} onChange={(e) => setProcesses(Number(e.target.value))}
                disabled={running} />
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
            <button id="start-btn" className="btn btn-primary" onClick={handleStart}
              disabled={running || !password.trim()}>
              {running ? <><span className="spinner" /> Running...</> : '▶ Start Cracking'}
            </button>
            {running && (
              <button className="btn btn-secondary" onClick={handleStop}>⏹ Stop</button>
            )}
          </div>
        </div>

        {/* Right: Live Progress & Result */}
        <div>
          {/* Progress Panel */}
          {(running || progress) && (
            <div className="progress-panel animate-slide-up">
              <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '0.5rem' }}>
                {running && <span className="animate-pulse">⏳</span>} Live Progress
              </h3>
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: running ? '60%' : '100%' }} />
              </div>
              <div className="progress-stats">
                <div className="stat-item">
                  <div className="stat-value">{formatNumber(progress?.tested || 0)}</div>
                  <div className="stat-label">Candidates Tested</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{(progress?.speed_kps || 0).toFixed(1)}</div>
                  <div className="stat-label">Speed (k/s)</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{(progress?.elapsed || 0).toFixed(1)}s</div>
                  <div className="stat-label">Elapsed</div>
                </div>
              </div>
              {progress?.last_candidate && (
                <div style={{
                  marginTop: '1rem', fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)', fontSize: '0.8rem',
                }}>
                  Last: <span style={{ color: 'var(--text-secondary)' }}>{progress.last_candidate}</span>
                </div>
              )}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className={`result-card ${result.found ? 'success' : 'failure'} animate-slide-up`}
              style={{ marginTop: '1.5rem' }}>
              <div className="result-icon">{result.found ? '✅' : '❌'}</div>
              <h3>{result.found ? 'Password Found!' : 'Not Found'}</h3>
              {result.found && (
                <div className="result-password">"{result.cracked}"</div>
              )}
              <div className="result-meta">
                <span>⏱ {result.time_secs?.toFixed(2)}s</span>
                <span>🔢 {formatNumber(result.tested)} tested</span>
                <span>⚡ {result.speed_kps?.toFixed(1)} k/s</span>
              </div>
            </div>
          )}

          {/* Idle state */}
          {!running && !progress && !result && (
            <div className="card" style={{ textAlign: 'center', padding: '3rem', opacity: 0.6 }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔐</div>
              <p>Configure and start a cracking run to see live progress here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
