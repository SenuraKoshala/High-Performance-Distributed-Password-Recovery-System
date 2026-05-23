import { useState, useRef, useEffect } from 'react'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const METHODS = ['Serial', 'OpenMP', 'MPI', 'MPI+OpenMP']
const COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#a855f7']

export default function Benchmark() {
  const [password, setPassword] = useState('abc')
  const [attackMode, setAttackMode] = useState(1)
  const [threads, setThreads] = useState(4)
  const [processes, setProcesses] = useState(4)
  const [running, setRunning] = useState(false)
  const [currentMethod, setCurrentMethod] = useState('')
  const [results, setResults] = useState(null)
  const wsRef = useRef(null)

  const handleRun = () => {
    if (!password.trim()) return
    setRunning(true)
    setResults(null)
    setCurrentMethod('')

    const wsUrl = `ws://localhost:8000/ws/benchmark`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({ password, attack_mode: attackMode, threads, processes }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'method_start') {
        setCurrentMethod(data.method)
      } else if (data.type === 'benchmark_complete') {
        setResults(data.results)
        setRunning(false)
        setCurrentMethod('')
      }
    }

    ws.onerror = () => setRunning(false)
    ws.onclose = () => setRunning(false)
  }

  useEffect(() => {
    return () => { if (wsRef.current) wsRef.current.close() }
  }, [])

  const formatNumber = (n) => n ? n.toLocaleString() : '0'

  const chartData = results ? {
    labels: results.map(r => r.method),
    datasets: [{
      label: 'Time (seconds)',
      data: results.map(r => r.time_secs),
      backgroundColor: COLORS.slice(0, results.length).map(c => c + '80'),
      borderColor: COLORS.slice(0, results.length),
      borderWidth: 2,
      borderRadius: 8,
    }]
  } : null

  const speedupData = results && results.length > 1 ? {
    labels: results.map(r => r.method),
    datasets: [{
      label: 'Speedup (vs Serial)',
      data: results.map(r => results[0].time_secs > 0 ? results[0].time_secs / (r.time_secs || 1) : 1),
      backgroundColor: COLORS.slice(0, results.length).map(c => c + '80'),
      borderColor: COLORS.slice(0, results.length),
      borderWidth: 2,
      borderRadius: 8,
    }]
  } : null

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: false },
    },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(99,102,241,0.1)' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(99,102,241,0.1)' } },
    },
  }

  return (
    <div className="animate-fade-in">
      <h1 className="section-title">Benchmark Comparison</h1>
      <p className="section-subtitle">Run all methods on the same password and compare performance</p>

      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ margin: 0, flex: 1, minWidth: '200px' }}>
            <label className="form-label">Password</label>
            <input id="bench-password" className="form-input" value={password}
              onChange={e => setPassword(e.target.value)} disabled={running}
              placeholder="Short password for brute force..." />
          </div>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Attack</label>
            <select className="form-select" value={attackMode}
              onChange={e => setAttackMode(Number(e.target.value))} disabled={running}>
              <option value={1}>Brute Force</option>
              <option value={2}>Dictionary</option>
              <option value={3}>Rule-Based</option>
              <option value={4}>All Three</option>
            </select>
          </div>
          <button id="bench-start" className="btn btn-success" onClick={handleRun} disabled={running || !password.trim()}>
            {running ? <><span className="spinner" /> Running {currentMethod}...</> : '🏁 Run Benchmark'}
          </button>
        </div>
      </div>

      {running && (
        <div className="card animate-slide-up" style={{ textAlign: 'center', padding: '3rem' }}>
          <div className="spinner" style={{ margin: '0 auto 1rem', width: 40, height: 40, borderWidth: 3 }} />
          <h3 style={{ color: 'var(--accent-cyan)' }}>Running: {currentMethod || 'Initializing...'}</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Testing Serial → OpenMP → MPI → MPI+OpenMP sequentially
          </p>
        </div>
      )}

      {results && (
        <>
          {/* Results Table */}
          <div className="card animate-slide-up">
            <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>📊 Results</h3>
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Result</th>
                  <th>Time</th>
                  <th>Candidates</th>
                  <th>Speed</th>
                  <th>Speedup</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => {
                  const speedup = results[0].time_secs > 0
                    ? (results[0].time_secs / (r.time_secs || 1)).toFixed(2) : '1.00'
                  return (
                    <tr key={i}>
                      <td style={{ color: COLORS[i], fontWeight: 600 }}>{r.method}</td>
                      <td>{r.found ? <span style={{ color: 'var(--accent-green)' }}>✔ Found</span>
                            : <span style={{ color: 'var(--accent-red)' }}>✘ Not found</span>}</td>
                      <td>{r.time_secs?.toFixed(3)}s</td>
                      <td>{formatNumber(r.tested)}</td>
                      <td>{r.speed_kps?.toFixed(1)} k/s</td>
                      <td style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{speedup}×</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Charts */}
          <div className="two-col" style={{ marginTop: '1.5rem' }}>
            <div className="chart-container animate-slide-up">
              <h4 style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>⏱ Execution Time</h4>
              {chartData && <Bar data={chartData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {display: false}}}} />}
            </div>
            <div className="chart-container animate-slide-up" style={{ animationDelay: '100ms' }}>
              <h4 style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>🚀 Speedup (vs Serial)</h4>
              {speedupData && <Bar data={speedupData} options={chartOptions} />}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
