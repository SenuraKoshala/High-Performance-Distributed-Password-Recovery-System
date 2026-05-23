import { Link } from 'react-router-dom'

const methods = [
  {
    id: 'serial', name: 'Serial', icon: '🔢',
    color: '#6366f1',
    description: 'Single-threaded brute-force baseline. Tests every candidate one by one.',
    parallelism: 'None', tag: 'Baseline',
  },
  {
    id: 'openmp', name: 'OpenMP', icon: '🧵',
    color: '#22d3ee',
    description: 'Shared-memory multi-threading. Fork-join model with dynamic scheduling.',
    parallelism: 'Threads', tag: 'Shared Memory',
  },
  {
    id: 'mpi', name: 'MPI', icon: '🌐',
    color: '#f59e0b',
    description: 'Distributed master-worker model with tag-based message passing.',
    parallelism: 'Processes', tag: 'Distributed',
  },
  {
    id: 'hybrid', name: 'MPI + OpenMP', icon: '⚡',
    color: '#a855f7',
    description: 'Two-layer parallelism: MPI distributes between nodes, OpenMP threads within.',
    parallelism: 'Processes × Threads', tag: 'Hybrid',
  },
  {
    id: 'cuda', name: 'MPI + CUDA', icon: '🚀',
    color: '#10b981',
    description: 'GPU-accelerated brute-force with device-side MD5 hashing in CUDA kernels.',
    parallelism: 'Processes × GPU Threads', tag: 'GPU',
  },
]

export default function Dashboard() {
  return (
    <div className="animate-fade-in">
      <div className="hero">
        <h1 className="hero-title">High-Performance Password Recovery</h1>
        <p className="hero-subtitle">
          Comparing 5 parallelization strategies for MD5 hash cracking —
          from serial baseline to GPU-accelerated distributed computing.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <Link to="/run" className="btn btn-primary">▶ Start Cracking</Link>
          <Link to="/benchmark" className="btn btn-secondary">📊 Benchmark All</Link>
        </div>
      </div>

      <h2 className="section-title">Implementations</h2>
      <p className="section-subtitle">Click any card to learn more about the parallelism model</p>

      <div className="methods-grid">
        {methods.map((m, i) => (
          <div
            key={m.id}
            className="card"
            style={{
              animationDelay: `${i * 100}ms`,
              animation: `slideUp 0.5s ease ${i * 100}ms both`,
              borderColor: `${m.color}22`,
            }}
          >
            <div className="card-header">
              <div className="card-icon" style={{ background: `${m.color}15` }}>
                {m.icon}
              </div>
              <div>
                <div className="card-title">{m.name}</div>
                <div className="card-badge" style={{
                  background: `${m.color}15`,
                  color: m.color,
                  borderColor: `${m.color}30`,
                }}>{m.tag}</div>
              </div>
            </div>
            <p className="card-description">{m.description}</p>
            <div style={{ marginTop: '0.75rem' }}>
              <span className="chip chip-cyan">{m.parallelism}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
        <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '0.5rem' }}>
          🎯 Attack Methods Supported
        </h3>
        <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: '2rem' }}>🔓</div>
            <div style={{ fontWeight: 600 }}>Brute Force</div>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>All combinations a-z A-Z 0-9</div>
          </div>
          <div>
            <div style={{ fontSize: '2rem' }}>📖</div>
            <div style={{ fontWeight: 600 }}>Dictionary</div>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Wordlist lookup</div>
          </div>
          <div>
            <div style={{ fontSize: '2rem' }}>🔧</div>
            <div style={{ fontWeight: 600 }}>Rule-Based</div>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>9 mutation rules per word</div>
          </div>
        </div>
      </div>
    </div>
  )
}
