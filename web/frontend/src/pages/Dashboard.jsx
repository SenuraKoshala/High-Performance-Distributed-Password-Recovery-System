import { Link } from 'react-router-dom'
import { useState } from 'react'

/* ── Method data with rich visuals ──────────────────────── */
const methods = [
  {
    id: 'serial', name: 'Serial', icon: '🔢',
    color: '#6366f1',
    description: 'Single-threaded brute-force baseline. Tests every candidate one by one on a single CPU core.',
    parallelism: 'None', tag: 'Baseline',
    speed: 1,
    diagram: [
      '┌────────────────────────────┐',
      '│        Single Core         │',
      '│   ┌──────────────────┐     │',
      '│   │   Thread 0  ▓▓▓  │     │',
      '│   └──────────────────┘     │',
      '│  Tests: a → b → c → d ... │',
      '└────────────────────────────┘',
    ],
    keyPoints: [
      'One thread, one core',
      'Sequential candidate testing',
      'Simplest implementation',
      'Reference for speedup calculation',
    ],
  },
  {
    id: 'openmp', name: 'OpenMP', icon: '🧵',
    color: '#22d3ee',
    description: 'Shared-memory multi-threading using fork-join. All threads share the same RAM and coordinate via atomic flags.',
    parallelism: 'Threads', tag: 'Shared Memory',
    speed: 4,
    diagram: [
      '┌──────────────────────────────────┐',
      '│        Single Machine (RAM)      │',
      '│  ┌────────┐  ┌────────┐          │',
      '│  │ Thr 0 ▓│  │ Thr 1 ▓│          │',
      '│  └────────┘  └────────┘          │',
      '│  ┌────────┐  ┌────────┐          │',
      '│  │ Thr 2 ▓│  │ Thr 3 ▓│          │',
      '│  └────────┘  └────────┘          │',
      '│     shared: found_flag (atomic)  │',
      '└──────────────────────────────────┘',
    ],
    keyPoints: [
      '#pragma omp parallel for',
      'Dynamic scheduling across cores',
      'Atomic flag for early exit',
      'Scales to all cores on one machine',
    ],
  },
  {
    id: 'mpi', name: 'MPI', icon: '🌐',
    color: '#f59e0b',
    description: 'Distributed master–worker model. Rank 0 dispatches work chunks; workers process and report back via message passing.',
    parallelism: 'Processes', tag: 'Distributed',
    speed: 4,
    diagram: [
      '┌──────────┐   MPI_Send    ┌──────────┐',
      '│  Rank 0  │──────────────▶│  Rank 1  │',
      '│ (Master) │   TAG_WORK    │ (Worker) │',
      '│          │◀──────────────│          │',
      '│          │   TAG_DONE    └──────────┘',
      '│          │               ┌──────────┐',
      '│          │──────────────▶│  Rank 2  │',
      '│          │◀──────────────│ (Worker) │',
      '└──────────┘               └──────────┘',
    ],
    keyPoints: [
      'Master dispatches via TAG_WORK',
      'Workers reply TAG_DONE / TAG_FOUND',
      'Isolated process memory (OS separated)',
      'Scales across network nodes',
    ],
  },
  {
    id: 'hybrid', name: 'MPI + OpenMP', icon: '⚡',
    color: '#a855f7',
    description: 'Two-level hybrid: MPI distributes work across processes, OpenMP threads crack in parallel within each process.',
    parallelism: 'Processes × Threads', tag: 'Hybrid',
    speed: 16,
    diagram: [
      '┌──────────┐        ┌────────────────────┐',
      '│  Rank 0  │──MPI──▶│     Rank 1         │',
      '│ (Master) │        │  ┌────┐  ┌────┐    │',
      '│          │        │  │Thr0│  │Thr1│    │',
      '│          │        │  └────┘  └────┘    │',
      '│          │        │  ┌────┐  ┌────┐    │',
      '│          │        │  │Thr2│  │Thr3│    │',
      '│          │        │  └────┘  └────┘    │',
      '│          │        └────────────────────┘',
      '│          │        ┌────────────────────┐',
      '│          │──MPI──▶│     Rank 2         │',
      '│          │        │  ┌────┐  ┌────┐    │',
      '│          │        │  │Thr0│  │Thr1│    │',
      '└──────────┘        └────────────────────┘',
    ],
    keyPoints: [
      'MPI splits by first character (ci)',
      'OpenMP splits second character (c2)',
      'Remaining chars via recursive gen()',
      'MPI_THREAD_FUNNELED for safety',
    ],
  },
  {
    id: 'cuda', name: 'MPI + CUDA', icon: '🚀',
    color: '#10b981',
    description: 'GPU-accelerated cracking. MPI distributes ranges to nodes; each node launches thousands of CUDA threads on the GPU.',
    parallelism: 'Processes × GPU Threads', tag: 'GPU',
    speed: 64,
    diagram: [
      '┌──────────┐        ┌─────────────────────────┐',
      '│  Rank 0  │──MPI──▶│  Rank 1 (Host CPU)      │',
      '│ (Master) │        │    │ cudaMemcpy ▼        │',
      '│          │        │  ┌─────────────────────┐ │',
      '│          │        │  │    GPU Device        │ │',
      '│          │        │  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │ │',
      '│          │        │  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │ │',
      '│          │        │  │  1000s of threads    │ │',
      '│          │        │  └─────────────────────┘ │',
      '└──────────┘        └─────────────────────────┘',
    ],
    keyPoints: [
      'Device-side MD5 in CUDA kernels',
      'Thousands of parallel GPU threads',
      'cudaMemcpy for host↔device transfer',
      'Highest raw throughput',
    ],
  },
]

/* ── Parallelism level bar ──────────────────────────────── */
function SpeedBar({ level, max, color }) {
  const pct = (level / max) * 100
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.5rem',
      marginTop: '0.5rem',
    }}>
      <span style={{
        fontSize: '0.7rem', color: 'var(--text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.05em',
        minWidth: '90px',
      }}>Parallelism</span>
      <div style={{
        flex: 1, height: '6px', borderRadius: '3px',
        background: 'rgba(99,102,241,0.1)',
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${pct}%`, height: '100%', borderRadius: '3px',
          background: `linear-gradient(90deg, ${color}, ${color}aa)`,
          transition: 'width 0.8s ease',
        }} />
      </div>
      <span style={{
        fontSize: '0.7rem', fontFamily: 'var(--font-mono)',
        color, minWidth: '30px', textAlign: 'right',
      }}>{level}×</span>
    </div>
  )
}

/* ── Expandable Architecture Diagram ────────────────────── */
function ArchDiagram({ lines, color, isOpen }) {
  return (
    <div style={{
      maxHeight: isOpen ? '400px' : '0',
      overflow: 'hidden',
      transition: 'max-height 0.4s ease, opacity 0.3s ease',
      opacity: isOpen ? 1 : 0,
    }}>
      <pre style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '0.72rem',
        lineHeight: '1.45',
        color: color,
        background: 'rgba(15, 23, 42, 0.8)',
        padding: '1rem 1.25rem',
        borderRadius: 'var(--radius-md)',
        marginTop: '0.75rem',
        overflowX: 'auto',
        border: `1px solid ${color}20`,
      }}>
        {lines.join('\n')}
      </pre>
    </div>
  )
}

/* ── Key Points List ────────────────────────────────────── */
function KeyPoints({ points, color, isOpen }) {
  return (
    <div style={{
      maxHeight: isOpen ? '200px' : '0',
      overflow: 'hidden',
      transition: 'max-height 0.4s ease 0.1s, opacity 0.3s ease 0.1s',
      opacity: isOpen ? 1 : 0,
    }}>
      <ul style={{
        listStyle: 'none', padding: 0,
        marginTop: '0.75rem',
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem',
      }}>
        {points.map((p, i) => (
          <li key={i} style={{
            fontSize: '0.78rem',
            color: 'var(--text-secondary)',
            display: 'flex', alignItems: 'center', gap: '0.4rem',
          }}>
            <span style={{
              width: '5px', height: '5px', borderRadius: '50%',
              background: color, flexShrink: 0,
            }} />
            {p}
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ── Attack Methods Section ─────────────────────────────── */
const attacks = [
  {
    icon: '🔓', name: 'Brute Force',
    desc: 'Exhaustively tests every combination from charset a-z A-Z 0-9. Best for passwords ≤ 4 characters.',
    detail: 'MPI splits by 1st char → OpenMP splits by 2nd char → gen() recurses remaining positions',
    color: '#22d3ee',
  },
  {
    icon: '📖', name: 'Dictionary',
    desc: 'Hashes words from a wordlist file (e.g. rockyou.txt) and compares against the target hash.',
    detail: 'Master streams 300-word batches → Workers hash with schedule(dynamic, 32)',
    color: '#f59e0b',
  },
  {
    icon: '🔧', name: 'Rule-Based',
    desc: 'Applies ~70 mutation rules per word — leet-speak, suffixes, prefixes, reversal, case changes.',
    detail: 'Same batch dispatch as Dictionary → mutate() generates variants in thread-local memory',
    color: '#a855f7',
  },
]

/* ═══════════════════════════════════════════════════════════
   DASHBOARD COMPONENT
═══════════════════════════════════════════════════════════ */
export default function Dashboard() {
  const [expanded, setExpanded] = useState(null)
  const maxSpeed = Math.max(...methods.map(m => m.speed))

  return (
    <div className="animate-fade-in">
      {/* ── Hero ─────────────────────────────────────────── */}
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

      {/* ── Implementations ──────────────────────────────── */}
      <h2 className="section-title">Implementations</h2>
      <p className="section-subtitle">
        Click any card to reveal its architecture diagram and key design points
      </p>

      <div className="methods-grid" style={{
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
      }}>
        {methods.map((m, i) => {
          const isOpen = expanded === m.id
          return (
            <div
              key={m.id}
              className="card"
              onClick={() => setExpanded(isOpen ? null : m.id)}
              style={{
                animationDelay: `${i * 100}ms`,
                animation: `slideUp 0.5s ease ${i * 100}ms both`,
                borderColor: isOpen ? `${m.color}50` : `${m.color}22`,
                cursor: 'pointer',
                boxShadow: isOpen ? `0 0 25px ${m.color}15` : 'none',
                transition: 'all 0.3s ease',
              }}
            >
              {/* Card Header */}
              <div className="card-header">
                <div className="card-icon" style={{ background: `${m.color}15` }}>
                  {m.icon}
                </div>
                <div style={{ flex: 1 }}>
                  <div className="card-title">{m.name}</div>
                  <div className="card-badge" style={{
                    background: `${m.color}15`,
                    color: m.color,
                    borderColor: `${m.color}30`,
                  }}>{m.tag}</div>
                </div>
                <span style={{
                  color: 'var(--text-muted)', fontSize: '0.8rem',
                  transition: 'transform 0.3s ease',
                  transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                }}>▼</span>
              </div>

              {/* Description */}
              <p className="card-description">{m.description}</p>

              {/* Speed Bar */}
              <SpeedBar level={m.speed} max={maxSpeed} color={m.color} />

              {/* Parallelism chip */}
              <div style={{ marginTop: '0.75rem' }}>
                <span className="chip chip-cyan">{m.parallelism}</span>
              </div>

              {/* Expandable Section */}
              <ArchDiagram lines={m.diagram} color={m.color} isOpen={isOpen} />
              <KeyPoints points={m.keyPoints} color={m.color} isOpen={isOpen} />
            </div>
          )
        })}
      </div>

      {/* ── Attack Methods ───────────────────────────────── */}
      <h2 className="section-title" style={{ marginTop: '1rem' }}>Attack Methods</h2>
      <p className="section-subtitle">
        Each implementation supports all three attack strategies
      </p>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '1.25rem',
        marginBottom: '2rem',
      }}>
        {attacks.map((a, i) => (
          <div key={i} className="card" style={{
            animation: `slideUp 0.5s ease ${(i + 5) * 100}ms both`,
            borderColor: `${a.color}22`,
          }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>{a.icon}</div>
            <div style={{
              fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.5rem',
            }}>{a.name}</div>
            <p className="card-description">{a.desc}</p>
            <div style={{
              marginTop: '0.75rem',
              padding: '0.6rem 0.85rem',
              background: 'rgba(15, 23, 42, 0.6)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              color: a.color,
              borderLeft: `3px solid ${a.color}`,
            }}>
              {a.detail}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
