export default function Architecture() {
  return (
    <div className="animate-fade-in">
      <h1 className="section-title">Architecture & Theory</h1>
      <p className="section-subtitle">How each parallelism model works under the hood</p>

      {/* Serial */}
      <div className="card arch-card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(99,102,241,0.15)' }}>🔢</div>
          <div className="card-title">1. Serial Baseline</div>
        </div>
        <p className="card-description" style={{ marginBottom: '1rem' }}>
          Single thread tests every candidate sequentially. This is the baseline for
          calculating speedup. All work happens in one process on one core.
        </p>
        <pre>{`
  ┌──────────────────────────────────────────┐
  │              Single Thread               │
  │  hash("a") → hash("b") → ... → hash("z")│
  │  hash("aa") → hash("ab") → ...          │
  └──────────────────────────────────────────┘
  Total time = N × (time per hash)
        `}</pre>
      </div>

      {/* OpenMP */}
      <div className="card arch-card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(34,211,238,0.15)' }}>🧵</div>
          <div className="card-title">2. OpenMP (Shared Memory)</div>
        </div>
        <p className="card-description" style={{ marginBottom: '1rem' }}>
          <strong>#pragma omp parallel for</strong> splits the candidate space across threads.
          Each thread has its own stack but shares the <code>found</code> flag via <code>std::atomic</code>.
          Uses <code>schedule(dynamic)</code> for load balancing.
        </p>
        <pre>{`
  ┌────────────────── Process ──────────────────┐
  │  ┌─Thread 0─┐ ┌─Thread 1─┐ ┌─Thread 2─┐   │
  │  │ hash a-m │ │ hash n-z │ │ hash A-M │   │
  │  └─────┬────┘ └─────┬────┘ └─────┬────┘   │
  │        └──── shared memory ──────┘          │
  │         std::atomic<bool> found             │
  └─────────────────────────────────────────────┘
        `}</pre>
        <div className="chip chip-cyan" style={{ marginTop: '0.5rem' }}>Speedup ≈ #threads (for embarrassingly parallel work)</div>
      </div>

      {/* MPI */}
      <div className="card arch-card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(245,158,11,0.15)' }}>🌐</div>
          <div className="card-title">3. MPI (Distributed Memory)</div>
        </div>
        <p className="card-description" style={{ marginBottom: '1rem' }}>
          Master (rank 0) distributes work chunks to workers via <code>MPI_Send</code>/<code>MPI_Recv</code>.
          Workers process independently and report results. Tag-based protocol:
          TAG_WORK, TAG_FOUND, TAG_DONE, TAG_STOP.
        </p>
        <pre>{`
  ┌─── Rank 0 (Master) ────┐    ┌─── Rank 1 (Worker) ───┐
  │  Distribute work       │───▶│  Process chunk          │
  │  Collect results       │◀───│  Report: FOUND / DONE   │
  │  Send: WORK / STOP     │    └─────────────────────────┘
  └────────────────────────┘    ┌─── Rank 2 (Worker) ───┐
           │                ───▶│  Process chunk          │
           └────────────────◀───│  Report: FOUND / DONE   │
                                └─────────────────────────┘
        `}</pre>
        <div className="chip chip-yellow" style={{ marginTop: '0.5rem' }}>Scales across machines via network</div>
      </div>

      {/* Hybrid MPI+OpenMP */}
      <div className="card arch-card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(168,85,247,0.15)' }}>⚡</div>
          <div className="card-title">4. Hybrid MPI + OpenMP</div>
        </div>
        <p className="card-description" style={{ marginBottom: '1rem' }}>
          Two-layer parallelism: MPI distributes between processes (Layer 1),
          OpenMP creates threads within each worker (Layer 2).
          Uses <code>MPI_Init_thread(MPI_THREAD_FUNNELED)</code> — only master thread calls MPI.
        </p>
        <pre>{`
  ┌────────── Node 0 ──────────┐  ┌────────── Node 1 ──────────┐
  │  Rank 0 (MASTER)           │  │  Rank 1 (WORKER)           │
  │  MPI_Send / MPI_Recv       │  │  ┌─OMP 0─┐ ┌─OMP 1─┐     │
  │  Distributes work          │  │  │ hash  │ │ hash  │     │
  │                            │  │  └───────┘ └───────┘     │
  └────────────────────────────┘  │  ┌─OMP 2─┐ ┌─OMP 3─┐     │
                                  │  │ hash  │ │ hash  │     │
                                  │  └───────┘ └───────┘     │
                                  └────────────────────────────┘
  Total parallelism = MPI_procs × OMP_threads
        `}</pre>
        <div className="chip chip-purple" style={{ marginTop: '0.5rem' }}>Best of both worlds: distributed + shared memory</div>
      </div>

      {/* MPI+CUDA */}
      <div className="card arch-card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(16,185,129,0.15)' }}>🚀</div>
          <div className="card-title">5. Hybrid MPI + CUDA</div>
        </div>
        <p className="card-description" style={{ marginBottom: '1rem' }}>
          MPI distributes across nodes, each worker offloads brute-force to GPU.
          Device-side MD5 (RFC 1321) runs in CUDA kernels — each GPU thread hashes
          one candidate. <code>atomicCAS</code> for thread-safe found flag on device.
        </p>
        <pre>{`
  ┌─── CPU: MPI Worker ──────┐    ┌─── GPU: CUDA Kernel ─────────┐
  │  Receive range from      │    │  Block 0: Thread 0..255       │
  │  master via MPI          │───▶│  Block 1: Thread 0..255       │
  │  Launch kernel           │    │  Block 2: Thread 0..255       │
  │  Copy result back        │◀───│  ...                          │
  └──────────────────────────┘    │  1M+ threads in parallel!     │
                                  │  Each: index→candidate→MD5    │
                                  └───────────────────────────────┘
        `}</pre>
        <div className="chip chip-green" style={{ marginTop: '0.5rem' }}>Massive parallelism: thousands of GPU cores</div>
      </div>

      {/* Amdahl's Law */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '2rem' }}>
        <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>📐 Amdahl's Law</h3>
        <p className="card-description" style={{ marginBottom: '1rem' }}>
          The theoretical maximum speedup is limited by the serial fraction of the program.
        </p>
        <div style={{
          background: 'rgba(15, 23, 42, 0.8)',
          padding: '1.5rem',
          borderRadius: 'var(--radius-md)',
          textAlign: 'center',
          fontFamily: 'var(--font-mono)',
          fontSize: '1.25rem',
          color: 'var(--accent-cyan)',
        }}>
          Speedup(N) = 1 / ( S + (1 - S) / N )
        </div>
        <p className="card-description" style={{ marginTop: '1rem' }}>
          Where <strong>S</strong> = serial fraction, <strong>N</strong> = number of processors.
          For password cracking, S is very small (only I/O is serial), so we get near-linear speedup.
        </p>
      </div>
    </div>
  )
}
