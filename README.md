# High-Performance Distributed Password Recovery System

**EE7218 / EC7207 · High Performance Computing**  
**Group 30**

A cutting-edge password recovery system designed to benchmark and demonstrate the scaling capabilities of High-Performance Computing (HPC) paradigms. The project implements a robust C++ password cracker utilizing multiple parallelization strategies and features a modern, real-time React/FastAPI web dashboard for orchestration and visual benchmarking.

---

## 🏗️ Architecture

The system is divided into two primary layers:

### 1. HPC Core (C++ / CUDA)
Five distinct parallel implementations for cracking MD5 hashes via Brute Force, Dictionary, and Rule-Based attacks:
- **Serial**: Single-threaded baseline implementation.
- **OpenMP**: Shared-memory multi-threading (Fork-Join model).
- **MPI**: Distributed-memory master-worker process architecture.
- **MPI + OpenMP (Hybrid)**: Two-layer parallelization (Nodes × Threads).
- **MPI + CUDA (Hybrid)**: GPU-accelerated brute force distributed across nodes.

### 2. Web Dashboard (React + FastAPI)
A dark-themed, glassmorphism UI that automatically orchestrates the HPC binaries in the background. It utilizes:
- **React + Vite** for the frontend UI and dynamic Chart.js benchmarking.
- **FastAPI** for the backend server.
- **WebSockets** to stream live stdout from the C++ binaries directly to the browser.

---

## ⚙️ Prerequisites & Installation (WSL / Ubuntu)

To run this project, you must be in a Linux/WSL environment to compile the MPI and CUDA binaries.

### 1. Install System Dependencies
```bash
# Install OpenMPI, OpenSSL, and standard build tools
sudo apt update
sudo apt install -y build-essential libssl-dev openmpi-bin libopenmpi-dev
```
*(Note: If you plan to compile the CUDA binary, you must also have the NVIDIA CUDA Toolkit `nvcc` installed).*

### 2. Download a Wordlist
The dictionary and rule-based attacks require a wordlist. We recommend the standard `rockyou.txt` dataset:
```bash
cd src
curl -L -o wordlist.txt https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt
cd ..
```

---

## 🚀 Compiling the HPC Binaries

Navigate to the `src/` directory and compile the 5 implementations:

```bash
cd src

# 1. Serial
g++ -O2 -std=c++17 password_recovery_serial.cpp -lssl -lcrypto -o recover_serial

# 2. OpenMP
g++ -O2 -std=c++17 -fopenmp password_recovery_openmp.cpp -lssl -lcrypto -o recover_omp

# 3. MPI
mpicxx -O2 -std=c++17 password_recovery_mpi.cpp -lssl -lcrypto -o recover_mpi

# 4. MPI + OpenMP
mpicxx -O2 -std=c++17 -fopenmp password_recovery_hybrid_mpi_openmp.cpp -lssl -lcrypto -o recover_hybrid

# 5. MPI + CUDA
nvcc -O2 -std=c++17 password_recovery_hybrid_mpi_cuda.cu -lssl -lcrypto $(mpicxx --showme:compile) $(mpicxx --showme:link) -o recover_mpi_cuda
```

---

## 🌐 Running the Web Dashboard

You can run the web dashboard using two terminals.

### Terminal 1: Python Backend
```bash
cd web/backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies and run
pip install -r requirements.txt
python main.py
```

### Terminal 2: React Frontend
```bash
cd web/frontend

# Install node modules and run the dev server
npm install
npm run dev
```

Once both are running, open your browser and navigate to **`http://localhost:3000`**. 

> **Note:** The UI bypasses the Vite proxy and connects directly to the FastAPI WebSocket on port `8000`. Ensure both servers are running.

---

## 📊 Benchmarking & Amdahl's Law

When testing the system, keep in mind **Parallel Overhead**. If you test a very short password (e.g., `123`), the Serial method will crack it instantly (0.01s). The MPI/OpenMP methods will appear "slower" because the time it takes to initialize the MPI environment and spawn thread pools (~0.2s) vastly outweighs the computation time.

To see the true scaling power of the parallel implementations, run the benchmark against a complex password located deep inside the 14-million word dictionary, or use a complex 5-character string for GPU Brute Forcing!

---

## 🛠️ MPI Architecture & Code Reference (viva Prep)

For the viva presentation, the MPI processes communicate using a master-worker dynamic scheduling architecture. Below is a reference of the core MPI functions utilized in the C++ binaries:

### 1. Environment & Process Setup
*   **`MPI_Init_thread`**: Initializes the MPI execution environment with thread support. In this project, `MPI_THREAD_FUNNELED` is specified to guarantee that only the main thread of each MPI process calls network operations, keeping the OpenMP sub-threads focused entirely on calculation.
*   **`MPI_Comm_rank`**: Acquires the unique ID ("rank") of the current process. **Rank 0** is designated as the **Master**, while all other ranks (**1 to N-1**) act as **Workers**.
*   **`MPI_Comm_size`**: Retrieves the total process count to split search space indices and calculate benchmark speeds.

### 2. Synchronization & Data Sharing
*   **`MPI_Bcast` (Broadcast)**: One-to-all communication. The Master (Rank 0) uses this to broadcast global parameters (target hash, max search length, and attack mode) to all worker nodes in one transaction.
*   **`MPI_Barrier`**: A global barrier synchronization. It blocks processes until all ranks reach the barrier. This ensures that different attack modes do not run concurrently during full comparison benchmarks, preventing CPU thread interference and keeping the measurements highly accurate.

### 3. Dynamic Work Scheduling
*   **`MPI_Send` / `MPI_Recv`**: Core point-to-point blocking transmission calls. The Master uses `MPI_Send` to dispatch chunks of work (first-character indices or dictionary buffers), and workers use them to send back candidate counts (`TAG_DONE`) or the verified password (`TAG_FOUND`).
*   **`MPI_Probe`**: Checks for incoming messages without pulling them out of the network queue. The Master uses it with wildcards (`MPI_ANY_SOURCE`, `MPI_ANY_TAG`) to monitor which worker process finished first and what type of report it sent (cracked or finished), allocating the correct buffer size dynamically before calling receive.

### 4. Termination & Shutdown
*   **`MPI_Abort`**: Instantly terminates all processes in the communicator group in the event of a fatal error (e.g., missing dictionary files or empty password entry).
*   **`MPI_Finalize`**: Safely shuts down the MPI session, frees memory buffers, and prevents active ranks from hanging at process termination.
