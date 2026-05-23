"""
FastAPI backend for HPC Password Cracker Dashboard.
Provides REST endpoints + WebSocket for live progress streaming.
"""

import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from models import CrackRequest, BenchmarkRequest, CrackResult, Method, AttackMode
from cracker import run_cracker, run_benchmark

app = FastAPI(
    title="HPC Password Cracker API",
    description="Backend for the High-Performance Password Recovery Dashboard",
    version="1.0.0",
)

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs: dict[str, dict] = {}


@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "HPC Password Cracker API is running"}


@app.get("/api/methods")
async def get_methods():
    """Return available cracking methods with metadata."""
    return [
        {
            "id": "serial",
            "name": "Serial",
            "description": "Single-threaded brute-force baseline",
            "parallelism": "None",
            "icon": "🔢",
            "color": "#6366f1",
        },
        {
            "id": "openmp",
            "name": "OpenMP",
            "description": "Shared-memory multi-threading with fork-join model",
            "parallelism": "Threads (shared memory)",
            "icon": "🧵",
            "color": "#22d3ee",
        },
        {
            "id": "mpi",
            "name": "MPI",
            "description": "Distributed master-worker across processes",
            "parallelism": "Processes (distributed memory)",
            "icon": "🌐",
            "color": "#f59e0b",
        },
        {
            "id": "hybrid",
            "name": "MPI + OpenMP",
            "description": "Two-layer: MPI between nodes, OpenMP within each worker",
            "parallelism": "Processes × Threads",
            "icon": "⚡",
            "color": "#a855f7",
        },
        {
            "id": "cuda",
            "name": "MPI + CUDA",
            "description": "GPU-accelerated brute-force with MPI distribution",
            "parallelism": "Processes × GPU threads",
            "icon": "🚀",
            "color": "#10b981",
        },
    ]


@app.post("/api/crack")
async def start_crack(req: CrackRequest):
    """Start a cracking job. Returns job_id for WebSocket progress."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "request": req.model_dump(),
        "result": None,
    }
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get the status/result of a job."""
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]


@app.post("/api/benchmark")
async def start_benchmark(req: BenchmarkRequest):
    """Run all methods and return comparison results."""
    results = await run_benchmark(
        password=req.password,
        attack_mode=req.attack_mode,
    )
    return {
        "password": req.password,
        "results": [r.model_dump() for r in results],
    }


@app.websocket("/ws/crack/{job_id}")
async def websocket_crack(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for live progress updates.
    Client connects after POST /api/crack, receives live updates.
    """
    await websocket.accept()

    if job_id not in jobs:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    job = jobs[job_id]
    req = job["request"]
    job["status"] = "running"

    try:
        method = Method(req["method"])
        async for update in run_cracker(
            method=method,
            password=req["password"],
            attack_mode=req["attack_mode"],
            threads=req["threads"],
            processes=req["processes"],
        ):
            await websocket.send_json(update.model_dump())

            if update.type == "complete":
                job["status"] = "complete"
                if update.result:
                    job["result"] = update.result.model_dump()
            elif update.type == "error":
                job["status"] = "error"

    except WebSocketDisconnect:
        job["status"] = "disconnected"
    except Exception as e:
        job["status"] = "error"
        await websocket.send_json({"type": "error", "message": str(e)})

    await websocket.close()


@app.websocket("/ws/benchmark")
async def websocket_benchmark(websocket: WebSocket):
    """
    WebSocket for live benchmark progress.
    Client sends { password, attack_mode } then receives updates for each method.
    """
    await websocket.accept()

    try:
        config = await websocket.receive_json()
        password = config.get("password", "test")
        attack_mode_val = config.get("attack_mode", 1)
        attack_mode = AttackMode(attack_mode_val)
        threads = config.get("threads", 4)
        processes = config.get("processes", 4)

        methods = [Method.SERIAL, Method.OPENMP, Method.MPI, Method.HYBRID]
        all_results = []

        for method in methods:
            # Notify which method is starting
            await websocket.send_json({
                "type": "method_start",
                "method": method.value,
            })

            final_result = None
            async for update in run_cracker(
                method=method,
                password=password,
                attack_mode=attack_mode,
                threads=threads,
                processes=processes,
            ):
                await websocket.send_json(update.model_dump())
                if update.type == "complete" and update.result:
                    final_result = update.result

            if final_result:
                all_results.append(final_result.model_dump())

        # Send final summary
        await websocket.send_json({
            "type": "benchmark_complete",
            "results": all_results,
        })

    except WebSocketDisconnect:
        pass

    await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
