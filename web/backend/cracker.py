"""
Subprocess runner that spawns C++ password cracker binaries,
feeds them input via stdin, parses their stdout line by line,
and yields progress/result updates.
"""

import asyncio
import os
import re
import time
from typing import AsyncGenerator, Optional
from models import CrackResult, ProgressUpdate, Method, AttackMode


# Path to compiled binaries (relative to project src/)
BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")


def _get_command(method: Method, processes: int) -> list[str]:
    """Build the shell command for each method."""
    bins = {
        Method.SERIAL:  [os.path.join(BIN_DIR, "recover_serial")],
        Method.OPENMP:  [os.path.join(BIN_DIR, "recover_omp")],
        Method.MPI:     ["mpirun", "-np", str(processes), "--oversubscribe",
                         os.path.join(BIN_DIR, "recover_mpi")],
        Method.HYBRID:  ["mpirun", "-np", str(processes), "--oversubscribe",
                         os.path.join(BIN_DIR, "recover_hybrid")],
        Method.CUDA:    ["mpirun", "-np", str(processes), "--oversubscribe",
                         os.path.join(BIN_DIR, "recover_mpi_cuda")],
    }
    return bins[method]


def _parse_progress_line(line: str) -> Optional[dict]:
    """
    Parse the live progress line from C++ output.
    Format: "  Tested:     12345  Speed:    456.7 k/s  Last: abc"
    """
    m = re.search(r'Tested:\s*(\d+)', line)
    speed_m = re.search(r'Speed:\s*([\d.]+)\s*k/s', line)
    last_m = re.search(r'Last:\s*(\S+)', line)

    if m:
        return {
            "tested": int(m.group(1)),
            "speed_kps": float(speed_m.group(1)) if speed_m else 0.0,
            "last_candidate": last_m.group(1) if last_m else "",
        }
    return None


def _parse_result_line(line: str) -> Optional[dict]:
    """
    Parse the result line from C++ output.
    Format: "  ✔  Found!  Password: "abc"  |  Time: 0.31s  |  Tested: 14,000 candidates"
    or:     "  ✘  Not found  |  Time: 4.20s  |  Tested: 1,200,000 candidates"
    """
    if "Found!" in line:
        pw_m = re.search(r'Password:\s*"([^"]*)"', line)
        time_m = re.search(r'Time:\s*([\d.]+)s', line)
        tested_m = re.search(r'Tested:\s*([\d,]+)', line)
        return {
            "found": True,
            "cracked": pw_m.group(1) if pw_m else "",
            "time_secs": float(time_m.group(1)) if time_m else 0.0,
            "tested": int(tested_m.group(1).replace(",", "")) if tested_m else 0,
        }
    elif "Not found" in line:
        time_m = re.search(r'Time:\s*([\d.]+)s', line)
        tested_m = re.search(r'Tested:\s*([\d,]+)', line)
        return {
            "found": False,
            "cracked": "",
            "time_secs": float(time_m.group(1)) if time_m else 0.0,
            "tested": int(tested_m.group(1).replace(",", "")) if tested_m else 0,
        }
    return None


def _method_display_name(method: Method) -> str:
    names = {
        Method.SERIAL: "Serial",
        Method.OPENMP: "OpenMP",
        Method.MPI: "MPI",
        Method.HYBRID: "MPI+OpenMP",
        Method.CUDA: "MPI+CUDA",
    }
    return names.get(method, str(method))


async def run_cracker(
    method: Method,
    password: str,
    attack_mode: AttackMode = AttackMode.ALL,
    threads: int = 4,
    processes: int = 4,
) -> AsyncGenerator[ProgressUpdate, None]:
    """
    Spawn a C++ cracker binary, feed it password + menu choice,
    and yield ProgressUpdate objects as output arrives.
    """
    cmd = _get_command(method, processes)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    display_name = _method_display_name(method)

    start_time = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=BIN_DIR,
        )

        # Send password and attack mode choice via stdin
        stdin_input = f"{password}\n{attack_mode.value}\n"
        proc.stdin.write(stdin_input.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        result_found = False

        result_found = False
        buffer = b""

        # Read stdout in chunks to handle \r and \n seamlessly
        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(1024), timeout=120.0)
            except asyncio.TimeoutError:
                yield ProgressUpdate(type="error", method=display_name)
                break

            if not chunk:
                break

            buffer += chunk
            lines = buffer.replace(b'\r', b'\n').split(b'\n')

            if len(lines) > 1:
                # All except the last element are complete lines
                for line in lines[:-1]:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue

                    # Strip ANSI escape codes
                    clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)

                    # Try parsing as result first (so progress doesn't swallow it)
                    res = _parse_result_line(clean_text)
                    if res and not result_found:
                        result_found = True
                        cr = CrackResult(
                            method=display_name,
                            found=res["found"],
                            cracked=res["cracked"],
                            tested=res["tested"],
                            time_secs=res["time_secs"],
                            speed_kps=res["tested"] / res["time_secs"] / 1000 if res["time_secs"] > 0 else 0,
                            threads=threads if method in (Method.OPENMP, Method.HYBRID) else 1,
                            processes=processes if method in (Method.MPI, Method.HYBRID, Method.CUDA) else 1,
                            accel="CUDA" if method == Method.CUDA else "CPU",
                        )
                        yield ProgressUpdate(
                            type="complete",
                            method=display_name,
                            tested=res["tested"],
                            elapsed=res["time_secs"],
                            result=cr,
                        )
                        continue

                    # Try parsing as progress
                    prog = _parse_progress_line(clean_text)
                    if prog:
                        yield ProgressUpdate(
                            type="progress",
                            method=display_name,
                            tested=prog["tested"],
                            speed_kps=prog["speed_kps"],
                            elapsed=time.time() - start_time,
                            last_candidate=prog["last_candidate"],
                        )
                        continue


                # Keep the last partial line in the buffer
                buffer = lines[-1]

        await proc.wait()

        # If no result was parsed, send a completion with no result
        if not result_found:
            yield ProgressUpdate(
                type="complete",
                method=display_name,
                elapsed=time.time() - start_time,
            )

    except FileNotFoundError:
        yield ProgressUpdate(
            type="error",
            method=display_name,
        )
    except Exception as e:
        yield ProgressUpdate(
            type="error",
            method=display_name,
        )


async def run_benchmark(
    password: str,
    attack_mode: AttackMode = AttackMode.BRUTE_FORCE,
    threads: int = 4,
    processes: int = 4,
) -> list[CrackResult]:
    """
    Run all available methods sequentially and collect results.
    Returns a list of CrackResult for comparison.
    """
    methods = [Method.SERIAL, Method.OPENMP, Method.MPI, Method.HYBRID]
    results = []

    for method in methods:
        final_result = None
        async for update in run_cracker(method, password, attack_mode, threads, processes):
            if update.type == "complete" and update.result:
                final_result = update.result

        if final_result:
            results.append(final_result)
        else:
            results.append(CrackResult(
                method=_method_display_name(method),
                found=False,
                time_secs=0,
                processes=processes if method in (Method.MPI, Method.HYBRID) else 1,
                threads=threads if method in (Method.OPENMP, Method.HYBRID) else 1,
            ))

    return results
