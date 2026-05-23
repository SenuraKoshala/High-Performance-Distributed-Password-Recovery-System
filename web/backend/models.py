from pydantic import BaseModel
from typing import Optional
from enum import Enum


class Method(str, Enum):
    SERIAL = "serial"
    OPENMP = "openmp"
    MPI = "mpi"
    HYBRID = "hybrid"
    CUDA = "cuda"


class AttackMode(int, Enum):
    BRUTE_FORCE = 1
    DICTIONARY = 2
    RULE_BASED = 3
    ALL = 4


class CrackRequest(BaseModel):
    password: str
    method: Method = Method.SERIAL
    attack_mode: AttackMode = AttackMode.ALL
    threads: int = 4
    processes: int = 4


class BenchmarkRequest(BaseModel):
    password: str
    attack_mode: AttackMode = AttackMode.BRUTE_FORCE


class CrackResult(BaseModel):
    method: str
    found: bool = False
    cracked: str = ""
    tested: int = 0
    time_secs: float = 0.0
    speed_kps: float = 0.0
    threads: int = 1
    processes: int = 1
    accel: str = "CPU"


class ProgressUpdate(BaseModel):
    type: str = "progress"  # "progress" | "complete" | "error"
    method: str = ""
    tested: int = 0
    speed_kps: float = 0.0
    elapsed: float = 0.0
    last_candidate: str = ""
    result: Optional[CrackResult] = None
