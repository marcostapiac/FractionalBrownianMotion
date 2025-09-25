# utils/resource_logger.py
import atexit
import json
import os
import psutil
import signal
import threading
import time
import weakref
from typing import Callable, Optional, Any, Dict, List

# ---- Optional NVML import ----
try:
    import pynvml  # type: ignore
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False

# ---- Global registry for no-instance runtime updates ----
_ACTIVE_LOGGERS: "weakref.WeakSet[ResourceLogger]"  # type: ignore[name-defined]
_ACTIVE_LOGGERS = weakref.WeakSet()

def set_runtime_global(**kv: Any) -> None:
    """Update runtime vars on all active ResourceLogger instances."""
    for rl in list(_ACTIVE_LOGGERS):
        try:
            rl.set_runtime(**kv)
        except Exception:
            pass


class ResourceLogger:
    """
    Single-GPU resource logger.
      - SIGINT/SIGTERM aware
      - Atomic writes (<outfile>.tmp -> <outfile>)
      - Auto-create parent dirs
      - Single-GPU via NVML (device index configurable)
      - Runtime vars without passing instance:
          * set_runtime_global(epoch=...), or
          * runtime_probe() -> dict, or
          * runtime_watch_path (JSON or key=value)
    """

    def __init__(
        self,
        interval: float = 120,
        outfile: str = "experiment_log.json",
        job_type: str = "unspecified",
        metadata: Optional[Dict[str, Any]] = None,
        flush_every: float = 60,
        gpu_index: int = 0,
        nvml_enable: bool = True,
        runtime_probe: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        runtime_watch_path: Optional[str] = None,
    ):
        self.interval = float(interval)
        self.flush_every = float(flush_every)
        self.outfile = outfile
        self.job_type = job_type
        self.metadata = metadata or {}

        # runtime vars (initial/current)
        self._runtime_initial: Dict[str, Any] = {}
        self._runtime_current: Dict[str, Any] = {}
        self.runtime_probe = runtime_probe
        self.runtime_watch_path = runtime_watch_path

        # thread state
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._finalized = False

        # CPU/RAM samples
        self.cpu: List[float] = []
        self.ram: List[float] = []

        # Single-GPU state
        self._gpu_enabled = bool(nvml_enable) and (_PYNVML_AVAILABLE) and (os.environ.get("RESOURCE_LOGGER_DISABLE_GPU", "") != "1")
        self._gpu_index = int(os.environ.get("RESOURCE_LOGGER_GPU_INDEX", gpu_index))
        self._nvml_inited = False
        self._gpu_handle = None
        self._gpu_name: Optional[str] = None
        self._gpu_mem_total_mb: Optional[float] = None
        self._gpu_util: List[float] = []
        self._gpu_mem_used_mb: List[float] = []

        # Signals
        for sig_name in ("SIGINT", "SIGTERM", "SIGUSR1", "SIGUSR2"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, RuntimeError):
                pass

    # ---------- context manager ----------

    def __enter__(self):
        self.start_time = time.time()
        print(f"ResourceLogger → {os.path.abspath(self.outfile)}", flush=True)
        atexit.register(self._write_log)
        _ACTIVE_LOGGERS.add(self)
        self.thread = threading.Thread(target=self._loop, name="ResourceLogger", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._finalized:
            return
        self._stop.set()
        if getattr(self, "thread", None):
            self.thread.join(timeout=self.interval + 2.0)
        self._write_log()
        self._finalized = True
        try:
            _ACTIVE_LOGGERS.discard(self)
        except Exception:
            pass
        if self._nvml_inited:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ---------- signals ----------

    def _handle_signal(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping ResourceLogger...", flush=True)
        self._stop.set()
        try:
            if getattr(self, "thread", None) and self.thread.is_alive():
                self.thread.join(timeout=2.0)
        finally:
            raise SystemExit(128 + int(signum))

    # ---------- NVML (single GPU) ----------

    def _ensure_nvml(self):
        if not self._gpu_enabled or self._nvml_inited:
            return
        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            # metadata
            name = pynvml.nvmlDeviceGetName(self._gpu_handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            self._gpu_name = str(name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            self._gpu_mem_total_mb = float(mem.total) / 1e6
            self._nvml_inited = True
        except Exception:
            # Permanently disable GPU sampling for this run
            self._gpu_enabled = False
            self._nvml_inited = False
            self._gpu_handle = None

    # ---------- runtime polling ----------

    def _coerce_scalar(self, s: str) -> Any:
        s = s.strip()
        if not s:
            return s
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return s

    def _read_runtime_watch_path(self) -> Dict[str, Any]:
        path = self.runtime_watch_path
        if not path or not os.path.exists(path):
            return {}
        try:
            raw = open(path, "r").read().strip()
        except Exception:
            return {}
        # JSON dict or scalar
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
            return {"value": obj}
        except Exception:
            pass
        # key=value lines
        updates: Dict[str, Any] = {}
        for line in raw.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                updates[k.strip()] = self._coerce_scalar(v)
        if updates:
            return updates
        if raw:
            return {"value": self._coerce_scalar(raw)}
        return {}

    def _poll_runtime(self) -> None:
        updates: Dict[str, Any] = {}
        if self.runtime_probe is not None:
            try:
                probed = self.runtime_probe()
                if isinstance(probed, dict):
                    updates.update(probed)
            except Exception:
                pass
        if self.runtime_watch_path is not None:
            updates.update(self._read_runtime_watch_path())
        if updates:
            self.set_runtime(**updates)

    # ---------- public API ----------

    def set_runtime(self, **kv: Any) -> None:
        """First value per key frozen in `initial`; latest in `current`."""
        with self._lock:
            for k, v in kv.items():
                if k not in self._runtime_initial:
                    self._runtime_initial[k] = v
                self._runtime_current[k] = v

    def flush(self):
        self._write_log()

    # ---------- sampling & writing ----------

    def _loop(self):
        self._last_flush = time.time()
        self._ensure_nvml()
        while not self._stop.is_set():
            self._sample_once()
            if (time.time() - self._last_flush) >= self.flush_every:
                self._write_log()
                self._last_flush = time.time()
            self._stop.wait(self.interval)
        self._sample_once()

    def _sample_once(self):
        # runtime polling (no need to pass the instance)
        self._poll_runtime()

        # CPU / RAM
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().used / 1e6
        with self._lock:
            self.cpu.append(cpu)
            self.ram.append(ram)

        # GPU
        if not self._gpu_enabled or not self._nvml_inited or self._gpu_handle is None:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle).used / 1e6
            with self._lock:
                self._gpu_util.append(float(util))
                self._gpu_mem_used_mb.append(float(mem))
        except Exception:
            # On any failure, disable GPU collection for remainder
            self._gpu_enabled = False

    def _stats(self) -> Dict[str, Any]:
        # non-blocking lock
        if not self._lock.acquire(timeout=0.2):
            cpu = list(self.cpu); ram = list(self.ram)
            gpu_util = list(self._gpu_util)
            gpu_mem_used = list(self._gpu_mem_used_mb)
            runtime_initial = dict(self._runtime_initial)
            runtime_current = dict(self._runtime_current)
        else:
            try:
                cpu = list(self.cpu); ram = list(self.ram)
                gpu_util = list(self._gpu_util)
                gpu_mem_used = list(self._gpu_mem_used_mb)
                runtime_initial = dict(self._runtime_initial)
                runtime_current = dict(self._runtime_current)
            finally:
                self._lock.release()

        def avg(xs): return sum(xs) / len(xs) if xs else None
        def peak(xs): return max(xs) if xs else None

        now = time.time()
        start = getattr(self, "start_time", now)

        return {
            "job_type": self.job_type,
            "start_time": start,
            "end_time": now,
            "wall_clock_sec": now - start,
            "cpu_percent_avg": avg(cpu),
            "cpu_percent_peak": peak(cpu),
            "ram_used_MB_avg": avg(ram),
            "ram_used_MB_peak": peak(ram),
            "gpu_util_avg": avg(gpu_util) if self._gpu_enabled else None,
            "gpu_util_peak": peak(gpu_util) if self._gpu_enabled else None,
            "gpu_mem_MB_avg": avg(gpu_mem_used) if self._gpu_enabled else None,
            "gpu_mem_MB_peak": peak(gpu_mem_used) if self._gpu_enabled else None,
            "gpu": {
                "enabled": bool(self._gpu_enabled and self._nvml_inited and self._gpu_handle is not None),
                "index": self._gpu_index if self._gpu_enabled else None,
                "name": self._gpu_name if self._gpu_enabled else None,
                "mem_total_MB": self._gpu_mem_total_mb if self._gpu_enabled else None,
                "samples": len(gpu_util) if self._gpu_enabled else 0,
            },
            "samples": len(cpu),
            "interval_sec": self.interval,
            "runtime": {
                "initial": runtime_initial or None,
                "current": runtime_current or None,
            },
            "metadata": self.metadata,
        }

    def _ensure_outdir(self):
        d = os.path.dirname(self.outfile)
        if d:
            os.makedirs(d, exist_ok=True)

    def _write_log(self):
        if self._finalized:
            return
        self._ensure_outdir()
        payload = self._stats()
        tmp = f"{self.outfile}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.outfile)
        print(f"Resource log written to {self.outfile}", flush=True)


__all__ = ["ResourceLogger", "set_runtime_global"]
# Wall-clock per epoch/idx from a ResourceLogger JSON (GPU or CPU).
# Use this to build your LaTeX table. No “throughput” — it’s hardware/loader-specific.

import json

# Wall-clock per epoch/idx + CPU/RAM/GPU details from a ResourceLogger JSON (GPU or CPU).

def _load(run):
    if isinstance(run, dict): return run
    with open(run, "r") as f: return json.load(f)

def _to_gib(mb):
    return (float(mb) / 1024.0) if (mb is not None) else None

def wallclock_and_system_metrics(run_json_or_path, inclusive=False):
    r = _load(run_json_or_path)

    # total wall-clock [h]
    wall_s = r.get("wall_clock_sec") or (float(r["end_time"]) - float(r["start_time"]))
    wall_h = wall_s / 3600.0

    # choose unit: epoch preferred, else idx
    rt = r.get("runtime") or {}
    ini, cur = rt.get("initial") or {}, rt.get("current") or {}
    if ("epoch" in ini) or ("epoch" in cur):
        name, a, b = "epoch", ini.get("epoch"), cur.get("epoch")
    else:
        name, a, b = "idx", ini.get("idx"), cur.get("idx")

    span = None
    if (a is not None) and (b is not None):
        span = float(b) - float(a)
        if inclusive:
            span += 1.0
        if span <= 0:
            span = None

    # per-unit wall-clock
    per_h = (wall_h / span) if span else None
    per_min = (per_h * 60.0) if per_h else None

    # CPU
    cpu_avg = r.get("cpu_percent_avg")
    cpu_peak = r.get("cpu_percent_peak")

    # RAM [GiB]
    ram_avg_gib  = _to_gib(r.get("ram_used_MB_avg"))
    ram_peak_gib = _to_gib(r.get("ram_used_MB_peak"))

    # GPU (None for CPU jobs)
    gpu_util_avg   = r.get("gpu_util_avg")
    gpu_util_peak  = r.get("gpu_util_peak")
    gpu_mem_avg_gib  = _to_gib(r.get("gpu_mem_MB_avg"))
    gpu_mem_peak_gib = _to_gib(r.get("gpu_mem_MB_peak"))

    return {
        "progress_name": name,                  # "epoch" or "idx"
        "span_units": span,                     # number of epochs/idx covered
        "wall_clock_h_total": wall_h,           # total hours
        "wall_clock_per_unit_h": per_min/60.,     # minutes per epoch/idx
        "cpu_avg_pct": cpu_avg,                 # CPU average [%]
        "cpu_peak_pct": cpu_peak,               # CPU peak [%]
        "ram_avg_gib": ram_avg_gib,             # RAM average [GiB]
        "ram_peak_gib": ram_peak_gib,           # RAM peak [GiB]
        "gpu_util_avg_pct": gpu_util_avg,       # GPU average util [%] (GPU jobs)
        "gpu_util_peak_pct": gpu_util_peak,     # GPU peak util [%] (GPU jobs)
        "gpu_mem_avg_gib": gpu_mem_avg_gib,     # GPU used mem avg [GiB] (GPU jobs)
        "gpu_mem_peak_gib": gpu_mem_peak_gib,   # GPU used mem peak [GiB] (GPU jobs)
    }

# Example:
