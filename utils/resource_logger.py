# utils/resource_logger.py
import atexit
import json
import os
import psutil
import signal
import threading
import time
from typing import List, Callable, Any, Optional

# ---- Global (no NVML init at import) ----
try:
    import pynvml  # type: ignore
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False



class ResourceLogger:
    """
    Signal-safe resource logger:
      - instant wake on SIGINT/SIGTERM
      - autosaves partial logs every `flush_every` seconds
      - atomic writes (<outfile>.tmp -> <outfile>)
      - creates parent dirs for <path>/<file>.json
      - multi-GPU autodetect (no args)
      - NVML hardened: timeouts + auto-fallback to CPU-only
    """

    def __init__(
        self,
        interval: float = 120,
        outfile: str = "experiment_log.json",
        job_type: str = "unspecified",
        metadata: dict | None = None,
        flush_every: float = 60,
        # advanced: override per-call NVML timeout (seconds)
        nvml_timeout_s: float = 0.25,
    ):
        self.interval = float(interval)
        self.flush_every = float(flush_every)
        self.outfile = outfile
        self.job_type = job_type
        self.metadata = metadata or {}
        self.nvml_timeout_s = float(nvml_timeout_s)

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._finalized = False

        self.cpu: List[float] = []
        self.ram: List[float] = []

        # GPU state (lazy NVML init)
        self._nvml_ok = False
        self._gpu_disabled = os.environ.get("RESOURCE_LOGGER_DISABLE_GPU", "") == "1"
        self.gpu_indices: List[int] = []
        self._gpu_meta: dict[int, dict] = {}
        self.gpu_util: dict[int, List[float]] = {}
        self.gpu_mem: dict[int, List[float]] = {}

        # Best-effort signal hooks (no heavy work inside handler)
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
        if self._nvml_ok:
            try:
                # Don't block here; ignore errors
                self._nvml_call(pynvml.nvmlShutdown, default=None, timeout=self.nvml_timeout_s)
            except Exception:
                pass

    # ---------- signal handling (no locks, no I/O) ----------

    def _handle_signal(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping ResourceLogger...", flush=True)
        self._stop.set()
        # Let thread exit; atexit/__exit__ does the write.
        try:
            if getattr(self, "thread", None) and self.thread.is_alive():
                self.thread.join(timeout=2.0)
        finally:
            raise SystemExit(128 + int(signum))

    # ---------- NVML hardening ----------

    def _with_timeout(self, fn: Callable[[], Any], default: Any, timeout: float) -> Any:
        """Run fn() in a daemon thread; return default on timeout/error."""
        box = {"res": default, "err": None}
        def run():
            try:
                box["res"] = fn()
            except BaseException as e:
                box["err"] = e
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive() or box["err"] is not None:
            return default
        return box["res"]

    def _nvml_call(self, f: Callable, *args, default: Any = None, timeout: Optional[float] = None):
        if not self._nvml_ok:
            return default
        return self._with_timeout(lambda: f(*args), default, self.nvml_timeout_s if timeout is None else timeout)

    def _ensure_nvml(self):
        if self._gpu_disabled or self._nvml_ok or not _PYNVML_AVAILABLE:
            return
        ok = self._with_timeout(lambda: (pynvml.nvmlInit() or True), False, self.nvml_timeout_s)
        if not ok:
            self._gpu_disabled = True
            return
        # minimal discovery: indices 0..count-1; avoid UUID/PCI lookups
        count = self._with_timeout(lambda: pynvml.nvmlDeviceGetCount(), 0, self.nvml_timeout_s)
        if not count:
            self._gpu_disabled = True
            self._with_timeout(lambda: pynvml.nvmlShutdown(), None, 0.1)
            return
        self.gpu_indices = list(range(int(count)))
        for i in self.gpu_indices:
            self.gpu_util[i] = []
            self.gpu_mem[i] = []
        # metadata is best-effort; time-boxed
        for i in self.gpu_indices:
            name = self._nvml_call(lambda: pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i)),
                                   default=None)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            total = self._nvml_call(lambda: pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total / 1e6,
                                    default=None)
            self._gpu_meta[i] = {"name": name, "mem_total_MB": total}
        self._nvml_ok = True

    # ---------- sampling & writing ----------

    def _loop(self):
        self._last_flush = time.time()
        # Lazy NVML init inside the worker thread
        self._ensure_nvml()
        while not self._stop.is_set():
            self._sample_once()
            if (time.time() - self._last_flush) >= self.flush_every:
                self._write_log()
                self._last_flush = time.time()
            self._stop.wait(self.interval)
        # last sample
        self._sample_once()

    def _sample_once(self):
        # CPU / RAM
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().used / 1e6
        with self._lock:
            self.cpu.append(cpu)
            self.ram.append(ram)

        # GPU
        if not self._nvml_ok or self._gpu_disabled or not self.gpu_indices:
            return

        # time-box each device query; if any timeout occurs, disable GPU sampling
        for i in self.gpu_indices:
            try:
                handle = self._nvml_call(pynvml.nvmlDeviceGetHandleByIndex, i, default=None)
                if handle is None:
                    self._gpu_disabled = True
                    break
                util = self._nvml_call(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                                       default=None)
                mem_used = self._nvml_call(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6,
                                           default=None)
                if util is None or mem_used is None:
                    self._gpu_disabled = True
                    break
                with self._lock:
                    self.gpu_util[i].append(util)
                    self.gpu_mem[i].append(mem_used)
            except Exception:
                self._gpu_disabled = True
                break

    def _stats(self):
        # non-blocking lock to avoid deadlocks during shutdown
        if not self._lock.acquire(timeout=0.2):
            # best-effort snapshot without lock; avoids freeze
            cpu = list(self.cpu)
            ram = list(self.ram)
            gu = {k: list(v) for k, v in self.gpu_util.items()}
            gm = {k: list(v) for k, v in self.gpu_mem.items()}
        else:
            try:
                cpu = list(self.cpu)
                ram = list(self.ram)
                gu = {k: list(v) for k, v in self.gpu_util.items()}
                gm = {k: list(v) for k, v in self.gpu_mem.items()}
            finally:
                self._lock.release()

        def avg(xs): return sum(xs) / len(xs) if xs else None
        def peak(xs): return max(xs) if xs else None

        now = time.time()
        start = getattr(self, "start_time", now)

        gpu_devices = {}
        if self._nvml_ok and not self._gpu_disabled and self.gpu_indices:
            for idx in self.gpu_indices:
                u = gu.get(idx, [])
                m = gm.get(idx, [])
                meta = self._gpu_meta.get(idx, {})
                gpu_devices[str(idx)] = {
                    "name": meta.get("name"),
                    "mem_total_MB": meta.get("mem_total_MB"),
                    "util_avg": avg(u),
                    "util_peak": peak(u),
                    "mem_used_MB_avg": avg(m),
                    "mem_used_MB_peak": peak(m),
                    "samples": len(u) or len(m) or 0,
                }

        util_avgs = [d["util_avg"] for d in gpu_devices.values() if d.get("util_avg") is not None]
        util_peaks = [d["util_peak"] for d in gpu_devices.values() if d.get("util_peak") is not None]
        mem_avgs = [d["mem_used_MB_avg"] for d in gpu_devices.values() if d.get("mem_used_MB_avg") is not None]
        mem_peaks = [d["mem_used_MB_peak"] for d in gpu_devices.values() if d.get("mem_used_MB_peak") is not None]

        return {
            "job_type": self.job_type,
            "start_time": start,
            "end_time": now,
            "wall_clock_sec": now - start,
            "cpu_percent_avg": avg(cpu),
            "cpu_percent_peak": peak(cpu),
            "ram_used_MB_avg": avg(ram),
            "ram_used_MB_peak": peak(ram),
            # overall GPU summaries (None if disabled/unavailable)
            "gpu_util_avg": (sum(util_avgs) / len(util_avgs)) if util_avgs else None,
            "gpu_util_peak": (max(util_peaks) if util_peaks else None),
            "gpu_mem_MB_avg": (sum(mem_avgs) / len(mem_avgs)) if mem_avgs else None,
            "gpu_mem_MB_peak": (max(mem_peaks) if mem_peaks else None),
            "gpu": {
                "enabled": bool(self._nvml_ok and not self._gpu_disabled and self.gpu_indices),
                "indices": self.gpu_indices,
                "devices": gpu_devices,
                "disabled_reason": (
                    "env/timeout/error" if (not self._nvml_ok or self._gpu_disabled) else None
                ),
            },
            "samples": len(cpu),
            "interval_sec": self.interval,
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

    def flush(self):
        self._write_log()
