# utils/resource_logger.py
import atexit
import json
import os
import psutil
import signal
import threading
import time

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class ResourceLogger:
    """
    Signal-safe resource logger that:
      - wakes instantly on SIGINT/SIGTERM (no blocking sleep)
      - autosaves partial logs every `flush_every` seconds
      - writes atomically (<outfile>.tmp -> <outfile>)
      - works when outfile is <path>/<filename>.json (creates parent dir)
    """
    def __init__(
        self,
        interval: float = 120,
        outfile: str = "experiment_log.json",
        job_type: str = "unspecified",
        metadata: dict | None = None,
        flush_every: float = 60,
    ):
        """
        interval: seconds between samples
        flush_every: seconds between autosaves (partial logs)
        """
        self.interval = float(interval)
        self.flush_every = float(flush_every)
        self.outfile = outfile
        self.job_type = job_type
        self.metadata = metadata or {}

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._finalized = False

        self.cpu = []
        self.ram = []
        self.gpu_util = []
        self.gpu_mem = []

        # Best-effort signal hooks (main thread only)
        for sig_name in ("SIGINT", "SIGTERM", "SIGUSR1", "SIGUSR2"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, RuntimeError):
                # Not in main thread or not supported platform
                pass

    # ---------- context manager ----------

    def __enter__(self):
        self.start_time = time.time()
        print(f"ResourceLogger → {os.path.abspath(self.outfile)}", flush=True)
        atexit.register(self._write_log)  # best-effort on clean interpreter exits
        self.thread = threading.Thread(
            target=self._loop, name="ResourceLogger", daemon=True
        )
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._finalized:
            return
        self._stop.set()
        if getattr(self, "thread", None):
            # returns quickly thanks to waitable sleep
            self.thread.join(timeout=self.interval + 2.0)
        self._write_log()
        self._finalized = True
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ---------- signal handling ----------

    def _handle_signal(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping ResourceLogger...", flush=True)
        self._stop.set()         # wake the sampling loop immediately
        try:
            self._write_log()    # dump partial log right now
            if getattr(self, "thread", None) and self.thread.is_alive():
                self.thread.join(timeout=2.0)  # short, non-blocking join
        finally:
            raise SystemExit(128 + int(signum))

    # ---------- sampling & writing ----------

    def _loop(self):
        self._last_flush = time.time()
        while not self._stop.is_set():
            self._sample_once()
            # periodic autosave: ensures something is written if killed later
            if (time.time() - self._last_flush) >= self.flush_every:
                self._write_log()
                self._last_flush = time.time()
            # waitable sleep (wakes instantly when _stop is set)
            self._stop.wait(self.interval)
        # take one last sample on the way out
        self._sample_once()

    def _sample_once(self):
        # CPU / RAM
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().used / 1e6  # MB (decimal), to match your original

        # GPU (index 0)
        gu = gm = None
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gu = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gm = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6  # MB
            except Exception:
                # If NVML hiccups, skip this tick
                pass

        with self._lock:
            self.cpu.append(cpu)
            self.ram.append(ram)
            if gu is not None:
                self.gpu_util.append(gu)
            if gm is not None:
                self.gpu_mem.append(gm)

    def _stats(self):
        with self._lock:
            cpu = list(self.cpu)
            ram = list(self.ram)
            gu = list(self.gpu_util)
            gm = list(self.gpu_mem)

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
            "gpu_util_avg": avg(gu),
            "gpu_util_peak": peak(gu),
            "gpu_mem_MB_avg": avg(gm),
            "gpu_mem_MB_peak": peak(gm),
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
        os.replace(tmp, self.outfile)  # atomic in same directory
        print(f"Resource log written to {self.outfile}", flush=True)

    # Optional: manual flush you can call yourself
    def flush(self):
        self._write_log()
